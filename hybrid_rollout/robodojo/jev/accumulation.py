"""Leaky visual evidence and persistent-uncertainty requests for Astra review.

Scores are evidence heuristics, not calibrated joint probabilities. Each fresh
executed endpoint contributes once; static refreshes never advance either budget.
"""
from dataclasses import dataclass
import math
import time

from .monitor import CONSTRAINTS, IDENTITY, ConstraintRuntime, MonitorConfig
from .runner import ReplanScheduler


@dataclass(frozen=True)
class AccumulationConfig(MonitorConfig):
    accumulation_revision: str = 'leaky_evidence_v1'
    evidence_decay: float = .9
    evidence_baseline: float = .2
    evidence_threshold: float = 1.5
    min_observations: int = 3
    uncertainty_observations: int = 12

    def __post_init__(self):
        super().__post_init__()
        if self.observation_profile != 'constraints':
            raise ValueError('Accumulation requires the constraints observation profile')
        for value in (self.evidence_decay, self.evidence_baseline, self.evidence_threshold):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Accumulation parameters must be finite numbers')
        if not 0 <= self.evidence_decay < 1 or not 0 <= self.evidence_baseline < 1 or self.evidence_threshold <= 0:
            raise ValueError('Invalid accumulation decay, baseline or threshold')
        for value in (self.min_observations, self.uncertainty_observations):
            if type(value) is not int or value < 2:
                raise ValueError('Review requires at least two distinct executed observations')
        if self.uncertainty_observations < self.min_observations:
            raise ValueError('Uncertainty budget must cover the minimum evidence window')


class AccumulatingRuntime(ConstraintRuntime):
    def __init__(self, config):
        super().__init__(config)
        self.invalidate()

    def invalidate(self):
        super().invalidate()
        self.evidence = {key: 0. for key in CONSTRAINTS}
        self.observations = self.uncertain_observations = 0

    def _update(self, frame, scores, now):
        # Reuse identity, duplicate, freshness and gap validation. Its discrete
        # confirmation votes remain diagnostics only, not intervention triggers.
        report = super()._update(frame, scores, now)
        config = self.config
        uncertain = any(c['status'] == 'unknown' for c in report['checks'])
        clear = all(c['status'] in ('clear', 'not_applicable') for c in report['checks'])
        eligible = not frame.get('inspection_only') and not frame.get('context_only')
        increments = {}
        if eligible:
            self.observations += 1
            self.uncertain_observations = self.uncertain_observations + 1 if uncertain else 0
            for key, (guard, _) in CONSTRAINTS.items():
                weight = scores[guard] if guard else 1.
                increments[key] = weight * (scores[key] - config.evidence_baseline)
                self.evidence[key] = max(0., config.evidence_decay * self.evidence[key] + increments[key])
        crossed = [key for key, value in self.evidence.items() if value >= config.evidence_threshold]
        ready = eligible and self.observations >= config.min_observations
        reason = ('accumulated_risk' if ready and crossed else
                  'persistent_uncertainty' if ready and self.uncertain_observations >= config.uncertainty_observations
                  else None)
        for check in report['checks']:
            check['legacy_status'] = check['status']
            if check['status'] == 'violated':
                check['status'] = 'suspected'
        report.update(violations=[], state='RUNNING' if clear else 'UNCERTAIN',
            review_requested=reason is not None, review_reason=reason,
            accumulation=dict(revision=config.accumulation_revision, eligible=eligible,
                observations=self.observations, uncertain_observations=self.uncertain_observations,
                uncertainty_limit=config.uncertainty_observations,
                scores=dict(self.evidence), increments=increments,
                threshold=config.evidence_threshold, crossed=crossed,
                decay=config.evidence_decay, baseline=config.evidence_baseline,
                confirmed_failure=False))
        return report


class AccumulatingScheduler(ReplanScheduler):
    """Continue bounded controls until evidence requests review; never pause on UNKNOWN alone."""
    progress_review_enabled = True

    def __init__(self, *args, **kwargs):
        kwargs.update(uncertainty_policy='continue', include_initial_context=True)
        super().__init__(*args, **kwargs)

    def progress_review(self, feedback):
        return feedback if feedback.get('review_requested') else None

    def recovery_context(self):
        return dict(review_source='temporal_accumulator',
                    review_reason=self.latest_feedback['review_reason'],
                    accumulated_evidence=self.latest_feedback['accumulation'])

    def validate_recovery_scene(self, scene):
        if time.time() - scene['captured_at'] <= self.monitor.config.max_age_s:
            return self.fresh(scene)
        # RoboDojo advances physics only on the owner's chunk_step calls. A
        # slow planner may outlive wall-clock freshness while the scene is held.
        # Obtain a real new observation and require identical measured state.
        # Native refresh renders without stepping physics. RTX sampling can
        # change pixels even in a held scene, so image byte equality is invalid.
        # Never rewrite timestamps or accept changed physics with an old plan.
        fresh = self.capture_scene()
        keys = (*IDENTITY, 'step_id', 'current_state', 'current_eef', 'task', 'subgoal')
        unchanged = all(fresh[key] == scene[key] for key in keys)
        if not unchanged:
            raise ValueError('Paused scene changed during Astra review; old decision rejected')
        self.event('planner_scene_revalidated', previous_observation_id=scene['observation_id'],
                   frame=fresh, adds_accumulation_vote=False)
