"""Task-stage choice questions and physical-time evidence, without actuator access."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

from .monitor import IDENTITY, JSONEndpoint, EvidenceScores, visual_frame


ANOMALIES = ('wrong_object', 'placement_error', 'wrong_order', 'collision', 'grasp_failure', 'regression', 'stalled')


def choice(instructions, **criteria):
    return dict(type='choice', instructions=instructions, criteria=criteria)


def subgoal_questions(frame):
    goals = frame['plan_goals']
    if not 1 <= len(goals) <= 3 or frame['subgoal_id'] not in [g['subgoal_id'] for g in goals]:
        raise ValueError('Monitoring requires one to three goals and an active goal')
    active = next(g for g in goals if g['subgoal_id'] == frame['subgoal_id'])
    result = dict(
        observability=choice('Are task-relevant objects visible enough to detect execution anomalies?',
            usable='The target and scene relations are identifiable and comparable.',
            occluded='The task-relevant target or outcome is occluded.',
            ambiguous='Targets are ambiguous or the views cannot be compared reliably.'))
    descriptions = dict(
        wrong_object=f"Is executed manipulation inconsistent with this instruction: {active['instruction']}? Preparation shared by stages is normal.",
        placement_error=('After an executed placement attempt, is a task object left toppled, unsupported or at the wrong destination? '
            'An unattempted object at its initial location is not a placement error. Required outcomes: ' +
            ' '.join(r['completion_check'] for r in frame.get('task_requirements', []))),
        wrong_order='Does the executed behavior violate a prerequisite or required order of the current plan?',
        collision='Is unintended contact or collision visible in executed motion? Ordinary required contact is not a collision.',
        grasp_failure='After an actual grasp/carry attempt, is a miss, slip or drop visible? Approach and an unopened grasp attempt are not failures.',
        regression='Do the current images contradict an Astra-confirmed completed requirement? Compare the completion references.',
        stalled='Do distinct executed endpoints show repeated ineffective attempts or idle behavior while work remains? Required waiting and normal withdrawal are not anomalies.')
    for key, question in descriptions.items():
        result[key] = choice(question, present='The specified anomaly is visible.',
            clear='No such anomaly is visible, including normal behavior or no relevant attempt.',
            insufficient_evidence='The images do not support a reliable determination.')
    return result


def stage_visual_frame(frame):
    value = visual_frame(frame)
    for key in ('plan_goals', 'completion_checks', 'completed_requirements', 'active_goal_index', 'task_requirements'):
        value[key] = deepcopy(frame.get(key, []))
    return value


def decode_answers(raw, questions):
    if not isinstance(raw, dict) or set(raw) != set(questions):
        raise ValueError('Valen must answer every subgoal question')
    values = {}
    for key, question in questions.items():
        answer = raw[key]
        if not isinstance(answer, dict) or answer.get('type') != 'choice':
            raise ValueError('Subgoal monitoring requires choice probabilities')
        probabilities = answer.get('probabilities')
        if not isinstance(probabilities, dict) or set(probabilities) != set(question['criteria']):
            raise ValueError('Subgoal answer candidates differ from the active plan')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or not 0 <= v <= 1 for v in probabilities.values()):
            raise ValueError('Choice probabilities must be finite numbers in [0, 1]')
        if not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-5):
            raise ValueError('Choice probabilities must sum to one')
        values[key] = probabilities['usable' if key == 'observability' else 'present']
    return EvidenceScores(values, raw)


class SubgoalValenMonitor(JSONEndpoint):
    def evaluate(self, frames):
        current = frames[-1]
        questions = subgoal_questions(current)
        context = dict(input_format='robodojo_anomaly_evidence_v2',
            frames=[stage_visual_frame(f) for f in frames],
            references=[stage_visual_frame(f) for f in current.get('reference_frames', [])],
            instruction='Detect anomalies in actual execution. Only Astra decides completion and stage transitions. References are earlier states.')
        content = [dict(type='text', text=json.dumps(context, allow_nan=False))]
        seen = set()
        for frame in [*current.get('reference_frames', []), *frames]:
            if frame['observation_id'] in seen:
                continue
            seen.add(frame['observation_id'])
            for image in frame['images']:
                content.extend([dict(type='text', text=f"Observation {frame['observation_id']}, step {frame['step_id']}: {image['camera']}"),
                    dict(type='image_url', image_url=dict(url=image['path']))])
        result = self.post(dict(state=dict(messages=[dict(role='user', content=content)]), questions=questions))
        if not isinstance(result, dict):
            raise ValueError('Invalid Valen subgoal response')
        return decode_answers(result.get('answers'), questions)


@dataclass(frozen=True)
class SubgoalMonitorConfig:
    high: float = .8
    max_age_s: float = 120.
    max_gap_s: float = 120.
    max_step_gap: int = 25
    history: int = 6
    control_dt: float = .04
    min_evidence_seconds: float = 3.
    uncertainty_seconds: float = 6.
    review_seconds: float = 12.
    review_cooldown_seconds: float = 6.
    acute_collision_threshold: float = .9
    observation_profile: str = 'subgoals'

    def __post_init__(self):
        for name in ('high', 'max_age_s', 'max_gap_s', 'control_dt', 'min_evidence_seconds',
                     'uncertainty_seconds', 'review_seconds', 'review_cooldown_seconds', 'acute_collision_threshold'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
                raise ValueError('Subgoal evidence budgets must be finite and positive')
        if not .5 < self.high <= self.acute_collision_threshold <= 1:
            raise ValueError('Invalid subgoal probability thresholds')
        if self.uncertainty_seconds < self.min_evidence_seconds or self.review_seconds < self.min_evidence_seconds:
            raise ValueError('Review windows must cover the minimum evidence interval')
        if self.review_cooldown_seconds > self.review_seconds:
            raise ValueError('Anomaly cooldown cannot exceed the periodic review interval')
        if type(self.max_step_gap) is not int or not 1 <= self.max_step_gap <= 25:
            raise ValueError('Subgoal chunks must contain 1..25 controls')
        if type(self.history) is not int or not 2 <= self.history <= 100:
            raise ValueError('Subgoal monitoring requires 2..100 historical endpoints')
        if self.observation_profile != 'subgoals':
            raise ValueError('Subgoal monitor profile is fixed')

    @property
    def program_id(self):
        spec = dict(config=asdict(self), questions=subgoal_questions(dict(plan_goals=[dict(
            subgoal_id='g1', instruction='Active source instruction')], subgoal_id='g1')),
            input_format='robodojo_anomaly_evidence_v2', revision=2)
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]


class SubgoalRuntime:
    def __init__(self, config):
        self.config, self.identity, self.last = config, None, None
        self.seen = set()
        self.invalidate()

    def invalidate(self):
        self.uncertain_seconds = 0.
        self.anomaly_seconds = {key: 0. for key in ANOMALIES}

    def update(self, frame, scores, now=None):
        try:
            return self._update(frame, scores, time.time() if now is None else now)
        except Exception:
            self.invalidate()
            raise

    def _update(self, frame, scores, now):
        raw = getattr(scores, 'answers', None)
        questions = subgoal_questions(frame)
        decode_answers(raw, questions)
        config = self.config
        identity = tuple(frame[k] for k in IDENTITY)
        if frame['program_id'] != config.program_id or self.identity is not None and identity != self.identity:
            raise ValueError('Wrong subgoal monitor generation')
        step, capture = frame['step_id'], frame['captured_at']
        start = frame.get('executed_start_step')
        if (type(step) is not int or type(start) is not int or not 0 <= start < step
                or not 0 < step-start <= config.max_step_gap):
            raise ValueError('Expected a bounded executed interval')
        if frame['observation_id'] in self.seen or self.last and step <= self.last['step_id']:
            raise ValueError('Repeated evidence cannot trigger a review')
        if not math.isfinite(capture) or not 0 <= now-capture <= config.max_age_s:
            raise ValueError('Stale subgoal evidence')
        if self.last and (capture <= self.last['captured_at'] or start < self.last['step_id']):
            raise ValueError('Non-increasing or overlapping subgoal evidence')
        gap = bool(self.last and (start != self.last['step_id'] or capture-self.last['captured_at'] > config.max_gap_s))
        if gap:
            self.invalidate()
        self.identity, self.last = identity, deepcopy(frame)
        self.seen.add(frame['observation_id'])
        eligible = not frame.get('inspection_only') and not frame.get('context_only')
        dt = (step-start)*config.control_dt if eligible else 0.
        def selected(key):
            probabilities = raw[key]['probabilities']
            best = max(probabilities, key=probabilities.get)
            return best if probabilities[best] >= config.high else 'insufficient_evidence'
        labels = {key: selected(key) for key in questions}
        usable = labels['observability'] == 'usable'
        if eligible:
            for key in ANOMALIES:
                present = labels[key] == 'present' and (usable or key == 'collision')
                self.anomaly_seconds[key] = self.anomaly_seconds[key]+dt if present else 0.
            # Diffuse candidate probabilities do not repeatedly summon Astra.
            obscured = labels['observability'] in ('occluded', 'ambiguous')
            self.uncertain_seconds = self.uncertain_seconds+dt if obscured else 0.
        acute = eligible and raw['collision']['probabilities']['present'] >= config.acute_collision_threshold
        crossed = [key for key, seconds in self.anomaly_seconds.items() if seconds+1e-9 >= config.min_evidence_seconds]
        reason = ('acute_collision' if acute else 'completed_result_lost' if 'regression' in crossed else
                  'subgoal_anomaly' if crossed else
                  'persistent_occlusion' if self.uncertain_seconds+1e-9 >= config.uncertainty_seconds else None)
        return dict(**{k: frame[k] for k in IDENTITY}, step_id=step, observation_id=frame['observation_id'],
            captured_at=capture, gap_reset=gap, scores=dict(scores), raw_answers=deepcopy(raw), labels=labels,
            violations=[], state='EVIDENCE', review_requested=eligible and reason is not None,
            review_reason=reason, interrupt_requested=acute,
            evidence=dict(anomaly_seconds=dict(self.anomaly_seconds), uncertain_seconds=self.uncertain_seconds,
                interval_seconds=dt, eligible=eligible, crossed=crossed, confirmed_failure=False))
