"""RoboDojo physical-time evidence and bounded, nonblocking Astra review."""
from copy import deepcopy
from dataclasses import dataclass
import math
import time
import uuid

from ..io import write_json
from .accumulation import AccumulationConfig, AccumulatingRuntime, AccumulatingScheduler
from .monitor import CONSTRAINTS, IDENTITY, ConstraintRuntime, PendingCall


@dataclass(frozen=True)
class TemporalConfig(AccumulationConfig):
    accumulation_revision: str = 'physical_time_v2'
    history: int = 6
    max_step_gap: int = 25
    evidence_baseline: float = .5
    evidence_threshold: float = .6  # probability-seconds after baseline removal
    evidence_decay: float = .85    # retention per simulated second
    control_dt: float = .04
    min_evidence_seconds: float = 3.
    uncertainty_seconds: float = 6.
    acute_collision_threshold: float = .9

    def __post_init__(self):
        super().__post_init__()
        for value in (self.control_dt,self.min_evidence_seconds,self.uncertainty_seconds,self.acute_collision_threshold):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Temporal limits must be finite and positive')
        if self.evidence_decay <= 0 or self.uncertainty_seconds < self.min_evidence_seconds:
            raise ValueError('Temporal decay must be positive and uncertainty window must cover minimum evidence')
        if not self.high <= self.acute_collision_threshold <= 1:
            raise ValueError('Acute collision threshold must be in [high, 1]')


class TemporalRuntime(AccumulatingRuntime):
    def invalidate(self):
        super().invalidate()
        self.evidence_seconds = 0.
        self.uncertain_seconds = {k:0. for k in CONSTRAINTS}

    def acknowledge_review(self):
        # Retain endpoint identity and time validation, and all visual history.
        self.evidence = {k:0. for k in CONSTRAINTS}
        self.evidence_seconds = 0.
        self.uncertain_seconds = {k:0. for k in CONSTRAINTS}
        self.observations = 0

    def _update(self, frame, scores, now):
        previous = self.last
        report = ConstraintRuntime._update(self,frame,scores,now)
        config = self.config
        start = frame.get('executed_start_step',previous['step_id'] if previous else 0)
        span = frame['step_id']-start
        if type(start) is not int or not 0 < span <= config.max_step_gap:
            raise ValueError('Executed interval must be bounded and positive')
        eligible = not frame.get('inspection_only') and not frame.get('context_only')
        dt = span*config.control_dt if eligible else 0.
        if eligible:
            self.observations += 1
            self.evidence_seconds += dt
        decay = config.evidence_decay**dt
        integral = -math.expm1(math.log(config.evidence_decay)*dt)/-math.log(config.evidence_decay)
        increments = {}
        for check in report['checks']:
            key = check['constraint_id']
            check['legacy_status'] = check['status']
            if eligible:
                self.uncertain_seconds[key] = self.uncertain_seconds[key]+dt if check['status']=='unknown' else 0.
                guard = CONSTRAINTS[key][0]
                weight = scores[guard] if guard else 1.
                increments[key] = weight*(scores[key]-config.evidence_baseline)*integral
                self.evidence[key] = max(0.,decay*self.evidence[key]+increments[key])
            if check['status']=='violated':check['status']='suspected'
        crossed = [k for k,v in self.evidence.items() if v >= config.evidence_threshold]
        persistent = [k for k,v in self.uncertain_seconds.items() if v+1e-9 >= config.uncertainty_seconds]
        ready = eligible and self.evidence_seconds+1e-9 >= config.min_evidence_seconds
        acute = eligible and scores['collision'] >= config.acute_collision_threshold
        reason = ('acute_collision' if acute else 'accumulated_risk' if ready and crossed else
                  'persistent_uncertainty' if ready and persistent else None)
        clear = all(c['status'] in ('clear','not_applicable') for c in report['checks'])
        report.update(violations=[],state='RUNNING' if clear else 'UNCERTAIN',review_requested=reason is not None,
            review_reason=reason,interrupt_requested=acute,
            accumulation=dict(revision=config.accumulation_revision,eligible=eligible,
                observations=self.observations,evidence_seconds=self.evidence_seconds,
                interval_seconds=dt,uncertain_seconds=dict(self.uncertain_seconds),
                uncertainty_limit_seconds=config.uncertainty_seconds,persistent_constraints=persistent,
                scores=dict(self.evidence),increments=increments,threshold=config.evidence_threshold,
                crossed=crossed,decay_per_second=config.evidence_decay,baseline=config.evidence_baseline,
                confirmed_failure=False))
        return report


class TemporalScheduler(AccumulatingScheduler):
    max_chunk_steps = 25

    def __init__(self,*args,review_cooldown_seconds=4.,max_async_seconds=6.,**kwargs):
        super().__init__(*args,**kwargs)
        for value in (review_cooldown_seconds,max_async_seconds):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Async budgets must be finite and positive')
        self.cooldown_steps = math.ceil(review_cooldown_seconds/self.monitor.config.control_dt)
        self.max_async_steps = math.floor(max_async_seconds/self.monitor.config.control_dt)
        if self.max_async_steps < self.chunk_steps:
            raise ValueError('Async continuation budget must cover one chunk')
        self.background = None
        self.background_request = None
        self.next_review_step = 0
        self.background_attempts = []
        self.followup_advice = None

    def seed_context(self):
        actual = self.rollout.meta.get('control_dt')
        if not isinstance(actual, (float, int)) or not math.isclose(actual, self.monitor.config.control_dt):
            raise ValueError('Simulator control_dt differs from temporal evidence configuration')
        super().seed_context()

    def progress_review(self,feedback):
        if feedback.get('interrupt_requested'):
            return feedback
        if self.background is not None or self.rollout.tick < self.next_review_step:
            return None
        if not feedback.get('review_requested'):
            return None
        if self.max_replans and self.replans >= self.max_replans:
            raise RuntimeError('Explicit Astra call budget exhausted')
        scene = self.current_frame()
        self.fresh(scene)
        request = dict(version=2,request_id=uuid.uuid4().hex,**self.identity,
            review_kind='background',review_reason=feedback['review_reason'],
            original_task=self.original_task,current_subgoal=self.subgoal,current_scene=scene,
            control_dt=self.monitor.config.control_dt,accumulated_evidence=deepcopy(feedback['accumulation']),
            failure=deepcopy(feedback),evidence_frames=deepcopy(list(self.monitor.frames)),
            previous_attempts=deepcopy(self.background_attempts[-2:]))
        self.replans += 1
        self.rollout.run.update(teacher_off=False,teacher_invocation='temporal_background_and_acute_pause')
        write_json(self.rollout.output/'run.json',self.rollout.run)
        write_json(self.rollout.output/f'background_{self.replans:03d}.json',request)
        self.background_request = request
        self.background = PendingCall(lambda:self.planner.replan(deepcopy(request)),self.planner_timeout,'astra-background')
        self.event('background_review_request',request=request,max_continuation_steps=self.max_async_steps)
        return None

    def poll_background(self,feedback):
        if self.background is None:
            return None
        request = self.background_request
        urgent = bool(feedback and feedback.get('interrupt_requested'))
        moved = self.rollout.tick-request['current_scene']['step_id']
        budget = moved+self.chunk_steps > self.max_async_steps
        if self.background.thread.is_alive() and not urgent and not budget and time.monotonic() < self.background.deadline:
            return None
        if self.background.thread.is_alive():
            self.transition('WAITING_FOR_BACKGROUND','Acute risk' if urgent else 'Bounded continuation budget reached')
        raw = self.background.get()
        self.background = self.background_request = None
        if (not isinstance(raw,dict) or set(raw) != {'request_id',*IDENTITY,'decision'}
                or any(type(raw[k]) is not type(request[k]) or raw[k]!=request[k] for k in ('request_id',*IDENTITY))):
            raise ValueError('Invalid background review identity')
        decision = raw['decision']
        if (not isinstance(decision,dict) or set(decision) != {'mode','reason'}
                or decision.get('mode') not in ('continue','request_recovery')
                or not isinstance(decision.get('reason'),str) or not 0 < len(decision['reason']) <= 1200):
            raise ValueError('Background review may only continue or request fresh recovery; no stale actions')
        attempt = dict(request_id=request['request_id'],source_step=request['current_scene']['step_id'],
            received_step=self.rollout.tick,decision=decision,result='advisory_only_no_action')
        self.background_attempts.append(attempt)
        self.event('background_review_response',response=raw,source_step=attempt['source_step'],moved_steps=moved)
        if any(request[k]!=v for k,v in self.identity.items()):
            self.event('background_review_discarded',reason='Plan generation changed while review was in flight')
            return feedback if urgent else None
        if urgent or decision['mode']=='request_recovery':
            self.followup_advice = decision
            return feedback or self.latest_feedback
        self.next_review_step = self.rollout.tick+self.cooldown_steps
        self.monitor.runtime.acknowledge_review()
        # At the continuation limit the snapshot may have aged during the wait.
        # Same-step native state revalidation avoids relabeling old timestamps.
        self.validate_recovery_scene(self.current_frame())
        self.event('background_continuation',next_review_step=self.next_review_step,claims_task_success=False)
        return None

    def recovery_context(self):
        context = super().recovery_context()
        if self.followup_advice:
            context.update(review_reason='astra_requested_fresh_recovery',background_advice=self.followup_advice)
        return context

    def recover(self,*args,**kwargs):
        response = super().recover(*args,**kwargs)
        self.followup_advice = None
        self.next_review_step = self.rollout.tick+self.cooldown_steps
        return response

    def run(self):
        try:
            result = super().run()
        finally:
            if self.background is not None:
                self.event('background_review_discarded',reason='Episode ended; pending answer cannot actuate')
                if hasattr(self.planner,'close'):self.planner.close()
        result.update(background_attempts=self.background_attempts,
            background_worker_pending=bool(self.background and self.background.thread.is_alive()),
            max_async_control_steps=self.max_async_steps,review_cooldown_steps=self.cooldown_steps)
        write_json(self.rollout.output/'jev_summary.json',result)
        return result
