"""Continuous Pi control with one asynchronous, latest-observation Valen worker.

Only the owner executes physics. Reports refer to historical endpoints; they
can request a pause but never directly authorize an action at a newer endpoint.
Pauses occur at bounded chunk boundaries. This models software latency, not
physical braking/inertia of a deployed robot.
"""
from collections import deque
from copy import deepcopy
import time

from ..robodojo.io import write_json
from ..robodojo.jev.monitor import IDENTITY, PendingCall, ConstraintRuntime
from ..robodojo.jev.runner import ReplanScheduler, automatic_response
from .task_context import TaskContext


class ContinuousScheduler(ReplanScheduler):
    progress_review_enabled = True

    def __init__(self, *args, astra_enabled=True, progress_budget=80,
                 subgoal_budget=80, max_monitor_lag=40, shadow_only=False,
                 stall_steps=30, stall_min_reports=4, max_review_interval=160, **kwargs):
        super().__init__(*args, **kwargs)
        if min(progress_budget, subgoal_budget, max_monitor_lag) < self.chunk_steps:
            raise ValueError('Continuous budgets must cover at least one chunk')
        if type(max_review_interval) is not int or max_review_interval < self.chunk_steps:
            raise ValueError('Review interval must cover at least one chunk')
        self.max_review_interval = max_review_interval
        self.last_astra_review_step = 0
        self.astra_enabled = astra_enabled
        if shadow_only and astra_enabled:
            raise ValueError('Shadow monitoring cannot invoke Astra')
        self.shadow_only = shadow_only
        self.progress_budget, self.subgoal_budget = progress_budget, subgoal_budget
        self.max_monitor_lag = max_monitor_lag
        self.observed = deque(maxlen=self.monitor.config.history)
        self.submitted = {}
        self.last_submitted_step = None
        self.goal_start_step = None
        self.trigger_metrics = []
        self.monitor_reports = 0
        self.max_report_lag = 0
        self.coalesced_observations = 0
        self._pause_context = None
        self._review_votes = deque()
        self.local_correction_pending = False
        self.task_context = TaskContext(self.monitor.config, stall_steps=stall_steps, min_reports=stall_min_reports, max_sample_gap=max_monitor_lag)

    def seed_context(self):
        super().seed_context()
        self.observed.clear()
        self.observed.append(deepcopy(self.current_frame()))
        self.last_submitted_step = self.rollout.tick
        self.goal_start_step = self.rollout.tick
        self._review_votes.clear()
        self.task_context.activate(self.current_frame(), self.goals)

    def record_context(self, report, source):
        if self.task_context.update(report, source):
            snapshot = self.task_context.snapshot()
            write_json(self.rollout.output/'task_context.json', snapshot)
            self.event('task_context_updated', context=snapshot)

    def activate_goal(self, reason, *, refresh=True):
        super().activate_goal(reason, refresh=refresh)
        self.latest_feedback = None
        self.submitted.clear()

    def offer_latest(self):
        """Never queue an unbounded stream behind a slow detector."""
        if self.monitor.pending is not None or not self.observed:
            return
        frame = self.observed[-1]
        if frame['step_id'] <= self.last_submitted_step:
            return
        skipped = max(0, (frame['step_id']-self.last_submitted_step)//self.chunk_steps-1)
        self.coalesced_observations += skipped
        self.monitor.frames.clear()
        self.monitor.frames.extend(deepcopy(list(self.observed)[:-1]))
        self.monitor.submit(frame)
        self.submitted[frame['observation_id']] = deepcopy(frame)
        self.last_submitted_step = frame['step_id']
        self.event('monitor_submitted', source_step=frame['step_id'],
                   historical_steps=[f['step_id'] for f in self.observed], skipped_endpoints=skipped)

    def poll_report(self, *, block=False):
        pending = self.monitor.pending
        if pending is None:
            return None
        if pending.result.empty() and not block:
            if time.monotonic() >= pending.deadline:
                raise TimeoutError('Asynchronous Valen deadline exceeded')
            return None
        report = self.monitor.review()
        source = self.submitted.pop(report['observation_id'], None)
        if (source is None or any(report[k] != self.identity[k] for k in IDENTITY)
                or report['step_id'] != source['step_id'] or report['step_id'] > self.rollout.tick):
            raise ValueError('Unrelated asynchronous monitor generation or endpoint')
        self.fresh(report)
        lag = self.rollout.tick-report['step_id']
        self.max_report_lag = max(self.max_report_lag, lag)
        self.monitor_reports += 1
        self.reports.append(report)
        self.latest_feedback = report
        self.record_context(report, source)
        self.event('monitor_report', report=report, received_at=time.time(),
                   observation_lag_steps=lag, model_latency_s=report['render_to_report_seconds'])
        return report

    def observation_reference_step(self):
        # A paused Astra review refreshes the current scene even when it keeps
        # the instruction (and therefore does not reset goal_start_step).
        return max(self.goal_start_step, self.last_astra_review_step,
                   self.latest_feedback['step_id'] if self.latest_feedback else 0)

    def progress_evidence(self, report):
        semantic = self.task_context.review()
        if semantic:
            return semantic
        if report and self.monitor.config.observation_profile == 'constraints':
            self._review_votes.append(report)
            while self._review_votes and report['step_id']-self._review_votes[0]['step_id'] > 80:
                self._review_votes.popleft()
            if (len(self._review_votes) >= 3
                    and report['step_id']-self._review_votes[0]['step_id'] >= 60
                    and sum(r['scores']['stuck'] >= .55 and r['scores']['progress_attempted'] >= .8
                            for r in self._review_votes)/len(self._review_votes) >= .75):
                return dict(trigger='sustained_stuck_evidence', confirmed_failure=False,
                            first_step=self._review_votes[0]['step_id'], last_step=report['step_id'], violations=[])
        recovery = self.subgoal_id != 'original-task'
        budget = min(self.subgoal_budget if recovery else self.progress_budget, self.max_review_interval)
        reference_step = max(self.goal_start_step, self.last_astra_review_step, (self.task_context.last_advancement_signal or {}).get('step_id', 0))
        if self.rollout.tick-reference_step >= budget:
            # RUNNING means constraint checks clear, never task progress.
            # A high advancement score or goal activation resets the sampling fallback.
            return dict(trigger='subgoal_execution_budget' if recovery else 'task_progress_review_budget',
                        confirmed_failure=False, first_step=reference_step,
                        last_step=self.rollout.tick, violations=[], executed_budget=budget,
                        reason='Evidence sampling budget reached; Astra must review the board and current scene.')
        return None

    def drain_monitor(self):
        if self.monitor.pending is not None:
            report = self.monitor.review()
            source = self.submitted.pop(report['observation_id'], None)
            self.event('monitor_drained_at_pause', report=report)
            # Historical evidence is retained, never used to advance a new goal.
            if all(report[k] == self.identity[k] for k in IDENTITY):
                self.reports.append(report)
                if source is not None:
                    self.record_context(report, source)

    def start_inference(self):
        # Preserve the final obligation in every recovery prompt seen by Pi.
        from ..robodojo.jev.runner import AsyncInference
        prompt = self.subgoal
        if self.subgoal_id != 'original-task' and self.monitor.config.observation_profile != 'task_progress':
            prompt = f'{self.original_task}. Next, {self.subgoal}'
        self.inference = AsyncInference(self.rollout, self.policy_timeout, prompt)

    def recovery_context(self):
        horizon = self.rollout.session.horizon
        self._decision_scene = deepcopy(self.current_frame())
        cap = {}
        if (self.monitor.config.observation_profile == 'task_progress'
                and self.subgoal_id != 'original-task'):
            cap['resume_original'] = dict(subgoal_id=self.subgoal_id,
                evidence_observation_ids=[self.rollout.observation_packet['observation_id']])
        return dict(planner_role='instruction_eef' if self.monitor.config.observation_profile == 'task_progress' else 'legacy_control',
                    instruction_execution=dict(start_step=self.goal_start_step,
                        current_step=self.rollout.tick, executed_controls=self.rollout.tick-self.goal_start_step),
                    plan_operations=cap,
                    recovery_policy='single_instruction' if self.monitor.config.observation_profile == 'task_progress' else 'legacy_plan',
                    recovery_instruction_policy='evidence_driven', max_review_interval=self.max_review_interval,
                    task_context=self.task_context.snapshot(),
                    board_review_required=self.monitor.config.observation_profile == 'task_progress',
                    execution_budget=dict(total_controls=horizon, current_step=self.rollout.tick,
                    remaining_controls=horizon-self.rollout.tick,
                    subgoal_max_controls=self.subgoal_budget,
                    remaining_teacher_calls=self.max_replans-self.replans),
                    asynchronous_execution=deepcopy(self._pause_context),
                    planning_guidance='Use the fresh paused current_scene for the recovery, not the older trigger '
                    'frame. Use eef for a grounded local geometric error, or replan with short concrete '
                    'Pi-executable instructions for instruction/order errors. Preserve the original task. '
                    'Return exactly one current recovery instruction. No future instruction queue. '
                    'The instruction persists across monitored chunks. New evidence or an execution-budget fallback '
                    'requests review; replan replaces the instruction, resume_original restores the original task. '
                    'Each subgoal has a host execution budget; unresolved completion will trigger another review. '
                    'Respect remaining_controls. A deadline review is not proof of failure.')

    def recover(self, failure, *, uncertainty=False, progress=False):
        paused = self.rollout.tick
        now = time.time()
        report = self.latest_feedback
        metric = dict(trigger=failure.get('trigger', 'confirmed_failure'),
            trigger_origin='execution_budget' if 'budget' in failure.get('trigger', '') else 'valen_report',
            source_step=report['step_id'] if report else None,
            source_captured_at=report['captured_at'] if report else None,
            detector_completed_at=report['completed_at'] if report else None,
            trigger_step=paused, trigger_at=now, pause_step=paused,
            controls_after_trigger=0, pause_boundary='end_of_at_most_5_control_chunk',
            controls_since_source=paused-report['step_id'] if report else None,
            detector_latency_s=report['render_to_report_seconds'] if report else None,
            delivery_latency_s=now-report['completed_at'] if report else None)
        self._pause_context = metric
        self.trigger_metrics.append(metric)
        self.event('pause_confirmed', **metric)
        self.discard_inference('Continuous execution paused; discard unexecuted candidate')
        release = getattr(self.rollout.student, 'release_matched_queries', None)
        if release:
            release()
        self.drain_monitor()
        attempt_count = len(self.attempts)
        result = super().recover(failure, uncertainty=uncertainty, progress=progress)
        if len(self.attempts) > attempt_count:
            self.last_astra_review_step = paused
            self.task_context.record_astra(self.attempts[-1].get('board_review'))
            self.task_context.remember('last_decision', self._decision_scene)
        self.task_context.reviewed(paused)
        self.monitor.runtime.invalidate()
        self.latest_feedback = None
        self.observed.clear()
        self.observed.append(deepcopy(self.current_frame()))
        if self.rollout.tick != paused:
            raise RuntimeError('Control advanced during paused recovery')
        self.event('pause_release', paused_step=paused, current_step=self.rollout.tick,
                   wait_wall_seconds=time.time()-now, controls_during_planning=0)
        return result

    def verify_completion(self):
        """Historical completion votes request a hold and fresh verification."""
        self.discard_inference('Verify goal completion against the current paused scene')
        self.drain_monitor()
        scene = self.capture_scene()
        frames = [*list(self.observed)[-2:], scene]
        scores = PendingCall(lambda: self.monitor.monitor.evaluate(frames),
                             self.monitor.timeout_s, 'completion-check').get()
        checked = ConstraintRuntime(self.monitor.config).update(scene, scores)
        clear = all(c['status'] in ('clear', 'not_applicable') for c in checked['checks'])
        passed = clear and scores['subgoal_complete'] >= self.monitor.config.high
        self.event('current_completion_verification', passed=passed, scores=scores, frame=scene)
        return passed

    def run(self):
        native_result = None
        try:
            args = self.rollout.next_call().copy(); args.pop('tool')
            self.rollout.start(**args)
            self.original_task = self.rollout.meta['instruction']
            self.subgoal = self.original_task
            self.goals.append(dict(subgoal_id='original-task', instruction=self.subgoal, success_condition=None))
            self.rollout.run.update(evaluation_method='pi05_valen_continuous', teacher_off=not self.astra_enabled,
                gate_policy='asynchronous_post_action', predictive_intent_gate=False,
                planner_role='instruction_eef' if self.monitor.config.observation_profile=='task_progress' else 'legacy_control',
                progress_budget=self.progress_budget, subgoal_budget=self.subgoal_budget,
                max_review_interval=self.max_review_interval, astra_call_budget=self.max_replans,
                max_monitor_lag=self.max_monitor_lag, observation_profile=self.monitor.config.observation_profile,
                stall_steps=self.task_context.stall_steps, stall_min_reports=self.task_context.min_reports)
            self.rollout.run['shadow_only'] = self.shadow_only
            write_json(self.rollout.output/'run.json', self.rollout.run)
            self.seed_context()
            self.transition('RUNNING')
            while self.rollout.phase != 'done':
                correction_review = None
                reference = self.observation_reference_step()
                lag_limit = self.rollout.tick-reference+self.chunk_steps > self.max_monitor_lag
                if lag_limit and self.monitor.pending:
                    self.event('monitor_backpressure', reference_step=reference,
                               maximum_lag=self.max_monitor_lag)
                report = self.poll_report(block=self.local_correction_pending or lag_limit)
                if self.local_correction_pending and self.monitor.config.observation_profile == 'task_progress':
                    if report is None or report['step_id'] != self.rollout.tick:
                        self.transition('STOPPED', 'Local EEF correction requires fresh endpoint evidence before Pi handback')
                        break
                    self.local_correction_pending = False
                    self.attempts[-1]['verification'] = dict(observation_id=report['observation_id'],
                        result='fresh_visual_evidence_received_not_task_success')
                    self.event('local_correction_review', report=report, semantic_verdict=False)
                    self.discard_inference('Correction executed; Pi must infer from the new scene')
                if self.local_correction_pending:
                    self.local_correction_pending = False
                    self.event('local_correction_review', report=report)
                    self.attempts[-1]['verification'] = dict(
                        observation_id=report['observation_id'] if report else None,
                        result='clear_monitor_checks_not_task_success' if report and report['state']=='RUNNING'
                        else 'not_verified')
                    # A correction changes the scene. Do not execute a new Pi chunk
                    # before its post-action evidence has actually arrived.
                    if report is None or report['state'] not in ('RUNNING', 'FAILED'):
                        self.transition('PAUSED_RECOVERY_UNVERIFIED',
                                        'Local correction needs clear post-action evidence before Pi handback')
                        correction_review = dict(trigger='local_correction_unverified',
                            confirmed_failure=False, violations=[], first_step=self.rollout.tick,
                            last_step=self.rollout.tick, observation_id=report['observation_id'] if report else None)
                        if not self.astra_enabled:
                            self.transition('STOPPED', 'Local correction unverified and Astra unavailable')
                            break
                if report and report['state'] == 'UNCERTAIN':
                    self.uncertain_continuations += 1
                failure = report if (self.monitor.config.observation_profile == 'constraints' and report and report['violations']) else None
                if failure and not self.astra_enabled:
                    if not self.shadow_only:
                        self.transition('STOPPED', 'Astra disabled: confirmed failure')
                        break
                    self.event('shadow_failure_logged', report=failure, controls_not_gated=True)
                    failure = None
                if (self.monitor.config.observation_profile == 'constraints' and report
                        and report['subgoal_complete'] and len(self.goals) > 1 and not failure):
                    if self.verify_completion():
                        completed = self.goals.popleft()
                        self.task_context.completed_subgoal(completed, self.current_frame())
                        self.event('subgoal_completed', subgoal=completed)
                        self.activate_goal('Current paused scene verified subgoal completion')
                        continue
                advisory = correction_review or self.progress_evidence(report)
                if advisory and not self.astra_enabled:
                    if not self.shadow_only:
                        self.transition('STOPPED', 'Required Astra evidence review unavailable')
                        break
                    self.event('progress_review_suppressed', evidence=advisory, astra_enabled=False)
                    self.goal_start_step = self.rollout.tick
                    self._review_votes.clear()
                    self.task_context.reviewed(self.rollout.tick)
                if self.astra_enabled and (failure or advisory):
                    if self.replans >= self.max_replans and not failure:
                        if correction_review or self.monitor.config.observation_profile == 'task_progress':
                            self.transition('STOPPED', 'Required Astra review budget exhausted')
                            break
                        self.event('teacher_budget_exhausted', restore_original_instruction=True)
                        self.drain_monitor()
                        self.discard_inference('Teacher budget exhausted; return to original instruction')
                        self.goals = deque([dict(subgoal_id='original-task', instruction=self.original_task, success_condition=None)])
                        self.activate_goal('Resume original task within remaining control budget')
                        self.astra_enabled = False
                        continue
                    self.event('progress_review_trigger' if not failure else 'failure_trigger', evidence=failure or advisory)
                    decision = self.recover(failure or advisory, progress=not bool(failure))
                    if decision is None:
                        break
                    if decision['mode'] in ('replan', 'advance_subgoal', 'resume_original', 'keep_instruction'):
                        continue
                else:
                    self.offer_latest()
                    reference = self.observation_reference_step()
                    if self.rollout.tick-reference+self.chunk_steps > self.max_monitor_lag:
                        # Wait only at the bounded lag limit. The next loop consumes
                        # the in-flight result; ordinary chunks never wait for Valen.
                        continue
                    self.start_inference()
                    self.finish_inference()
                    # Check again after Pi inference; a completed detector report
                    # must be handled before any newly prepared chunk executes.
                    if self.monitor.pending and not self.monitor.pending.result.empty():
                        self.retire_prepared_proposal()
                        continue
                    steps = self.chunk_steps
                    decision = automatic_response(self.rollout.request, steps, self.latest_feedback)
                before = self.rollout.tick
                before_frame = deepcopy(self.current_frame())
                args = self.rollout.next_call().copy(); args.pop('tool'); args['response'] = decision
                self.rollout.execute(**args)
                frame = self.current_frame()
                self.task_context.record_execution_images(before_frame, frame, decision['mode'])
                self.frames.append(frame); self.observed.append(deepcopy(frame))
                if self.attempts and self.attempts[-1]['decision']['mode'] == 'replan':
                    attempt = self.attempts[-1]
                    attempt.setdefault('executed_subgoals', []).append(dict(subgoal_id=self.subgoal_id,
                        instruction=self.subgoal, start_step=before, end_step=self.rollout.tick))
                    attempt['result'] = 'subgoal_controls_executed'
                self.event('executed_chunk', frame=frame, decision_author=decision['decision_author'])
                self.offer_latest()
                if decision['mode'] in ('eef', 'edit') or correction_review:
                    self.local_correction_pending = True
                    self.attempts[-1]['result'] = 'local_controls_executed'
                    self.attempts[-1]['executed_steps'] = self.rollout.tick-before
            if self.rollout.phase == 'done':
                self.transition('COMPLETED', 'Native original-task episode ended')
            # Drain only after execution is terminal; this cannot delay controls.
            self.drain_monitor()
        except Exception as error:
            self.transition('STOPPED', f'{type(error).__name__}: {error}')
        finally:
            if self.rollout.episode is not None and self.rollout.phase != 'done':
                native_result = self.rollout.finish('continuous_scheduler_stop')
            else:
                native_result = getattr(self.rollout, 'observation_packet', {}).get('result')
        summary = dict(state=self.state, reason=self.reason, **self.identity, step_id=self.rollout.tick,
            native_result=native_result, replan_count=self.replans, attempts=self.attempts,
            remaining_plan=list(self.goals), uncertain_continuation_count=self.uncertain_continuations,
            monitor_reports=self.monitor_reports, max_observation_lag_steps=self.max_report_lag,
            coalesced_observations=self.coalesced_observations, triggers=self.trigger_metrics,
            task_context=self.task_context.snapshot(),
            monitor_worker_pending=bool(self.monitor.pending and self.monitor.pending.thread.is_alive()),
            policy_worker_pending=bool(self.inference and self.inference.call.thread.is_alive()))
        write_json(self.rollout.output/'jev_summary.json', summary)
        write_json(self.rollout.output/'task_context.json', summary['task_context'])
        return summary
