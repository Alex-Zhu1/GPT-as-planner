"""One owner executes Pi under a bounded Astra language plan and Valen evidence."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import math
import uuid

from ..io import write_json
from ..prompt_context import task_context
from .accumulation import AccumulatingScheduler
from .monitor import PendingCall
from .runner import ReplanScheduler, automatic_response
from .subgoal_plan import (PLAN_VERSION, PLANNER_ROLE, PI_INSTRUCTION_CONTRACT,
                          planning_context, task_requirements, validate_planning_response)


class SubgoalScheduler(ReplanScheduler):
    max_chunk_steps = 25

    def __init__(self, *args, **kwargs):
        kwargs.update(include_initial_context=True, uncertainty_policy='continue')
        super().__init__(*args, **kwargs)
        self.plan_goals = []
        self.requirements = []
        self.completed_requirements = []
        self.completion_references = []
        self.initial_reference = self.goal_reference = None
        self.planner_call = None
        self.last_review_step = self.stage_start_step = 0
        self.final_stage_confirmed = self.collision_latched = False
        self.assessments = []
        self.context = {}
        self.instruction_history = []

    @staticmethod
    def reference(frame):
        value = deepcopy(frame)
        value.pop('reference_frames', None)
        return value

    def current_frame(self):
        frame = super().current_frame()
        frame.update(plan_goals=deepcopy(self.plan_goals),
            task_requirements=deepcopy(self.requirements),
            completion_checks=deepcopy(self.goals[0].get('completion_checks', [])),
            completed_requirements=deepcopy(self.completed_requirements),
            active_goal_index=next((i for i, g in enumerate(self.plan_goals) if g['subgoal_id'] == self.subgoal_id), 0))
        frame['reference_frames'] = deepcopy([f for f in (
            self.initial_reference, self.goal_reference, *self.completion_references) if f is not None])
        return frame

    def validate_recovery_scene(self, scene):
        return AccumulatingScheduler.validate_recovery_scene(self, scene)

    def activate_stage(self, reason, *, new_stage=True):
        self.subgoal_id = self.goals[0]['subgoal_id']
        self.subgoal = self.goals[0]['instruction']
        coverage = self.goals[0]['requirements']
        previous = self.instruction_history[-1] if self.instruction_history else None
        if previous is None or (previous['instruction'], previous['requirements']) != (self.subgoal, coverage):
            if previous and set(previous['requirements']) == set(coverage) and previous['instruction'] != self.subgoal:
                self.event('subgoal_instruction_rewritten', previous_instruction=previous['instruction'],
                           instruction=self.subgoal, previous_executed_controls=previous['executed_controls'],
                           reason=reason, decision_author='astra')
            self.instruction_history.append(dict(instruction=self.subgoal, requirements=deepcopy(coverage),
                plan_revision=self.revision, start_step=self.rollout.tick, end_step=self.rollout.tick,
                executed_controls=0, reason=reason))
        self.monitor.activate()
        self.frames.clear()
        self.reports.clear()
        self.latest_feedback = None
        if new_stage:
            self.stage_start_step = self.rollout.tick
            self.final_stage_confirmed = False
            self.goal_reference = self.reference(self.current_frame())
        self.seed_context()
        write_json(self.rollout.output/f'jev_plan_{self.revision:03d}_{self.subgoal_id}.json',
            dict(**self.identity, original_task=self.original_task, plan_goals=self.plan_goals,
                 remaining_plan=list(self.goals), completed_requirements=self.completed_requirements))
        self.event('subgoal_activated' if new_stage else 'subgoal_retained', instruction=self.subgoal,
                   completion_checks=self.goals[0]['completion_checks'], reason=reason)
        self.transition('RUNNING', reason)

    def plan(self, kind, feedback=None):
        if self.max_replans and self.replans >= self.max_replans:
            raise RuntimeError('Astra language planning call budget exhausted')
        initial = kind == 'initial_plan'
        if not initial:
            self.discard_inference('Paused language review invalidates speculative Pi inference')
            if self.rollout.phase == 'execute':
                self.retire_prepared_proposal()
        scene = self.current_frame() if initial else self.capture_scene()
        paused_step = self.rollout.tick
        completed = {r['requirement_id'] for r in self.completed_requirements}
        request = dict(version=PLAN_VERSION, planner_role=PLANNER_ROLE,
            request_id=uuid.uuid4().hex, **self.identity, review_kind=kind,
            original_task=self.original_task, current_subgoal=self.subgoal,
            current_scene=scene, task_context=self.context,
            all_requirements=deepcopy(self.requirements), previous_assessments=deepcopy(self.assessments),
            pi_instruction_contract=deepcopy(PI_INSTRUCTION_CONTRACT),
            instruction_history=deepcopy(self.instruction_history[-6:]),
            remaining_requirements=[deepcopy(r) for r in self.requirements if r['requirement_id'] not in completed],
            completed_requirements=deepcopy(self.completed_requirements),
            remaining_plan=deepcopy(list(self.goals)) if not initial else [],
            reference_frames=deepcopy(scene['reference_frames']), evidence_frames=list(self.frames)[-6:],
            monitor_evidence=deepcopy(feedback), review_reason=(feedback or {}).get('review_reason') or (
                'initial_decomposition' if initial else 'periodic_stage_check'),
            execution_budget=dict(current_step=paused_step,
                stage_executed_controls=paused_step-self.stage_start_step,
                remaining_controls=self.rollout.obs['remaining_steps'], max_subgoals=3,
                remaining_planner_calls=None if not self.max_replans else self.max_replans-self.replans))
        self.replans += 1
        write_json(self.rollout.output/f'jev_replan_{self.replans:03d}.json', request)
        self.event('subgoal_plan_request', request=request)
        self.transition('PLANNING', kind)
        self.planner_call = PendingCall(lambda: self.planner.replan(deepcopy(request)), self.planner_timeout, 'astra-subgoal-plan')
        raw = self.planner_call.get()
        self.planner_call = None
        decision = validate_planning_response(raw, request)
        self.validate_recovery_scene(scene)
        if self.rollout.tick != paused_step:
            raise RuntimeError('Physics advanced during language planning')
        # An old initial inference is reusable only for the exact same instruction
        # AND observation. A scene refreshed after a long call requires new Pi.
        reuse = (initial and decision['mode'] == 'plan' and decision['subgoals'][0]['instruction'] == self.original_task
                 and self.rollout.observation_packet['observation_id'] == scene['observation_id'])
        if initial and not reuse:
            self.discard_inference('Initial source-task proposal differs from the activated subgoal or observation')
        if not initial:
            self.assessments = deepcopy(decision['assessments'])
            satisfied = {a['requirement_id'] for a in self.assessments if a['status'] == 'satisfied'}
            reopened = completed-satisfied
            if reopened:
                self.event('requirements_reopened', requirements=sorted(reopened), decision_author='astra')
                self.completion_references.clear()
            self.completed_requirements = [deepcopy(r) for r in self.requirements if r['requirement_id'] in satisfied]
            if satisfied-completed:
                self.completion_references.append(self.reference(scene))
                self.completion_references = self.completion_references[-2:]
            self.event('requirements_assessed', assessments=self.assessments, decision_author='astra')
        if decision['mode'] == 'plan':
            self.revision += 1
            self.plan_goals = deepcopy(decision['subgoals'])
            self.goals = deque(dict(g, success_condition=' '.join(g['completion_checks'])) for g in self.plan_goals)
        self.attempts.append(dict(request_id=request['request_id'], plan_revision=self.revision,
            review_kind=kind, decision=deepcopy(decision), source_step=paused_step,
            result=dict(plan='language_plan_installed', keep_plan='language_plan_retained', advance='astra_stage_advance')[decision['mode']]))
        self.event('subgoal_plan_response', response=raw, controls_during_planning=0,
                   reused_initial_inference=reuse)
        self.last_review_step = self.rollout.tick
        if decision['mode'] == 'advance':
            self.advance(scene)
        else:
            self.final_stage_confirmed = False
            self.activate_stage(decision['reason'],
                                new_stage=decision['mode'] == 'plan')

    def advance(self, scene):
        satisfied = {r['requirement_id'] for r in self.completed_requirements}
        changed = False
        while len(self.goals) > 1 and set(self.goals[0]['requirements']) <= satisfied:
            completed = self.goals.popleft()
            self.subgoal_id = completed['subgoal_id']
            self.event('subgoal_completed', subgoal=completed, observation_id=scene['observation_id'],
                       decision_author='astra', assessments=self.assessments)
            changed = True
        final = len(self.goals) == 1 and set(self.goals[0]['requirements']) <= satisfied
        self.activate_stage('Astra confirmed completion from current visual evidence', new_stage=changed)
        self.final_stage_confirmed = final
        if final:
            # Keep the last instruction for settling/arm return until native termination.
            self.event('final_stage_confirmed', decision_author='astra', native_success_claimed=False)

    def review_due(self, feedback):
        config = self.monitor.config
        elapsed = (self.rollout.tick-self.last_review_step)*config.control_dt
        acute = bool(feedback and feedback.get('interrupt_requested'))
        new_collision = acute and not self.collision_latched
        if feedback is not None:
            self.collision_latched = acute
        if feedback and feedback['review_requested']:
            if new_collision or self.replans == 1 or elapsed+1e-9 >= config.review_cooldown_seconds:
                return 'subgoal_review', feedback
            self.event('review_deferred', review_reason=feedback['review_reason'], elapsed_seconds=elapsed)
        if elapsed+1e-9 >= config.review_seconds:
            return ('final_review' if self.final_stage_confirmed else 'progress_review',
                    dict(feedback or {}, review_reason='periodic_stage_check'))
        return None

    def run(self):
        native_result = None
        try:
            args = self.rollout.next_call().copy(); args.pop('tool')
            self.rollout.start(**args)
            actual_dt = self.rollout.meta.get('control_dt')
            if isinstance(actual_dt, bool) or not isinstance(actual_dt, (float, int)) or not math.isclose(actual_dt, self.monitor.config.control_dt):
                raise ValueError('Simulator control_dt differs from subgoal evidence configuration')
            self.original_task = self.subgoal = self.rollout.meta['instruction']
            self.goals.append(dict(subgoal_id='original-task', instruction=self.original_task, success_condition=None))
            self.context = planning_context(self.rollout.task, self.original_task, task_context(self.rollout.task))
            self.requirements = task_requirements(self.original_task, self.context)
            self.initial_reference = self.reference(self.current_frame())
            self.rollout.run.update(evaluation_method='pi05_valen_subgoal_plan', teacher_off=False,
                teacher_invocation='initial_plan_anomaly_or_periodic_stage_review', max_subgoals=3,
                completion_authority='astra', valen_role='anomaly_trigger', task_contract=self.context,
                planner_role=PLANNER_ROLE, planner_protocol_version=PLAN_VERSION,
                teacher_reasoning_effort='high', gate_policy='subgoal_post_action', predictive_intent_gate=False,
                monitor_program=asdict(self.monitor.config), original_task=self.original_task,
                initial_pi_policy='speculative_original_instruction_no_execution', max_planner_calls=self.max_replans)
            write_json(self.rollout.output/'run.json', self.rollout.run)
            self.start_inference()
            self.plan('initial_plan')
            while self.rollout.phase != 'done':
                if self.inference is None:
                    self.start_inference()
                feedback = self.review() if self.monitor.pending else None
                due = self.review_due(feedback)
                if due:
                    self.plan(*due)
                    continue
                self.transition('WAITING_FOR_POLICY')
                self.finish_inference()
                self.fresh(self.rollout.observation_packet)
                response = automatic_response(self.rollout.request, self.chunk_steps, feedback)
                response['evaluation_method'] = 'pi05_valen_subgoal_plan'
                response['reason'] = 'Execute the active Astra subgoal under endpoint monitoring; completion remains unverified.'
                before = self.rollout.tick
                args = self.rollout.next_call().copy(); args.pop('tool'); args['response'] = response
                self.transition('RUNNING')
                self.rollout.execute(**args)
                if not 0 < self.rollout.tick-before <= self.chunk_steps:
                    raise RuntimeError('Simulator violated the subgoal chunk bound')
                self.instruction_history[-1]['executed_controls'] += self.rollout.tick-before
                self.instruction_history[-1]['end_step'] = self.rollout.tick
                frame = self.current_frame()
                self.frames.append(self.reference(frame))
                self.event('executed_chunk', frame=frame, decision_author=response['decision_author'], instruction=self.subgoal)
                if self.rollout.phase != 'done':
                    self.monitor.submit(frame)
            native_result = self.rollout.observation_packet.get('result')
            terminal = bool(native_result and native_result.get('complete'))
            self.transition('COMPLETED' if terminal else 'STOPPED',
                'Native rollout ended' if terminal else 'Decision budget exhausted before native termination')
        except Exception as error:
            state = ('PAUSED_MONITOR_UNAVAILABLE' if self.state == 'WAITING_FOR_MONITOR' else
                     'PAUSED_POLICY_UNAVAILABLE' if self.state == 'WAITING_FOR_POLICY' else 'STOPPED')
            self.transition(state, f'{type(error).__name__}: {error}')
        finally:
            if self.rollout.episode is not None and self.rollout.phase != 'done':
                try:
                    native_result = self.rollout.finish('subgoal_scheduler_stop')
                except Exception as error:
                    native_result = dict(complete=False, success=None, finish_error=f'{type(error).__name__}: {error}')
            elif native_result is None:
                native_result = getattr(self.rollout, 'observation_packet', {}).get('result')
        result = dict(state=self.state, reason=self.reason, **self.identity, step_id=self.rollout.tick,
            native_result=native_result, replan_count=self.replans, attempts=self.attempts,
            remaining_plan=list(self.goals), completed_requirements=self.completed_requirements,
            plan_goals=self.plan_goals, assessments=self.assessments, completion_authority='astra',
            instruction_history=deepcopy(self.instruction_history),
            final_stage_confirmed=self.final_stage_confirmed, max_subgoals=3, max_unreviewed_control_steps=self.chunk_steps,
            planner_worker_pending=bool(self.planner_call and self.planner_call.thread.is_alive()),
            policy_worker_pending=bool(self.inference and self.inference.call.thread.is_alive()),
            monitor_worker_pending=bool(self.monitor.pending and self.monitor.pending.thread.is_alive()))
        write_json(self.rollout.output/'jev_summary.json', result)
        return result
