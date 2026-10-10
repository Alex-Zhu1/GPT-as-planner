"""One simulator owner, asynchronous evidence, and event-triggered Astra recovery."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import time
import uuid

from ..io import write_json
from .monitor import IDENTITY, PendingCall
from .plans import validate_decision


class AsyncInference:
    """Only pi05 runs in the worker; FK and all simulator RPC stay on the owner."""
    def __init__(self, rollout, timeout, instruction=None):
        if rollout.phase != 'infer':
            raise RuntimeError('Inference requires a current unconsumed observation')
        self.rollout = rollout
        self.directory = rollout.output/'jev_inference'/uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.path = self.directory/'actions.npz'
        snapshot = deepcopy(rollout.obs)
        self.instruction = instruction or snapshot['instruction']
        snapshot['instruction'] = self.instruction
        student = rollout.student
        self.call = PendingCall(lambda: student.infer(snapshot, self.path), timeout, 'pi05-inference')

    def finish(self, discard=False):
        result = self.call.get(discard=discard)
        if discard:
            return None
        self.rollout.actions, self.rollout.prediction = result
        self.rollout.proposal_path = self.path
        return self.rollout._inference_packet(self.directory, student_instruction=self.instruction)


def automatic_response(request, steps, feedback):
    """Explicit unknown assessment, not a fabricated Astra or predictive safety verdict."""
    task = request.get('student_instruction', request['instruction'])
    return dict(request_id=request['request_id'], mode='student', steps=steps,
        decision_author='jev_scheduler', evaluation_method='pi05_valen_replan',
        reason='Continue the student under bounded post-action monitoring; no confirmed failure.',
        assessment=dict(task_progress=dict(verified_completed=[], currently_attempting=task, remaining=[task]),
            current_subgoal=task, execution_status='not_started' if request['step_id'] == 0 else 'uncertain',
            execution_evidence='No prior execution.' if feedback is None else
                'See the versioned Valen constraint report; no task completion is inferred.',
            expected_next_intent=task, predicted_next_intent='Not assessed by the post-action monitor.',
            intent_status='uncertain', intent_evidence='No predictive intent review was performed.'))


def public_frame(rollout, identity, subgoal):
    if 'captured_at' not in rollout.obs:
        raise ValueError('Updated simulator capture timestamps are required')
    packet = rollout.observation_packet
    keys = ('episode_id', 'step_id', 'observation_id', 'captured_at', 'images', 'current_state', 'current_eef')
    frame = {key: deepcopy(packet[key]) for key in keys}
    frame.update(identity, task=packet['instruction'], subgoal=subgoal)
    if rollout.history:
        import numpy as np
        record = rollout.history[-1]
        with np.load(rollout.output/f"execution_{record['decision']:03d}.npz", allow_pickle=False) as data:
            frame['executed_actions'] = data['actions'].tolist()
        frame['action_source'] = record['source_detail']
        frame['executed_start_step'] = record['start_tick']
    return frame


def validate_plan(raw, request, rollout, max_chunk_steps):
    required = {'request_id', *IDENTITY, 'decision'}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError('Expected a versioned recovery envelope with a decision')
    if any(type(raw[key]) is not type(request[key]) or raw[key] != request[key]
           for key in ('request_id', *IDENTITY)):
        raise ValueError('Stale or unrelated Planner response')
    decision = deepcopy(raw['decision'])
    validate_decision(decision, rollout.request, max_chunk_steps, rollout.validate)
    decision.update(evaluation_method='pi05_valen_replan', decision_author='planner')
    return decision


class ReplanScheduler:
    def __init__(self, rollout, monitor, planner, *, chunk_steps=1, max_replans=3,
                 policy_timeout=120., planner_timeout=300., max_observation_refreshes=1, emit=None):
        if type(chunk_steps) is not int or not 1 <= chunk_steps <= 5:
            raise ValueError('Monitored chunks must contain 1..5 controls')
        if monitor.config.max_step_gap < chunk_steps:
            raise ValueError('Step-gap limit must cover a monitored chunk')
        if type(max_replans) is not int or max_replans < 1:
            raise ValueError('Recovery budget must be a positive integer')
        if type(max_observation_refreshes) is not int or not 0 <= max_observation_refreshes <= 3:
            raise ValueError('Supplemental observation budget must be an integer in [0, 3]')
        for timeout in (policy_timeout, planner_timeout, monitor.timeout_s):
            if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('Deadlines must be finite and positive')
        if planner is None or not callable(getattr(planner, 'replan', None)):
            raise ValueError('Configure an explicit Planner before starting')
        self.rollout, self.monitor, self.planner = rollout, monitor, planner
        self.chunk_steps, self.max_replans = chunk_steps, max_replans
        self.policy_timeout, self.planner_timeout = policy_timeout, planner_timeout
        self.max_observation_refreshes = max_observation_refreshes
        self.emit = emit or (lambda event: None)
        self.run_id, self.revision, self.replans = uuid.uuid4().hex, 0, 0
        self.subgoal_id, self.subgoal = 'original-task', None
        self.state, self.reason = 'INITIALIZING', None
        self.reports, self.frames, self.attempts = deque(maxlen=100), deque(maxlen=100), []
        self.inference = None
        self.latest_feedback = None
        self.goals = deque()
        self.verify_recovery = False
        self.original_task = None

    @property
    def identity(self):
        return dict(run_id=self.run_id, episode_id=self.rollout.episode,
                    plan_revision=self.revision, subgoal_id=self.subgoal_id,
                    program_id=self.monitor.config.program_id)

    def event(self, kind, **fields):
        self.emit(dict(kind=kind, **self.identity, step_id=self.rollout.tick, **fields))

    def transition(self, state, reason=None):
        self.state, self.reason = state, reason
        self.event('scheduler_state', state=state, reason=reason)

    def fresh(self, frame):
        if not 0 <= time.time() - frame['captured_at'] <= self.monitor.config.max_age_s:
            raise ValueError('Observation exceeded the feedback age budget')

    def review(self):
        self.transition('WAITING_FOR_MONITOR')
        report = self.monitor.review()
        frame = self.rollout.observation_packet
        if (any(report[key] != value for key, value in self.identity.items())
                or report['step_id'] != self.rollout.tick
                or report['observation_id'] != frame['observation_id']):
            raise ValueError('Stale monitor report does not match the paused execution site')
        self.fresh(report)
        self.reports.append(report)
        self.latest_feedback = report
        self.event('monitor_report', report=report)
        self.transition(report['state'])
        return report

    def start_inference(self):
        self.inference = AsyncInference(self.rollout, self.policy_timeout, self.subgoal)

    def finish_inference(self, discard=False):
        result = self.inference.finish(discard)
        self.inference = None
        return result

    def current_frame(self):
        frame = public_frame(self.rollout, self.identity, self.subgoal)
        frame['success_condition'] = self.goals[0]['success_condition']
        return frame

    def discard_inference(self, reason):
        if self.inference is not None:
            self.event('policy_discarded', reason=reason)
            self.finish_inference(discard=True)

    def capture_scene(self):
        old = self.rollout.observation_packet
        self.rollout._observation(self.rollout.output/'jev_refresh'/uuid.uuid4().hex, refresh=True)
        scene = self.current_frame()
        self.fresh(scene)
        if (scene['observation_id'] == old['observation_id'] or scene['step_id'] != old['step_id']
                or scene['episode_id'] != old['episode_id'] or scene['captured_at'] <= old['captured_at']):
            raise ValueError('A fresh non-stepping capture of the same scene is required')
        return scene

    def resolve_uncertainty(self, *, recovery=False):
        paused = 'PAUSED_RECOVERY_UNVERIFIED' if recovery else 'PAUSED_UNCERTAIN'
        self.transition(paused, 'Hold physics while collecting supplemental evidence')
        self.discard_inference('Uncertain evidence invalidates the speculative proposal')
        for _ in range(self.max_observation_refreshes):
            scene = self.capture_scene()
            self.event('supplemental_observation', frame=scene, adds_confirmation_vote=False)
            self.monitor.submit(scene, inspection=True)
            feedback = self.review()
            if feedback['state'] == 'RUNNING':
                return feedback
        self.transition(paused, 'Supplemental observations did not establish clear continuation')
        return None

    def activate_goal(self, reason, *, refresh=True):
        self.revision += 1
        self.subgoal_id = self.goals[0]['subgoal_id']
        self.subgoal = self.goals[0]['instruction']
        self.monitor.activate()
        self.reports.clear()
        self.frames.clear()
        self.verify_recovery = False
        if refresh:
            self.capture_scene()
        write_json(self.rollout.output/f'jev_plan_{self.revision:03d}.json',
                   dict(**self.identity, original_task=self.original_task, remaining_plan=list(self.goals)))
        self.transition('RUNNING', reason)

    def retire_prepared_proposal(self):
        # Preserve, rather than overwrite, the pre-replan request and hardlink.
        destination = self.rollout.output/'jev_discarded'/uuid.uuid4().hex
        destination.mkdir(parents=True)
        index = len(self.rollout.history)
        for name in (f'proposal_{index:03d}.npz', f'request_{index:03d}.json'):
            (self.rollout.output/name).rename(destination/name)
        self.rollout.phase = 'infer'
        self.event('policy_discarded', reason='High-level replan changes the policy instruction',
                   archive=str(destination))

    def recover(self, failure):
        self.transition('PAUSED_FOR_REPLAN', 'Confirmed constraint violation')
        if self.replans >= self.max_replans:
            self.transition('STOPPED', 'Recovery budget exhausted')
            return None
        # Drain the only policy call before reusing its stateful connection. No FK
        # or actions from the speculative result are installed, even if it succeeded.
        self.discard_inference('Confirmed failure; speculative proposal is never installed')
        scene = self.capture_scene()
        self.start_inference()
        proposal = self.finish_inference()
        # This new proposal is grounded in the paused scene, not the discarded one.
        # Astra may select it, edit it, provide an EEF target, or decline recovery.
        request = dict(version=1, request_id=uuid.uuid4().hex, **self.identity,
            original_task=self.rollout.meta['instruction'], current_subgoal=self.subgoal,
            remaining_task=self.rollout.meta['instruction'], current_scene=scene,
            remaining_plan=deepcopy(list(self.goals)),
            failure=failure, confirmation_reports=list(self.reports), evidence_frames=list(self.frames),
            previous_attempts=deepcopy(self.attempts), native_request=deepcopy(self.rollout.request),
            proposal_path=proposal['proposal_path'], max_chunk_steps=self.chunk_steps,
            monitor_program=dict(config=asdict(self.monitor.config), program_id=self.monitor.config.program_id),
            contract=getattr(self.rollout, 'recovery_contract',
                     'Return exactly the echoed identity and request_id plus decision. Decision uses the '
                     'student/edit/eef/stop contract, or mode=replan with reason and 1-8 declarative '
                     'subgoals (subgoal_id, instruction, success_condition), and native_request.request_id. '
                     'Do not change monitoring limits, original task, or execute code as a plan. '
                     'A local correction is not original-task completion.'))
        self.replans += 1
        self.rollout.run['teacher_off'] = False
        write_json(self.rollout.output/'run.json', self.rollout.run)
        path = self.rollout.output/f'jev_replan_{self.replans:03d}.json'
        write_json(path, request)
        self.event('replan_request', request=request)
        self.transition('PLANNING')
        call = PendingCall(lambda: self.planner.replan(deepcopy(request)), self.planner_timeout, 'jev-planner')
        raw = call.get()
        self.transition('VALIDATING_PLAN')
        response = validate_plan(raw, request, self.rollout, self.chunk_steps)
        self.fresh(scene)
        self.attempts.append(dict(request_id=request['request_id'], plan_revision=self.revision,
                                  failure=failure, decision=response, result='not_executed'))
        self.event('replan_response', response=raw)
        if response['mode'] == 'stop':
            self.transition('STOPPED', 'Planner declined recovery')
            return None
        if response['mode'] == 'replan':
            self.retire_prepared_proposal()
            original = dict(subgoal_id='original-task', instruction=self.original_task, success_condition=None)
            self.goals = deque([*deepcopy(response['subgoals']), original])
            self.activate_goal('Validated high-level recovery plan installed')
            self.attempts[-1]['result'] = 'plan_installed_without_execution'
        else:
            self.activate_goal('Validated local recovery decision installed', refresh=False)
        return response

    def run(self):
        try:
            args = self.rollout.next_call().copy()
            args.pop('tool')
            self.rollout.start(**args)
            if 'captured_at' not in self.rollout.obs:
                raise ValueError('Updated simulator capture timestamps are required')
            self.subgoal = self.rollout.meta['instruction']
            self.original_task = self.subgoal
            self.goals.append(dict(subgoal_id='original-task', instruction=self.subgoal, success_condition=None))
            self.rollout.run.update(evaluation_method='pi05_valen_replan', gate_policy='valen-post-action',
                                   teacher_off=True, teacher_invocation='confirmed_failure_only',
                                   predictive_intent_gate=False, monitor_program=asdict(self.monitor.config))
            write_json(self.rollout.output/'run.json', self.rollout.run)
            self.transition('RUNNING')
            while self.rollout.phase != 'done':
                # After a Planner action, observe first. Even speculative policy
                # inference waits for the recovery verification boundary.
                if not self.verify_recovery:
                    self.start_inference()
                feedback = None
                if self.monitor.pending is not None:
                    feedback = self.review()
                if self.verify_recovery and feedback and not feedback['violations']:
                    if feedback['state'] != 'RUNNING':
                        feedback = self.resolve_uncertainty(recovery=True)
                        if feedback is None:
                            break
                    self.verify_recovery = False
                    self.attempts[-1]['verification'] = dict(observation_id=feedback['observation_id'],
                        result='clear_monitor_checks_not_task_success')
                    self.event('recovery_verified', report=feedback)
                elif feedback and feedback['state'] == 'UNCERTAIN':
                    feedback = self.resolve_uncertainty()
                    if feedback is None:
                        break
                if feedback and feedback['violations']:
                    response = self.recover(feedback)
                    if response is None:
                        break
                    if response['mode'] == 'replan':
                        continue
                elif feedback and feedback.get('subgoal_complete') and len(self.goals) > 1:
                    self.discard_inference('Subgoal completed; old instruction proposal is obsolete')
                    completed = self.goals.popleft()
                    self.event('subgoal_completed', subgoal=completed)
                    self.activate_goal('Advance to the next verified subgoal')
                    continue
                else:
                    self.transition('WAITING_FOR_POLICY')
                    if self.inference is None:
                        self.start_inference()
                    self.finish_inference()
                    response = automatic_response(self.rollout.request, self.chunk_steps, feedback)
                    self.transition('SUSPECTED' if feedback and feedback['state'] == 'SUSPECTED' else 'RUNNING')
                self.fresh(self.rollout.observation_packet)
                before = self.rollout.tick
                arguments = self.rollout.next_call().copy()
                arguments.pop('tool')
                arguments['response'] = response
                self.rollout.execute(**arguments)
                if not 0 < self.rollout.tick - before <= self.chunk_steps:
                    raise RuntimeError('Simulator violated the monitored chunk bound')
                frame = self.current_frame()
                self.frames.append(frame)
                if self.attempts and self.attempts[-1]['result'] == 'not_executed':
                    self.attempts[-1]['result'] = dict(step_id=self.rollout.tick, observation_id=frame['observation_id'])
                self.event('executed_chunk', frame=frame, decision_author=response['decision_author'])
                self.verify_recovery = response['decision_author'] == 'planner'
                self.monitor.submit(frame)
            if self.rollout.phase == 'done':
                # Terminal success/timeout belongs to the simulator, never to Valen.
                if self.monitor.pending is not None:
                    self.review()
                terminal = self.rollout.observation_packet.get('result', {}).get('complete', False)
                self.transition('COMPLETED' if terminal else 'STOPPED',
                                'Native rollout ended; no terminal replan' if terminal else 'Decision budget exhausted')
        except Exception as error:
            state = ('PAUSED_MONITOR_UNAVAILABLE' if self.state == 'WAITING_FOR_MONITOR' else
                     'PAUSED_POLICY_UNAVAILABLE' if self.state == 'WAITING_FOR_POLICY' else 'STOPPED')
            self.transition(state, f'{type(error).__name__}: {error}')
        finally:
            if self.rollout.episode is not None and self.rollout.phase != 'done':
                native_result = self.rollout.finish('jev_scheduler_stop')
            else:
                native_result = getattr(self.rollout, 'observation_packet', {}).get('result')
        result = dict(state=self.state, reason=self.reason, **self.identity,
                      step_id=self.rollout.tick, replan_count=self.replans,
                      native_result=native_result,
                      max_unreviewed_control_steps=self.chunk_steps,
                      policy_worker_pending=bool(self.inference and self.inference.call.thread.is_alive()),
                      monitor_worker_pending=bool(self.monitor.pending and self.monitor.pending.thread.is_alive()),
                      latest_feedback=self.latest_feedback, attempts=self.attempts)
        result['remaining_plan'] = list(self.goals)
        write_json(self.rollout.output/'jev_summary.json', result)
        return result
