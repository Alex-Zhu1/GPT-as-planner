"""Recovery backends. Neither backend is given a simulator or policy connection."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock

from ..io import InputError, write_json
from .monitor import IDENTITY, JSONEndpoint
from .plans import recovery_response_schema, validate_decision


class HTTPPlanner(JSONEndpoint):
    def replan(self, request):
        return self.post(request)


class DecisionOnlyRollout:
    """Reuse Codex's existing decision tools without executing their decisions."""
    def __init__(self, request, output):
        self.envelope, self.output = request, Path(output)
        self.request = deepcopy(request['native_request'])
        self.episode, self.tick = request['episode_id'], self.request['step_id']
        self.phase, self.counters = 'execute', {}
        self.observation_path = self.output/'paused_scene.json'
        write_json(self.observation_path, request['current_scene'])
        self.response = None

    def next_call(self):
        if self.phase == 'done':
            return None
        return dict(tool='robodojo_execute', proposal_path=self.envelope['proposal_path'],
                    output_dir=str(self.output/'selected_decision'),
                    response='Select one bounded recovery decision. This tool records it without execution.')

    def start(self, **kwargs):
        raise InputError('Episode already running and paused. Reset is forbidden.')

    def infer(self, **kwargs):
        raise InputError('A fresh paused-scene proposal is already supplied; select a recovery decision.')

    def execute(self, **kwargs):
        from ..robodojo_server.client import validate_dual_response
        expected = self.next_call()
        if expected is None or any(kwargs.get(key) != expected[key] for key in ('proposal_path', 'output_dir')):
            raise InputError('Stale decision submission')
        decision = kwargs.get('response')
        try:
            validate_decision(decision, self.request, self.envelope['max_chunk_steps'], validate_dual_response)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise InputError(str(error)) from error
        self.response, self.phase = deepcopy(decision), 'done'
        return dict(rollout_finished=True, decision_recorded=True, simulation_executed=False,
                    step_id=self.tick, next_call=None,
                    note='Planning completed. The owning scheduler will validate and execute separately.')


class CodexPlanner:
    """Invoke Astra for a scheduler-requested evidence or recovery review."""
    def __init__(self, output, codex, timeout=300., policy_factory=None):
        self.output, self.codex, self.timeout = Path(output), codex, timeout
        self.policy_factory = policy_factory
        self.worker = None
        self._lock, self._closed = Lock(), False

    def replan(self, request):
        if self.policy_factory is None:
            from ..skill.run import CodexPolicy
            factory = CodexPolicy
        else:
            factory = self.policy_factory
        directory = self.output/f"jev_planner_{request['request_id']}"
        worker = factory(directory, self.codex, timeout=self.timeout, recovery_schema=recovery_response_schema())
        with self._lock:
            if self._closed:
                worker.close()
                raise RuntimeError('Planner was closed while starting')
            self.worker = worker
        try:
            path = directory/'replan_request.json'
            write_json(path, request)
            deferred = DecisionOnlyRollout(request, directory)
            review = request.get('review_kind', 'confirmed_failure')
            context = ('The same RoboDojo episode is paused after a confirmed Valen constraint failure. '
                       if review == 'confirmed_failure' else
                       'The same RoboDojo episode is paused for an evidence review, not a confirmed failure. '
                       'Read review_reason and accumulated_evidence when present. Persistent uncertainty '
                       'alone is not grounds to invent a grasp failure or collision. ')
            worker.run(deferred, initial_prompt=(
                context +
                'Act only as the recovery Planner, not the rollout owner. Read the complete request at '
                f'{path}. Inspect current_scene images/proprio separately from historical failure evidence. '
                'Use the unchanged outcome/intent gate and EEF geometry contract. '
                'Do not reset, connect to the simulator, or execute actions through shell/code. '
                'The supplied pi05 proposal was freshly inferred at this paused scene. '
                'Choose student self-recovery, edit, EEF correction, high-level replan, or stop. '
                'For mode=replan return 1-8 ordered subgoals with subgoal_id, instruction, and '
                'observable success_condition. These instructions will actually be sent to pi05. '
                'Preserve the remaining original task; the host retains original-task completion as a final obligation. '
                f"At most {request['max_chunk_steps']} control steps may be selected. "
                'The robodojo_execute tool here records one decision WITHOUT physics execution; '
                'after it returns, finish your turn. The host validates and executes separately. '
                'A recovered grasp is not completion of the original task. Write public text in English. '
                'Next call: ' + json.dumps(deferred.next_call())))
            if deferred.response is None:
                raise ValueError('Planner ended without a structured decision')
            return dict(**{key: request[key] for key in ('request_id', *IDENTITY)}, decision=deferred.response)
        finally:
            self._close_worker()

    def _close_worker(self):
        with self._lock:
            worker = self.worker
            self.worker = None
        if worker is not None:
            worker.close()

    def close(self):
        with self._lock:
            self._closed = True
        self._close_worker()
