"""Deterministic protocol and scheduling tests; no MuJoCo or GPU dependencies."""
from copy import deepcopy
import errno
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Thread

import numpy as np
import pytest

from ..robodojo.jev.monitor import AsyncMonitor, IDENTITY, MonitorConfig, QUESTIONS, ValenMonitor
from ..robodojo.jev.runner import ReplanScheduler, automatic_response
from .contract import CONFIG, HORIZON, PROFILE, action_chunk, execution_actions, validate_response
from .policy import Pi05Client
from .rollout import LiberoRollout, archive_proposal
from .session import LiberoSession, measured_observation


def test_proposal_archive_on_filesystem_without_hardlinks(tmp_path, monkeypatch):
    source, destination = tmp_path / 'worker.npz', tmp_path / 'proposal.npz'
    actions = np.arange(70, dtype=np.float32).reshape(10, 7)
    np.savez_compressed(source, actions=actions)
    def no_links(*args):
        raise OSError(errno.EPERM, 'Hard links unsupported')
    monkeypatch.setattr('hybrid_rollout.libero.rollout.os.link', no_links)
    archive_proposal(source, destination)
    assert source.read_bytes() == destination.read_bytes()
    with np.load(destination) as saved:
        np.testing.assert_array_equal(saved['actions'], actions)
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        archive_proposal(source, destination)
    assert destination.read_bytes() == original


def raw_observation():
    return dict(robot0_eef_pos=np.array([.1, .2, .3]), robot0_eef_quat=np.array([0., 0., 0., 1.]),
                robot0_gripper_qpos=np.array([.02, -.02]),
                agentview_image=np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
                robot0_eye_in_hand_image=np.full((4, 4, 3), 7, dtype=np.uint8),
                privileged_object_pose='MUST_NOT_REACH_MODEL')


class Environment:
    def __init__(self, success_at=15):
        self.steps, self.resets, self.restores, self.refreshes = 0, 0, 0, 0
        self.success_at, self.actions, self.closed = success_at, [], False
        self.env = self
        self.sim = SimpleNamespace(data=SimpleNamespace(time=0.))

    def reset(self):
        self.resets += 1

    def set_init_state(self, value):
        self.restores += 1
        return raw_observation()

    def step(self, action):
        self.steps += 1
        self.sim.data.time += .05
        self.actions.append(action)
        return raw_observation(), 999, self.check_success(), {'hidden': 'MUST_NOT_REACH_MODEL'}

    def check_success(self):
        return self.steps >= self.success_at

    def _update_observables(self, force=False):
        assert force
        self.refreshes += 1

    def _get_observations(self):
        return raw_observation()

    def close(self):
        self.closed = True


class Student:
    metadata = {'config': CONFIG}

    def __init__(self):
        self.instructions = []

    def infer(self, observation, output):
        self.instructions.append(observation['instruction'])
        actions = np.zeros((HORIZON, 7), np.float32)
        np.savez_compressed(output, actions=actions)
        return actions, {'prediction_id': str(len(self.instructions))}


def scores(**kwargs):
    return {**{k: .95 if k.endswith('attempted') else .05 for k in QUESTIONS}, **kwargs}


@pytest.mark.parametrize('success_at,expected_steps,expected_code', [(17, 7, 0), (999, 12, 2)])
def test_pi_only_baseline_runs_until_native_outcome(tmp_path, monkeypatch, success_at, expected_steps, expected_code):
    from . import baseline
    env = Environment(success_at=success_at)
    session = LiberoSession(env, [0], 'put the bowl on the plate', 12)
    class PureStudent(Student):
        def close(self):
            pass
    monkeypatch.setattr(baseline, 'Pi05Client', lambda *args, **kwargs: PureStudent())
    monkeypatch.setattr(baseline, 'create_session', lambda *args: session)
    monkeypatch.setattr(baseline, 'configure_assets', lambda *args: None)
    output = tmp_path / 'baseline'
    code = baseline.main(['--libero-root', str(tmp_path), '--checkpoint', str(tmp_path),
                          '--output', str(output)])
    result = json.loads((output / 'result.json').read_text())
    assert code == expected_code and result['step_id'] == expected_steps
    assert result['complete'] and result['success'] is (expected_code == 0)
    assert env.resets == 1 and env.restores == 1 and env.closed
    assert 'MUST_NOT_REACH_MODEL' not in (output / 'frames.jsonl').read_text()
    assert not (output / 'jev_events.jsonl').exists()


def setup(tmp_path, decision='edit', monitor_fn=None, planner_fn=None, **options):
    env = Environment(success_at=16)
    session = LiberoSession(env, [0], 'put the bowl on the plate', 20)
    student = Student()
    rollout = LiberoRollout(tmp_path, student, lambda: session, max_decisions=options.pop('max_decisions', 0))
    requests, windows = [], []

    class Monitor:
        def evaluate(self, frames):
            windows.append(deepcopy(frames))
            if monitor_fn:
                return monitor_fn(frames)
            return scores(collision=.95 if frames[-1]['plan_revision'] == 0 else .05)

    class Planner:
        def replan(self, request):
            requests.append(deepcopy(request))
            steps = env.steps
            if planner_fn:
                result = planner_fn(request)
            else:
                response = automatic_response(request['native_request'], 1, {})
                response.pop('decision_author')
                response.pop('evaluation_method')
                response['mode'] = decision
                response['assessment']['execution_status'] = 'failed'
                if decision == 'edit':
                    response['edit'] = {'delta': [.1, 0., 0., 0., 0., 0.], 'gripper': 'open'}
                result = {**{k: request[k] for k in ('request_id', *IDENTITY)}, 'decision': response}
            assert env.steps == steps
            return result

    observer = AsyncMonitor(Monitor(), MonitorConfig(), 2.)
    scheduler = ReplanScheduler(rollout, observer, Planner(), **options)
    return scheduler, rollout, env, student, requests, windows


def uncertainty_decision(request, mode='student'):
    decision = automatic_response(request['native_request'], 1, {})
    decision.pop('decision_author')
    decision.pop('evaluation_method')
    decision['mode'] = mode
    decision['assessment']['intent_status'] = 'aligned'
    decision['assessment']['intent_evidence'] = 'Fixture: visible target and a bounded approach.'
    return {**{k: request[k] for k in ('request_id', *IDENTITY)}, 'decision': decision}


def test_uncertainty_review_executes_only_one_student_step_then_monitors_again(tmp_path):
    scheduler, rollout, env, student, requests, windows = setup(tmp_path,
        max_decisions=3, max_uncertainty_reviews=2, include_initial_context=True,
        monitor_fn=lambda frames: scores(collision=.5), planner_fn=uncertainty_decision)
    result = scheduler.run()
    assert result['state'] == 'STOPPED' and result['reason'] == 'Decision budget exhausted'
    assert result['step_id'] == 3 and result['replan_count'] == 0
    assert result['uncertainty_review_count'] == 2
    assert all(r['review_kind'] == 'uncertainty' and not r['failure']['violations'] for r in requests)
    assert all(a['decision']['mode'] == 'student' and a['decision']['steps'] == 1
               for a in result['attempts'])
    assert all('verification' not in a for a in result['attempts'])
    assert windows[0][0]['context_only'] and windows[0][0]['step_id'] == 0
    assert {w[-1]['step_id'] for w in windows} == {1, 2, 3}
    assert result['latest_feedback']['state'] == 'UNCERTAIN'


@pytest.mark.parametrize('fault', ['edit', 'unaligned', 'stale', 'stop', 'timeout'])
def test_uncertainty_review_cannot_bypass_validation(tmp_path, fault):
    def planner(request):
        if fault == 'timeout':
            raise TimeoutError('Fixture: reviewer unavailable')
        response = uncertainty_decision(request, 'stop' if fault == 'stop' else 'student')
        if fault == 'edit':
            response['decision'].update(mode='edit', edit={'delta': [0.] * 6, 'gripper': 'keep'})
            response['decision']['assessment']['execution_status'] = 'failed'
        if fault == 'unaligned':
            response['decision']['assessment']['intent_status'] = 'uncertain'
        if fault == 'stale':
            response['request_id'] = 'old'
        return response
    scheduler, *_ = setup(tmp_path, max_decisions=3, max_uncertainty_reviews=2,
        monitor_fn=lambda frames: scores(collision=.5), planner_fn=planner)
    result = scheduler.run()
    assert result['step_id'] == 1
    assert result['replan_count'] == 0 and result['uncertainty_review_count'] == 1
    assert result['native_result']['success'] is False


def test_uncertainty_review_budget_is_enforced(tmp_path):
    scheduler, *_ = setup(tmp_path, max_decisions=4, max_uncertainty_reviews=1,
        monitor_fn=lambda frames: scores(collision=.5), planner_fn=uncertainty_decision)
    result = scheduler.run()
    assert result['step_id'] == 2 and result['uncertainty_review_count'] == 1
    assert result['reason'] == 'Uncertainty review budget exhausted'


def test_initial_context_never_casts_a_confirmation_vote(tmp_path):
    scheduler, *_ = setup(tmp_path, include_initial_context=True)
    result = scheduler.run()
    assert result['attempts'][0]['failure']['step_id'] == 2
    assert result['attempts'][0]['review_kind'] == 'confirmed_failure'


def test_uncertainty_option_does_not_bypass_correction_verification(tmp_path):
    scheduler, *_ = setup(tmp_path, max_uncertainty_reviews=5,
        monitor_fn=lambda frames: scores(collision=.95 if frames[-1]['plan_revision'] == 0 else .5))
    result = scheduler.run()
    assert result['state'] == 'PAUSED_RECOVERY_UNVERIFIED'
    assert result['step_id'] == 3 and result['replan_count'] == 1
    assert result['uncertainty_review_count'] == 0
    assert 'verification' not in result['attempts'][0]


def test_observation_rotation_proprio_and_no_privileged_fields():
    raw = raw_observation()
    obs = measured_observation(raw, 'task')
    np.testing.assert_array_equal(obs['agentview'], raw['agentview_image'][::-1, ::-1])
    np.testing.assert_allclose(obs['states'], [.1, .2, .3, 0., 0., 0., .02, -.02])
    assert 'privileged_object_pose' not in obs
    raw['robot0_eef_quat'] = [0., 0., np.sin(.25), np.cos(.25)]
    np.testing.assert_allclose(measured_observation(raw, 'task')['states'][3:6], [0., 0., .5])


def test_initialization_and_refresh_never_rewind_or_step():
    env = Environment()
    session = LiberoSession(env, [1], 'task', 10)
    first = session.start()
    assert env.steps == 10 and session.tick == 0
    assert all(a == [0.] * 6 + [-1.] for a in env.actions)
    next_obs = session.observe(refresh=True)
    assert next_obs['captured_at'] > first['captured_at']
    assert env.steps == 10 and env.refreshes == 1
    with pytest.raises(ValueError, match='initialization'):
        session.start()
    assert env.resets == env.restores == 1


def test_refresh_detects_physics_advance():
    env = Environment()
    session = LiberoSession(env, [1], 'task', 10)
    session.start()
    env._update_observables = lambda **_: setattr(env.sim.data, 'time', 999)
    with pytest.raises(RuntimeError, match='advanced'):
        session.observe(refresh=True)
    assert session.poisoned


def test_simulator_stays_on_owner_and_execution_errors_poison_session():
    env = Environment()
    session = LiberoSession(env, [1], 'task', 10)
    session.start()
    errors = []
    def worker():
        try:
            session.observe(refresh=True)
        except RuntimeError as error:
            errors.append(str(error))
    thread = Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and env.refreshes == 0
    def fail_after_unknown_execution(action):
        raise RuntimeError('transport/physics failure')
    env.step = fail_after_unknown_execution
    with pytest.raises(RuntimeError):
        session.execute(np.zeros((1, 7)))
    assert session.poisoned
    with pytest.raises(ValueError):
        session.execute(np.zeros((1, 7)))


def test_terminal_chunk_stops_immediately_and_success_is_native():
    env = Environment(success_at=12)
    session = LiberoSession(env, [1], 'task', 10)
    session.start()
    actions = session.execute(np.zeros((5, 7)))
    assert actions.shape == (2, 7) and session.tick == 2 and session.success
    with pytest.raises(ValueError):
        session.execute(np.zeros((1, 7)))


def test_confirmed_failure_edits_once_and_returns_to_student(tmp_path):
    scheduler, rollout, env, student, requests, windows = setup(tmp_path)
    result = scheduler.run()
    assert result['state'] == 'COMPLETED' and result['native_result']['success']
    assert len(requests) == 1 and env.resets == env.restores == 1
    assert [h['source_detail'] for h in rollout.history] == ['student', 'student', 'edit', 'student', 'student', 'student']
    np.testing.assert_allclose(env.actions[12], [.1, 0, 0, 0, 0, 0, -1])
    assert requests[0]['current_scene']['step_id'] == 2
    assert 'LIBERO' in requests[0]['contract']
    assert 'MUST_NOT_REACH_MODEL' not in json.dumps(requests + windows)
    assert 'reward' not in json.dumps(windows)
    assert result['attempts'][0]['verification']['result'] == 'clear_monitor_checks_not_task_success'
    rollout.close()
    assert env.closed


def test_high_level_replan_changes_policy_prompt_without_physics(tmp_path):
    def planner(request):
        return {**{k: request[k] for k in ('request_id', *IDENTITY)}, 'decision': {
            'request_id': request['native_request']['request_id'], 'mode': 'replan', 'reason': 'recover',
            'subgoals': [{'subgoal_id': 'retry', 'instruction': 'reacquire the bowl',
                         'success_condition': 'the bowl is visibly lifted'}]}}
    scheduler, rollout, env, student, requests, _ = setup(tmp_path, planner_fn=planner)
    result = scheduler.run()
    assert result['state'] == 'COMPLETED'
    assert 'reacquire the bowl' in student.instructions
    assert result['remaining_plan'][-1]['subgoal_id'] == 'original-task'
    assert env.resets == env.restores == 1
    assert len(list((tmp_path / 'jev_discarded').glob('*/request_002.json'))) == 1
    assert all(h['source_detail'] == 'student' for h in rollout.history)


def test_uncertainty_pauses_without_planner(tmp_path):
    scheduler, rollout, env, _, requests, _ = setup(tmp_path, monitor_fn=lambda _: scores(collision=.5))
    result = scheduler.run()
    assert result['state'] == 'PAUSED_UNCERTAIN'
    assert rollout.tick == 1 and env.steps == 11 and not requests
    assert not result['native_result']['success']


def test_stale_plan_and_dual_arm_eef_are_rejected(tmp_path):
    scheduler, rollout, env, _, requests, _ = setup(tmp_path, decision='eef')
    result = scheduler.run()
    assert result['state'] == 'STOPPED' and rollout.tick == 2
    assert 'unsupported' in result['reason']


def test_stale_outer_identity_cannot_execute(tmp_path):
    def planner(request):
        return {**{k: request[k] for k in ('request_id', *IDENTITY)}, 'run_id': 'old', 'decision': {}}
    scheduler, rollout, env, _, _, _ = setup(tmp_path, planner_fn=planner)
    assert scheduler.run()['state'] == 'STOPPED'
    assert rollout.tick == 2


def test_budget_exhaustion_is_not_task_success(tmp_path):
    scheduler, rollout, env, _, requests, _ = setup(tmp_path, max_decisions=2, monitor_fn=lambda _: scores())
    result = scheduler.run()
    assert result['state'] == 'STOPPED'
    assert not result['native_result']['complete'] and not result['native_result']['success']
    assert rollout.tick == 2


def test_planner_stop_does_not_execute_or_claim_success(tmp_path):
    scheduler, rollout, env, _, _, _ = setup(tmp_path, decision='stop')
    result = scheduler.run()
    assert result['state'] == 'STOPPED' and not result['native_result']['success']
    assert rollout.tick == 2 and env.steps == 12


def test_monitor_failure_halts_without_replanning(tmp_path):
    def broken(_):
        raise RuntimeError('monitor down')
    scheduler, rollout, env, _, requests, _ = setup(tmp_path, monitor_fn=broken)
    result = scheduler.run()
    assert result['state'] == 'PAUSED_MONITOR_UNAVAILABLE'
    assert rollout.tick == 1 and not requests


def test_native_horizon_without_success(tmp_path):
    scheduler, rollout, env, _, _, _ = setup(tmp_path, monitor_fn=lambda _: scores())
    env.success_at = 100
    result = scheduler.run()
    assert result['native_result']['complete'] and not result['native_result']['success']
    assert rollout.tick == 20


@pytest.mark.parametrize('offset', [float('nan'), .251, True])
def test_invalid_edit_is_rejected_before_execution(offset):
    request = {'request_id': 'r', 'step_id': 2, 'instruction': 'task', 'max_chunk_steps': 1}
    response = automatic_response(request, 1, {})
    response.update(mode='edit', edit={'delta': [offset, 0, 0, 0, 0, 0], 'gripper': 'keep'})
    response['assessment']['execution_status'] = 'failed'
    with pytest.raises(ValueError, match='six finite'):
        validate_response(response, request)


def test_edit_clipping_and_gripper_semantics():
    proposal = np.full((10, 7), .95)
    response = {'steps': 1, 'mode': 'edit', 'edit': {'delta': [.1] * 6, 'gripper': 'open'}}
    values = execution_actions(proposal, response)
    np.testing.assert_allclose(values, [[1, 1, 1, 1, 1, 1, -1]])
    np.testing.assert_allclose(proposal, .95)


class SocketClient:
    def __init__(self, host, port):
        self.closed = False
        self._ws = SimpleNamespace(ping_timeout=20, close=lambda: setattr(self, 'closed', True))
        self.metadata = dict(config=CONFIG, robot_profile=PROFILE, action_dim=7, action_horizon=10,
                             inferences=0, checkpoint=str(Path('/tmp/libero').resolve()), checkpoint_sha256='digest')
        self.shape = (10, 7)

    def get_server_metadata(self):
        return self.metadata

    def infer(self, inputs):
        self.inputs = inputs
        return {'actions': np.zeros(self.shape), 'policy_identity': {'checkpoint_sha256': 'digest', 'inference_index': 0}}


def test_policy_uses_libero_inputs_and_preserves_actions(tmp_path):
    socket = SocketClient('', 0)
    policy = Pi05Client('/tmp/libero', client_factory=lambda *_: socket)
    actions, identity = policy.infer(measured_observation(raw_observation(), 'pick bowl'), tmp_path / 'p.npz')
    assert actions.shape == (10, 7)
    assert set(socket.inputs) == {'observation/state', 'observation/image', 'observation/wrist_image', 'prompt'}
    assert socket.inputs['observation/image'].shape == (224, 224, 3)
    assert socket.inputs['prompt'] == 'pick bowl'
    assert socket._ws.ping_timeout is None
    with pytest.raises(ValueError, match='Stale'):
        policy.infer(measured_observation(raw_observation(), 'task'), tmp_path / 'p2.npz')


def test_policy_rejects_robodojo_server_and_bad_shapes(tmp_path):
    socket = SocketClient('', 0)
    socket.metadata['action_dim'] = 14
    with pytest.raises(ValueError, match='incompatible'):
        Pi05Client('/tmp/libero', client_factory=lambda *_: socket)
    assert socket.closed
    for shape in ((50, 14), (10, 32), (7,), (0, 7)):
        with pytest.raises(ValueError):
            action_chunk(np.zeros(shape), horizon=10)


def test_valen_request_uses_two_libero_views_and_scope():
    captured = []
    monitor = ValenMonitor('http://127.0.0.1:8001/predict', 1, scope='LIBERO test')
    monitor.post = lambda payload: captured.append(payload) or {'answers': {k: {'type': 'noul', 'noul': v} for k, v in scores().items()}}
    monitor.evaluate([{'step_id': 1, 'images': [{'camera': n, 'path': f'/{n}.png'} for n in ('agentview', 'wrist')]}])
    content = captured[0]['state']['messages'][0]['content']
    assert json.loads(content[0]['text'])['scope'] == 'LIBERO test'
    assert len([v for v in content if v['type'] == 'image_url']) == 2


def test_http_valen_adapter_round_trip():
    from http.server import HTTPServer
    from .valen_server import make_handler
    requests = []
    def predict(payload):
        requests.append(payload)
        return {'answers': {k: {'type': 'noul', 'noul': v} for k, v in scores().items()}}
    with HTTPServer(('127.0.0.1', 0), make_handler(predict)) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            monitor = ValenMonitor(f'http://127.0.0.1:{server.server_port}/predict', 2, scope='LIBERO')
            assert monitor.evaluate([{'step_id': 1, 'images': []}]) == scores()
            assert set(requests[0]['questions']) == set(QUESTIONS)
        finally:
            server.shutdown()
            worker.join(timeout=2)


def test_asset_configuration_is_private_and_noninteractive(tmp_path, monkeypatch):
    from .run import configure_assets
    root = tmp_path / 'LIBERO'
    for folder in ('bddl_files', 'init_files', 'assets'):
        (root / 'libero/libero' / folder).mkdir(parents=True)
    output = tmp_path / 'run'
    output.mkdir()
    monkeypatch.setenv('LIBERO_CONFIG_PATH', 'original')
    configure_assets(root, output)
    import os
    config = json.loads((Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml').read_text())
    assert config['assets'] == str(root / 'libero/libero/assets')
    assert Path(os.environ['LIBERO_CONFIG_PATH']).parent == output
