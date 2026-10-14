"""MAX adapter regressions: native scoring, complete video coverage and ablation."""
import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from .max_run import MaxPolicy, MaxSession, NoAstraScheduler, ProgressWatchdog, encode_video
from .session import LiberoSession
from .max_render import validate_camera_images
from .max_compare import pair_exit_code
from .max_fork import assert_equal_branch


def test_branch_refuses_state_or_budget_mismatch():
    reference = dict(branch_step=146, remaining_controls=374, integration_sha256='full-state',
                     controller_sha256='osc', camera_sha256={'agent':'changed'})
    assert_equal_branch(reference.copy(), reference)
    for key in reference:
        changed = {**reference, key:None}
        with pytest.raises(ValueError, match='branch state mismatch'):
            assert_equal_branch(changed, reference)


def progress_report(step, **extra):
    return dict(run_id='r', episode_id='e', plan_revision=0, subgoal_id='g', program_id='p',
        step_id=step, scores=dict(progress_attempted=.95, stuck=.65), **extra)


def test_progress_requires_temporal_evidence_not_unknown_alone():
    guard = ProgressWatchdog()
    for step in range(5,85,5):
        assert guard.update(progress_report(step)) is None
    result = guard.update(progress_report(85))
    assert result['state']=='PROGRESS_REVIEW' and not result['confirmed_failure']
    assert result['violations']==[] and result['total_reports']==17
    assert guard.update(progress_report(85)) is None
    for step in range(90,170,5):
        assert guard.update(progress_report(step)) is None


def test_progress_does_not_escalate_clear_motion_or_mix_generations():
    guard = ProgressWatchdog()
    for step in range(5,205,5):
        report = progress_report(step)
        report['scores']['stuck'] = .1
        assert guard.update(report) is None
    guard = ProgressWatchdog()
    for step in range(5,85,5):
        assert guard.update(progress_report(step)) is None
    changed = progress_report(85)
    changed['plan_revision'] = 1
    assert guard.update(changed) is None
    assert len(guard.rows)==1


def test_unverified_progress_deadline_cannot_wait_forever():
    guard = ProgressWatchdog()
    for step in range(5,170,5):
        report = progress_report(step, state='UNCERTAIN', subgoal_complete=False)
        report['scores']['stuck'] = .4
        result = guard.update(report)
        if step < 165:
            assert result is None
    assert result['trigger']=='unverified_progress_budget'
    assert not result['confirmed_failure']


def test_smoke_exit_requires_both_native_tasks_to_succeed():
    success = dict(status='episode_finished', success=True)
    failed = dict(status='episode_finished', success=False)
    assert pair_exit_code(dict(control=success, intervention=success)) == 0
    assert pair_exit_code(dict(control=success, intervention=failed)) == 2
    assert pair_exit_code(dict(control=success)) == 1
    assert pair_exit_code(dict(control=success,
        intervention=dict(status='infrastructure_error', success=False))) == 1


class FakeEnv:
    def __init__(self, *, success=False, done=False, fail=False):
        self.success, self.done, self.fail = success, done, fail
        self.sim = SimpleNamespace(data=SimpleNamespace(time=0.),
            get_state=lambda: np.zeros(4))
        self.runtime = SimpleNamespace(applied=False)
        self.events = []
        self.calls = 0

    def step(self, action):
        self.calls += 1
        if self.fail:
            raise RuntimeError('uncertain native step')
        self.sim.data.time += .05
        pixels = np.tile(np.arange(16, dtype=np.uint8)[None,:,None], (16,1,3))
        return {'agentview_image': pixels,
            'robot0_eye_in_hand_image': pixels}, 0, self.done, {}

    def check_success(self):
        return self.success


def session(tmp_path, env, horizon=2):
    value = MaxSession.__new__(MaxSession)
    LiberoSession.__init__(value, env, np.zeros(4), 'complete original task', horizon)
    value.started = True
    value.label = 'test'
    value.frames = tmp_path/'frames'
    value.frames.mkdir()
    value.steps = io.StringIO()
    return value


def test_horizon_does_not_claim_success_and_records_each_step(tmp_path):
    value = session(tmp_path, FakeEnv())
    controls = value.execute(np.zeros((5,7)))
    assert value.terminal and not value.success
    assert len(controls)==value.tick==2
    assert len(list(value.frames.glob('*.png')))==2
    assert len(value.steps.getvalue().splitlines())==2


def test_done_is_not_original_instruction_success(tmp_path):
    value = session(tmp_path, FakeEnv(done=True))
    value.execute(np.zeros((5,7)))
    assert value.terminal and not value.success and value.tick==1


def test_native_success_stops_chunk_immediately(tmp_path):
    value = session(tmp_path, FakeEnv(success=True))
    assert len(value.execute(np.zeros((5,7))))==1
    assert value.success and value.terminal
    with pytest.raises(ValueError):
        value.execute(np.zeros((1,7)))


def test_uncertain_execution_cannot_retry(tmp_path):
    env = FakeEnv(fail=True)
    value = session(tmp_path, env)
    with pytest.raises(RuntimeError):
        value.execute(np.zeros((1,7)))
    with pytest.raises(ValueError):
        value.execute(np.zeros((1,7)))
    assert value.poisoned and env.calls==1


def test_no_astra_recovery_never_calls_planner():
    scheduler = NoAstraScheduler.__new__(NoAstraScheduler)
    events = []
    scheduler.discard_inference = lambda reason: events.append('discard')
    scheduler.event = lambda *args, **kw: events.append(kw)
    scheduler.transition = lambda state, reason: events.append(state)
    def forbidden(*args):
        raise AssertionError('No-Astra arm called a planner')
    scheduler.planner = SimpleNamespace(replan=forbidden)
    assert scheduler.recover({'violations':['stuck']}) is None
    assert events[-1]=='STOPPED' and events[1]['astra_called'] is False


def test_video_rejects_missing_control_frame(tmp_path):
    (tmp_path/'frames').mkdir()
    (tmp_path/'controls.jsonl').write_text('{"sim_dt":0.05}\n')
    with pytest.raises(ValueError, match='every executed control'):
        encode_video(tmp_path)


def test_stable_black_images_are_not_valid_evidence():
    pixels = np.zeros((256,256,3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match='Empty/flat EGL'):
        validate_camera_images(dict(agentview_image=pixels, robot0_eye_in_hand_image=pixels))


def test_dynamic_replays_base_then_resumes_model_without_oracle_input(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'openpi_client', SimpleNamespace(image_tools=SimpleNamespace(
        resize_with_pad=lambda value, *args: value, convert_to_uint8=lambda value: value)))
    calls = []
    def infer(request):
        calls.append(request)
        return {'actions':np.ones((10,7), np.float32)}
    policy = MaxPolicy.__new__(MaxPolicy)
    policy.matched_queries = None
    policy.client = SimpleNamespace(infer=infer)
    policy.index, policy.seed = 0, 195
    policy.replay = {0:dict(instruction='original instruction', actions=np.zeros((10,7)).tolist())}
    obs = dict(agentview=np.zeros((16,16,3),np.uint8), wrist=np.zeros((16,16,3),np.uint8),
        states=np.zeros(8,np.float32), instruction='original instruction', max_policy_step=0,
        max_event_applied=False, sim_state_sha256='audit-only-not-model-input')
    actions, meta = policy.infer(obs, tmp_path/'first.npz')
    assert not calls and np.all(actions==0) and meta['max_query_source']=='control_replay'
    obs.update(max_event_applied=True, max_policy_step=5)
    actions, meta = policy.infer(obs, tmp_path/'second.npz')
    assert len(calls)==1 and np.all(actions==1) and meta['max_query_source']=='model'
    assert set(calls[0])=={'observation/image','observation/wrist_image','observation/state',
        'prompt','libero_max_noise_seed'}


def test_dynamic_missing_base_prefix_is_not_silently_replaced(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'openpi_client', SimpleNamespace(image_tools=SimpleNamespace(
        resize_with_pad=lambda value, *args: value, convert_to_uint8=lambda value: value)))
    policy = MaxPolicy.__new__(MaxPolicy)
    policy.matched_queries = None
    policy.replay = {}
    obs = dict(agentview=np.zeros((16,16,3),np.uint8), wrist=np.zeros((16,16,3),np.uint8),
        instruction='original instruction', max_policy_step=0, max_event_applied=False)
    with pytest.raises(ValueError, match='no matching policy query'):
        policy.infer(obs, tmp_path/'missing.npz')
