"""CPU-only geometric and end-to-end correction tests; no model/simulator calls."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .eef import command_preview, controller_spec, target_action, validate_target
from .session import LiberoSession
from .test_integration import Environment, raw_observation, scores
from .test_continuous import make_runner
from .contract import validate_response
from ..robodojo.jev.runner import automatic_response
from ..robodojo.jev.monitor import IDENTITY


def controller(**kwargs):
    return SimpleNamespace(**dict(dict(name='OSC_POSE', use_delta=True, use_ori=True,
        impedance_mode='fixed', position_limits=None, orientation_limits=None,
        input_min=[-1.] * 6, input_max=[1.] * 6,
        output_min=[-.05] * 3 + [-.5] * 3, output_max=[.05] * 3 + [.5] * 3,
        control_freq=20, eef_name='grip_site'), **kwargs))


def current():
    return dict(position=[.1, .2, .3], quaternion_xyzw=[0., 0., 0., 1.], frame='MuJoCo world')


def target():
    return dict(current(), position=[.13, .2, .3], gripper='keep')


def test_world_rotation_order_and_nominal_preview():
    spec = controller_spec(controller())
    initial = dict(current(), quaternion_xyzw=Rotation.from_euler('x', 70, degrees=True).as_quat().tolist())
    expected = Rotation.from_rotvec([0., 0., .2]) * Rotation.from_quat(initial['quaternion_xyzw'])
    goal = dict(target(), quaternion_xyzw=expected.as_quat().tolist())
    validate_target(goal, initial, spec)
    a = target_action(goal, initial, spec, -.8)
    np.testing.assert_allclose(a, [.6, 0., 0., 0., 0., .4, -.8], atol=1e-7)
    preview = command_preview(np.asarray([a]), initial, spec)
    np.testing.assert_allclose(preview['trajectory'][0]['position'], goal['position'], atol=1e-8)
    assert (Rotation.from_quat(preview['trajectory'][0]['quaternion_xyzw'])*expected.inv()).magnitude() < 1e-7
    assert 'not FK' in preview['interpretation']
    # q and -q denote the same pose.
    goal['quaternion_xyzw'] = (-expected.as_quat()).tolist()
    np.testing.assert_allclose(target_action(goal, initial, spec, -.8), a, atol=1e-7)


def test_nondefault_scaling_and_direction_preserving_saturation():
    spec = controller_spec(controller(output_min=[-.01]*3+[-.1]*3, output_max=[.01]*3+[.1]*3))
    goal = dict(target(), position=[.13, .22, .3], gripper='close')
    np.testing.assert_allclose(target_action(goal, current(), spec, -1), [1., 2/3, 0, 0, 0, 0, 1], atol=1e-7)


@pytest.mark.parametrize('changes', [dict(use_delta=False), dict(impedance_mode='variable'),
    dict(output_min=[-.04]*6), dict(input_max=[2.]*6), dict(position_limits=np.zeros((2,3)))])
def test_unknown_controller_semantics_fail_closed(changes):
    with pytest.raises(ValueError):
        controller_spec(controller(**changes))


@pytest.mark.parametrize('changes', [dict(position=[.151, .2, .3]), dict(frame='camera'),
    dict(quaternion_xyzw=[0, 0, 0, 2]), dict(position=[float('nan'), 0, 0]),
    dict(quaternion_xyzw=Rotation.from_rotvec([0, 0, .351]).as_quat().tolist()),
    dict(gripper='closed'), dict(left={})])
def test_reject_invalid_or_unbounded_targets(changes):
    with pytest.raises(ValueError):
        validate_target(dict(target(), **changes), current(), controller_spec(controller()))


def test_eef_assessment_and_stale_identity_gate():
    req = dict(request_id='fresh', step_id=4, max_chunk_steps=5, instruction='move pot',
               current_eef=current(), eef_control=controller_spec(controller()))
    response = automatic_response(req, 3, {})
    response.update(mode='eef', target=target())
    with pytest.raises(ValueError, match='Takeover requires'):
        validate_response(response, req)
    response['assessment']['intent_status'] = 'misaligned'
    validate_response(response, req)
    response['request_id'] = 'old'
    with pytest.raises(ValueError, match='Stale'):
        validate_response(response, req)


class GeometricEnvironment(Environment):
    """A deliberately imperfect plant: achieves only half each OSC command."""
    def __init__(self, success_at=999):
        super().__init__(success_at)
        self.position = np.array(current()['position'])
        self.rotation = Rotation.identity()
        self.robots = [SimpleNamespace(controller=controller())]
        self.sim.data.get_site_xpos = lambda name: self.position.copy()
        self.sim.data.get_site_xmat = lambda name: self.rotation.as_matrix()

    def obs(self):
        return dict(raw_observation(), robot0_eef_pos=self.position.copy(),
                    robot0_eef_quat=self.rotation.as_quat())

    def set_init_state(self, value):
        super().set_init_state(value)
        return self.obs()

    def _get_observations(self):
        return self.obs()

    def step(self, action):
        _, reward, done, info = super().step(action)
        self.position += .5 * .05 * np.asarray(action[:3])
        self.rotation = Rotation.from_rotvec(.5*.5*np.asarray(action[3:6])) * self.rotation
        return self.obs(), reward, done, info


def test_remeasure_each_control_keep_gripper_and_stop_at_terminal():
    env = GeometricEnvironment()
    session = LiberoSession(env, [0], 'move pot', 2, settle_steps=0)
    session.start()
    proposal = np.zeros((10, 7)); proposal[:, 6] = np.arange(10)/10
    executed = session.execute_eef(target(), proposal, 5, current())
    # .03m error -> .6; next measured .015m error -> .3. Never replay .6.
    np.testing.assert_allclose(executed[:, 0], [.6, .3], atol=1e-7)
    np.testing.assert_allclose(executed[:, 6], [0, .1], atol=1e-7)
    assert session.tick == 2 and len(session.eef_trace) == 2
    assert env.resets == env.restores == 1


def test_mismatched_live_pose_executes_nothing():
    env = GeometricEnvironment()
    session = LiberoSession(env, [0], 'move pot', 10, settle_steps=0)
    session.start()
    env.position[0] += .01
    with pytest.raises(ValueError, match='does not match'):
        session.execute_eef(target(), np.zeros((10, 7)), 2, current())
    assert session.tick == env.steps == 0


@pytest.mark.parametrize('uncertain', [False, True])
def test_replan_then_eef_then_fresh_pi_or_hold(tmp_path, uncertain):
    env = GeometricEnvironment()
    def monitor(frames):
        if frames[-1].get('action_source') == 'eef':
            before = env.steps, len(student.instructions)
            time.sleep(.025)
            assert before == (env.steps, len(student.instructions)), 'Pi handback preceded correction evidence'
        return scores(collision=.5 if uncertain and frames[-1].get('action_source') == 'eef' else .05)
    runner, rollout, student, _, windows, events = make_runner(tmp_path, env, monitor,
        astra=True, horizon=35, progress_budget=10, subgoal_budget=10, max_replans=2)
    calls = []
    def plan(request):
        calls.append(deepcopy(request))
        native = request['native_request']
        assert 'student_eef_command_preview' in native
        if len(calls) == 1:
            decision = dict(request_id=native['request_id'], mode='replan', reason='Fix task instruction',
                subgoals=[dict(subgoal_id='recovery-1', instruction='Lift the pot.', success_condition='Pot suspended')])
        else:
            decision = automatic_response(native, 3, {})
            decision.pop('decision_author'); decision.pop('evaluation_method')
            decision.update(mode='eef', target=dict(native['current_eef'], gripper='close'))
            decision['target'] = {k:v for k,v in decision['target'].items() if k in target()}
            decision['target']['position'][0] += .03
            decision['assessment']['execution_status'] = 'failed'
            decision['reason'] = 'Visible local misalignment; correct end effector then reobserve'
        return {**{k:request[k] for k in ('request_id', *IDENTITY)}, 'decision':decision}
    runner.planner = SimpleNamespace(replan=plan)
    result = runner.run()
    eef_index = next(i for i,r in enumerate(rollout.history) if r['source_detail']=='eef')
    assert calls[1]['current_subgoal'] == 'Lift the pot.'
    assert rollout.history[eef_index]['executed_steps'] == 3
    trace = json.loads(next(tmp_path.glob('eef_*.json')).read_text())
    np.testing.assert_allclose([t['executed_action'][0] for t in trace], [.6,.3,.15], atol=1e-6)
    assert any(e['kind']=='local_correction_review' for e in events)
    assert all('proposal_actions' not in f for w in windows for f in w)
    if uncertain:
        assert result['state']=='STOPPED'
        assert result['reason']=='Required Astra review budget exhausted'
        assert result['native_result']['complete'] is False
        assert rollout.history[-1]['source_detail']=='eef'
    else:
        assert result['native_result']['complete'] and result['step_id']==35
        next_request = json.loads((tmp_path/f'request_{eef_index+1:03d}.json').read_text())
        assert next_request['step_id']==23
        assert next_request['current_eef']['position'][0] > .12
        assert next_request['student_instruction'].endswith('Next, Lift the pot.')
        assert rollout.history[eef_index+1]['source_detail']=='student'
    assert env.resets == env.restores == 1


def test_partial_eef_failure_keeps_audit_and_never_retries(tmp_path):
    from .rollout import LiberoRollout
    from .test_integration import Student
    from ..robodojo.jev.runner import AsyncInference
    class FailingEnvironment(GeometricEnvironment):
        fail_at = 999
        def step(self, action):
            if self.steps == self.fail_at:
                raise RuntimeError('Fixture transport failure; execution uncertain')
            return super().step(action)
    env = FailingEnvironment()
    session = LiberoSession(env, [0], 'move pot', 20, settle_steps=0)
    rollout = LiberoRollout(tmp_path, Student(), lambda: session, chunk_steps=5)
    rollout.start()
    # A failure assessment is meaningful only after prior execution.
    session.execute(np.zeros((1, 7)))
    rollout._observation(tmp_path/'after-initial-control')
    env.fail_at = 2
    AsyncInference(rollout, 2).finish()
    response = automatic_response(rollout.request, 3, {})
    response.update(mode='eef', target=target())
    response['assessment']['execution_status']='failed'
    with pytest.raises(RuntimeError, match='transport failure'):
        rollout.execute(response=response)
    assert rollout.tick == session.tick == 2 and session.poisoned
    assert len(json.loads((tmp_path/'eef_000.json').read_text())) == 1
    with pytest.raises(ValueError):
        session.execute(np.zeros((1, 7)))
    assert env.steps == 2


def test_controller_site_reference_keeps_pi_body_quaternion_unchanged():
    from .session import measured_observation
    class BodyQuaternionEnvironment(GeometricEnvironment):
        def obs(self):
            raw = super().obs()
            raw['robot0_eef_quat'] = (self.rotation * Rotation.from_euler('z', 90, degrees=True)).as_quat()
            return raw
    env = BodyQuaternionEnvironment()
    session = LiberoSession(env, [0], 'put the cup on the plate', 10, settle_steps=0)
    obs = session.start()
    expected_pi = measured_observation(session.raw, session.instruction)
    np.testing.assert_array_equal(obs['states'], expected_pi['states'])
    np.testing.assert_allclose(obs['eef_quaternion_xyzw'], [0,0,0,1])
    assert not np.allclose(obs['eef_quaternion_xyzw'], expected_pi['eef_quaternion_xyzw'])
    reference = dict(position=obs['eef_position'].tolist(),
                     quaternion_xyzw=obs['eef_quaternion_xyzw'].tolist(), frame='MuJoCo world')
    goal = dict(deepcopy(reference), gripper='open'); goal['position'][0] += .01
    assert len(session.execute_eef(goal, np.zeros((1,7)), 1, reference)) == 1


def test_controller_metadata_through_offscreen_wrapper():
    env = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(controller=controller())]))
    session = LiberoSession(env, [], 'test', 10)
    assert session.eef_control()['supported']


@pytest.mark.parametrize('module', ['planner_server', 'stage1_planner'])
def test_both_planner_schemas_support_only_validated_single_arm_eef(tmp_path, module):
    import importlib
    from .test_planner import request
    r = request(tmp_path)
    r['native_request']['eef_control'] = controller_spec(controller())
    variants = importlib.import_module('hybrid_rollout.libero.'+module).decision_schema(r)['properties']['decision']['anyOf']
    eef = next(v for v in variants if v['properties']['mode']['enum']==['eef'])
    assert set(eef['properties']['target']['properties']) == set(target())
    r['review_kind']='uncertainty'
    schema = importlib.import_module('hybrid_rollout.libero.'+module).decision_schema(r)
    assert schema['properties']['decision']['properties']['mode']['enum']==['student','stop']
