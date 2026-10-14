"""Offline model-tool protocol and measured EEF handback; no live services."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .control_review import (archive_control_record, diagnostics, read_control_record,
                             record_path, validate_eef_decision)
from .eef import controller_spec
from .instruction_planner import EEF_ROLE, EEF_INPUT_FORMAT
from .planner_server import AstraPlanner, decision_schema
from .session import LiberoSession
from .test_eef import controller, current, GeometricEnvironment
from .test_episode_thread import FakeTransport
from .test_instruction_planner import instruction_request
from .test_recovery_protocol import assessment, response, student
from .test_task_context import make_semantic_runner, observed
from ..robodojo.jev.plans import validate_recovery_decision
from .contract import validate_response


def request(tmp_path):
    r = instruction_request(tmp_path)
    r['planner_role'] = EEF_ROLE
    r['max_chunk_steps'] = 5
    n = r['native_request']
    n['max_chunk_steps'] = 5
    n.update(observation_id=r['current_scene']['observation_id'], prediction_id='prediction-1',
        student_instruction=r['current_subgoal'], proposal_actions=[[.1234567]*6+[1.]]*10,
        current_eef=current(), eef_control=controller_spec(controller()))
    r['current_scene'].update(current_eef=current(), executed_actions=[[.7654321]*7])
    r['control_diagnostics'] = diagnostics(r)
    return r


def decision(r):
    target = deepcopy(r['native_request']['current_eef'])
    target = {k:target[k] for k in ('position','quaternion_xyzw','frame')}
    target['position'][0] += .03
    target['gripper'] = 'open'
    return dict(mode='eef', request_id=r['native_request']['request_id'],
        observation_id=r['current_scene']['observation_id'], steps=3, target=target,
        basis='execution_failed', reason='The visible grasp missed; reopen and correct local alignment.')


def test_immutable_candidate_read_is_paginated_bound_and_not_executed(tmp_path):
    r = request(tmp_path)
    active = archive_control_record(tmp_path, r)
    args = dict(review_id=active['review_id'], record='pi_proposal', offset=2, limit=3)
    data = read_control_record(tmp_path, args, active)
    assert data['actions'] == r['native_request']['proposal_actions'][2:5]
    assert data['prediction_executed'] is False and data['prediction_id'] == 'prediction-1'
    assert data['sha256'] == active['sha256']
    context = read_control_record(tmp_path, dict(args, record='eef_context', offset=0, limit=1), active)
    assert context['current_eef'] == current()
    assert context['controller']['output_max'] == [.05]*3+[.5]*3
    assert archive_control_record(tmp_path, r) == active
    changed = deepcopy(r); changed['native_request']['proposal_actions'][0][0] = -.5
    with pytest.raises(ValueError, match='reused'):
        archive_control_record(tmp_path, changed)
    for extra in (dict(review_id='other'), dict(offset=True), dict(limit=11),
                  dict(record='edit'), dict(path='/tmp/another-file')):
        with pytest.raises(ValueError):
            read_control_record(tmp_path, dict(args, **extra), active)
    path = record_path(tmp_path, active['review_id'])
    raw = json.loads(path.read_text()); raw['actions'][0][0] = 999
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='changed'):
        read_control_record(tmp_path, args, active)


@pytest.mark.parametrize('change', ['stale_request','stale_scene','oversize','nan','rotation',
                                  'steps','keep','edit','unsupported','basis'])
def test_eef_rejects_stale_unbounded_or_unavailable_correction(tmp_path, change):
    r = request(tmp_path); d = decision(r)
    validate_eef_decision(d, r)
    if change == 'stale_request': d['request_id'] = 'old'
    elif change == 'stale_scene': d['observation_id'] = 'old'
    elif change == 'oversize': d['target']['position'][0] += .03
    elif change == 'nan': d['target']['position'][0] = float('nan')
    elif change == 'rotation': d['target']['quaternion_xyzw'] = [0,0,1,0]
    elif change == 'steps': d['steps'] = 6
    elif change == 'keep': d['target']['gripper'] = 'keep'
    elif change == 'edit': d['actions'] = [[0]*7]
    elif change == 'unsupported': r['native_request']['eef_control'] = {'supported':False}
    elif change == 'basis': d['basis'] = 'uncertain'
    with pytest.raises(ValueError):
        validate_recovery_decision(d, r, validate_response)


def test_new_schema_offers_eef_but_never_student_or_edit(tmp_path):
    r = request(tmp_path)
    def modes():
        return {m for v in decision_schema(r)['properties']['decision']['anyOf']
                for m in v['properties']['mode']['enum']}
    assert modes() == {'keep_instruction','stop','replan','eef'}
    r['native_request']['eef_control'] = {'supported':False}
    assert 'eef' not in modes()
    for mode in ('edit','student'):
        with pytest.raises(ValueError, match='cannot select robot controls'):
            validate_recovery_decision(dict(decision(r), mode=mode), r, validate_response)


def test_eef_respects_remaining_budget_and_native_chunk_bound(tmp_path):
    r=request(tmp_path); r['execution_budget']={'remaining_controls':2}
    assert diagnostics(r)['max_steps']==2
    with pytest.raises(ValueError, match='chunk bound'):
        validate_eef_decision(decision(r), r)
    r['native_request']['max_chunk_steps']=1
    assert diagnostics(r)['max_steps']==1
    r['execution_budget']['remaining_controls']=0
    assert not diagnostics(r)['eef_supported']
    assert all('eef' not in v['properties']['mode']['enum']
               for v in decision_schema(r)['properties']['decision']['anyOf'])


def test_last_eef_step_overshoot_stops_and_preserves_measured_trace():
    class Overshoot(GeometricEnvironment):
        def step(self, action):
            result=super().step(action)
            self.position[0]+=.06
            return result
    env=Overshoot()
    session=LiberoSession(env,[0],'pick up the cup',20,settle_steps=0)
    session.start()
    reference=current(); goal=dict(deepcopy(reference),gripper='open')
    goal['position'][0]+=.03
    with pytest.raises(ValueError,match='after execution'):
        session.execute_eef(goal,np.zeros((1,7)),1,reference)
    assert session.tick==1 and len(session.eef_trace)==1
    assert session.eef_trace[0]['measured_after']['position'][0] > reference['position'][0]+.05


def test_real_adapter_tools_read_current_and_executed_controls_without_prompt_arrays(tmp_path):
    r = request(tmp_path); made=[]
    class ToolTransport(FakeTransport):
        def request(self, method, params, timeout):
            result = super().request(method, params, timeout)
            if method == 'turn/start':
                tid, turn = params['threadId'], result['turn']['id']
                calls = [('read_control_diagnostics', dict(review_id=r['native_request']['request_id'],
                    record='pi_proposal',offset=0,limit=2)),
                    ('read_control_diagnostics',dict(review_id=r['native_request']['request_id'],
                    record='eef_context',offset=0,limit=1)),
                    ('read_observation_actions', dict(observation_id=r['current_scene']['observation_id'],offset=0,limit=1))]
                self.events[:0] = [dict(id=i, method='item/tool/call',params=dict(
                    threadId=tid,turnId=turn,tool=tool,arguments=args)) for i,(tool,args) in enumerate(calls)]
            return result
    def factory(*args):
        t=ToolTransport(*args,payload=dict(decision=decision(r),board_assessment=assessment(r)))
        made.append(t); return t
    planner=AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=factory)
    try:
        assert planner.replan(r)['decision'] == decision(r)
        start=next(p for m,p in made[0].calls if m=='thread/start')
        assert {t['name'] for t in start['dynamicTools']} == {
            'read_episode_record','read_observation_actions','read_control_diagnostics'}
        turn=next(p for m,p in made[0].calls if m=='turn/start')
        packet=json.loads(turn['input'][0]['text'])
        assert packet['input_format'] == EEF_INPUT_FORMAT
        for content in (turn['input'][0]['text'], Path(packet['history_path']).read_text(),
                        Path(packet['board_path']).read_text()):
            for forbidden in ('proposal_actions','executed_actions','current_eef','current_state','0.7654321','0.1234567'):
                assert forbidden not in content
        replies=[p[1] for m,p in made[0].calls if m=='reply']
        assert len(replies)==3 and all(p['success'] for p in replies)
        values=[json.loads(p['contentItems'][0]['text']) for p in replies]
        assert len(values[0]['actions'])==2
        assert values[1]['current_eef']==current()
        assert values[2]['executed_actions']==r['current_scene']['executed_actions']
    finally:
        planner.close()


def test_old_instruction_episode_cannot_gain_control_tools_on_resume(tmp_path):
    planner=AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=FakeTransport)
    old=instruction_request(tmp_path)
    # Fake transport needs the instruction-only output envelope.
    planner.threads.factory=lambda *args:FakeTransport(*args,payload=dict(
        decision=student(old),board_assessment=assessment(old)))
    try:
        planner.replan(old)
        new=request(tmp_path); new['request_id']='next-review'
        with pytest.raises(RuntimeError, match='older input format'):
            planner.replan(new)
    finally:
        planner.close()


def test_eef_waits_for_visual_endpoint_then_pi_uses_new_scene(tmp_path):
    runner,_,events=make_semantic_runner(tmp_path/'rollout',horizon=50)
    env=GeometricEnvironment()
    runner.rollout.session_factory=lambda:LiberoSession(env,[0],'pick up the cup',50,settle_steps=0)
    runner.progress_budget=runner.subgoal_budget=20
    runner.max_replans=3
    seen=[]; reviewed=[]
    def monitor(frames):
        reviewed.append(frames[-1]['step_id'])
        return observed(target_visible=.5,scene_comparable=.5)
    runner.monitor.monitor=SimpleNamespace(evaluate=monitor)
    def plan(r):
        seen.append(deepcopy(r))
        assert runner.rollout.tick == r['current_scene']['step_id']
        d=decision(r) if len(seen)==1 else student(r)
        if len(seen)>1: d['mode']='stop'
        return response(r,d)
    runner.planner=SimpleNamespace(replan=plan)
    infer=runner.rollout.student.infer
    after=[]
    def tracked_infer(obs, output):
        if runner.rollout.history and runner.rollout.history[-1]['source_detail']=='eef':
            assert runner.rollout.tick in reviewed, 'Pi resumed before correction image evidence arrived'
            after.append(deepcopy(obs))
        return infer(obs,output)
    runner.rollout.student.infer=tracked_infer
    result=runner.run()
    assert result['state']=='STOPPED' and result['reason']=='Planner declined recovery'
    i=next(i for i,h in enumerate(runner.rollout.history) if h['source_detail']=='eef')
    assert runner.rollout.history[i]['executed_steps']==3
    trace=json.loads((runner.rollout.output/f'eef_{i:03d}.json').read_text())
    np.testing.assert_allclose([t['executed_action'][0] for t in trace],[.6,.3,.15],atol=1e-6)
    assert all(t['executed_action'][6]==-1 for t in trace)
    assert after and after[0]['instruction']=='pick up the cup'
    assert runner.rollout.history[i+1]['source_detail']=='student'
    assert result['attempts'][0]['verification']['result']=='fresh_visual_evidence_received_not_task_success'
    assert all(t['controls_during_planning']==0 for t in events if t['kind']=='pause_release')
    assert env.resets==env.restores==1
