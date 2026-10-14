"""Offline instruction-only planner and asynchronous pause-boundary regressions."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from .instruction_planner import INPUT_FORMAT, EEF_INPUT_FORMAT, QUESTIONS, SKILL_PATH, review_brief
from .planner_server import AstraPlanner, decision_schema
from .test_episode_thread import FakeTransport, next_request
from .test_planner import request
from .test_recovery_protocol import assessment, response, student
from .test_task_context import make_semantic_runner, observed, frame, goals
from .task_context import TaskContext
from .test_integration import Environment
from .session import LiberoSession
from ..robodojo.jev.monitor import MonitorConfig
from ..robodojo.jev.plans import validate_recovery_decision
from .contract import validate_response


def instruction_request(tmp_path):
    r=next_request(request(tmp_path),10)
    r.update(planner_role='instruction_only', recovery_policy='single_instruction',
        board_review_required=True, original_task='place cup on shelf',
        current_subgoal='grasp cup and lift slightly',
        instruction_execution=dict(start_step=0,current_step=10,executed_controls=10),
        review_kind='progress', failure=dict(trigger='subgoal_execution_budget',first_step=0,last_step=10))
    r['current_scene']['success_condition']='cup visibly clear of table'
    return r


def test_actual_rpc_uses_skill_and_omits_actions_even_from_readable_history(tmp_path):
    r=instruction_request(tmp_path)
    r['native_request'].update(proposal_actions=[[12345]*7], student_eef_command_preview={'secret':'trajectory'},
        current_eef={'position':[45678]*3}, eef_control={'supported':True}, assessment_example={'legacy':'gate'})
    r['current_scene'].update(executed_actions=[[54321]*7],executed_start_step=5,action_source='student')
    row=dict(evidence=dict(observation_id='old',step_id=5,images=[]), scores={'collision':.1},
             raw_answers={},executed_actions=[[54321]*7]*5)
    r['task_context']=dict(new_evidence=[row],recent_history=[row])
    made=[]
    def factory(*args):
        t=FakeTransport(*args,payload=dict(decision=student(r),board_assessment=assessment(r)))
        made.append(t);return t
    p=AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=factory)
    assert p.replan(r)['decision']['mode']=='keep_instruction'
    start=next(v for m,v in made[0].calls if m=='thread/start')
    turn=next(v for m,v in made[0].calls if m=='turn/start')
    assert start['developerInstructions']==SKILL_PATH.read_text()
    assert [t['name'] for t in start['dynamicTools']]==['read_episode_record']
    packet=json.loads(turn['input'][0]['text'])
    assert next(iter(packet))=='review_brief'
    assert packet['input_format']==INPUT_FORMAT
    assert packet['review_brief']['instruction_execution']['executed_controls']==10
    assert packet['review_brief']['success_condition']=='cup visibly clear of table'
    assert packet['new_evidence'][0]['executed_action_count']==5
    assert packet['previous_result']['executed_steps']==1
    texts=[turn['input'][0]['text'],Path(packet['board_path']).read_text(),Path(packet['history_path']).read_text()]
    for text in texts:
        for forbidden in ('proposal_actions','executed_actions','student_eef_command_preview','eef_control','current_eef','54321','12345','45678'):
            assert forbidden not in text
    # Original raw input remains available to host audits, not to model tools.
    assert json.loads(next((tmp_path/'audit').glob('*/request.json')).read_text())==r
    folder=Path(packet['board_path']).parent
    event=dict(id=1,params=dict(threadId='thread-1',turnId='turn-1',tool='read_observation_actions',
        arguments=dict(observation_id='old',offset=0,limit=1)))
    p.threads.read_record(event,folder,'thread-1','turn-1')
    assert not made[0].calls[-1][1][1]['success']


@pytest.mark.parametrize('mode',['student','eef','edit','advance_subgoal'])
def test_schema_and_host_both_reject_action_modes(tmp_path,mode):
    r=instruction_request(tmp_path)
    variants=decision_schema(r)['properties']['decision']['anyOf']
    assert {m for v in variants for m in v['properties']['mode']['enum']}=={'keep_instruction','stop','replan'}
    with pytest.raises(ValueError,match='cannot select robot controls'):
        validate_recovery_decision(dict(mode=mode,request_id=r['native_request']['request_id']),r,validate_response)


def test_recovery_id_schema_avoids_unsupported_lookaround_and_host_reserves_original(tmp_path):
    r=instruction_request(tmp_path)
    schema=decision_schema(r)
    def check(node):
        if isinstance(node,dict):
            if 'pattern' in node:
                assert not any(token in node['pattern'] for token in ('(?=', '(?!', '(?<=', '(?<!'))
            for value in node.values():check(value)
        elif isinstance(node,list):
            for value in node:check(value)
    check(schema)
    decision=dict(mode='replan',request_id=r['native_request']['request_id'],reason='Recover the grasp',
                  subgoals=[dict(subgoal_id='original-task',instruction='grasp cup',success_condition='cup lifted')])
    with pytest.raises(ValueError,match='original-task is reserved'):
        validate_recovery_decision(decision,r,validate_response)


@pytest.mark.parametrize('trigger',list(QUESTIONS))
def test_each_trigger_delivers_its_question_and_current_instruction(tmp_path,trigger):
    r=instruction_request(tmp_path);r['failure']['trigger']=trigger
    brief=review_brief(r)
    assert brief['question']==QUESTIONS[trigger]
    assert brief['trigger']==trigger and brief['trigger_is_not_a_verdict']
    assert brief['current_instruction']==r['current_subgoal']


def test_old_action_thread_cannot_resume_as_instruction_thread(tmp_path):
    r=next_request(request(tmp_path),1)
    p=AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=FakeTransport)
    p.replan(r)
    fresh=instruction_request(tmp_path)
    with pytest.raises(RuntimeError,match='input format'):
        p.replan(fresh)


def test_coalesced_anomaly_and_completion_request_review_without_constraint_confirmation():
    for field,trigger in [('collision','anomaly_scores_review'),('subgoal_complete','completion_scores_review')]:
        cfg=MonitorConfig(max_step_gap=5,observation_profile='task_progress')
        board=TaskContext(cfg,max_sample_gap=40);board.activate(frame(0),goals())
        for step in (5,20):
            f=frame(step)
            report={k:f[k] for k in ('run_id','episode_id','plan_revision','subgoal_id','program_id','step_id','observation_id')}
            report['scores']=observed(**{field:.95});board.update(report,f)
        assert board.review()['trigger']==trigger
        assert not board.review()['confirmed_failure']


def test_async_execution_then_pause_refresh_and_no_pi_inference_during_astra(tmp_path):
    runner,calls,events=make_semantic_runner(tmp_path,horizon=100)
    moved=Event()
    class Moving(Environment):
        def step(self,action):
            result=super().step(action)
            if self.steps>=25: moved.set()
            return result
    env=Moving(999)
    runner.rollout.session_factory=lambda:LiberoSession(env,[0],'pick up the cup',100)
    runner.progress_budget=20
    def monitor(frames):
        if frames[-1]['step_id']==5:
            assert moved.wait(1), 'Pi blocked behind the first JEV call'
        return observed(target_visible=.5,scene_comparable=.5)
    runner.monitor.monitor=SimpleNamespace(evaluate=monitor)
    count_at_pause=[]
    def emit(event):
        events.append(event)
        if event['kind']=='pause_confirmed':
            count_at_pause.append(len(runner.rollout.student.instructions))
    runner.emit=emit
    def plan(r):
        calls.append(deepcopy(r))
        assert len(runner.rollout.student.instructions)==count_at_pause[-1]+1
        assert r['current_scene']['step_id']==runner.rollout.tick
        assert '/jev_refresh/' in r['current_scene']['images'][0]['path']
        assert r['asynchronous_execution']['controls_after_trigger']==0
        assert r['native_request']['step_id']==r['current_scene']['step_id']
        assert r['control_diagnostics']['prediction_executed'] is False
        assert len(r['native_request']['proposal_actions']) == 10
        d=student(r);d['mode']='stop'
        return response(r,d)
    runner.planner=SimpleNamespace(replan=plan)
    result=runner.run()
    assert len(calls)==1 and result['state']=='STOPPED'
    assert result['reason']=='Planner declined recovery'
    assert result['step_id']==calls[0]['current_scene']['step_id']
    assert any(e['kind']=='monitor_report' and e['observation_lag_steps']>=10 for e in events)
    assert all(e['controls_during_planning']==0 for e in events if e['kind']=='pause_release')


def test_slow_monitor_applies_backpressure_only_at_lag_limit(tmp_path):
    runner,calls,events=make_semantic_runner(tmp_path,horizon=50,shadow=True)
    at_limit=Event();runner.max_monitor_lag=15
    def emit(event):
        events.append(event)
        if event['kind']=='monitor_backpressure':at_limit.set()
    runner.emit=emit
    def monitor(frames):
        if frames[-1]['step_id']==5:
            assert at_limit.wait(1)
            assert runner.rollout.tick==15
        return observed(task_advancing=.95)
    runner.monitor.monitor=SimpleNamespace(evaluate=monitor)
    result=runner.run()
    assert result['step_id']==50 and result['native_result']['complete']
    assert not calls
    assert any(e['kind']=='monitor_backpressure' for e in events)


def test_keep_original_after_lag_budget_resumes_without_expiring_instruction(tmp_path):
    runner,calls,_=make_semantic_runner(tmp_path,horizon=65)
    runner.max_replans=5;runner.max_monitor_lag=10;runner.progress_budget=20
    runner.monitor.monitor=SimpleNamespace(evaluate=lambda frames: observed(
        target_visible=.5,scene_comparable=.5))
    def plan(r):
        calls.append(deepcopy(r))
        return response(r,student(r))
    runner.planner=SimpleNamespace(replan=plan)
    result=runner.run()
    assert result['step_id']==65 and result['native_result']['complete']
    assert [r['current_scene']['step_id'] for r in calls]==[20,40,60]
    assert all(r['instruction_execution']['start_step']==0 for r in calls)
    assert all(r['plan_revision']==0 for r in calls)


def test_scheduler_through_real_planner_adapter_replans_keeps_and_resumes(tmp_path):
    runner,_,_=make_semantic_runner(tmp_path/'rollout',horizon=95)
    runner.max_replans=5;runner.progress_budget=runner.subgoal_budget=20
    runner.monitor.monitor=SimpleNamespace(evaluate=lambda frames: observed(
        target_visible=.5,scene_comparable=.5))
    packets=[];made=[]
    class InstructionTransport(FakeTransport):
        def request(self,method,params,timeout):
            if method=='turn/start':
                r=json.loads(params['input'][0]['text']);packets.append(r)
                d=student(r)
                if len(packets)==1:
                    d.update(mode='replan',subgoals=[dict(subgoal_id='grasp-cup',
                        instruction='grasp the cup and lift slightly',success_condition='cup clears table')])
                elif len(packets)==3:
                    d.update(mode='resume_original',plan_revision=r['plan_revision'],subgoal_id=r['subgoal_id'],
                        evidence_observation_ids=[r['current_scene']['observation_id']])
                elif len(packets)==4:
                    d['mode']='stop'
                self.payload=dict(decision=d,board_assessment=assessment(r))
            return super().request(method,params,timeout)
    def factory(*args):
        t=InstructionTransport(*args);made.append(t);return t
    planner=AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=factory)
    runner.planner=planner
    try:
        result=runner.run()
    finally:
        planner.close()
    assert result['state']=='STOPPED' and result['step_id']==80
    assert [r['current_subgoal'] for r in packets]==[
        'pick up the cup','grasp the cup and lift slightly','grasp the cup and lift slightly','pick up the cup']
    assert [a['decision']['mode'] for a in result['attempts']]==[
        'replan','keep_instruction','resume_original','stop']
    assert all(p['input_format']==EEF_INPUT_FORMAT and 'proposal_actions' not in p['native_request'] for p in packets)
    assert len([m for m,p in made[0].calls if m=='thread/start'])==1
    assert all(t['controls_after_trigger']==0 for t in result['triggers'])
