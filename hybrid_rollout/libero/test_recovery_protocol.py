"""CPU regressions for explicit Astra plan operations and bounded review liveness."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import importlib
import json
import time

import pytest

from .test_task_context import make_semantic_runner, observed, frame, goals
from .test_eef import GeometricEnvironment
from .test_integration import Student
from .test_planner import request as planner_request
from .task_context import TaskContext, validate_board_assessment
from .continuous import ContinuousScheduler
from .rollout import LiberoRollout
from .session import LiberoSession
from .contract import validate_response
from ..robodojo.jev.monitor import AsyncMonitor, MonitorConfig, ConstraintRuntime, IDENTITY
from ..robodojo.jev.runner import automatic_response, validate_plan


def assessment(r):
    return dict(robot_status='At reviewed pose', task_status='in_progress',
        task_reason='Original task remains', scene_context='Target visible', uncertainties='Grasp uncertain',
        evidence_observation_ids=[r['current_scene']['observation_id']])


def response(r, d):
    return {**{k:r[k] for k in ('request_id', *IDENTITY)},
            'decision':d, 'board_assessment':assessment(r)}


def student(r, steps=1):
    if r.get('planner_role') in ('instruction_only', 'instruction_eef'):
        return dict(request_id=r['native_request']['request_id'], mode='keep_instruction',
                    reason='Current instruction remains appropriate')
    d=automatic_response(r['native_request'],steps,{})
    d.pop('decision_author');d.pop('evaluation_method')
    return d


def advance(r):
    return dict(mode='advance_subgoal',request_id=r['native_request']['request_id'],
        plan_revision=r['plan_revision'],subgoal_id=r['subgoal_id'],reason='Current subgoal complete',
        evidence_observation_ids=[r['current_scene']['observation_id']])


def test_single_instruction_persists_and_resumes_only_on_astra(tmp_path):
    runner,calls,events=make_semantic_runner(tmp_path,horizon=240)
    runner.max_replans=10;runner.max_review_interval=40
    def plan(r):
        calls.append(deepcopy(r))
        if len(calls) in (1,3):
            d=dict(mode='replan',request_id=r['native_request']['request_id'],reason='Current scene requires new instruction',
                subgoals=[dict(subgoal_id='recovery-1',instruction='pick up the cup' if len(calls)==1 else 'place the cup upright',success_condition='upright')])
        elif len(calls)==2:
            result=response(r,student(r))
            result['board_assessment']['task_status']='completed'
            return result
        elif len(calls)==4:
            d=advance(r);d['mode']='resume_original'
        else:
            d=student(r);d['mode']='stop'
        return response(r,d)
    runner.planner=SimpleNamespace(replan=plan)
    result=runner.run()
    assert len(calls)==5
    assert [r['current_subgoal'] for r in calls[:4]]==['pick up the cup','pick up the cup','pick up the cup','place the cup upright']
    for previous,current in zip(calls[:3],calls[1:4]):
        assert current['current_scene']['step_id']-previous['current_scene']['step_id']>=25
        assert current['failure']['trigger']=='low_progress_scores_review'
        assert len(current['remaining_plan'])==1
    assert calls[4]['subgoal_id']=='original-task'
    assert all(r['original_task']=='pick up the cup' for r in calls)
    assert all('Next,' not in s for s in runner.rollout.student.instructions)
    assert not any(e['kind']=='subgoal_completed' for e in events)
    assert result['attempts'][3]['result']=='original_instruction_restored_without_execution'
    assert not result['native_result']['success']


def operation_request(tmp_path):
    r=planner_request(tmp_path)
    r.update(plan_revision=3,subgoal_id='recovery-1',board_review_required=True)
    r['current_scene'].update(observation_id='fresh',step_id=10)
    r['plan_operations']={'advance_subgoal':dict(subgoal_id='recovery-1',evidence_observation_ids=['fresh'])}
    return r


@pytest.mark.parametrize('bad', ['revision','goal','evidence','proposal','unavailable','extra','uncertainty'])
def test_host_rejects_invalid_advance_before_mutation(tmp_path,bad):
    r=operation_request(tmp_path);d=advance(r)
    if bad=='revision':d['plan_revision']=2
    if bad=='goal':d['subgoal_id']='original-task'
    if bad=='evidence':d['evidence_observation_ids']=['historical']
    if bad=='proposal':d['request_id']='old'
    if bad=='unavailable':r['plan_operations']={}
    if bad=='extra':d['steps']=1
    if bad=='uncertainty':r['review_kind']='uncertainty'
    rollout=SimpleNamespace(request=r['native_request'],validate=validate_response,
                            validate_board_assessment=validate_board_assessment)
    with pytest.raises(ValueError):validate_plan(response(r,d),r,rollout,1)


@pytest.mark.parametrize('module',['planner_server','stage1_planner'])
@pytest.mark.parametrize('stale',[False,True])
def test_both_planner_paths_validate_advance(tmp_path,monkeypatch,module,stale):
    mod=importlib.import_module('hybrid_rollout.libero.'+module)
    r=operation_request(tmp_path);d=advance(r)
    variants=mod.decision_schema(r)['properties']['decision']['anyOf']
    assert any(v['properties']['mode']['enum']==['advance_subgoal'] for v in variants)
    if stale:d['plan_revision']=2
    from .test_episode_thread import FakeTransport
    class Transport(FakeTransport):
        def request(self,method,params,timeout):
            self.payload=dict(decision=d,board_assessment=assessment(r))
            return super().request(method,params,timeout)
    planner=mod.AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=Transport)
    if stale:
        with pytest.raises(ValueError,match='Stale'):planner.replan(r)
    else:assert planner.replan(r)['decision']==d


def test_coalesced_reports_trigger_review_without_relaxing_safety_confirmation():
    cfg=MonitorConfig(max_step_gap=5,observation_profile='task_progress')
    board=TaskContext(cfg,stall_steps=30,max_sample_gap=40)
    board.activate(frame(0),goals());runtime=ConstraintRuntime(cfg)
    reports=[]
    for step in (5,15,25,35,45,55):
        f=frame(step);f['program_id']=cfg.program_id
        if step==5:board.activate(f,goals())
        r=runtime.update(f,observed(collision=.95))
        reports.append(r);board.update(r,f)
    assert reports[-1]['gap_reset'] and not reports[-1]['violations']
    assert board.review()['trigger']=='anomaly_scores_review'
    assert board.review()['report_count']==2
    assert not board.review()['confirmed_failure']


@pytest.mark.parametrize('gap',['steps','time','generation'])
def test_sampling_does_not_bridge_stale_or_new_plan_evidence(gap):
    cfg=MonitorConfig(max_step_gap=5,observation_profile='task_progress')
    board=TaskContext(cfg,stall_steps=30,max_sample_gap=40);board.activate(frame(0),goals())
    for step in (5,15,25):
        f=frame(step);r={**{k:f[k] for k in (*IDENTITY,'step_id','observation_id')},'scores':observed()}
        board.update(r,f)
    f=frame(75 if gap=='steps' else 35,1 if gap=='generation' else 0)
    if gap=='time':f['captured_at']+=cfg.max_gap_s+1
    if gap=='generation':board.activate(f,goals())
    r={**{k:f[k] for k in (*IDENTITY,'step_id','observation_id')},'scores':observed()}
    board.update(r,f)
    assert len(board.votes)==1 and board.review() is None


@pytest.mark.parametrize('limit',[2,10])
def test_high_progress_cannot_reset_hard_review_interval(tmp_path,limit):
    runner,calls,_=make_semantic_runner(tmp_path,advancing=True)
    runner.max_review_interval=20;runner.max_replans=limit
    def plan(r):calls.append(deepcopy(r));return response(r,student(r,5))
    runner.planner=SimpleNamespace(replan=plan)
    result=runner.run()
    assert not calls and result['step_id']==90
    assert not result['native_result']['success']


@pytest.mark.parametrize('mode',['eef','edit','student'])
def test_instruction_scheduler_rejects_low_level_decisions_before_execution(tmp_path,mode):
    runner,calls,events=make_semantic_runner(tmp_path,horizon=90)
    def plan(r):
        calls.append(deepcopy(r))
        assert r['planner_role'] == 'instruction_eef'
        assert r['control_diagnostics']['prediction_executed'] is False
        d=dict(request_id=r['native_request']['request_id'],mode=mode,reason='Forbidden control')
        return response(r,d)
    runner.planner=SimpleNamespace(replan=plan)
    result=runner.run()
    assert len(calls)==1
    assert result['state']=='STOPPED'
    assert ('Bounded EEF is unavailable' if mode == 'eef' else 'cannot select robot controls') in result['reason']
    assert result['step_id']==calls[0]['current_scene']['step_id']
    assert all(h['source_detail']=='student' for h in runner.rollout.history)
    assert not any(e['kind']=='local_correction_review' for e in events)


@pytest.mark.parametrize('module',['planner_server','stage1_planner'])
@pytest.mark.parametrize('count',[1,2])
def test_single_instruction_schema_and_both_validators(tmp_path,monkeypatch,module,count):
    mod=importlib.import_module('hybrid_rollout.libero.'+module)
    r=operation_request(tmp_path);r['recovery_policy']='single_instruction'
    r['plan_operations']={'resume_original':r['plan_operations']['advance_subgoal']}
    variants=mod.decision_schema(r)['properties']['decision']['anyOf']
    repl=next(v for v in variants if v['properties']['mode']['enum']==['replan'])
    assert repl['properties']['subgoals']['maxItems']==1
    assert any(v['properties']['mode']['enum']==['resume_original'] for v in variants)
    assert not any(v['properties']['mode']['enum']==['advance_subgoal'] for v in variants)
    d=dict(mode='replan',request_id=r['native_request']['request_id'],reason='Current scene',
           subgoals=[dict(subgoal_id=f'recovery-{i+1}',instruction='pick up cup',success_condition='held') for i in range(count)])
    from .test_episode_thread import FakeTransport
    class Transport(FakeTransport):
        def request(self,method,params,timeout):
            self.payload=dict(decision=d,board_assessment=assessment(r))
            return super().request(method,params,timeout)
    planner=mod.AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=Transport)
    rollout=SimpleNamespace(request=r['native_request'],validate=validate_response,validate_board_assessment=validate_board_assessment)
    if count==2:
        with pytest.raises(ValueError,match='exactly one'):planner.replan(r)
        with pytest.raises(ValueError,match='exactly one'):validate_plan(response(r,d),r,rollout,1)
    else:
        assert validate_plan(planner.replan(r),r,rollout,1)['mode']=='replan'
        r['request_id']='next-review'
        d=advance(r);d['mode']='resume_original'
        assert validate_plan(planner.replan(r),r,rollout,1)['mode']=='resume_original'
        d['plan_revision']-=1
        with pytest.raises(ValueError,match='Stale'):validate_plan(response(r,d),r,rollout,1)


@pytest.mark.parametrize('budget',[0,-1])
def test_invalid_astra_budget_rejected_before_any_runtime_import(budget,capsys):
    from .max_run import main
    with pytest.raises(SystemExit) as error:
        main(['--case','unused','--scenario','unused','--arm','intervention','--astra','on',
              '--output','unused','--checkpoint','unused','--astra-call-budget',str(budget)])
    assert error.value.code==2
    assert '--astra-call-budget must be positive' in capsys.readouterr().err


@pytest.mark.parametrize('budget',[2,6])
def test_instruction_replacement_keep_and_explicit_resume(tmp_path,budget):
    runner,calls,events=make_semantic_runner(tmp_path,horizon=180)
    runner.max_replans=budget
    def plan(r):
        calls.append(deepcopy(r))
        if len(calls)==1:
            d=dict(mode='replan',request_id=r['native_request']['request_id'],reason='Try the current scene',
                subgoals=[dict(subgoal_id='recovery-1',instruction='grasp the cup',success_condition='held')])
        elif len(calls)==2:
            d=student(r)
        elif len(calls)==3:
            d=advance(r);d['mode']='resume_original'
        else:
            d=student(r);d['mode']='stop'
        return response(r,d)
    runner.planner=SimpleNamespace(replan=plan)
    result=runner.run()
    assert all(h['source_detail']=='student' for h in runner.rollout.history)
    assert calls[1]['current_subgoal']=='grasp the cup'
    assert calls[1]['instruction_execution']['executed_controls']>0
    assert result['attempts'][1]['result']=='instruction_retained_without_execution'
    if budget==2:
        assert len(calls)==2 and result['reason']=='Required Astra review budget exhausted'
        assert result['remaining_plan'][0]['instruction']=='grasp the cup'
    else:
        assert len(calls)==4
        assert calls[2]['current_subgoal']=='grasp the cup'
        assert calls[2]['plan_revision']==calls[1]['plan_revision']
        assert calls[2]['instruction_execution']['start_step']==calls[1]['instruction_execution']['start_step']
        assert calls[3]['subgoal_id']=='original-task'
        assert runner.rollout.student.instructions[-1]=='pick up the cup'
    assert not any(e['kind']=='subgoal_completed' for e in events)
    assert result['native_result']['success'] is False


@pytest.mark.parametrize('value',[None,4,20])
def test_astra_budget_cli_accepts_default_and_override(monkeypatch,value):
    import argparse
    from .max_run import main
    parse=argparse.ArgumentParser.parse_args
    class Parsed(Exception):pass
    def inspect(parser,args):
        parsed=parse(parser,args)
        assert parsed.astra_call_budget==(12 if value is None else value)
        raise Parsed
    monkeypatch.setattr(argparse.ArgumentParser,'parse_args',inspect)
    args=['--case','unused','--scenario','unused','--arm','intervention','--astra','on',
          '--output','unused','--checkpoint','unused']
    if value is not None:args+=['--astra-call-budget',str(value)]
    with pytest.raises(Parsed):main(args)
