"""Semantic monitoring fixtures only: no GPU, model calls, or object oracle."""
from copy import deepcopy
from dataclasses import replace
import time
from types import SimpleNamespace

import pytest

from ..robodojo.jev.monitor import (AsyncMonitor, ConstraintRuntime, IDENTITY,
    MonitorConfig, QUESTIONS, TASK_QUESTIONS, ValenMonitor)
from .task_context import TaskContext
from .test_integration import Environment, Student, scores
from .rollout import LiberoRollout
from .session import LiberoSession
from .continuous import ContinuousScheduler


def observed(**kwargs):
    return {**scores(), **dict.fromkeys(TASK_QUESTIONS, .05),
        'target_visible':.95, 'scene_comparable':.95, 'target_on_support':.95, 'arm_approaching':.95, **kwargs}


def frame(step, revision=0):
    return dict(run_id='run', episode_id='episode', plan_revision=revision,
        subgoal_id='original-task' if revision==0 else 'recovery-1', program_id='program',
        step_id=step, observation_id=f'{revision}-{step}', captured_at=time.time(), images=[],
        current_eef={'position':[.1,.2,.3]}, current_state=[0]*8,
        task='pick up the cup', subgoal='pick up the cup' if revision==0 else 'align with the handle',
        success_condition=None)


def goals():
    return [dict(subgoal_id='original-task', instruction='pick up the cup', success_condition=None)]


def ledger():
    c = TaskContext(MonitorConfig(max_step_gap=5, observation_profile='task_progress'), stall_steps=15)
    c.activate(frame(0), goals())
    return c


def update(c, step, revision=0, **kwargs):
    f = frame(step, revision)
    r = {k:f[k] for k in (*IDENTITY, 'step_id', 'observation_id')}
    r.update(scores=observed(**kwargs))
    c.update(r, f)
    return r, f


def test_stalled_is_task_no_progress_not_robot_motion_or_generic_uncertainty():
    c=ledger()
    for step in (5,10,15,20,25):
        update(c,step)
        assert c.review() is None
    update(c,30)
    r=c.review()
    assert r['trigger']=='low_progress_scores_review' and not r['confirmed_failure'] and not r['violations']
    assert r['report_count']==6
    assert c.snapshot()['recent_history'][-1]['scores']['arm_approaching']==.95
    assert 'status' not in c.snapshot()['task']
    assert c.snapshot()['recent_history'][-1]['scores']['target_on_support']==.95


def test_two_progress_reports_reset_window_but_one_does_not():
    c=ledger()
    for step in (5,10,15): update(c,step)
    update(c,20,task_advancing=.95)
    assert c.last_advancement_signal is None and len(c.valid_votes)==4
    update(c,25,task_advancing=.95)
    assert c.last_advancement_signal['step_id']==25 and len(c.valid_votes)==1
    for step in (30,35,40,45):
        update(c,step)
        assert c.review() is None
    update(c,50)
    assert c.review()['trigger']=='low_progress_scores_review'


def test_pending_is_not_stalled():
    c=ledger()
    for step in range(5,51,5):
        update(c,step,progress_attempted=.05,arm_approaching=.05)
    assert 'status' not in c.snapshot()['task'] and c.review() is None


def test_observed_repeated_approach_is_attempt_evidence_despite_ambiguous_generic_guard():
    c=ledger()
    for step in (5,10,15,20,25,30):
        update(c,step,progress_attempted=.73,arm_approaching=.95)
    assert c.review()['trigger']=='low_progress_scores_review'


def test_view_change_is_unobservable_not_false_object_progress():
    c=ledger()
    for step in range(5,36,5):
        update(c,step,scene_comparable=.05,task_advancing=.95)
    assert c.last_advancement_signal is None
    assert c.review()['trigger']=='ambiguous_observations_review'
    assert c.snapshot()['recent_history'][-1]['scores']['scene_comparable']==.05


def test_occluded_or_conflicting_target_facts_are_unknown():
    c=ledger()
    update(c,5,target_visible=.05,target_held=.95)
    assert c.snapshot()['recent_history'][-1]['scores']['target_held']==.95
    update(c,10,target_held=.95,target_on_support=.95)
    assert c.snapshot()['recent_history'][-1]['scores']['target_held']==.95
    assert c.snapshot()['recent_history'][-1]['scores']['target_on_support']==.95


def test_duplicate_gap_and_generation_do_not_create_votes():
    c=ledger()
    r,f=update(c,5)
    assert not c.update(r,f) and len(c.votes)==1
    update(c,10)
    update(c,30)
    assert len(c.votes)==1 and c.review() is None
    wrong=deepcopy(r); wrong['episode_id']='other'
    with pytest.raises(ValueError,match='unrelated'):
        c.update(wrong,f)


def test_replan_preserves_episode_memory_but_not_old_target_as_current_fact():
    c=ledger()
    update(c,5,task_advancing=.95)
    update(c,10)
    before=c.snapshot()
    c.reviewed(10)
    c.activate(frame(10,1),[dict(subgoal_id='recovery-1',instruction='align with the handle',success_condition='aligned')])
    after=c.snapshot()
    assert after['milestones']==before['milestones']
    assert after['recent_history']==before['recent_history']
    assert after['scheduling']['last_high_advancement_score'] is None
    assert after['recent_history'][-1]['evidence']['plan_revision']==0
    assert after['plan_history'][-1]['plan_revision']==1
    assert len(after['plan_history'])==2 and c.review() is None
    for step in (15,20,25,30,35,40):update(c,step,revision=1)
    assert c.review()['first_step']==15


def test_valen_never_sets_or_rewrites_astra_status():
    c=ledger()
    update(c,5,task_satisfied=.95)
    update(c,10,task_satisfied=.95)
    assert 'status' not in c.snapshot()['task']
    assert not c.snapshot()['astra_interpretations']
    review=dict(author='astra',source_step=10,assessment={'task_status':'completed'})
    c.record_astra(review)
    update(c,15,task_satisfied=.05)
    update(c,20,task_satisfied=.05)
    assert c.snapshot()['astra_interpretations']==[review]
    assert c.snapshot()['recent_history'][-1]['scores']['task_satisfied']==.05
    assert 'status' not in c.snapshot()['task']


def test_extended_valen_http_packet_and_runtime_separate_status_from_constraints(monkeypatch):
    cfg=MonitorConfig(observation_profile='task_progress')
    monitor=ValenMonitor('http://localhost/predict',1,observation_profile='task_progress')
    payloads=[]
    def post(payload):
        payloads.append(payload)
        from .test_valen_evidence import answers
        return {'answers':answers(payload['questions'])}
    monkeypatch.setattr(monitor,'post',post)
    f=frame(5);f['program_id']=cfg.program_id
    evidence=monitor.evaluate([f])
    assert set(payloads[0]['questions'])==set(QUESTIONS)|set(TASK_QUESTIONS)
    assert sum(q['type']=='choice' for q in payloads[0]['questions'].values())==5
    assert evidence.answers
    r=ConstraintRuntime(cfg).update(f,evidence)
    assert r['state']=='EVIDENCE' and r['raw_answers']==evidence.answers  # A clear safety report is not task progress.
    assert cfg.program_id != replace(cfg,observation_profile='constraints').program_id
    monkeypatch.setattr(monitor,'post',lambda _: {'answers':{k:{'noul':v} for k,v in scores().items()}})
    with pytest.raises(ValueError,match='every compiled question'):
        monitor.evaluate([f])


def make_semantic_runner(tmp_path, advancing=False, shadow=False, completion=False, horizon=90):
    class SlowEnvironment(Environment):
        def step(self, action):
            time.sleep(.001)
            return super().step(action)
    env=SlowEnvironment(999)
    rollout=LiberoRollout(tmp_path,Student(),lambda:LiberoSession(env,[0],'pick up the cup',horizon),
                          max_decisions=0,chunk_steps=5)
    calls=[];events=[]
    class Monitor:
        def evaluate(self,frames):
            f=frames[-1]
            return observed(task_advancing=.95 if advancing or f['step_id']==5 else .05,
                            subgoal_complete=.95 if completion else .05,
                            task_satisfied=.95 if completion else .05)
    class Planner:
        def replan(self,r):
            calls.append(deepcopy(r))
            return {**{k:r[k] for k in ('request_id',*IDENTITY)},'decision':dict(
                request_id=r['native_request']['request_id'],mode='replan',reason='Try another grasp',
                subgoals=[dict(subgoal_id='recovery-1',instruction='align with the handle',success_condition='aligned')]),
                'board_assessment':dict(robot_status='Repeated approach',task_status='in_progress',
                    task_reason='No established lift',scene_context='Cup appears on support',uncertainties='Grasp uncertain',
                    evidence_observation_ids=[r['current_scene']['observation_id']])}
    config=MonitorConfig(history=6,max_step_gap=5,observation_profile='task_progress')
    runner=ContinuousScheduler(rollout,AsyncMonitor(Monitor(),config,2),Planner(),
        chunk_steps=5,include_initial_context=True,uncertainty_policy='continue',max_replans=2,
        progress_budget=40,subgoal_budget=40,stall_steps=15,astra_enabled=not shadow,
        shadow_only=shadow,emit=events.append)
    return runner,calls,events


def test_continuous_semantic_trigger_and_cross_replan_request(tmp_path):
    runner,calls,events=make_semantic_runner(tmp_path,horizon=180)
    result=runner.run()
    assert result['step_id']<180 and len(calls)==2
    assert result['state']=='STOPPED' and result['reason']=='Required Astra review budget exhausted'
    assert [r['failure']['trigger'] for r in calls]==['low_progress_scores_review','low_progress_scores_review']
    assert all(not r['failure']['confirmed_failure'] for r in calls)
    first=calls[0]['task_context'];second=calls[1]['task_context']
    assert first['scheduling']['last_high_advancement_score'] is None
    assert second['scheduling']['last_high_advancement_score'] is None
    assert len(second['plan_history'])==2 and second['recent_history']
    assert second['astra_interpretations'][0]['author']=='astra'
    assert second['astra_interpretations'][0]['assessment']['task_status']=='in_progress'
    assert calls[1]['previous_attempts'][0]['decision']['mode']=='replan'
    assert calls[1]['native_request']['step_id']==calls[1]['current_scene']['step_id']
    assert calls[1]['task_context']['latest_source_step']<=calls[1]['current_scene']['step_id']
    assert (tmp_path/'task_context.json').exists()
    assert any(e['kind']=='task_context_updated' for e in events)


def test_continued_real_progress_does_not_fire_budget_or_semantic_review(tmp_path):
    runner,calls,_=make_semantic_runner(tmp_path,advancing=True)
    result=runner.run()
    assert result['step_id']==90 and not calls
    assert 'status' not in result['task_context']['task']
    assert not result['task_context']['astra_interpretations']
    assert result['native_result']['success'] is False


def test_shadow_logs_semantic_stall_without_astra(tmp_path):
    runner,calls,events=make_semantic_runner(tmp_path,shadow=True)
    assert runner.run()['step_id']==90 and not calls
    assert any(e['kind']=='progress_review_suppressed' and e['evidence']['trigger']=='low_progress_scores_review' for e in events)


def test_completion_scores_go_to_astra_without_host_goal_advance(tmp_path):
    runner,calls,events=make_semantic_runner(tmp_path,completion=True)
    result=runner.run()
    assert len(calls)==2 and [r['failure']['trigger'] for r in calls]==['completion_scores_review','completion_scores_review']
    assert not any(e['kind']=='subgoal_completed' for e in events)
    assert result['native_result']['success'] is False


@pytest.mark.parametrize('module',['planner_server','stage1_planner'])
def test_planner_attaches_episode_memory_images_and_checks_root(tmp_path,module):
    import importlib
    from .test_planner import request
    r=request(tmp_path)
    old=tmp_path/'old.png';old.write_bytes(b'fixture')
    r['task_context']={'evidence_frames':[dict(step_id=5,images=[dict(camera='agentview',path=str(old))])]}
    fn=importlib.import_module('hybrid_rollout.libero.'+module).evidence_images
    images=fn(r,tmp_path)
    assert images[-1]['label']=='episode_context_step_5'
    r['task_context']['evidence_frames'][0]['images'][0]['path']=str(tmp_path.parent/'escape.png')
    with pytest.raises(ValueError,match='evidence-root'):
        fn(r,tmp_path)


def test_completion_scores_only_request_astra_review_even_with_high_advancement():
    c=ledger()
    update(c,5,task_satisfied=.95,task_advancing=.95)
    update(c,10,task_satisfied=.95,task_advancing=.95)
    assert c.review()['trigger']=='completion_scores_review'
    assert 'status' not in c.snapshot()['task']
    c.reviewed(10)
    assert c.review() is None


@pytest.mark.parametrize('module',['planner_server','stage1_planner'])
def test_board_assessment_schema_and_real_response_path(tmp_path,monkeypatch,module):
    import importlib,json
    from .test_planner import request
    from ..robodojo.jev.runner import automatic_response,validate_plan
    from .contract import validate_response
    from .task_context import validate_board_assessment
    planner_module=importlib.import_module('hybrid_rollout.libero.'+module)
    r=request(tmp_path)
    r['current_scene']['observation_id']='current'
    r['board_review_required']=True
    schema=planner_module.decision_schema(r)
    assert 'board_assessment' in schema['required']
    decision=automatic_response(r['native_request'],1,{})
    decision.pop('evaluation_method');decision.pop('decision_author')
    assessment=dict(robot_status='Approaching',task_status='uncertain',task_reason='Need clearer grasp evidence',
                    scene_context='Target visible',uncertainties='Whether held',evidence_observation_ids=['current'])
    from .test_episode_thread import FakeTransport
    raw=planner_module.AstraPlanner('/unused',tmp_path/'audit',tmp_path,
        transport_factory=lambda *args: FakeTransport(*args,payload=dict(decision=decision,board_assessment=assessment))).replan(r)
    rollout=SimpleNamespace(request=r['native_request'],validate=validate_response,
                            validate_board_assessment=validate_board_assessment)
    assert validate_plan(raw,r,rollout,1)['mode']=='student'
    raw['board_assessment']['evidence_observation_ids']=['invented']
    with pytest.raises(ValueError,match='cite supplied evidence'):
        validate_plan(raw,r,rollout,1)
    raw.pop('board_assessment')
    with pytest.raises(ValueError,match='envelope'):
        validate_plan(raw,r,rollout,1)
