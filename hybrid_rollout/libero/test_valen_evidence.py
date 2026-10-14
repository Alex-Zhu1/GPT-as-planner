"""Offline protocol tests: actual payloads, explicit unknowns and fresh evidence windows."""
from copy import deepcopy
import json
import pytest
from ..robodojo.jev.monitor import (AsyncMonitor, ValenMonitor, ConstraintRuntime,
                                  MonitorConfig, IDENTITY, VISUAL_INPUT_FORMAT)
from .test_task_context import frame, ledger, update, observed


def answers(questions, **selected):
    defaults=dict(scene_comparable='comparable', task_advancing='attempt_no_effect',
                  stuck='no_repeated_attempt', grasp_failure='no_attempt', subgoal_complete='not_applicable')
    values=observed()
    result={}
    for key,q in questions.items():
        if q['type']=='choice':
            choice=selected.get(key,defaults[key])
            probs={k:(.97 if k==choice else .03/(len(q['criteria'])-1)) for k in q['criteria']}
            result[key]=dict(type='choice',probabilities=probs,choice=choice)
        else:result[key]=dict(type='noul',noul=values[key])
    return result


@pytest.mark.parametrize('profile', ['constraints', 'task_progress'])
@pytest.mark.parametrize('route', ['direct', 'async', 'inspection'])
def test_model_receives_only_scene_evidence_without_mutating_audit(monkeypatch, profile, route):
    config = MonitorConfig(observation_profile=profile, max_step_gap=5)
    monitor = ValenMonitor('http://localhost/predict', 1, observation_profile=profile)
    frames = [frame(step) for step in (0, 5)]
    for f in frames:
        f.update(program_id=config.program_id, success_condition='The cup is visibly off the table.',
                 executed_actions=[[.1] * 7], proposal_actions=[[.2] * 7],
                 executed_start_step=max(0, f['step_id'] - 5), action_source='eef',
                 previous_result={'native_success': True},
                 extra_control_metadata={'current_state': [123]})
        f['images'] = [dict(camera=c, path=f"/{f['step_id']}-{c}.png",
                            current_eef={'position': [456]}) for c in ('agentview', 'wrist')]
    original = deepcopy(frames)
    packets = []
    def post(payload):
        packets.append(payload)
        return dict(answers=answers(payload['questions']))
    monkeypatch.setattr(monitor, 'post', post)
    if route == 'direct':
        monitor.evaluate(frames)
    else:
        worker = AsyncMonitor(monitor, config, 2)
        worker.seed_context(frames[0])
        worker.submit(frames[1])
        worker.review()
        if route == 'inspection':
            worker.submit(frames[1], inspection=True)
            worker.review()
    assert frames == original
    for payload in packets:
        content = payload['state']['messages'][0]['content']
        body = json.loads(content[0]['text'])
        assert body['input_format'] == VISUAL_INPUT_FORMAT
        for sent in body['frames']:
            source = frames[0 if sent['step_id'] == 0 else 1]
            assert set(sent) <= {*IDENTITY, 'step_id', 'observation_id', 'captured_at',
                'task', 'subgoal', 'success_condition', 'executed_start_step',
                'context_only', 'inspection_only', 'images'}
            for key in (*IDENTITY, 'step_id', 'observation_id', 'captured_at',
                        'task', 'subgoal', 'success_condition'):
                assert sent[key] == source[key]
            assert sent['images'] == [{k: im[k] for k in ('camera', 'path')} for im in source['images']]
        media = [v['image_url']['url'] for v in content if v['type'] == 'image_url']
        assert media == [im['path'] for f in body['frames'] for im in f['images']]
    if route == 'inspection':
        assert body['frames'][-1]['inspection_only'] is True
    if route != 'direct':
        assert body['frames'][0]['context_only'] is True


def test_six_dual_camera_endpoints_and_full_candidate_distributions(monkeypatch):
    monitor=ValenMonitor('http://localhost/predict',1,observation_profile='task_progress')
    frames=[frame(step) for step in range(0,30,5)]
    for f in frames:f['images']=[dict(camera=c,path=f"/{f['step_id']}-{c}.png") for c in ('agentview','wrist')]
    packets=[]
    def post(payload):
        packets.append(payload)
        return dict(answers=answers(payload['questions'],task_advancing='insufficient_evidence'))
    monkeypatch.setattr(monitor,'post',post)
    result=monitor.evaluate(frames)
    media=[v['image_url']['url'] for v in packets[0]['state']['messages'][0]['content'] if v['type']=='image_url']
    assert media==[im['path'] for f in frames for im in f['images']]
    assert len(media)==12
    cfg=MonitorConfig(observation_profile='task_progress');f=frames[-1];f['program_id']=cfg.program_id
    report=ConstraintRuntime(cfg).update(f,result)
    assert report['raw_answers']['task_advancing']['probabilities']['insufficient_evidence']==.97
    assert report['state']=='EVIDENCE' and not report['violations'] and not report['subgoal_complete']


@pytest.mark.parametrize('bad',['missing','nan','sum','type'])
def test_candidate_response_rejects_corruption(monkeypatch,bad):
    monitor=ValenMonitor('http://localhost/predict',1,observation_profile='task_progress')
    raw=answers(monitor.compiled);q=raw['task_advancing']
    if bad=='missing':del q['probabilities']['insufficient_evidence']
    if bad=='nan':q['probabilities']['advancing']=float('nan')
    if bad=='sum':q['probabilities']['advancing']=.9
    if bad=='type':q['type']='noul'
    monkeypatch.setattr(monitor,'post',lambda _:dict(answers=raw))
    with pytest.raises(ValueError):monitor.evaluate([frame(5)])


def test_unknown_is_not_low_progress_and_raw_answers_survive_board(monkeypatch):
    monitor=ValenMonitor('http://localhost/predict',1,observation_profile='task_progress')
    monkeypatch.setattr(monitor,'post',lambda p:dict(answers=answers(p['questions'],task_advancing='insufficient_evidence')))
    board=ledger()
    for step in range(5,36,5):
        f=frame(step);s=monitor.evaluate([f]);r={**{k:f[k] for k in (*IDENTITY,'step_id','observation_id')},'scores':dict(s),'raw_answers':s.answers}
        board.update(r,f)
    assert not board.valid_votes
    assert board.review() is None
    assert board.snapshot()['recent_history'][-1]['raw_answers']==s.answers


def test_four_of_six_low_progress_and_one_high_does_not_erase_window():
    b=ledger()
    for step in (5,10,15,20):update(b,step)
    update(b,25,task_advancing=.95)
    update(b,30,task_advancing=.5)
    review=b.review()
    assert review['trigger']=='low_progress_scores_review' and review['low_progress_votes']==4
    assert review['report_count']==6


def test_recovery_clears_votes_not_history_and_old_reports_cannot_retrigger():
    b=ledger()
    for step in (5,10,15,20,25,30):update(b,step)
    history=deepcopy(b.snapshot()['recent_history']);b.reviewed(35)
    update(b,35,collision=.95)
    assert not b.votes and b.review() is None
    assert b.snapshot()['recent_history'][:6]==history
    update(b,40,collision=.95)
    assert b.review() is None
    update(b,45,collision=.95)
    assert b.review()['trigger']=='anomaly_scores_review'
    assert not b.review()['confirmed_failure']


def test_slow_detector_samples_chronologically_without_periodic_astra(tmp_path):
    import time
    from types import SimpleNamespace
    from .test_task_context import make_semantic_runner
    runner,calls,_=make_semantic_runner(tmp_path,advancing=True,horizon=40)
    seen=[]
    def evaluate(frames):
        time.sleep(.01)
        seen.append(frames[-1]['step_id'])
        return observed(task_advancing=.95)
    runner.monitor.monitor=SimpleNamespace(evaluate=evaluate)
    result=runner.run()
    assert seen==sorted(set(seen)) and seen[0]==5
    assert all(step in range(5,41,5) for step in seen)
    assert not result['monitor_worker_pending']
    assert not calls and result['coalesced_observations']==0


def test_unknown_after_four_low_reports_cannot_reuse_old_window(monkeypatch):
    monitor=ValenMonitor('http://localhost/predict',1,observation_profile='task_progress')
    b=ledger()
    for step in (5,10,15,20,25,30):update(b,step)
    assert b.review()['trigger']=='low_progress_scores_review'
    b.reviewed(30)
    monkeypatch.setattr(monitor,'post',lambda p:dict(answers=answers(p['questions'],scene_comparable='insufficient_evidence')))
    for step in range(35,111,5):
        f=frame(step);scores=monitor.evaluate([f])
        b.update({**{k:f[k] for k in (*IDENTITY,'step_id','observation_id')},'scores':scores,'raw_answers':scores.answers},f)
        if step < 110: assert b.review() is None
    review=b.review()
    assert review['trigger']=='ambiguous_observations_review'
    assert all(r['step_id']>30 for r in review['evidence'])


def test_anomaly_report_uses_candidates_not_confirmed_failure():
    cfg=MonitorConfig(observation_profile='task_progress',max_step_gap=5)
    runtime=ConstraintRuntime(cfg)
    for step in (5,10):
        f=frame(step);f['program_id']=cfg.program_id
        report=runtime.update(f,observed(collision=.95))
    assert report['state']=='EVIDENCE' and not report['violations']
    collision=next(c for c in report['checks'] if c['constraint_id']=='collision')
    assert collision['status']=='high_score_candidate'


def test_diffuse_candidate_distribution_is_unresolved_not_no_progress(monkeypatch):
    monitor=ValenMonitor('http://localhost/predict',1,observation_profile='task_progress')
    raw=answers(monitor.compiled)
    raw['task_advancing']['probabilities']=dict(advancing=.27,attempt_no_effect=.27,no_attempt=.27,insufficient_evidence=.19)
    monkeypatch.setattr(monitor,'post',lambda _:dict(answers=raw))
    board=ledger()
    for step in (5,10,15):
        f=frame(step);s=monitor.evaluate([f])
        board.update({**{k:f[k] for k in (*IDENTITY,'step_id','observation_id')},'scores':s,'raw_answers':s.answers},f)
    assert not board.valid_votes
    assert board.review() is None


def test_trigger_window_ids_remain_citable_after_recent_history_eviction():
    from .task_context import TaskContext, board_evidence_ids, validate_board_assessment
    from .test_task_context import goals
    board=TaskContext(MonitorConfig(max_step_gap=5,observation_profile='task_progress'))
    board.activate(frame(0),goals());frames=[]
    for step in range(5,81,5):
        f=frame(step);f['executed_start_step']=step-5;frames.append(f)
        board.update({**{k:f[k] for k in (*IDENTITY,'step_id','observation_id')},
                      'scores':observed(scene_comparable=.95 if step%15==5 else .05)},f)
    req=dict(current_scene=frame(80),evidence_frames=frames[-6:],task_context=board.snapshot(),failure=board.review())
    assert req['failure']['trigger']=='low_progress_scores_review'
    assert '0-5' not in [r['evidence']['observation_id'] for r in req['task_context']['recent_history']]
    assert '0-5' in board_evidence_ids(req)
    validate_board_assessment(dict(robot_status='Review',task_status='uncertain',task_reason='Review',
        scene_context='Review',uncertainties='Review',evidence_observation_ids=['0-5']),req)


def test_held_on_support_does_not_invalidate_comparable_progress():
    board=ledger()
    for step in (5,10,15):update(board,step,target_held=.95,target_on_support=.95,task_advancing=.95)
    assert board.review() is None
    assert board.last_advancement_signal['step_id']==15


def test_completion_requires_two_votes_for_the_same_condition():
    board=ledger()
    update(board,5,task_satisfied=.95,subgoal_complete=.05)
    update(board,10,task_satisfied=.05,subgoal_complete=.95)
    assert board.review() is None
    update(board,15,task_satisfied=.05,subgoal_complete=.95)
    assert board.review()['completion_candidates']==['subgoal_complete']


@pytest.mark.parametrize('chunk,span',[(1,30),(3,30),(5,40),(5,60)])
def test_short_chunks_and_long_spans_can_trigger(chunk,span):
    from .task_context import TaskContext
    from .test_task_context import goals
    board=TaskContext(MonitorConfig(max_step_gap=5,observation_profile='task_progress'),stall_steps=span)
    board.activate(frame(0),goals())
    for step in range(chunk,span+chunk,chunk):
        f=frame(step);f['executed_start_step']=step-chunk
        board.update({**{k:f[k] for k in (*IDENTITY,'step_id','observation_id')},'scores':observed()},f)
        if step<span:assert board.review() is None
    review=board.review()
    assert review['trigger']=='low_progress_scores_review' and review['report_count']==6
    assert review['last_step']-review['first_step']>=span


@pytest.mark.parametrize('shadow',[False,True])
def test_unavailable_astra_stops_required_review_except_explicit_shadow(tmp_path,shadow):
    from types import SimpleNamespace
    from .test_task_context import make_semantic_runner
    runner,calls,events=make_semantic_runner(tmp_path,horizon=30)
    runner.astra_enabled=False;runner.shadow_only=shadow
    runner.monitor.monitor=SimpleNamespace(evaluate=lambda _:observed(collision=.95))
    result=runner.run()
    assert not calls
    if shadow:
        assert result['step_id']==30
        assert any(e['kind']=='progress_review_suppressed' for e in events)
    else:
        assert result['step_id']==10 and result['state']=='STOPPED'
        assert result['reason']=='Required Astra evidence review unavailable'
