"""Offline regressions for the premature-review failure seen in the real smoke."""
import json
from pathlib import Path
from types import SimpleNamespace

from .test_task_context import ledger, frame, goals, update, observed, make_semantic_runner
from ..robodojo.jev.monitor import IDENTITY, EvidenceScores


def test_real_first_sixty_scores_do_not_request_rapid_visual_review():
    data=json.loads((Path(__file__).parent/'fixtures/dynamic_smoke_v2_first60_scores.json').read_text())
    board=ledger()
    # Replay recorded scores only: this makes no claim about the trajectory
    # that would result from different Astra decisions.
    for row in data['reports']:
        f=frame(row['step_id']);f['observation_id']=row['observation_id']
        board.update({**{k:f[k] for k in (*IDENTITY,'step_id','observation_id')},
                      'scores':row['scores'],'raw_answers':row['raw_answers']},f)
        assert board.review() is None
    assert len(board.recent)==12 and not board.valid_votes
    assert 'status' not in board.snapshot()['task']


def test_visual_review_span_and_repeat_interval_survive_replan():
    board=ledger()
    for step in range(5,31,5):
        update(board,step,scene_comparable=.05)
        if step<30: assert board.review() is None
    assert board.review()['trigger']=='ambiguous_observations_review'
    board.reviewed(30)
    board.activate(frame(30,1),goals())
    for step in range(35,111,5):
        update(board,step,revision=1,target_visible=.05)
        if step<110: assert board.review() is None
    assert board.review()['trigger']=='ambiguous_observations_review'
    assert all(f['step_id']>30 for f in board.review()['evidence'])


def test_anomaly_and_completion_bypass_visual_review_interval():
    for field,trigger in [('collision','anomaly_scores_review'),('task_satisfied','completion_scores_review')]:
        board=ledger();board.reviewed(30)
        update(board,35,**{field:.95})
        update(board,40,**{field:.95})
        assert board.review()['trigger']==trigger
        assert not board.review()['confirmed_failure']


def test_unknown_progress_reaches_labelled_budget_then_fresh_pi(tmp_path):
    runner,calls,events=make_semantic_runner(tmp_path,horizon=90)
    runner.progress_budget=runner.subgoal_budget=80
    def evaluate(frames):
        return EvidenceScores(observed(task_advancing=.18), answers={
            'task_advancing':{'type':'choice','probabilities':{
                'advancing':.18,'attempt_no_effect':.6,'no_attempt':.04,'insufficient_evidence':.18}}})
    runner.monitor.monitor=SimpleNamespace(evaluate=evaluate)
    result=runner.run()
    assert [r['current_scene']['step_id'] for r in calls]==[80]
    assert calls[0]['failure']['trigger']=='task_progress_review_budget'
    assert not calls[0]['failure']['confirmed_failure']
    assert result['step_id']==90
    assert any(e['kind']=='executed_chunk' and e['step_id']>80 for e in events)
    assert not any(e['kind']=='subgoal_completed' for e in events)
