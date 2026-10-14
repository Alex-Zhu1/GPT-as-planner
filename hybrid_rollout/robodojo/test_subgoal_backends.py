"""Both planner transports and the Valen payload use the new language contract."""
from copy import deepcopy
import json

import pytest

from .jev.subgoal_monitor import SubgoalValenMonitor, SubgoalMonitorConfig
from .jev.subgoal_plan import goal_for
from .test_subgoals import request, runtime_frame, evidence, plan_response
from .test_api_planner import source
from .test_task_instructions import planner_for


@pytest.mark.parametrize('backend', ['compact','api'])
@pytest.mark.parametrize('invalid', [False, True])
def test_transports_share_planning_prompt_schema_and_validate_before_reply(tmp_path, backend, invalid):
    req = source(tmp_path)
    req.update(request())
    rows = req['remaining_requirements']
    answer = dict(mode='plan', reason='Three source-language stages.', assessments=[], subgoals=[
        goal_for(rows[:2],'g1'), goal_for(rows[2:4],'g2'), goal_for(rows[4:],'g3')])
    if invalid: answer['subgoals'].pop()
    sent={}
    planner=planner_for(backend,tmp_path/'audit',answer,sent)
    try:
        if invalid:
            with pytest.raises(ValueError,match='omitted'): planner.replan(req)
            assert not list((tmp_path/'audit').glob('astra*/response.json'))
        else:
            result=planner.replan(req)
            assert result['decision']==answer
        if backend=='api':
            prompt=sent['instructions']; schema=sent['text']['format']['schema']
            packet=json.loads(sent['input'][0]['content'][0]['text'])
        else:
            prompt=sent['thread/start']['developerInstructions'];schema=sent['turn/start']['outputSchema']
            packet=json.loads(sent['turn/start']['input'][0]['text'])
        assert 'ONE TO THREE' in prompt and 'Do not use tools' in prompt
        assert 'proposal' not in packet and 'native_request' not in packet
        assert packet['remaining_requirements']==rows
        assert packet['pi_instruction_contract']['max_words_per_requirement']==24
        assert schema['properties']['decision']['properties']['mode']['enum']==['plan']
        assert schema['properties']['decision']['properties']['subgoals']['maxItems']==3
        assert json.loads(next((tmp_path/'audit').glob('astra*/schema.json')).read_text())==schema
    finally: planner.close()


def test_valen_receives_plan_context_references_and_choice_candidates_without_hidden_state():
    config=SubgoalMonitorConfig()
    current=runtime_frame(50,config)
    previous=runtime_frame(25,config)
    current.update(current_state='SECRET', native_success=True, proposal_actions='SECRET')
    for frame in (previous,current):
        frame['images']=[dict(camera=k,path=f"/images/{frame['step_id']}-{k}.png") for k in ('head','left','right')]
    current['reference_frames']=[deepcopy(previous)]
    monitor=SubgoalValenMonitor('http://127.0.0.1:8001/predict',1)
    sent=[]
    def post(payload):
        sent.append(payload)
        return dict(answers=evidence(current).answers)
    monitor.post=post
    result=monitor.evaluate([previous,current])
    assert result.answers['placement_error']['type']=='choice'
    encoded=json.dumps(sent[0])
    assert 'SECRET' not in encoded and 'native_success' not in encoded and 'proposal_actions' not in encoded
    content=sent[0]['state']['messages'][0]['content']
    context=json.loads(content[0]['text'])
    assert context['frames'][-1]['plan_goals']==current['plan_goals']
    assert context['frames'][-1]['completion_checks']==current['completion_checks']
    assert len(context['references'])==1
    assert len([item for item in content if item['type']=='image_url'])==6


@pytest.mark.parametrize('backend',['compact','api'])
@pytest.mark.parametrize('mode',['advance','plan'])
def test_transports_support_astra_completion_and_repair_assessments(tmp_path,backend,mode):
    req=source(tmp_path);req.update(request())
    req.update(review_kind='subgoal_review',remaining_plan=plan_response(req,[2,2,3])['decision']['subgoals'])
    answer=plan_response(req,mode=mode)['decision']
    if mode=='plan':answer['subgoals'][0]['instruction']='Pick up the fallen figurine and place it upright on the stand. Finish the remaining placements and reset the robot arm.'
    sent={}
    planner=planner_for(backend,tmp_path/'audit',answer,sent)
    try:
        assert planner.replan(req)['decision']==answer
        packet=json.loads(sent['input'][0]['content'][0]['text'] if backend=='api' else sent['turn/start']['input'][0]['text'])
        assert packet['all_requirements']==req['all_requirements']
        assert packet['review_kind']=='subgoal_review'
    finally:planner.close()


@pytest.mark.parametrize('backend',['compact','api'])
def test_transports_deliver_wording_history_and_reject_pi_preservation_suffix(tmp_path,backend):
    req=source(tmp_path);req.update(request())
    req.update(review_kind='progress_review',remaining_plan=plan_response(req,[2,2,3])['decision']['subgoals'],
               instruction_history=[dict(instruction='Put the figurine on the stand.',executed_controls=300)])
    answer=plan_response(req)['decision']
    answer['subgoals'][0]['instruction']='Put the figurine upright without disturbing the clock.'
    sent={};planner=planner_for(backend,tmp_path/'audit',answer,sent)
    try:
        with pytest.raises(ValueError,match='preservation commentary'):planner.replan(req)
        packet=json.loads(sent['input'][0]['content'][0]['text'] if backend=='api' else sent['turn/start']['input'][0]['text'])
        assert packet['instruction_history']==req['instruction_history']
        assert packet['pi_instruction_contract']['max_sentences_per_requirement']==1
        assert not list((tmp_path/'audit').glob('astra*/response.json'))
    finally:planner.close()


def test_default_cli_wires_subgoals_with_native_budget_before_connection(tmp_path,monkeypatch):
    from .jev import run, compact_planner, subgoals
    from .test_api_planner import cli_args
    captured={}
    class Selected(Exception):pass
    monkeypatch.setattr(compact_planner,'CompactCodexPlanner',lambda *a,**k:object())
    def scheduler(rollout,monitor,planner,**kwargs):
        captured.update(config=monitor.config,monitor=monitor.monitor,**kwargs)
        raise Selected()
    monkeypatch.setattr(subgoals,'SubgoalScheduler',scheduler)
    with pytest.raises(Selected):run.main(cli_args(tmp_path)+['--codex','/fake/codex'])
    assert isinstance(captured['monitor'],SubgoalValenMonitor)
    assert captured['config'].observation_profile=='subgoals'
    assert captured['chunk_steps']==25 and captured['max_replans']==12
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('flags', [
    ['--astra-action-chunk','none'], ['--astra-output-format','mode-specific'],
    ['--evidence-threshold','0.2'], ['--max-async-seconds','2'],
    ['--monitor-policy','temporal','--subgoal-review-seconds','20']])
def test_inapplicable_mode_options_are_rejected(tmp_path,flags):
    from .jev.run import main
    from .test_api_planner import cli_args
    with pytest.raises(SystemExit):main(cli_args(tmp_path)+['--codex','/fake/codex',*flags])
    assert not (tmp_path/'run').exists()
