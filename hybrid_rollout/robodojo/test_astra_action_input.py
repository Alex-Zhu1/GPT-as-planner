"""Selectable Astra trajectory disclosure, bounded to the upcoming execution prefix."""
from copy import deepcopy
import json

import pytest

from .jev.compact_planner import CompactCodexPlanner, compact_packet, recovery_output_schema
from .jev.monitor import IDENTITY
from .jev.task_instructions import ARRANGE_NUMBERS, replan_candidates


def request(kind='progress', budget=25):
    scene = dict(step_id=100, observation_id='current', images=[dict(camera='front', path='/synthetic/front.png')],
                 current_eef=dict(left=dict(position=[0, 0, 1])))
    return dict(request_id='review', **{k: 'test' for k in IDENTITY}, review_kind=kind,
                current_scene=scene, evidence_frames=[], original_task=ARRANGE_NUMBERS, current_subgoal=ARRANGE_NUMBERS,
                max_chunk_steps=budget, native_request=dict(request_id='proposal', step_id=100,
                    student_step_limit=budget, current_eef=scene['current_eef'],
                    student_eef_trajectory=[dict(index=i, left=dict(position=[i/100, 0, 1]),
                        right=dict(gripper_closed=i>=2)) for i in range(50)], actions='RAW_ACTION_SENTINEL'))


@pytest.mark.parametrize('mode,steps,indices', [
    ('none', 5, []), ('sampled', 5, [0, 12, 25, 49]), ('recent', 3, [0, 1, 2]),
    ('recent', 25, list(range(25))),
])
def test_disclosure_modes_preserve_identity_and_exclude_raw_actions(mode, steps, indices):
    source = request()
    original = deepcopy(source)
    packet, images = compact_packet(source, action_chunk_mode=mode, action_chunk_steps=steps)
    proposal = packet['proposal']
    assert [p['index'] for p in proposal.get('sampled_fk', [])] == indices
    assert proposal['request_id']=='proposal' and proposal['max_student_steps']==25
    assert 'RAW_ACTION_SENTINEL' not in json.dumps(packet)
    if mode=='none':
        assert 'sampled_fk' not in proposal
    assert source == original and len(images)==1


def test_recent_prefix_respects_host_and_student_budgets_and_short_trajectory():
    source = request(budget=3)
    source['native_request']['student_step_limit']=2
    packet, _ = compact_packet(source, action_chunk_mode='recent', action_chunk_steps=10)
    assert [p['index'] for p in packet['proposal']['sampled_fk']]==[0, 1]
    source['native_request']['student_eef_trajectory']=[]
    assert compact_packet(source, action_chunk_mode='recent')[0]['proposal']['sampled_fk']==[]


@pytest.mark.parametrize('mode', ['none', 'sampled', 'recent'])
def test_background_never_discloses_predicted_actions(mode):
    packet, _ = compact_packet(request('background'), action_chunk_mode=mode)
    assert 'proposal' not in packet and 'sampled_fk' not in json.dumps(packet)


@pytest.mark.parametrize('mode,steps', [('bad', 5), ('recent', 0), ('recent', 26), ('none', True)])
def test_invalid_options_rejected_before_transport(tmp_path, mode, steps):
    with pytest.raises(ValueError):
        CompactCodexPlanner(tmp_path, 'codex', action_chunk_mode=mode, action_chunk_steps=steps)


def test_mode_specific_output_retains_evidence_and_only_required_control_payload():
    schema = recovery_output_schema(25, 'mode-specific')['properties']['decision']
    modes = {branch['properties']['mode']['enum'][0]:branch for branch in schema['anyOf']}
    for mode in ('student', 'edit', 'eef', 'stop'):
        properties = modes[mode]['properties']
        assert 'assessment' in properties
        assert properties['steps']['maximum']==(5 if mode in ('edit', 'eef') else 25)
        assert ('edit' in properties)==(mode=='edit')
        assert ('target' in properties)==(mode=='eef')
        assert set(properties)==set(modes[mode]['required'])
    assert 'subgoals' in modes['replan']['properties']
    with pytest.raises(ValueError):
        recovery_output_schema(25, 'invalid')


@pytest.mark.parametrize('mode', ['none', 'sampled', 'recent'])
def test_planner_sends_selected_input_and_records_mode(tmp_path, mode):
    sent = {}
    class Transport:
        def __init__(self, *args): pass
        def notify(self, *args): pass
        def close(self): pass
        def request(self, method, params, timeout):
            sent[method] = params
            if method=='thread/start':
                return dict(model='gpt-6-astra', reasoningEffort='high', thread=dict(id='thread'))
            if method=='turn/start':
                return dict(turn=dict(id='turn'))
            return {}
        def next_message(self, timeout):
            if 'completed' not in sent:
                sent['completed']=True
                return dict(method='item/completed', params=dict(threadId='thread',turnId='turn',
                    item=dict(type='agentMessage', text=json.dumps(dict(decision=dict(
                        request_id='proposal', mode='replan', reason='Retry.',
                        subgoals=replan_candidates(ARRANGE_NUMBERS)))))))
            return dict(method='turn/completed',params=dict(threadId='thread',turn=dict(id='turn',status='completed')))
    planner = CompactCodexPlanner(tmp_path, 'codex', transport_factory=Transport,
                                 action_chunk_mode=mode, action_chunk_steps=3)
    try:
        planner.replan(request())
        packet = json.loads(sent['turn/start']['input'][0]['text'])
        assert len(packet['proposal'].get('sampled_fk', [])) == {'none':0, 'sampled':4, 'recent':3}[mode]
        timing = json.loads((tmp_path/'astra_review/timing.json').read_text())
        assert timing['action_chunk_mode']==mode and timing['action_chunk_steps']==3
        if mode=='none':
            assert 'mark intent_status uncertain' in sent['thread/start']['developerInstructions']
    finally:
        planner.close()
