"""CPU-only contracts for compact evidence, execution receipts and action reads."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from .compact_input import (INPUT_FORMAT, action_record_path, archive_actions,
                            compact_evidence, execution_receipt, read_actions)
from .episode_thread import delta_packet
from .planner_server import AstraPlanner
from .test_episode_thread import FakeTransport, next_request, setup
from .test_planner import request
from .visual_evidence import evidence_images


def decode_scores_and_answers(format_, row):
    """Independent reader of the documented wire format."""
    schema = format_['schemas'][row['schema']]
    answers = {}
    for spec, value in zip(schema['questions'], row['answers']):
        name = spec['name']
        if spec['encoding'] == 'noul':
            answers[name] = dict(type='noul', noul=value)
        elif spec['encoding'] == 'choice':
            probabilities, choice, confidence = value
            answers[name] = dict(type='choice', probabilities=dict(zip(spec['candidates'], probabilities)),
                                 choice=choice, confidence=confidence)
        else:
            answers[name] = value
    scores = {}
    for name in schema['score_questions']:
        if name in row.get('score_overrides', {}):
            scores[name] = row['score_overrides'][name]
        elif answers[name]['type'] == 'noul':
            scores[name] = answers[name]['noul']
        else:
            scores[name] = min(1., sum(answers[name]['probabilities'][k]
                                      for k in format_['score_projections'][name]))
    return scores, answers


def test_all_recorded_probabilities_scores_choice_and_confidence_round_trip():
    reports = json.loads((Path(__file__).parent /
                          'fixtures/dynamic_smoke_v2_first60_scores.json').read_text())['reports']
    rows = [dict(r, evidence=dict(observation_id=r['observation_id'], step_id=r['step_id'],
                                  run_id='run', episode_id='episode'), scope={'instruction':'task'})
            for r in reports]
    # Preserve the wire representation through actual JSON serialization.
    format_, compact = json.loads(json.dumps(compact_evidence(rows)))
    assert len(format_['contexts']) == len(format_['schemas']) == 1
    for source, encoded in zip(rows, compact):
        assert decode_scores_and_answers(format_, encoded) == (source['scores'], source['raw_answers'])
        assert encoded['observation_id'] == source['observation_id']
        assert 'score_overrides' not in encoded


def test_mixed_schema_context_and_disagreeing_scores_are_never_dropped():
    rows = [dict(observation_id='a', evidence=dict(plan_revision=0), scope={'instruction':'old'},
                 scores={'collision':.8, 'future':.3}, raw_answers={
                     'collision':dict(type='noul', noul=.7999999999999999),
                     'future':dict(type='new-format', probabilities={'a':.3, 'b':.7}, extra='keep')}),
            dict(observation_id='b', evidence=dict(plan_revision=1), scope={'instruction':'new'},
                 scores={'collision':.25}, raw_answers={})]
    before = deepcopy(rows)
    format_, encoded = compact_evidence(rows)
    assert len(format_['contexts']) == len(format_['schemas']) == 2
    for original, row in zip(rows, encoded):
        assert decode_scores_and_answers(format_, row) == (original['scores'], original['raw_answers'])
    assert rows == before


def with_actions(tmp_path, step=5):
    r = next_request(request(tmp_path), step)
    actions = [[i / 31.] * 7 for i in range(25)]
    row = dict(evidence=dict(observation_id='history-actions', step_id=step),
               scores={'collision':.1}, raw_answers={}, executed_actions=actions)
    r['task_context'] = dict(new_evidence=[row], recent_history=[row])
    r['current_scene'].update(executed_actions=actions[-5:], executed_start_step=step-5,
                              action_source='student')
    return r, actions


def test_prompt_has_no_history_actions_but_audit_and_index_survive_eviction_restart(tmp_path):
    planner, made = setup(tmp_path)
    r, actions = with_actions(tmp_path)
    before = deepcopy(r)
    planner.replan(r)
    turn = next(v for m, v in made[0].calls if m == 'turn/start')
    packet = json.loads(turn['input'][0]['text'])
    assert 'executed_actions' not in turn['input'][0]['text']
    assert packet['new_evidence'][0]['executed_action_count'] == 25
    folder = Path(packet['board_path']).parent
    assert json.loads(Path(packet['board_path']).read_text()) == r['task_context']
    assert json.loads(next((tmp_path/'audit').glob('*/request.json')).read_text()) == before
    assert json.loads((folder/'history.jsonl').read_text().splitlines()[0])['request'] == before
    assert read_actions(folder, dict(observation_id='history-actions', offset=0, limit=20))['executed_actions'] == actions[:20]
    assert read_actions(folder, dict(observation_id='history-actions', offset=20, limit=20))['executed_actions'] == actions[20:]
    planner.close()
    restarted, transports = setup(tmp_path)
    fresh = next_request(r, 10)
    fresh['task_context'] = {}
    restarted.replan(fresh)
    assert any(m == 'thread/resume' for m, _ in transports[0].calls)
    assert read_actions(folder, dict(observation_id='history-actions', offset=0, limit=20))['total_controls'] == 25
    assert r == before


def test_action_tool_is_registered_and_served_during_real_rpc_loop(tmp_path):
    made = []
    class ReadingTransport(FakeTransport):
        def request(self, method, params, timeout):
            result = super().request(method, params, timeout)
            if method == 'turn/start':
                self.events.insert(0, dict(id=123, method='item/tool/call', params=dict(
                    threadId=params['threadId'], turnId=result['turn']['id'],
                    tool='read_observation_actions', arguments=dict(
                        observation_id='history-actions', offset=2, limit=3))))
            return result
    def factory(*args):
        t = ReadingTransport(*args); made.append(t); return t
    planner = AstraPlanner('/unused', tmp_path/'audit', tmp_path, transport_factory=factory)
    r, actions = with_actions(tmp_path)
    planner.replan(r)
    start = next(v for m, v in made[0].calls if m == 'thread/start')
    assert {t['name'] for t in start['dynamicTools']} == {'read_episode_record','read_observation_actions'}
    reply = next(v[1] for m, v in made[0].calls if m == 'reply')
    assert reply['success']
    assert json.loads(reply['contentItems'][0]['text'])['executed_actions'] == actions[2:5]


@pytest.mark.parametrize('args', [
    dict(observation_id='missing', offset=0, limit=5),
    dict(observation_id='../other-episode', offset=0, limit=5),
    dict(observation_id='history-actions', offset=True, limit=5),
    dict(observation_id='history-actions', offset=-1, limit=5),
    dict(observation_id='history-actions', offset=0, limit=21),
    dict(observation_id='history-actions', offset=0, limit=False),
    dict(observation_id='history-actions', offset=0, limit=5, path='/tmp/other'),
])
def test_action_tool_rejects_invalid_or_foreign_reads(tmp_path, args):
    r, _ = with_actions(tmp_path)
    folder = tmp_path/'episode'; folder.mkdir()
    archive_actions(folder, r)
    with pytest.raises(ValueError): read_actions(folder, args)
    foreign = tmp_path/'foreign'; foreign.mkdir()
    with pytest.raises(ValueError, match='this episode'):
        read_actions(foreign, dict(observation_id='history-actions', offset=0, limit=5))


def test_action_index_rejects_conflicts_and_symlink_escape(tmp_path):
    r, _ = with_actions(tmp_path)
    folder = tmp_path/'episode'; folder.mkdir()
    archive_actions(folder, r)
    changed = deepcopy(r)
    changed['task_context']['new_evidence'][0]['executed_actions'][0][0] = 123
    with pytest.raises(ValueError, match='different executed actions'): archive_actions(folder, changed)
    path = action_record_path(folder, 'history-actions')
    outside = tmp_path/'outside.json'; outside.write_bytes(path.read_bytes())
    path.unlink(); path.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        read_actions(folder, dict(observation_id='history-actions', offset=0, limit=5))


def test_action_lookup_also_indexes_legacy_evidence_frames_without_board(tmp_path):
    r = next_request(request(tmp_path), 5)
    r['evidence_frames'] = [dict(observation_id='legacy', step_id=4,
                                 executed_actions=[[.12345678901234567] * 7])]
    folder = tmp_path/'episode'; folder.mkdir()
    archive_actions(folder, r)
    result = read_actions(folder, dict(observation_id='legacy', offset=0, limit=5))
    assert result['executed_actions'] == r['evidence_frames'][0]['executed_actions']
    assert result['step_id'] == 4


def test_revised_probabilities_are_delivered_again_without_repeating_actions(tmp_path):
    planner, made = setup(tmp_path)
    r, _ = with_actions(tmp_path); planner.replan(r)
    fresh = next_request(r, 10)
    row = fresh['task_context']['new_evidence'][0]
    row.update(scores={'collision':.23456789012345678},
               raw_answers={'collision':dict(type='noul', noul=.23456789012345678)})
    planner.replan(fresh)
    turns = [p for m,p in made[0].calls if m == 'turn/start']
    packet = json.loads(turns[-1]['input'][0]['text'])
    assert len(packet['new_evidence']) == 1
    assert decode_scores_and_answers(packet['evidence_format'], packet['new_evidence'][0]) == (
        row['scores'], row['raw_answers'])
    assert 'executed_actions' not in turns[-1]['input'][0]['text']


@pytest.mark.parametrize('thread,turn', [('foreign','t'),('thread-1','old')])
def test_action_rpc_rejects_foreign_thread_and_turn(tmp_path, thread, turn):
    planner, _ = setup(tmp_path)
    r, _ = with_actions(tmp_path); planner.replan(r)
    folder = next((tmp_path/'audit/episodes').iterdir())
    event = dict(id=1, params=dict(threadId=thread, turnId=turn, tool='read_observation_actions',
                                  arguments=dict(observation_id='history-actions', offset=0, limit=5)))
    with pytest.raises(RuntimeError, match='another episode or turn'):
        planner.threads.read_record(event, folder, 'thread-1', 't')


def test_previous_result_uses_actual_execution_not_previous_astra_interpretation(tmp_path):
    r, _ = with_actions(tmp_path, step=205)
    r['previous_attempts'] = [dict(result='original_instruction_restored_without_execution',
                                   decision={'reason':'OLD-SEMANTICS'}, board_review={'text':'OLD-BOARD'})]
    actual = dict(decision=41, start_tick=200, end_tick=205, executed_steps=5, source_detail='eef')
    r['native_request']['previous_result'] = actual
    receipt = execution_receipt(r)
    assert all(receipt[k] == v for k, v in actual.items())
    assert 'OLD-' not in json.dumps(receipt)
    del r['native_request']['previous_result']
    assert execution_receipt(r)['source_detail'] == 'student'
    r['current_scene'].pop('executed_actions')
    assert execution_receipt(r) is None  # Never infer execution from a replan result.


def test_compact_manifest_matches_actual_attachment_order_and_keeps_aliases(tmp_path):
    planner, made = setup(tmp_path)
    r = next_request(request(tmp_path), 5)
    r['task_context'] = dict(keyframes={'last_decision':dict(
        r['current_scene'], observation_id='alias-observation')})
    planner.replan(r)
    turn = next(v for m, v in made[0].calls if m == 'turn/start')
    packet = json.loads(turn['input'][0]['text'])
    full = json.loads(next((tmp_path/'audit').glob('*/image_manifest.json')).read_text())
    assert len(packet['image_manifest']) == len(full) == len(turn['input'])-1 == 2
    for i, entry in enumerate(packet['image_manifest']):
        assert entry['attachment_index'] == i
        assert entry['camera'] == full[i]['camera']
        assert turn['input'][i+1]['path'] == full[i]['path']
        assert 'path' not in entry and 'references' not in entry
        assert entry['aliases'][0]['observation_id'] == 'alias-observation'


def test_v1_episode_cannot_silently_switch_protocol_or_start_new_thread(tmp_path):
    planner, made = setup(tmp_path)
    r = next_request(request(tmp_path), 5); planner.replan(r)
    path = next((tmp_path/'audit/episodes').glob('*/thread.json'))
    state = json.loads(path.read_text()); state.pop('input_format'); path.write_text(json.dumps(state))
    before = list(made[0].calls)
    with pytest.raises(RuntimeError, match='older input format'):
        planner.replan(next_request(r, 10))
    assert made[0].calls == before


def test_current_candidate_dedup_preserves_distinct_measurements(tmp_path):
    r = next_request(request(tmp_path), 5)
    r['current_scene'].update(current_state=[1,2], current_eef={'position':[1,2,3]})
    r['native_request'].update(current_state=[1,2], current_eef={'position':[9,8,7]})
    packet = delta_packet(r, evidence_images(r,tmp_path), {}, tmp_path)[0]
    assert 'current_state' not in packet['native_request']
    assert packet['native_request']['current_scene_ref']['fields'] == ['current_state']
    assert packet['native_request']['current_eef'] == {'position':[9,8,7]}
    assert packet['input_format'] == INPUT_FORMAT


def test_actual_rollout_supplies_last_execution_receipt_after_replan(tmp_path):
    from .test_task_context import make_semantic_runner
    from .instruction_planner import public_evidence
    runner, calls, _ = make_semantic_runner(tmp_path, horizon=90)
    runner.progress_budget = runner.subgoal_budget = 30
    runner.run()
    assert calls
    for r in calls:
        receipt = r['native_request']['previous_result']
        assert receipt['end_tick'] == r['current_scene']['step_id']
        assert receipt['executed_steps'] == receipt['end_tick'] - receipt['start_tick']
        assert receipt['source_detail'] == 'student'
        assert execution_receipt(r)['end_tick'] == receipt['end_tick']
        for row in public_evidence(r)['task_context']['new_evidence']:
            actual = row['execution_receipt']
            assert actual['end_tick'] == row['evidence']['step_id']
            assert actual['executed_steps'] == row['executed_action_count']
            assert 'executed_actions' not in row
