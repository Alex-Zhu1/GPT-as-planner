"""Constrain actual Pi language input across both compact transports."""
import json

import pytest

from .jev.api_planner import CompactAPIPlanner
from .jev.compact_planner import CompactCodexPlanner, compact_instructions, compact_packet
from .jev.task_instructions import ARRANGE_NUMBERS, ORGANIZE_TABLE, replan_candidates
from .test_api_planner import source, result


def planner_for(backend, output, answer, sent):
    def client(body):
        sent.update(body)
        decision = answer(json.loads(body['input'][0]['content'][0]['text'])) if callable(answer) else answer
        return result(dict(decision=decision))

    class Transport:
        def __init__(self, *args): self.delivered = False
        def notify(self, *args): pass
        def close(self): pass
        def request(self, method, params, timeout):
            sent[method] = params
            if method == 'thread/start':
                return dict(model='gpt-6-astra', reasoningEffort='high', thread=dict(id='thread'))
            if method == 'turn/start': return dict(turn=dict(id='turn'))
            return {}
        def next_message(self, timeout):
            if not self.delivered:
                self.delivered = True
                decision = answer(json.loads(sent['turn/start']['input'][0]['text'])) if callable(answer) else answer
                return dict(method='item/completed', params=dict(threadId='thread', turnId='turn',
                    item=dict(type='agentMessage', text=json.dumps(dict(decision=decision)))))
            return dict(method='turn/completed', params=dict(threadId='thread',
                        turn=dict(id='turn', status='completed')))

    options = dict(action_chunk_mode='none', output_format='mode-specific')
    if backend == 'api': return CompactAPIPlanner(output, client=client, **options)
    return CompactCodexPlanner(output, 'codex', transport_factory=Transport, **options)


@pytest.mark.parametrize('backend', ['compact', 'api'])
@pytest.mark.parametrize('bad', [None, 'queue', 'move_digit', 'paraphrase', 'arm', 'condition',
                               'identity', 'unknown_task', 'stale', 'reason'])
def test_transports_only_accept_one_exact_task_goal_and_keep_nine_images(tmp_path, backend, bad):
    request = source(tmp_path)
    request['original_task'] = request['current_subgoal'] = ORGANIZE_TABLE
    goal = replan_candidates(ORGANIZE_TABLE)[0]
    answer = dict(request_id='proposal', mode='replan', reason='The mouse remains off the pad.', subgoals=[goal])
    if bad == 'queue': answer['subgoals'].append(replan_candidates(ORGANIZE_TABLE)[1])
    if bad == 'move_digit':
        request['original_task'] = ARRANGE_NUMBERS
        goal['instruction'] = 'Move 2 from the second pad to the third pad.'
    if bad == 'paraphrase': goal['instruction'] = 'Put the mouse onto its mouse pad.'
    if bad == 'arm': goal['instruction'] = 'Place the mouse on the mouse pad using the right arm.'
    if bad == 'condition': goal['success_condition'] = 'The keyboard rests inside the frame.'
    if bad == 'identity': goal['subgoal_id'] = 'original-task'
    if bad == 'unknown_task': request['original_task'] = 'An unreviewed task.'
    if bad == 'stale': answer['request_id'] = 'old-proposal'
    if bad == 'reason': answer['reason'] = 'x'*601
    sent = {}
    output = tmp_path/'audit'
    planner = planner_for(backend, output, answer, sent)
    try:
        if bad:
            with pytest.raises(ValueError): planner.replan(request)
            assert not list(output.glob('astra*/response.json'))
        else:
            assert planner.replan(request)['decision'] == answer
            timing = json.loads(next(output.glob('astra*/timing.json')).read_text())
            assert timing['attached_images'] == 9 and timing['output_format'] == 'mode-specific'
            assert timing['action_chunk_mode'] == 'none'
        schema = sent['turn/start']['outputSchema'] if backend == 'compact' else sent['text']['format']['schema']
        branches = {b['properties']['mode']['enum'][0]: b for b in schema['properties']['decision']['anyOf']}
        assert 'edit' not in branches['student']['properties'] and 'target' not in branches['student']['properties']
        if bad == 'unknown_task':
            assert 'replan' not in branches
        else:
            goals = branches['replan']['properties']['subgoals']
            assert goals['minItems'] == goals['maxItems'] == 1
            encoded = [{k: v['enum'][0] for k, v in b['properties'].items()} for b in goals['items']['anyOf']]
            assert encoded == replan_candidates(request['original_task'])
    finally:
        planner.close()


def test_number_task_never_introduces_digit_specific_commands():
    choices = replan_candidates(ARRANGE_NUMBERS)
    assert len(choices) == 1 and choices[0]['instruction'] == ARRANGE_NUMBERS
    assert not any(character.isdigit() for character in choices[0]['instruction'])


def test_category_destinations_come_from_episode_instruction_not_hardcoded_order():
    original = ('Put watch objects into the left basket, wooden_toy objects into the middle basket, '
                'and car objects into the right basket, then reset the robot arm.')
    choices = replan_candidates(original)
    assert [g['instruction'] for g in choices] == [
        'Put watch objects into the left basket.', 'Put wooden_toy objects into the middle basket.',
        'Put car objects into the right basket.', 'Reset the robot arm.']
    choices[0]['instruction'] = 'Changed by caller.'
    assert replan_candidates(original)[0]['instruction'] == 'Put watch objects into the left basket.'


def test_background_input_and_decisions_do_not_get_a_language_plan(tmp_path):
    request = source(tmp_path, 'background')
    request['original_task'] = ORGANIZE_TABLE
    packet, images = compact_packet(request, action_chunk_mode='none')
    _, schema = compact_instructions(request, output_format='mode-specific')
    assert 'replan_candidates' not in packet and len(images) == 9
    assert schema['properties']['mode']['enum'] == ['continue', 'request_recovery']


@pytest.mark.parametrize('backend', ['compact', 'api'])
def test_one_template_reaches_pi_then_original_task_resumes(tmp_path, monkeypatch, backend):
    from .test_contract import Student
    from .test_jev import LongSim, scores, setup
    original_request = LongSim.request
    def simulator_request(self, op, **args):
        value = original_request(self, op, **args)
        if op in ('metadata', 'teacher_observation'):
            value['instruction'] = ORGANIZE_TABLE
        return value
    monkeypatch.setattr(LongSim, 'request', simulator_request)
    pi_instructions = []
    class TrackedStudent(Student):
        def infer(self, observation, output):
            pi_instructions.append(observation['instruction'])
            return super().infer(observation, output)
    def monitor(frames):
        frame = frames[-1]
        return scores(grasp_failure=.95 if frame['plan_revision'] == 0 else .05,
                      subgoal_complete=.95 if frame.get('success_condition') else .05)
    (tmp_path/'controller').mkdir()
    runner, rollout, _, _, events = setup(tmp_path/'controller', student=TrackedStudent(), monitor_fn=monitor)
    goal = replan_candidates(ORGANIZE_TABLE)[0]
    def answer(packet):
        return dict(request_id=packet['proposal']['request_id'], mode='replan', reason='The mouse is off its pad.',
                    subgoals=[goal])
    runner.planner = planner_for(backend, tmp_path/'planner', answer, {})
    try:
        summary = runner.run()
        assert summary['state'] == 'COMPLETED', summary['reason']
        assert goal['instruction'] in pi_instructions
        assert set(pi_instructions) == {ORGANIZE_TABLE, goal['instruction']}
        assert pi_instructions[-1] == rollout.meta['instruction'] == ORGANIZE_TABLE
        assert [e['subgoal'] for e in events if e['kind'] == 'subgoal_completed'] == [goal]
        assert len(summary['remaining_plan']) == 1
        assert summary['remaining_plan'][0]['subgoal_id'] == 'original-task'
    finally:
        runner.planner.close()
