"""Stateless image transport, existing RoboDojo gates, and CLI/scheduler wiring."""
import base64
from copy import deepcopy
import json
from pathlib import Path
from threading import Event

import pytest
from PIL import Image

from .jev.api_planner import CompactAPIPlanner
from .jev.compact_planner import compact_packet, compact_instructions
from .jev.monitor import IDENTITY
from .test_astra_action_input import request as action_request
from .test_contract import POSE, response


def source(tmp_path, kind='progress'):
    request = action_request(kind)
    request['version'] = 2 if kind == 'background' else 1
    request['current_scene']['current_eef'] = {arm: deepcopy(POSE) for arm in ('left', 'right')}
    request['native_request']['current_eef'] = request['current_scene']['current_eef']
    frames = []
    for step in (0, 25, 50, 75, 100):
        scene = deepcopy(request['current_scene'])
        scene.update(step_id=step, observation_id=f'obs-{step}', images=[])
        for camera in ('front', 'left', 'right'):
            path = tmp_path/f'{step}-{camera}.png'
            Image.new('RGB', (8, 8), (step, 50, 200)).save(path)
            scene['images'].append(dict(camera=camera, path=str(path)))
        frames.append(scene)
    request.update(current_scene=frames[-1], evidence_frames=frames)
    return request


def result(payload):
    return dict(model='gpt-6-astra', status='completed', id='test-response', usage={'total_tokens': 123},
                output=[dict(type='message', content=[dict(type='output_text', text=json.dumps(payload))])])


@pytest.mark.parametrize('kind', ['background', 'progress'])
@pytest.mark.parametrize('mode,points', [('none', 0), ('sampled', 4), ('recent', 3)])
@pytest.mark.parametrize('output_format', ['full', 'mode-specific'])
def test_explicit_compact_prompt_images_schema_and_no_agent(tmp_path, monkeypatch, kind, mode, points, output_format):
    from .skill.transport import StdioAppServer
    monkeypatch.setattr(StdioAppServer, '__init__', lambda *a, **k: pytest.fail('API must not start Codex'))
    request = source(tmp_path, kind)
    sent = []
    answer = (dict(mode='continue', reason='Progress visible.') if kind == 'background' else
              dict(decision=response(request['native_request'])))
    if kind != 'background' and output_format == 'mode-specific':
        for field in ('edit', 'target'):
            answer['decision'].pop(field)
    def client(body):
        sent.append(deepcopy(body))
        return result(answer)
    planner = CompactAPIPlanner(tmp_path/'audit', client=client, action_chunk_mode=mode,
                                action_chunk_steps=3, output_format=output_format)
    output = planner.replan(request)
    assert {k: output[k] for k in ('request_id', *IDENTITY)} == {k: request[k] for k in ('request_id', *IDENTITY)}
    body = sent[0]
    assert body['tools'] == [] and body['store'] is False and body['stream'] is False
    assert body['reasoning'] == {'effort': 'high'}
    assert 'previous_response_id' not in body and 'conversation' not in body
    packet, images = compact_packet(request, action_chunk_mode=mode, action_chunk_steps=3)
    instructions, schema = compact_instructions(request, action_chunk_mode=mode, output_format=output_format)
    assert body['instructions'] == instructions and body['text']['format']['schema'] == schema
    content = body['input'][0]['content']
    assert json.loads(content[0]['text']) == packet
    assert len(content) == 10  # Three timestamps, three cameras each.
    for item, image in zip(content[1:], images):
        assert item['type'] == 'input_image' and item['detail'] == 'high'
        assert base64.b64decode(item['image_url'].split(',', 1)[1]) == Path(image['path']).read_bytes()
    assert len(packet.get('proposal', {}).get('sampled_fk', [])) == (0 if kind == 'background' else points)
    second = deepcopy(request)
    second['request_id'] = 'second'
    planner.replan(second)
    assert sent[0] == sent[1]  # No implicit previous answer or image deduplication.
    assert not list((tmp_path/'audit').rglob('AGENTS.md'))
    timing = json.loads(next((tmp_path/'audit').glob('astra_api_*/timing.json')).read_text())
    assert timing['backend'] == 'api' and timing['status'] == 'completed'
    assert timing['token_usage'] == {'total_tokens': 123} and timing['attached_images'] == 9


@pytest.mark.parametrize('bad', ['background_action', 'stale', 'eef_bound', 'student_bound', 'refusal', 'tool', 'incomplete', 'model'])
def test_invalid_model_results_rejected_before_execution(tmp_path, bad):
    request = source(tmp_path, 'background' if bad == 'background_action' else 'progress')
    decision = response(request['native_request'], 'eef' if bad == 'eef_bound' else 'student')
    if bad == 'stale': decision['request_id'] = 'old-proposal'
    if bad == 'eef_bound': decision['target']['left']['position'][0] = .051
    if bad == 'student_bound': decision['steps'] = 26
    raw = result(dict(mode='eef', reason='Stale action.') if bad == 'background_action' else dict(decision=decision))
    if bad == 'refusal': raw['output'][0]['content'] = [dict(type='refusal', refusal='No.')]
    if bad == 'tool': raw['output'] = [dict(type='function_call', name='execute')]
    if bad == 'incomplete': raw['status'] = 'incomplete'
    if bad == 'model': raw['model'] = 'other'
    planner = CompactAPIPlanner(tmp_path/'audit', client=lambda body: raw)
    with pytest.raises(ValueError): planner.replan(request)
    assert not list((tmp_path/'audit').glob('*/response.json'))
    assert json.loads(next((tmp_path/'audit').glob('*/timing.json')).read_text())['status'] == 'failed'


def test_failed_call_is_not_automatically_retried(tmp_path):
    calls = []
    def client(body):
        calls.append(body)
        raise TimeoutError('synthetic timeout')
    planner = CompactAPIPlanner(tmp_path/'audit', client=client)
    request = source(tmp_path, 'background')
    with pytest.raises(TimeoutError): planner.replan(request)
    with pytest.raises(FileExistsError): planner.replan(request)
    assert len(calls) == 1


def test_eef_still_requires_observed_failure_or_wrong_intent(tmp_path):
    request = source(tmp_path)
    request['native_request']['gate_policy'] = 'failure-or-intent'
    decision = response(request['native_request'], 'eef')
    decision['assessment'].update(execution_status='uncertain', intent_status='uncertain')
    planner = CompactAPIPlanner(tmp_path/'audit', client=lambda body: result(dict(decision=decision)))
    with pytest.raises(ValueError, match='Takeover requires observed failure'):
        planner.replan(request)
    assert not list((tmp_path/'audit').glob('*/response.json'))


def test_closed_planner_discards_inflight_result(tmp_path):
    def client(body):
        planner.close()
        return result(dict(mode='continue', reason='Late answer.'))
    planner = CompactAPIPlanner(tmp_path/'audit', client=client)
    with pytest.raises(RuntimeError, match='discard result'): planner.replan(source(tmp_path, 'background'))
    assert not list((tmp_path/'audit').glob('*/response.json'))


def cli_args(tmp_path):
    return ['--output', str(tmp_path/'run'), '--task', 'arrange_largest_number',
            '--checkpoint', str(tmp_path/'checkpoint'), '--valen-url', 'http://127.0.0.1:8001/predict']


@pytest.mark.parametrize('flags,expected', [
    (['--planner-backend', 'api', '--astra-action-chunk', 'recent', '--astra-action-steps', '3'], 'api'),
    (['--codex', '/fake/codex'], 'compact'),
    (['--codex', '/fake/codex', '--planner-backend', 'compact'], 'compact'),
    (['--codex', '/fake/codex', '--monitor-policy', 'legacy'], 'agent'),
    (['--planner-url', 'http://127.0.0.1:8003/replan'], 'http'),
])
def test_cli_selects_backend_before_any_simulator_connection(tmp_path, monkeypatch, flags, expected):
    from .jev import run, compact_planner, api_planner
    calls = []
    class Selected(Exception): pass
    def factory(name):
        def build(*args, **kwargs):
            calls.append((name, args, kwargs))
            raise Selected()
        return build
    monkeypatch.setattr(api_planner, 'CompactAPIPlanner', factory('api'))
    monkeypatch.setattr(compact_planner, 'CompactCodexPlanner', factory('compact'))
    monkeypatch.setattr(run, 'CodexPlanner', factory('agent'))
    monkeypatch.setattr(run, 'HTTPPlanner', factory('http'))
    with pytest.raises(Selected): run.main(cli_args(tmp_path) + flags)
    assert calls[0][0] == expected and not (tmp_path/'run').exists()
    if expected == 'api':
        assert calls[0][1][1] == 'http://127.0.0.1:8317/v1'
        assert calls[0][2]['action_chunk_steps'] == 3


@pytest.mark.parametrize('flags', [[], ['--api-base-url', 'http://localhost/v1'],
    ['--planner-backend', 'api', '--codex', '/fake/codex'],
    ['--planner-backend', 'api', '--planner-url', 'http://localhost/replan'],
    ['--codex', '/fake/codex', '--planner-backend', 'agent'],
    ['--planner-url', 'http://localhost/replan', '--astra-action-chunk', 'none'],
])
def test_cli_rejects_ambiguous_or_unsupported_flags(tmp_path, flags):
    from .jev.run import main
    with pytest.raises(SystemExit) as error: main(cli_args(tmp_path) + flags)
    assert error.value.code == 2 and not (tmp_path/'run').exists()


def test_api_scheduler_background_then_fresh_paused_recovery(tmp_path):
    from .test_jev_temporal import make_runner
    from .test_jev import scores
    advanced = Event()
    def monitor(frames):
        if frames[-1]['step_id'] >= 200: advanced.set()
        return scores(grasp_attempted=.4)
    runner, rollout, _, _, events = make_runner(tmp_path, horizon=400, monitor_fn=monitor, max_async=2)
    calls = []
    def client(body):
        packet = json.loads(body['input'][0]['content'][0]['text'])
        calls.append(packet)
        if packet['review_kind'] == 'background':
            assert advanced.wait(2), 'Background API call froze the robot'
            assert 'proposal' not in packet
            return result(dict(mode='request_recovery', reason='Review the fresh grasp.'))
        step = rollout.tick
        assert step >= 200 and packet['proposal']['step_id'] == step
        decision = response(packet['proposal'])
        decision.update(steps=25)
        decision['assessment']['execution_status'] = 'uncertain'
        assert rollout.tick == step
        return result(dict(decision=decision))
    runner.planner = CompactAPIPlanner(tmp_path/'planner', client=client)
    try:
        summary = runner.run()
        assert summary['state'] == 'COMPLETED', summary['reason']
        assert len(calls) >= 2 and calls[0]['review_kind'] == 'background'
        assert rollout.sim.calls.count('reset') == 1
        assert any(e['kind'] == 'background_review_response' and e['moved_steps'] >= 50 for e in events)
        assert runner.attempts and runner.attempts[0]['decision']['mode'] == 'student'
    finally:
        runner.planner.close()
