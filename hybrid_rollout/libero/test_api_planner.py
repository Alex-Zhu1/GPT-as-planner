"""API contracts: actual pixels, independent input, unchanged host gates."""
import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest
from PIL import Image

from .api_planner import API_INSTRUCTIONS, INPUT_FORMAT, structured_payload
from .planner_server import AstraPlanner
from .test_episode_thread import next_request
from .test_planner import request as legacy_request
from .test_instruction_planner import instruction_request
from .test_control_review import request as eef_request, decision as eef_decision
from .test_recovery_protocol import student, assessment


def images(request):
    for i, image in enumerate(request['current_scene']['images']):
        Image.new('RGB', (16, 16), (50+i*100, 20, 80)).save(image['path'])
    return request


def reply(payload):
    return dict(id='response-id', status='completed', model='gpt-6-astra',
        output=[dict(type='message', content=[dict(type='output_text', text=json.dumps(payload))])],
        usage=dict(input_tokens=100, output_tokens=20, total_tokens=120))


class Model:
    def __init__(self, payload):
        self.payload, self.calls = payload, []
    def __call__(self, body):
        self.calls.append(deepcopy(body))
        return reply(self.payload)


def planner(tmp_path, model):
    return AstraPlanner(None, tmp_path/'audit', tmp_path, backend='api',
        api_base_url='http://127.0.0.1:8317/v1', api_key_file=tmp_path/'unused-key', api_client=model,
        transport_factory=lambda *a: pytest.fail('API must never start Codex'))


def packet(call):
    return json.loads(call['input'][0]['content'][0]['text'])


def test_images_schema_and_no_codex_skill_or_tools(tmp_path, monkeypatch):
    r = images(instruction_request(tmp_path))
    model = Model(dict(decision=student(r), board_assessment=assessment(r)))
    original = Path.read_text
    def read(path, *a, **kw):
        if path.name in ('SKILL.md', 'AGENTS.md'):
            pytest.fail('API must not load agent instruction files')
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'read_text', read)
    result = planner(tmp_path, model).replan(r)
    assert result['decision']['mode'] == 'keep_instruction'
    call = model.calls[0]
    assert call['model'] == 'gpt-6-astra' and call['reasoning'] == {'effort': 'xhigh'}
    assert call['tools'] == [] and call['instructions'] == API_INSTRUCTIONS
    assert call['store'] is False and call['stream'] is False
    assert call['text']['format']['type'] == 'json_schema' and call['text']['format']['strict'] is True
    assert len(call['input']) == 1
    parts = call['input'][0]['content'][1:]
    assert len(parts) == 2
    for part, image in zip(parts, r['current_scene']['images']):
        assert base64.b64decode(part['image_url'].split(',', 1)[1]) == Path(image['path']).read_bytes()
    p = packet(call)
    assert p['input_format'] == INPUT_FORMAT
    assert [im['attachment_index'] for im in p['image_manifest']] == [0, 1]
    assert str(tmp_path) not in json.dumps(p)
    assert 'board_path' not in p and 'history_path' not in p
    assert not list((tmp_path/'audit').rglob('thread.json'))


def test_each_call_contains_explicit_history_and_all_selected_images(tmp_path):
    r = images(next_request(legacy_request(tmp_path), 10))
    historical = tmp_path/'historical.png'
    Image.new('RGB', (16, 16), 'white').save(historical)
    r['evidence_frames'] = [dict(step_id=0, observation_id='old-view', images=[
        dict(camera='agentview', path=str(historical))])]
    r['task_context'] = {'recent_history': [{'evidence': {'observation_id': 'old-view'}}],
                         'astra_interpretations': [{'assessment': 'EXPLICIT_HOST_HISTORY'}]}
    model = Model(dict(decision=student(r)))
    p = planner(tmp_path, model)
    p.replan(r)
    second = next_request(r, 15)
    model.payload = dict(decision=student(second))
    p.replan(second)
    assert all(len(c['input'][0]['content']) == 4 for c in model.calls)
    assert 'EXPLICIT_HOST_HISTORY' in json.dumps(packet(model.calls[1]))
    assert 'proposal-10' not in json.dumps(packet(model.calls[1]))
    other = images(next_request(legacy_request(tmp_path), 1))
    other['episode_id'] = 'different-experiment'
    model.payload = dict(decision=student(other))
    p.replan(other)
    assert 'EXPLICIT_HOST_HISTORY' not in json.dumps(packet(model.calls[2]))
    assert len(model.calls[2]['input'][0]['content']) == 3


def test_eef_inlines_measured_context_without_tools(tmp_path):
    r = images(eef_request(tmp_path))
    model = Model(dict(decision=eef_decision(r), board_assessment=assessment(r)))
    assert planner(tmp_path, model).replan(r)['decision']['mode'] == 'eef'
    p = packet(model.calls[0])
    assert p['control_context']['current_eef'] == r['native_request']['current_eef']
    assert p['control_context']['actions'] == r['native_request']['proposal_actions']
    assert p['control_context']['prediction_executed'] is False
    assert 'tool' not in p['control_diagnostics']
    assert 'proposal_actions' not in p['native_request']


@pytest.mark.parametrize('bad', ['stale', 'oversize', 'descriptor'])
def test_original_eef_and_identity_validators_still_apply(tmp_path, bad):
    r = images(eef_request(tmp_path))
    d = eef_decision(r)
    if bad == 'stale': d['request_id'] = 'wrong-proposal'
    elif bad == 'oversize': d['target']['position'][0] += 1
    else: r['control_diagnostics']['sha256'] = 'wrong'
    model = Model(dict(decision=d, board_assessment=assessment(r)))
    with pytest.raises(ValueError): planner(tmp_path, model).replan(r)
    assert not list((tmp_path/'audit').rglob('response.json'))
    if bad == 'descriptor': assert not model.calls


def test_retry_cache_binds_actual_image_bytes_and_survives_restart(tmp_path):
    r = images(instruction_request(tmp_path))
    model = Model(dict(decision=student(r), board_assessment=assessment(r)))
    p = planner(tmp_path, model)
    first = p.replan(r)
    assert p.replan(r) == first and planner(tmp_path, model).replan(r) == first
    assert len(model.calls) == 1
    Image.new('RGB', (16, 16), 'black').save(r['current_scene']['images'][0]['path'])
    with pytest.raises(ValueError, match='reused'): p.replan(r)


def test_failure_is_not_retried_or_converted_into_a_decision(tmp_path):
    r = images(instruction_request(tmp_path))
    calls = []
    def offline(body):
        calls.append(body)
        raise TimeoutError('offline')
    p = planner(tmp_path, offline)
    with pytest.raises(TimeoutError): p.replan(r)
    with pytest.raises(RuntimeError, match='reconcile'): p.replan(r)
    assert len(calls) == 1
    record = json.loads(next((tmp_path/'audit').rglob('invocation.json')).read_text())
    assert record['status'] == 'failed'


def test_continuous_rollout_consumes_api_decision_at_paused_boundary(tmp_path):
    from .test_task_context import make_semantic_runner
    runner, _, events = make_semantic_runner(tmp_path, horizon=100)
    calls = []
    def model(body):
        calls.append(deepcopy(body))
        p = packet(body)
        assert p['current_scene']['step_id'] == runner.rollout.tick
        assert p['control_context']['prediction_executed'] is False
        assert len(body['input'][0]['content']) >= 3
        return reply(dict(decision=dict(mode='stop', request_id=p['native_request']['request_id'],
                         reason='Protocol fixture: decline recovery'),
            board_assessment=dict(robot_status='Uncertain', task_status='in_progress',
                task_reason='No established success', scene_context='Fixture views',
                uncertainties='Synthetic observation',
                evidence_observation_ids=[p['current_scene']['observation_id']])))
    p = planner(tmp_path, model)
    runner.planner = p
    try:
        result = runner.run()
    finally:
        p.close()
    assert len(calls) == 1
    assert result['state'] == 'STOPPED' and result['reason'] == 'Planner declined recovery'
    assert result['step_id'] == packet(calls[0])['current_scene']['step_id']
    assert not result['native_result']['success']
    assert all(e['controls_during_planning'] == 0 for e in events if e['kind'] == 'pause_release')


@pytest.mark.parametrize('change', [
    {'model': 'other'}, {'status': 'incomplete'},
    {'output': [{'type': 'function_call', 'name': 'execute'}]},
    {'output': [{'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'no'}]}]},
])
def test_invalid_api_output_rejected(change):
    with pytest.raises(ValueError): structured_payload(dict(reply({'decision': {}}), **change))
