import json

import pytest

from .contract import PROFILE, validate_response
from .planner_server import AstraPlanner, decision_schema, evidence_images
from ..robodojo.jev.monitor import IDENTITY
from ..robodojo.jev.plans import validate_decision
from ..robodojo.jev.runner import automatic_response


def request(tmp_path):
    images = []
    for camera in ('agentview', 'wrist'):
        path = tmp_path / (camera + '.png')
        path.write_bytes(b'image fixture')
        images.append(dict(camera=camera, path=str(path)))
    native = dict(request_id='proposal', step_id=1, max_chunk_steps=1,
                  instruction='put the bowl on the plate', robot_profile=PROFILE)
    return dict(version=1, request_id='outer', **{k: 'generation' for k in IDENTITY},
                max_chunk_steps=1, native_request=native, current_scene={'images': images})


def test_schema_has_only_libero_modes(tmp_path):
    r = request(tmp_path)
    schema = decision_schema(r)
    variants = schema['properties']['decision']['anyOf']
    assert {m for v in variants for m in v['properties']['mode']['enum']} == {
        'student', 'stop', 'edit', 'replan'}
    edit = variants[1]['properties']['edit']['properties']
    assert set(edit) == {'delta', 'gripper'}
    assert edit['delta']['maxItems'] == 6
    assert edit['delta']['items']['maximum'] == .25
    identifiers = variants[2]['properties']['subgoals']['items']['properties']['subgoal_id']['enum']
    assert identifiers == [f'recovery-{i}' for i in range(1, 9)]
    assert 'original-task' not in identifiers


def test_evidence_rejects_symlink_escape(tmp_path):
    inside = tmp_path / 'inside'
    inside.mkdir()
    r = request(inside)
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'outside')
    image = inside / 'agentview.png'
    image.unlink()
    image.symlink_to(outside)
    with pytest.raises(ValueError, match='evidence-root'):
        evidence_images(r, inside)


def test_uncertainty_schema_forbids_replan_edit_and_long_chunks(tmp_path):
    r = request(tmp_path)
    r.update(review_kind='uncertainty', max_chunk_steps=5)
    decision = decision_schema(r)['properties']['decision']['properties']
    assert decision['mode']['enum'] == ['student', 'stop']
    assert decision['steps']['enum'] == [1]
    assert 'edit' not in decision and 'subgoals' not in decision


@pytest.mark.parametrize('stale', [False, True])
def test_planner_validates_model_identity_before_return(tmp_path, monkeypatch, stale):
    r = request(tmp_path)
    decision = automatic_response(r['native_request'], 1, {})
    decision.pop('evaluation_method')
    decision.pop('decision_author')
    if stale:
        decision['request_id'] = 'old-proposal'

    from .test_episode_thread import FakeTransport
    planner = AstraPlanner('/codex', tmp_path / 'audit', tmp_path,
        transport_factory=lambda *args: FakeTransport(*args, payload={'decision':decision}))
    if stale:
        with pytest.raises(ValueError, match='Stale'):
            planner.replan(r)
    else:
        result = planner.replan(r)
        assert result['request_id'] == 'outer'
        assert result['decision']['request_id'] == 'proposal'


def test_bounded_correction_still_requires_failure_evidence(tmp_path):
    r = request(tmp_path)['native_request']
    d = automatic_response(r, 1, {})
    d.pop('evaluation_method')
    d.pop('decision_author')
    d.update(mode='edit', edit={'delta': [0.] * 6, 'gripper': 'keep'})
    with pytest.raises(ValueError, match='Takeover requires'):
        validate_decision(d, r, 1, validate_response)


@pytest.mark.parametrize('state,reason,expected', [
    ('STOPPED', 'Decision budget exhausted', 2),
    ('COMPLETED', 'Native rollout ended; no terminal replan', 2),
    ('PAUSED_MONITOR_UNAVAILABLE', 'Model timed out', 2),
    ('STOPPED', 'Uncertain observation budget exhausted', 2),
])
def test_smoke_does_not_hide_pipeline_failure(tmp_path, monkeypatch, state, reason, expected):
    from . import smoke
    summary = dict(state=state, reason=reason, step_id=1, native_result={'success': False},
                   replan_count=0, latest_feedback={'state': 'UNCERTAIN'})
    (tmp_path / 'jev_summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(smoke.run, 'main', lambda argv: 2)
    monkeypatch.setattr(smoke, 'planner_probe', lambda *args: {
        'status': 'passed', 'mode': 'student', 'simulation_executed': False})
    code = smoke.main(['--output', str(tmp_path), '--planner-url', 'http://127.0.0.1:8003/replan'])
    assert code == expected
    report = json.loads((tmp_path / 'smoke_report.json').read_text())
    assert report['native_result']['success'] is False
    assert report['status'] == ('passed' if expected == 0 else 'incomplete')


def test_task_smoke_requires_native_success_and_separates_probe(tmp_path, monkeypatch):
    from . import smoke
    summary = dict(state='COMPLETED', reason='Native rollout ended', step_id=70,
        native_result={'success': True, 'complete': True, 'execution_uncertain': False},
        replan_count=0, latest_feedback={'state': 'RUNNING'})
    (tmp_path / 'jev_summary.json').write_text(json.dumps(summary))
    monkeypatch.setattr(smoke.run, 'main', lambda argv: 0)
    monkeypatch.setattr(smoke, 'planner_probe', lambda *args: {'status': 'failed', 'error': 'offline'})
    assert smoke.main(['--output', str(tmp_path), '--planner-url', 'http://127.0.0.1:8003/replan']) == 0
    report = json.loads((tmp_path / 'smoke_report.json').read_text())
    assert report['task_success'] and report['status'] == 'passed'
    assert report['component_check'] == 'incomplete'
