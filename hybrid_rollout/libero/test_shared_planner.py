import json
import pytest
import numpy as np
import sys
from types import SimpleNamespace

from .shared_planner import RecordedHTTPPlanner
from .max_run import MaxPolicy, digest


def test_shared_planner_records_only_its_request_and_rejects_other_response(tmp_path, monkeypatch):
    planner = RecordedHTTPPlanner('http://127.0.0.1:8003/replan', 1200, tmp_path)
    monkeypatch.setattr(planner, 'post', lambda request: {'request_id': 'another-experiment'})
    with pytest.raises(ValueError, match='another request identity'):
        planner.replan({'request_id': 'ours'})
    assert json.loads((tmp_path/'call-000/request.json').read_text()) == {'request_id': 'ours'}
    assert (tmp_path/'call-000/error.json').exists()
    assert not (tmp_path/'call-000/response.json').exists()
    monkeypatch.setattr(planner, 'post', lambda request: {'request_id': request['request_id']})
    assert planner.replan({'request_id': 'ours-2'}) == {'request_id': 'ours-2'}
    timing = json.loads((tmp_path/'call-001/timing.json').read_text())
    assert timing['request_id'] == 'ours-2'
    assert timing['shared_service_owned_by_client'] is False


@pytest.mark.parametrize('release_at_step', [None, 151])
def test_matched_policy_replays_only_identical_inputs_then_releases(tmp_path, monkeypatch, release_at_step):
    monkeypatch.setitem(sys.modules, 'openpi_client', SimpleNamespace(image_tools=SimpleNamespace(
        convert_to_uint8=lambda x: x, resize_with_pad=lambda x, *args: x)))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    obs = dict(agentview=image, wrist=image, states=np.zeros(8), instruction='original',
               max_policy_step=146, sim_state_sha256='same', max_event_applied=True)
    policy = MaxPolicy.__new__(MaxPolicy)
    policy.index, policy.seed, policy.replay = 30, 195, None
    policy.matched_until = release_at_step
    policy.matched_queries = {146: dict(instruction='original', sim_state_sha256='same',
        policy_image_sha256=digest(np.concatenate((image, image), axis=1)), actions=np.zeros((10, 7)))}
    actions, prediction = policy.infer(obs, tmp_path/'first.npz')
    assert prediction['max_query_source'] == 'matched_pre_review_replay'
    assert not actions.any()
    with pytest.raises(ValueError, match='differs from baseline'):
        policy.infer(dict(obs, sim_state_sha256='different'), tmp_path/'bad.npz')
    policy.client = SimpleNamespace(infer=lambda request: dict(actions=np.ones((10, 7))))
    if release_at_step is None:
        policy.release_matched_queries()
    actions, prediction = policy.infer(dict(obs, instruction='recovery', max_policy_step=151), tmp_path/'recovery.npz')
    assert prediction['max_query_source'] == 'model'
    assert actions.all()
