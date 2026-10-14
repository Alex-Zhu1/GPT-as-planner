"""Accumulation thresholds, uncertainty escalation and same-episode recovery."""
from dataclasses import replace

import pytest

from .jev.accumulation import AccumulationConfig, AccumulatingRuntime, AccumulatingScheduler
from .test_contract import response
from .test_jev import frame, scores, setup, LongSim
from .jev.monitor import IDENTITY


def update(runtime, step, **values):
    return runtime.update(frame(step, runtime.config), scores(**values), now=100.1+step)


def accumulated_setup(tmp_path, *, config=None, **kwargs):
    old, rollout, requests, windows, events = setup(tmp_path,
        config=config or AccumulationConfig(uncertainty_observations=3), max_replans=0, **kwargs)
    old.monitor.runtime_factory = AccumulatingRuntime
    old.monitor.activate()
    runner = AccumulatingScheduler(rollout, old.monitor, old.planner, max_replans=0,
                                  emit=events.append)
    return runner, rollout, requests, windows, events


def student_review(request):
    decision = response(request['native_request'], 'student')
    decision['steps'] = 1
    decision['assessment']['execution_status'] = 'uncertain'
    return dict(**{key: request[key] for key in ('request_id', *IDENTITY)}, decision=decision)


def test_moderate_evidence_accumulates_without_any_high_threshold_vote():
    runtime = AccumulatingRuntime(AccumulationConfig(uncertainty_observations=100))
    values = []
    for step in range(1, 5):
        report = update(runtime, step, target_mismatch=.65)
        values.append(report['accumulation']['scores']['target_mismatch'])
        assert report['review_requested'] == (step == 4)
        assert not report['violations']
    assert values == pytest.approx([.45, .855, 1.2195, 1.54755])
    assert report['review_reason'] == 'accumulated_risk'


def test_isolated_spike_decays_and_normal_evidence_drains_score():
    runtime = AccumulatingRuntime(AccumulationConfig())
    assert not update(runtime, 1, collision=.99)['review_requested']
    for step in range(2, 8):
        report = update(runtime, step)
        assert not report['review_requested']
    assert report['accumulation']['scores']['collision'] == 0


def test_uncertain_applicability_does_not_turn_low_failure_into_high_risk():
    runtime = AccumulatingRuntime(AccumulationConfig(uncertainty_observations=3))
    for step in range(1, 4):
        report = update(runtime, step, progress_attempted=.65, stuck=.0057)
        assert report['accumulation']['scores']['stuck'] == 0
        assert report['review_requested'] == (step == 3)
    assert report['review_reason'] == 'persistent_uncertainty'
    assert report['accumulation']['confirmed_failure'] is False


def test_clear_endpoint_resets_uncertainty_streak_without_erasing_risk():
    runtime = AccumulatingRuntime(AccumulationConfig())
    update(runtime, 1, target_mismatch=.65)
    report = update(runtime, 2)
    assert report['accumulation']['uncertain_observations'] == 0
    assert report['accumulation']['scores']['target_mismatch'] > 0


def test_static_inspection_is_not_an_accumulation_vote():
    runtime = AccumulatingRuntime(AccumulationConfig(uncertainty_observations=3))
    for step in range(1, 6):
        report = runtime.update(frame(step, runtime.config, inspection_only=True),
                                scores(collision=.99), now=100.1+step)
        assert not report['review_requested']
        assert report['accumulation']['observations'] == 0
        assert report['accumulation']['scores']['collision'] == 0


@pytest.mark.parametrize('fault', ['gap', 'duplicate', 'stale', 'generation', 'invalid'])
def test_invalid_or_discontinuous_evidence_resets_accumulation(fault):
    runtime = AccumulatingRuntime(AccumulationConfig())
    update(runtime, 1, collision=.99)
    f = frame(2, runtime.config)
    value = scores(collision=.99)
    if fault == 'gap':
        f['step_id'] = 3
        report = runtime.update(f, value, now=102.1)
        assert report['gap_reset'] and report['accumulation']['observations'] == 1
        assert report['accumulation']['scores']['collision'] == pytest.approx(.79)
    else:
        if fault == 'duplicate': f['observation_id'] = 'frame-1'
        if fault == 'stale': f['captured_at'] = 1.
        if fault == 'generation': f['plan_revision'] = 1
        if fault == 'invalid': value['collision'] = float('nan')
        with pytest.raises(ValueError):runtime.update(f, value, now=202.1)
        assert runtime.evidence['collision'] == 0 and runtime.observations == 0


def test_persistent_uncertainty_calls_astra_and_continues_same_episode(tmp_path):
    runner, rollout, requests, windows, events = accumulated_setup(tmp_path,
        monitor_fn=lambda frames: scores(grasp_attempted=.4, grasp_failure=.3),
        planner_fn=student_review)
    result = runner.run()
    assert result['state'] == 'COMPLETED' and result['native_result']['success']
    assert [r['current_scene']['step_id'] for r in requests] == [3, 6]
    assert all(r['review_reason'] == 'persistent_uncertainty' for r in requests)
    assert all(r['review_kind'] == 'progress' and not r['failure']['violations'] for r in requests)
    assert all(r['accumulated_evidence']['uncertain_observations'] == 3 for r in requests)
    assert rollout.sim.calls.count('reset') == 1
    assert rollout.sim.calls.count('refresh_observation') == 2  # Only at Astra reviews.
    assert len(windows[0]) == 2 and windows[0][0]['context_only']
    assert all(r['failure']['accumulation']['observations'] == 3 for r in requests)
    assert not any(e.get('state') == 'PAUSED_UNCERTAIN' for e in events)
    assert any(e['kind'] == 'recovery_observed_uncertain' for e in events)


def test_accumulated_risk_requests_review_before_continuing(tmp_path):
    runner, rollout, requests, _, _ = accumulated_setup(tmp_path,
        config=AccumulationConfig(uncertainty_observations=100),
        monitor_fn=lambda frames: scores(target_mismatch=.65 if frames[-1]['plan_revision'] == 0 else .05),
        planner_fn=student_review)
    result = runner.run()
    assert result['state'] == 'COMPLETED' and rollout.tick == LongSim.horizon
    assert len(requests) == 1 and requests[0]['current_scene']['step_id'] == 4
    assert requests[0]['review_reason'] == 'accumulated_risk'


def test_native_terminal_endpoint_never_launches_an_extra_review(tmp_path):
    runner, _, requests, _, _ = accumulated_setup(tmp_path,
        config=AccumulationConfig(uncertainty_observations=LongSim.horizon),
        monitor_fn=lambda frames: scores(grasp_attempted=.4, grasp_failure=.3),
        planner_fn=student_review)
    assert runner.run()['state'] == 'COMPLETED'
    assert not requests


@pytest.mark.parametrize('changes', [dict(evidence_decay=1.), dict(evidence_baseline=-.1),
    dict(evidence_threshold=0), dict(evidence_threshold=float('nan')),
    dict(min_observations=True), dict(uncertainty_observations=1)])
def test_invalid_accumulation_configuration_is_rejected(changes):
    with pytest.raises(ValueError): replace(AccumulationConfig(), **changes)


@pytest.mark.parametrize('changed', [False, True])
def test_slow_planner_requires_fresh_identical_scene(tmp_path, changed):
    runner, rollout, requests, _, events = accumulated_setup(tmp_path,
        monitor_fn=lambda frames: scores(grasp_attempted=.4, grasp_failure=.3),
        planner_fn=student_review)
    validate = runner.validate_recovery_scene
    def expired_scene(scene):
        from copy import deepcopy
        old = deepcopy(scene)
        old['captured_at'] -= 2 * runner.monitor.config.max_age_s
        if changed:
            old['current_state'][0] += .01
        return validate(old)
    runner.validate_recovery_scene = expired_scene
    result = runner.run()
    if changed:
        assert result['state'] == 'STOPPED' and rollout.tick == 3
        assert 'scene changed' in result['reason']
        assert not result['attempts']
    else:
        assert result['state'] == 'COMPLETED' and rollout.tick == LongSim.horizon
        assert len([e for e in events if e['kind'] == 'planner_scene_revalidated']) == 2
    assert rollout.sim.calls.count('reset') == 1
