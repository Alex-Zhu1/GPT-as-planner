"""The fixed-time control must not silently become a progress trigger."""
import pytest

from .max_timing_ablation import FixedStepWatchdog


def report(step, score=0, **fields):
    return dict(step_id=step, scores=dict(progress_attempted=score, stuck=score), **fields)


@pytest.mark.parametrize('score', [0, .5, 1])
def test_fixed_trigger_ignores_scores_and_fires_once(score):
    guard = FixedStepWatchdog()
    for step in range(5, 85, 5):
        assert guard.update(report(step, score)) is None
    event = guard.update(report(85, score))
    assert event['last_step'] == 85
    assert event['trigger'] == 'fixed_step_85'
    assert event['violations'] == [] and event['confirmed_failure'] is False
    assert guard.update(report(85, score)) is None
    assert guard.update(report(90, score, plan_revision=1)) is None


def test_inspection_is_not_a_control_boundary():
    guard = FixedStepWatchdog()
    assert guard.update(report(85, inspection_only=True)) is None
    assert guard.update(report(85)) is not None


def test_missed_exact_boundary_is_an_error():
    with pytest.raises(RuntimeError, match='exact control boundary'):
        FixedStepWatchdog().update(report(90))
