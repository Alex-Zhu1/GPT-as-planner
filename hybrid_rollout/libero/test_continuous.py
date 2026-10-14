from copy import deepcopy
from threading import Event
import time
import pytest

from .continuous import ContinuousScheduler
from .test_integration import Environment, Student, scores
from .session import LiberoSession
from .rollout import LiberoRollout
from ..robodojo.jev.monitor import AsyncMonitor, MonitorConfig, IDENTITY


def make_runner(tmp_path, env, monitor_fn, *, astra=False, horizon=40, observation_profile='constraints', **kwargs):
    student = Student()
    rollout = LiberoRollout(tmp_path, student,
        lambda: LiberoSession(env, [0], 'put the bowl on the plate', horizon), max_decisions=0, chunk_steps=5)
    calls, windows, events = [], [], []
    class Monitor:
        def evaluate(self, frames):
            windows.append(deepcopy(frames))
            return monitor_fn(frames)
    class Planner:
        def replan(self, request):
            paused = rollout.tick
            calls.append(deepcopy(request))
            time.sleep(.01)
            assert rollout.tick == paused
            return {**{k:request[k] for k in ('request_id', *IDENTITY)}, 'decision':dict(
                request_id=request['native_request']['request_id'], mode='replan', reason='Fixture recovery',
                subgoals=[dict(subgoal_id='recovery-1', instruction='Open the gripper.',
                              success_condition='Fingers are open.')])}
    runner = ContinuousScheduler(rollout, AsyncMonitor(Monitor(), MonitorConfig(history=6,max_step_gap=5,
        observation_profile=observation_profile), 3),
        Planner(), astra_enabled=astra, chunk_steps=5, include_initial_context=True,
        uncertainty_policy='continue', emit=events.append, **kwargs)
    return runner, rollout, student, calls, windows, events


@pytest.mark.parametrize('profile', ['constraints','task_progress'])
def test_slow_valen_does_not_block_execution_and_queue_coalesces(tmp_path, profile):
    advanced = Event()
    class MovingEnvironment(Environment):
        def step(self, action):
            result = super().step(action)
            if self.steps >= 25:  # 10 initialization controls + 15 actual controls
                advanced.set()
            return result
    def monitor(frames):
        if frames[-1]['step_id'] == 5:
            assert advanced.wait(2), 'Pi waited for the pending detector instead of executing'
        from .test_task_context import observed
        return observed(task_advancing=.95) if profile=='task_progress' else scores(stuck=.5)
    runner, _, _, calls, _, events = make_runner(tmp_path, MovingEnvironment(999), monitor,
        observation_profile=profile)
    result = runner.run()
    assert result['native_result']['complete'] and result['step_id']==40
    assert not calls
    reports = [e for e in events if e['kind']=='monitor_report']
    first = next(e for e in reports if e['report']['step_id']==5)
    assert first['step_id'] >= 15 and first['observation_lag_steps'] >= 10
    assert result['coalesced_observations'] > 0


def test_running_reports_do_not_cancel_subgoal_timeout_and_teacher_receives_current_budget(tmp_path):
    runner, rollout, student, calls, _, events = make_runner(tmp_path, Environment(999),
        lambda frames:scores(), astra=True, progress_budget=10, subgoal_budget=10)
    result=runner.run()
    assert result['native_result']['complete'] and not result['native_result']['success']
    assert [c['current_scene']['step_id'] for c in calls]==[10,20,30]
    assert [c['execution_budget']['remaining_controls'] for c in calls]==[30,20,10]
    assert calls[1]['failure']['trigger']=='subgoal_execution_budget'
    assert all(c['asynchronous_execution']['controls_after_trigger']==0 for c in calls)
    assert all(c['current_scene']['observation_id'] != c['confirmation_reports'][-1]['observation_id']
               for c in calls if c['confirmation_reports'])
    assert any('put the bowl on the plate' in p and 'Open the gripper' in p for p in student.instructions)
    assert not any(e['kind']=='subgoal_completed' for e in events)


def test_late_completion_requires_fresh_current_scene_verification(tmp_path):
    def monitor(frames):
        f=frames[-1]
        refreshed=any('/jev_refresh/' in i['path'] for i in f['images'])
        return scores(subgoal_complete=.05 if refreshed else .95)
    runner, _, _, calls, _, events = make_runner(tmp_path, Environment(999), monitor,
        astra=True, horizon=55, progress_budget=10, subgoal_budget=30)
    result=runner.run()
    assert result['native_result']['complete']
    checks=[e for e in events if e['kind']=='current_completion_verification']
    assert checks and all(not e['passed'] for e in checks)
    assert not any(e['kind']=='subgoal_completed' for e in events)
    assert calls[1]['failure']['trigger']=='subgoal_execution_budget'


def test_shadow_comparator_uses_full_budget_despite_confirmed_failure(tmp_path):
    runner, _, _, calls, _, events = make_runner(tmp_path, Environment(999),
        lambda frames:scores(stuck=.95), shadow_only=True, horizon=50)
    result=runner.run()
    assert result['step_id']==50 and result['native_result']['complete']
    assert not calls
    assert any(e['kind']=='shadow_failure_logged' for e in events)
