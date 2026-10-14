"""Physical-time evidence, chunk bounds and delayed advisory contracts."""
from copy import deepcopy
from threading import Event
import time

import pytest

from .jev.temporal import TemporalConfig, TemporalRuntime, TemporalScheduler
from .jev.monitor import AsyncMonitor, IDENTITY
from .jev.compact_planner import compact_packet
from .robodojo_server.client import RoboDojoTools
from .test_contract import Student, response
from .test_jev import LongSim, frame, scores


def update(runtime, step, span, **values):
    return runtime.update(frame(step, runtime.config, executed_start_step=step-span),
                          scores(**values), now=100.1+step)


@pytest.mark.parametrize('chunk', [1, 15, 25])
def test_evidence_depends_on_simulated_time_not_chunk_count(chunk):
    runtime = TemporalRuntime(TemporalConfig(max_gap_s=1000))
    first = None
    for step in range(chunk, 301, chunk):
        report = update(runtime, step, chunk, target_mismatch=.7)
        if report['review_requested'] and first is None:first = step*.04
    # Exact exponential integration gives the same final evidence at 12 seconds.
    expected = (.7-.5) * (1-.85**12) / -__import__('math').log(.85)
    assert report['accumulation']['scores']['target_mismatch'] == pytest.approx(expected)
    assert 4 <= first <= 5
    assert not report['violations']


def test_uncertainty_waits_six_seconds_for_the_same_constraint():
    runtime = TemporalRuntime(TemporalConfig())
    for step in range(25, 151, 25):
        report = update(runtime, step, 25, grasp_attempted=.4)
        assert report['review_requested'] == (step == 150)
    assert report['review_reason'] == 'persistent_uncertainty'
    runtime = TemporalRuntime(TemporalConfig())
    for step in range(25, 301, 25):
        key = 'grasp_attempted' if (step//25)%2 else 'progress_attempted'
        assert not update(runtime, step, 25, **{key:.4})['review_requested']


def test_acute_collision_bypasses_minimum_window_and_static_views_do_not_vote():
    runtime = TemporalRuntime(TemporalConfig())
    report = update(runtime, 25, 25, collision=.95)
    assert report['interrupt_requested'] and report['review_reason'] == 'acute_collision'
    runtime.acknowledge_review()
    report = runtime.update(frame(50, runtime.config, executed_start_step=25, inspection_only=True),
                            scores(collision=.99), now=150.1)
    assert not report['review_requested'] and report['accumulation']['evidence_seconds']==0


def make_runner(tmp_path, *, chunk=25, horizon=250, planner_fn=None, monitor_fn=None, max_async=6):
    class Sim(LongSim):pass
    Sim.horizon = horizon
    rollout = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=Sim,
                           max_decisions=0, student_step_limit=chunk)
    windows, requests, events = [], [], []
    class Monitor:
        def evaluate(self, frames):
            windows.append(deepcopy(frames))
            return monitor_fn(frames) if monitor_fn else scores(grasp_attempted=.4)
    class Planner:
        closed = False
        def replan(self, request):
            requests.append(deepcopy(request))
            decision = planner_fn(request, rollout) if planner_fn else dict(mode='continue', reason='Advancing.')
            return dict(**{k:request[k] for k in ('request_id',*IDENTITY)}, decision=decision)
        def close(self):self.closed=True
    monitor = AsyncMonitor(Monitor(), TemporalConfig(), 2, runtime_factory=TemporalRuntime)
    runner = TemporalScheduler(rollout, monitor, Planner(), chunk_steps=chunk, max_replans=0,
                               planner_timeout=3, max_async_seconds=max_async, emit=events.append)
    return runner, rollout, requests, windows, events


@pytest.mark.parametrize('chunk', [15, 25])
def test_full_student_chunks_and_long_visual_window(tmp_path, chunk):
    runner, rollout, requests, windows, events = make_runner(tmp_path, chunk=chunk)
    result = runner.run()
    assert result['state']=='COMPLETED', result['reason']
    assert rollout.tick==250
    assert all(h['end_tick']-h['start_tick']==chunk for h in rollout.history[:-1])
    assert windows[0][0]['context_only'] and max(map(len, windows))==6
    assert requests[0]['current_scene']['step_id'] >= 150
    assert rollout.sim.calls.count('reset')==1
    assert not any(e.get('state')=='PAUSED_UNCERTAIN' for e in events)


def test_delayed_astra_runs_while_robot_advances_and_fresh_recovery_is_required(tmp_path):
    advanced = Event()
    def monitor(frames):
        if frames[-1]['step_id']>=200:advanced.set()
        return scores(grasp_attempted=.4)
    def planner(request, rollout):
        if request['review_kind']=='background':
            assert advanced.wait(2), 'Robot froze while advisory was running'
            return dict(mode='request_recovery',reason='Inspect current grasp.')
        assert request['current_scene']['step_id']>=200
        assert request['native_request']['step_id']==request['current_scene']['step_id']
        value = response(request['native_request']);value['steps']=25
        value['assessment']['execution_status']='uncertain'
        return value
    runner, rollout, requests, windows, events = make_runner(tmp_path, horizon=400,
        planner_fn=planner, monitor_fn=monitor, max_async=2)
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert len(requests)>=2 and requests[0]['current_scene']['step_id']==150
    assert requests[1]['current_scene']['step_id']>=200
    assert any(e['kind']=='background_review_response' and e['moved_steps']>=50 for e in events)
    assert rollout.sim.calls.count('reset')==1


def test_pending_advice_is_closed_at_native_terminal(tmp_path):
    release=Event()
    def planner(request, rollout):
        release.wait(2)
        return dict(mode='request_recovery',reason='Historical answer.')
    runner, rollout, requests, _, events=make_runner(tmp_path,horizon=175,planner_fn=planner)
    try:
        assert runner.run()['state']=='COMPLETED'
        assert runner.planner.closed and rollout.tick==175
        assert any(e['kind']=='background_review_discarded' for e in events)
        assert not runner.attempts
    finally:release.set()


def test_background_rejects_action_payload(tmp_path):
    def planner(request, rollout):return dict(mode='eef',reason='Stale action.')
    runner, rollout, _, _, _=make_runner(tmp_path,horizon=400,planner_fn=planner)
    result=runner.run()
    assert result['state']=='STOPPED' and 'no stale actions' in result['reason']
    assert not runner.attempts and rollout.tick < 400


def test_compact_input_is_bounded_and_excludes_source_artifacts():
    frames=[dict(step_id=i, observation_id=str(i), images=[dict(camera=c,path=f'/synthetic/{i}-{c}.png')
            for c in ('front','left','right')], executed_actions=['unused']) for i in range(20)]
    packet,images=compact_packet(dict(current_scene=frames[-1],evidence_frames=frames,review_kind='background',
        original_task='Arrange.',current_subgoal='Arrange.',secret_debug='must not enter prompt'))
    assert len(images)==9 and len(packet['scenes'])==3
    assert [s['step_id'] for s in packet['scenes']]==[0,10,19]
    assert 'secret_debug' not in packet and 'executed_actions' not in str(packet)


def test_student_limit_is_host_owned_and_corrections_remain_bounded():
    from .robodojo_server.validation import validate_response
    request=dict(request_id='r')
    decision=dict(request_id='r', mode='student', steps=25, reason='Continue.', student_step_limit=25)
    with pytest.raises(ValueError, match='1..15'):validate_response(decision,request)
    validate_response(decision, dict(request, student_step_limit=25))
    for invalid in (True, 0, 26):
        with pytest.raises(ValueError):validate_response(decision, dict(request, student_step_limit=invalid))
    decision.update(mode='eef', steps=6)
    with pytest.raises(ValueError,match='1..5'):validate_response(decision, dict(request, student_step_limit=25))


def test_async_limit_waits_without_executing_another_chunk(tmp_path):
    reached=Event()
    def monitor(frames):
        if frames[-1]['step_id']==200:reached.set()
        return scores(grasp_attempted=.4)
    def planner(request, rollout):
        assert reached.wait(2)
        time.sleep(.05)
        assert rollout.tick==200
        return dict(mode='continue',reason='Continue after bounded wait.')
    runner, rollout, _, _, events=make_runner(tmp_path,horizon=250,max_async=2,
        planner_fn=planner,monitor_fn=monitor)
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert any(e.get('state')=='WAITING_FOR_BACKGROUND' for e in events)
    assert runner.background_attempts[0]['received_step']==200


def test_acute_risk_ignores_review_cooldown(tmp_path):
    def planner(request, rollout):
        value=response(request['native_request']);value['steps']=1
        value['assessment']['execution_status']='uncertain'
        return value
    def monitor(frames):return scores(collision=.99 if frames[-1]['step_id']==25 else .05)
    runner, rollout, requests, _, _=make_runner(tmp_path,horizon=100,planner_fn=planner,monitor_fn=monitor)
    runner.next_review_step=10000
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert requests[0]['current_scene']['step_id']==25
    assert requests[0]['review_kind']=='progress'


def test_temporal_rejects_wrong_simulator_control_time_before_action(tmp_path):
    runner, rollout, _, _, _=make_runner(tmp_path)
    class WrongTimeSim(rollout.rpc_factory):
        def request(self, op, **args):
            result=super().request(op, **args)
            if op=='metadata':result['control_dt']=.02
            return result
    rollout.rpc_factory=WrongTimeSim
    result=runner.run()
    assert result['state']=='STOPPED' and 'control_dt' in result['reason']
    assert rollout.tick==0


def test_closed_simulator_does_not_mask_original_failure(tmp_path):
    runner,rollout,_,_,_=make_runner(tmp_path)
    class BrokenSim(rollout.rpc_factory):
        def request(self,op,**args):
            if op=='chunk_step':raise RuntimeError('Original simulator failure')
            if op=='finish_pilot':raise RuntimeError('RPC already closed')
            return super().request(op,**args)
    rollout.rpc_factory=BrokenSim
    result=runner.run()
    assert result['state']=='STOPPED' and 'Original simulator failure' in result['reason']
    assert result['native_result']['complete'] is False
    assert 'RPC already closed' in result['native_result']['finish_error']
    assert (tmp_path/'jev_summary.json').is_file()
