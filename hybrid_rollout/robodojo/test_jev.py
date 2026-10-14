"""CPU-only asynchronous monitoring/recovery contracts; no model or physics."""
from copy import deepcopy
import json
from threading import Event, get_ident
import time

import numpy as np
import pytest

from .jev.monitor import (AsyncMonitor, ConstraintRuntime, IDENTITY, MonitorConfig,
                          QUESTIONS, ValenMonitor)
from .jev.planner import CodexPlanner, DecisionOnlyRollout
from .jev.runner import ReplanScheduler
from .robodojo_server.client import RoboDojoTools
from .test_contract import Sim, Student, response


def scores(**overrides):
    return dict({key: .95 if key.endswith('attempted') else .05 for key in QUESTIONS}, **overrides)


def frame(step, config=None, **kwargs):
    config = config or MonitorConfig()
    return dict(run_id='r', episode_id='e', plan_revision=0, subgoal_id='g', program_id=config.program_id,
                step_id=step, captured_at=100. + step, observation_id=f'frame-{step}', images=[], **kwargs)


def test_confirmation_requires_same_constraint_and_applicable_phase():
    runtime = ConstraintRuntime(MonitorConfig())
    for step in (1, 2):
        result = runtime.update(frame(step), scores(grasp_attempted=.05, grasp_failure=.95), now=100.1+step)
        assert not result['violations']
    assert runtime.update(frame(3), scores(grasp_failure=.95), now=103.1)['state'] == 'SUSPECTED'
    assert not runtime.update(frame(4), scores(collision=.95), now=104.1)['violations']
    assert not runtime.update(frame(5), scores(grasp_failure=.95), now=105.1)['violations']
    result = runtime.update(frame(6), scores(grasp_failure=.95), now=106.1)
    assert [v['constraint_id'] for v in result['violations']] == ['grasp_failure']


@pytest.mark.parametrize('interrupt', ['unknown', 'guard_unknown', 'replay', 'old', 'invalid', 'gap'])
def test_invalid_or_unknown_evidence_breaks_confirmation(interrupt):
    runtime = ConstraintRuntime(MonitorConfig())
    runtime.update(frame(1), scores(grasp_failure=.95), now=101.1)
    middle = frame(2)
    values = scores(grasp_failure=.95)
    if interrupt == 'unknown':
        values['grasp_failure'] = .5
    elif interrupt == 'guard_unknown':
        values['grasp_attempted'] = .5
    elif interrupt == 'replay':
        middle['observation_id'] = 'frame-1'
    elif interrupt == 'old':
        middle['captured_at'] = 101.
    elif interrupt == 'invalid':
        values['grasp_failure'] = float('nan')
    elif interrupt == 'gap':
        middle['step_id'] = 3
    if interrupt in ('replay', 'old', 'invalid'):
        with pytest.raises(ValueError):
            runtime.update(middle, values, now=102.1)
    else:
        result = runtime.update(middle, values, now=102.1)
        assert not result['violations']
    # Skip another endpoint after the gap case so neither sample gets two votes.
    step = 5 if interrupt == 'gap' else 3
    assert not runtime.update(frame(step), scores(grasp_failure=.95), now=100.1+step)['violations']


def test_stuck_needs_three_samples_and_generation_cannot_mix():
    runtime = ConstraintRuntime(MonitorConfig())
    for step in (1, 2):
        assert not runtime.update(frame(step), scores(stuck=.95), now=100.1+step)['violations']
    assert runtime.update(frame(3), scores(stuck=.95), now=103.1)['violations']
    stale = frame(4)
    stale['plan_revision'] = 1
    with pytest.raises(ValueError, match='generation'):
        runtime.update(stale, scores(stuck=.95), now=104.1)


class LongSim(Sim):
    horizon = 7
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = get_ident()

    def request(self, op, **args):
        assert get_ident() == self.owner, 'Simulator RPC escaped its owner'
        if op == 'refresh_observation':
            self.calls.append(op)
            return self.request('teacher_observation')
        if op == 'chunk_step':
            self.calls.append(op)
            rows = []
            for action in args['actions']:
                self.tick += 1
                rows.append(dict(valid=True, executed_action=action, obs=dict(states=np.zeros(14)),
                    terminated=self.tick >= self.horizon, truncated=False, success=self.tick >= self.horizon))
                if self.tick >= self.horizon:
                    break
            return dict(step_id=self.tick, steps=rows)
        if op == 'finish_pilot':
            self.calls.append(op)
            return dict(step_id=self.tick, success=self.tick >= self.horizon,
                        terminated=self.tick >= self.horizon, truncated=False)
        result = super().request(op, **args)
        if op == 'metadata':
            result['max_episode_steps'] = self.horizon
        if op == 'teacher_observation':
            result['remaining_steps'] = self.horizon-self.tick
            result['captured_at'] = time.time()
        return result


def setup(tmp_path, monitor_fn=None, planner_fn=None, student=None, **kwargs):
    rollout = RoboDojoTools(tmp_path, 'arrange_largest_number', student or Student(),
                           rpc_factory=LongSim, max_decisions=0)
    requests, windows, events = [], [], []
    class Monitor:
        def evaluate(self, frames):
            windows.append(deepcopy(frames))
            return monitor_fn(frames) if monitor_fn else scores(grasp_failure=.95 if frames[-1]['plan_revision'] == 0 else .05)
    class Planner:
        def replan(self, request):
            requests.append(request)
            before = rollout.tick
            if planner_fn:
                result = planner_fn(request)
            else:
                decision = response(request['native_request'], 'eef')
                decision['steps'] = 1
                decision['assessment']['execution_status'] = 'failed'
                result = dict(**{key: request[key] for key in ('request_id', *IDENTITY)}, decision=decision)
            assert rollout.tick == before
            return result
    config = kwargs.pop('config', MonitorConfig())
    observer = AsyncMonitor(Monitor(), config, kwargs.pop('monitor_timeout', 2.))
    runner = ReplanScheduler(rollout, observer, Planner(), emit=events.append, **kwargs)
    return runner, rollout, requests, windows, events


def test_same_scene_recovery_uses_native_eef_and_returns_to_student(tmp_path):
    runner, rollout, requests, windows, events = setup(tmp_path)
    result = runner.run()
    assert result['state'] == 'COMPLETED'
    assert result['replan_count'] == 1 and result['plan_revision'] == 1
    assert rollout.sim.calls.count('reset') == 1
    assert rollout.sim.calls.count('refresh_observation') == 1
    assert rollout.counters['recovery_steps'] == 1
    assert [h['source_detail'] for h in rollout.history] == ['student', 'student', 'eef'] + ['student']*4
    req = requests[0]
    assert req['current_scene']['step_id'] == req['failure']['step_id'] == 2
    assert req['current_scene']['observation_id'] != req['failure']['observation_id']
    assert len(req['confirmation_reports']) == 2
    assert req['original_task'] == req['remaining_task']
    assert all(f['plan_revision'] == w[-1]['plan_revision'] for w in windows for f in w)
    assert all('native_success' not in json.dumps(w) and 'previous_result' not in json.dumps(w) for w in windows)
    assert any(e.get('state') == 'SUSPECTED' for e in events)
    # The speculative step-2 inference never gets a native request or action.
    assert len(list((tmp_path/'jev_inference').glob('*/actions.npz'))) == len(rollout.history)+1
    assert len(list(tmp_path.glob('request_*.json'))) == len(rollout.history)


def test_monitor_and_student_overlap_without_execution_leading(tmp_path):
    running, reviewed = Event(), Event()
    class SlowStudent(Student):
        count = 0
        def infer(self, obs, output):
            self.count += 1
            if self.count == 2:
                running.set()
                assert reviewed.wait(2)
            return super().infer(obs, output)
    def monitor(frames):
        if frames[-1]['step_id'] == 1:
            assert running.wait(2)
            assert rollout.tick == 1
            reviewed.set()
        return scores()
    runner, rollout, requests, _, _ = setup(tmp_path, monitor_fn=monitor, student=SlowStudent())
    assert runner.run()['state'] == 'COMPLETED'
    assert running.is_set() and not requests


def test_unknown_does_not_request_planner(tmp_path):
    runner, rollout, requests, _, _ = setup(tmp_path, monitor_fn=lambda fs: scores(grasp_failure=.95 if fs[-1]['step_id'] % 2 else .5))
    assert runner.run()['state'] == 'PAUSED_UNCERTAIN'
    assert rollout.tick == 2 and rollout.sim.calls.count('refresh_observation') == 1
    assert not requests


def test_unknown_continues_to_native_success_without_discard_or_teacher(tmp_path):
    runner, rollout, requests, windows, events = setup(tmp_path,
        monitor_fn=lambda fs: scores(grasp_failure=.5), uncertainty_policy='continue')
    result = runner.run()
    assert result['native_result']['success'] and rollout.tick == LongSim.horizon
    assert result['uncertain_continuation_count'] == LongSim.horizon - 1
    assert not requests and not rollout.sim.calls.count('refresh_observation')
    assert not any(e['kind'] == 'policy_discarded' for e in events)
    assert len(windows) == LongSim.horizon
    assert all(e['report']['state'] == 'UNCERTAIN' for e in events if e['kind'] == 'monitor_report')


def test_progress_review_calls_planner_without_fabricating_failure(tmp_path):
    def planner(request):
        decision = response(request['native_request'], 'student')
        decision['steps'] = 1
        decision['assessment']['execution_status'] = 'uncertain'
        return dict(**{k:request[k] for k in ('request_id', *IDENTITY)}, decision=decision)
    runner, rollout, requests, _, events = setup(tmp_path,
        monitor_fn=lambda fs: scores(grasp_failure=.5), planner_fn=planner,
        uncertainty_policy='continue')
    runner.progress_review = lambda report: (dict(state='PROGRESS_REVIEW', violations=[],
        first_step=1, confirmed_failure=False) if report['step_id']==2 else None)
    result = runner.run()
    assert result['native_result']['success'] and len(requests)==1
    assert requests[0]['review_kind']=='progress' and not requests[0]['failure']['violations']
    assert result['attempts'][0]['review_kind']=='progress'
    assert result['attempts'][0]['result'] != 'not_executed'
    assert any(e['kind']=='recovery_observed_uncertain' for e in events)
    assert not any(e['kind']=='recovery_verified' for e in events)


def test_continue_unknown_still_recovers_confirmed_failure(tmp_path):
    def monitor(fs):
        current = fs[-1]
        return scores(grasp_failure=(.5 if current['step_id'] == 1 else .95)
                      if current['plan_revision'] == 0 else .05)
    runner, rollout, requests, _, events = setup(tmp_path, monitor_fn=monitor,
                                               uncertainty_policy='continue')
    result = runner.run()
    assert result['native_result']['success'] and len(requests) == 1
    assert requests[0]['failure']['violations']
    assert result['uncertain_continuation_count'] == 1
    assert any(e['kind'] == 'recovery_verified' for e in events)


@pytest.mark.parametrize('fault', ['exception', 'stale', 'timeout'])
@pytest.mark.parametrize('uncertainty_policy', ['pause', 'continue'])
def test_monitor_failure_stops_without_planner(tmp_path, fault, uncertainty_policy):
    release = Event()
    def monitor(frames):
        if fault == 'exception':
            raise RuntimeError('unavailable')
        if fault == 'timeout':
            release.wait(2)
        elif fault == 'stale':
            # Expire the monitor sample, not the pre-action policy proposal.
            # A 20 ms wall-clock budget also expired unrelated setup on busy hosts.
            frames[-1]['captured_at'] -= 2 * config.max_age_s
        return scores(grasp_failure=.95)
    config = MonitorConfig()
    runner, rollout, requests, _, _ = setup(tmp_path, monitor_fn=monitor, config=config,
                                           monitor_timeout=.01 if fault == 'timeout' else 2.,
                                           uncertainty_policy=uncertainty_policy)
    try:
        result = runner.run()
        assert result['state'] == 'PAUSED_MONITOR_UNAVAILABLE'
        assert rollout.tick == 1 and not requests
    finally:
        release.set()
        if runner.monitor.pending:
            runner.monitor.pending.thread.join(2)


@pytest.mark.parametrize('fault', ['wrong_id', 'wrong_revision', 'bounds', 'too_long', 'malformed'])
def test_invalid_planner_cannot_execute_or_resume(tmp_path, fault):
    def planner(req):
        decision = response(req['native_request'], 'eef')
        decision['steps'] = 1
        decision['assessment']['execution_status'] = 'failed'
        raw = dict(**{key: req[key] for key in ('request_id', *IDENTITY)}, decision=decision)
        if fault == 'wrong_id':
            raw['request_id'] = 'old'
        elif fault == 'wrong_revision':
            raw['plan_revision'] = True
        elif fault == 'bounds':
            decision['target']['right']['position'][0] = .06
        elif fault == 'too_long':
            decision['steps'] = 2
        elif fault == 'malformed':
            raw['code'] = 'not executed'
        return raw
    runner, rollout, requests, _, _ = setup(tmp_path, planner_fn=planner)
    assert runner.run()['state'] == 'STOPPED'
    assert rollout.tick == 2 and len(requests) == 1
    assert rollout.counters['recovery_steps'] == 0
    assert rollout.sim.calls.count('reset') == 1


def test_planner_timeout_cannot_install_late_result(tmp_path):
    release, finished = Event(), Event()
    def planner(req):
        release.wait(2)
        finished.set()
        return {}
    runner, rollout, requests, _, events = setup(tmp_path, planner_fn=planner, planner_timeout=.01)
    runner.planner.diagnostics = lambda: dict(request_id='a'*32, stage='waiting_gateway_headers')
    try:
        result = runner.run()
        assert result['state'] == 'STOPPED' and rollout.tick == 2
        timeout = next(e for e in events if e['kind'] == 'planner_timeout')
        assert timeout['worker_alive'] and timeout['timeout_seconds'] == .01
        assert timeout['transport'] == runner.planner.diagnostics()
    finally:
        release.set()
        assert finished.wait(2)
    assert len(requests) == 1 and runner.revision == 0


def test_replan_budget_and_confirmation_reset(tmp_path):
    runner, rollout, requests, _, _ = setup(tmp_path,
        monitor_fn=lambda fs: scores(grasp_failure=.05 if fs[-1]['step_id'] == 3 else .95), max_replans=1)
    result = runner.run()
    assert result['state'] == 'STOPPED' and result['reason'] == 'Recovery budget exhausted'
    assert rollout.tick == 5 and len(requests) == 1
    if runner.inference:
        runner.inference.call.thread.join(2)


def test_pending_timeout_cannot_mutate_scheduler(tmp_path):
    release, finished = Event(), Event()
    class SlowStudent(Student):
        def infer(self, obs, output):
            release.wait(2)
            result = super().infer(obs, output)
            finished.set()
            return result
    runner, rollout, requests, _, _ = setup(tmp_path, student=SlowStudent(), policy_timeout=.01)
    try:
        assert runner.run()['state'] == 'PAUSED_POLICY_UNAVAILABLE'
        assert rollout.tick == 0 and not requests
    finally:
        release.set()
        assert finished.wait(2)
    assert rollout.tick == 0 and not list(tmp_path.glob('request_*.json'))


def test_decision_only_planner_has_no_actuation_handle(tmp_path):
    runner, _, requests, _, _ = setup(tmp_path)
    runner.run()
    proxy = DecisionOnlyRollout(requests[0], tmp_path)
    with pytest.raises(ValueError, match='Reset'):
        proxy.start()
    decision = response(proxy.request, 'eef')
    decision['steps'] = 1
    decision['assessment']['execution_status'] = 'failed'
    args = proxy.next_call()
    args.pop('tool')
    args['response'] = decision
    result = proxy.execute(**args)
    assert result['simulation_executed'] is False and proxy.phase == 'done'
    assert not hasattr(proxy, 'sim') and not hasattr(proxy, 'student')


def test_valen_request_has_all_three_views_and_no_native_outcome():
    sent = []
    monitor = ValenMonitor('http://localhost:8001/predict', 2)
    def post(payload):
        sent.append(payload)
        return {'answers': {key: {'type': 'noul', 'noul': value} for key, value in scores().items()}}
    monitor.post = post
    item = frame(1)
    item['images'] = [dict(camera=c, path=f'/shared/{c}.png') for c in RoboDojoTools.camera_keys]
    assert monitor.evaluate([item]) == scores()
    content = sent[0]['state']['messages'][0]['content']
    assert len([c for c in content if c['type'] == 'image_url']) == 3
    assert set(sent[0]['questions']) == set(QUESTIONS)


def test_no_reset_when_configuration_is_invalid(tmp_path):
    with pytest.raises(ValueError):
        setup(tmp_path, chunk_steps=2)
    assert not (tmp_path/'run.json').exists()


def test_monitor_timeout_does_not_wait_for_blocked_policy(tmp_path):
    release, running = Event(), Event()
    class BlockedStudent(Student):
        calls = 0
        def infer(self, obs, output):
            self.calls += 1
            if self.calls == 2:
                running.set()
                release.wait(2)
            return super().infer(obs, output)
    def monitor(frames):
        assert running.wait(2)
        release.wait(2)
        return scores(grasp_failure=.95)
    runner, rollout, requests, _, _ = setup(tmp_path, student=BlockedStudent(),
                                           monitor_fn=monitor, monitor_timeout=.05)
    try:
        result = runner.run()
        assert result['state'] == 'PAUSED_MONITOR_UNAVAILABLE'
        assert result['policy_worker_pending'] and rollout.tick == 1
        assert not requests
    finally:
        release.set()
        runner.inference.call.thread.join(2)
        runner.monitor.pending.thread.join(2)
    assert rollout.tick == 1


def test_native_refresh_does_not_reset_or_step_or_relabel_cached_time():
    from .robodojo_server.session import RoboDojoSession
    session = RoboDojoSession.__new__(RoboDojoSession)
    session.obs, session.observed_at = {'sample': 1}, 123.
    session.finished, session.step_id = False, 8
    session._check_identity = lambda episode, step: None
    calls = []
    def observe(record):
        calls.append(record)
        session.obs, session.observed_at = {'sample': 2}, 456.
    session._observe = observe
    args = dict(episode_id='episode', step_id=8)
    assert session.dispatch('teacher_observation', args)['captured_at'] == 123.
    assert session.dispatch('teacher_observation', args)['captured_at'] == 123.
    refreshed = session.dispatch('refresh_observation', args)
    assert refreshed == {'sample': 2, 'captured_at': 456.}
    assert calls == [False] and session.step_id == 8


def test_codex_backend_uses_decision_only_proxy_and_closes(tmp_path):
    runner, _, requests, _, _ = setup(tmp_path)
    runner.run()
    seen = []
    class FakeCodex:
        def __init__(self, directory, codex, timeout, recovery_schema):
            directory.mkdir()
            assert len(recovery_schema['anyOf']) == 2
        def run(self, proxy, *, initial_prompt):
            seen.append(initial_prompt)
            assert not hasattr(proxy, 'sim')
            args = proxy.next_call()
            args.pop('tool')
            decision = response(proxy.request)
            decision['steps'] = 1
            decision['assessment']['execution_status'] = 'failed'
            args['response'] = decision
            assert proxy.execute(**args)['simulation_executed'] is False
        def close(self):
            seen.append('closed')
    planner = CodexPlanner(tmp_path, '/fake/codex', policy_factory=FakeCodex)
    raw = planner.replan(requests[0])
    assert raw['request_id'] == requests[0]['request_id']
    assert raw['decision']['mode'] == 'student'
    assert seen[-1] == 'closed' and 'WITHOUT physics execution' in seen[0]


def test_multiple_violations_are_one_planner_event(tmp_path):
    def monitor(frames):
        return scores(grasp_failure=.95, collision=.95) if frames[-1]['plan_revision'] == 0 else scores()
    runner, _, requests, _, _ = setup(tmp_path, monitor_fn=monitor)
    assert runner.run()['state'] == 'COMPLETED'
    assert len(requests) == 1
    assert len(requests[0]['failure']['violations']) == 2


def test_terminal_endpoint_does_not_call_planner(tmp_path):
    def monitor(frames):
        return scores(grasp_failure=.95 if frames[-1]['step_id'] >= 6 else .05)
    runner, rollout, requests, _, _ = setup(tmp_path, monitor_fn=monitor)
    result = runner.run()
    assert result['state'] == 'COMPLETED' and result['native_result']['success']
    assert not requests and rollout.tick == 7


def test_global_decision_budget_does_not_claim_native_completion(tmp_path):
    runner, rollout, _, _, _ = setup(tmp_path, monitor_fn=lambda fs: scores())
    rollout.max_decisions = 2
    result = runner.run()
    assert result['state'] == 'STOPPED' and rollout.tick == 2
    assert not result['native_result']['complete']


def test_supplemental_clear_resumes_with_a_new_proposal(tmp_path):
    def monitor(frames):
        last = frames[-1]
        return scores(grasp_failure=.5 if last['step_id'] == 1 and not last.get('inspection_only') else .05)
    runner, rollout, requests, _, events = setup(tmp_path, monitor_fn=monitor)
    assert runner.run()['state'] == 'COMPLETED'
    assert not requests and rollout.sim.calls.count('reset') == 1
    supplemental = [e for e in events if e['kind'] == 'supplemental_observation']
    assert len(supplemental) == 1 and not supplemental[0]['adds_confirmation_vote']
    adopted = json.loads((tmp_path/'request_001.json').read_text())
    assert all('/jev_refresh/' in image['path'] for image in adopted['images'])
    assert len([e for e in events if e['kind'] == 'policy_discarded']) == 1


def test_repeated_paused_high_scores_cannot_confirm_a_failure(tmp_path):
    def monitor(frames):
        return scores(grasp_failure=.95 if frames[-1].get('inspection_only') else .5)
    runner, rollout, requests, _, events = setup(tmp_path, monitor_fn=monitor, max_observation_refreshes=3)
    assert runner.run()['state'] == 'PAUSED_UNCERTAIN'
    assert rollout.tick == 1 and not requests
    reports = [e['report'] for e in events if e['kind'] == 'monitor_report']
    assert len(reports) == 4 and not any(r['violations'] for r in reports)
    assert all(not r['subgoal_complete'] for r in reports)
    assert rollout.sim.calls.count('refresh_observation') == 3


@pytest.mark.parametrize('uncertainty_policy', ['pause', 'continue'])
def test_unverified_eef_correction_never_hands_back_to_vla(tmp_path, uncertainty_policy):
    runner, rollout, requests, _, events = setup(tmp_path, monitor_fn=lambda fs: scores(grasp_failure=.95),
                                               uncertainty_policy=uncertainty_policy)
    result = runner.run()
    assert result['state'] == 'PAUSED_RECOVERY_UNVERIFIED'
    assert rollout.tick == 3 and len(requests) == 1
    assert [r['source_detail'] for r in rollout.history] == ['student', 'student', 'eef']
    assert not any(e['kind'] == 'recovery_verified' for e in events)
    assert runner.inference is None


def test_vla_is_regenerated_only_after_recovery_verification(tmp_path):
    calls = []
    class TrackedStudent(Student):
        def infer(self, obs, output):
            calls.append(obs['instruction'])
            return super().infer(obs, output)
    def monitor(frames):
        last = frames[-1]
        if last['step_id'] == 3:
            # Two executed student proposals, one discarded speculation,
            # and one paused-scene proposal used by the Planner. No handback yet.
            assert len(calls) == 4
        return scores(grasp_failure=.95 if last['plan_revision'] == 0 else .05)
    runner, _, _, _, events = setup(tmp_path, student=TrackedStudent(), monitor_fn=monitor)
    result = runner.run()
    assert result['state'] == 'COMPLETED'
    assert any(e['kind'] == 'recovery_verified' for e in events)
    assert result['attempts'][0]['verification']['result'] == 'clear_monitor_checks_not_task_success'


def high_level_response(req):
    return dict(**{key: req[key] for key in ('request_id', *IDENTITY)}, decision=dict(
        request_id=req['native_request']['request_id'], mode='replan', reason='Retry before placing.',
        subgoals=[dict(subgoal_id='retry', instruction='Regrasp the digit.', success_condition='The digit is held.'),
                  dict(subgoal_id='place', instruction='Place the digit.', success_condition='The digit is at its destination.')]))


def test_high_level_replan_reaches_actual_policy_and_retains_original_task(tmp_path):
    calls = []
    class TrackedStudent(Student):
        def infer(self, obs, output):
            calls.append(obs['instruction'])
            return super().infer(obs, output)
    def monitor(frames):
        last = frames[-1]
        return scores(grasp_failure=.95 if last['plan_revision'] == 0 else .05,
                      subgoal_complete=.95 if last.get('success_condition') else .05)
    runner, rollout, requests, windows, events = setup(tmp_path, monitor_fn=monitor,
        planner_fn=high_level_response, student=TrackedStudent())
    result = runner.run()
    assert result['state'] == 'COMPLETED' and result['native_result']['success']
    assert result['plan_revision'] == 3 and len(requests) == 1
    assert rollout.sim.calls.count('reset') == 1
    assert 'Regrasp the digit.' in calls and 'Place the digit.' in calls
    assert calls[-1] == rollout.meta['instruction'] == 'Arrange the digits.'
    adopted = [json.loads((tmp_path/f'request_{i:03d}.json').read_text()) for i in range(7)]
    assert [r['student_instruction'] for r in adopted] == (
        ['Arrange the digits.']*2 + ['Regrasp the digit.']*2 + ['Place the digit.']*2 + ['Arrange the digits.'])
    assert all(r['instruction'] == 'Arrange the digits.' for r in adopted)
    assert len(list((tmp_path/'jev_discarded').glob('*/request_002.json'))) == 1
    assert all(f['subgoal_id'] == w[-1]['subgoal_id'] for w in windows for f in w)
    assert [e['subgoal']['subgoal_id'] for e in events if e['kind'] == 'subgoal_completed'] == ['retry', 'place']
    assert result['remaining_plan'][-1]['subgoal_id'] == 'original-task'
    assert result['attempts'][0]['result'] == 'subgoal_controls_executed'
    assert {'retry','place'} <= {x['subgoal_id'] for x in result['attempts'][0]['executed_subgoals']}


@pytest.mark.parametrize('fault', ['no_condition', 'duplicate_id', 'reserved_id', 'code', 'stale'])
def test_invalid_high_level_plan_stops_without_execution(tmp_path, fault):
    def planner(req):
        raw = high_level_response(req)
        decision = raw['decision']
        if fault == 'no_condition':
            decision['subgoals'][0].pop('success_condition')
        elif fault == 'duplicate_id':
            decision['subgoals'][1]['subgoal_id'] = 'retry'
        elif fault == 'reserved_id':
            decision['subgoals'][0]['subgoal_id'] = 'original-task'
        elif fault == 'code':
            decision['code'] = 'not executable'
        else:
            decision['request_id'] = 'stale'
        return raw
    runner, rollout, _, _, _ = setup(tmp_path, planner_fn=planner)
    assert runner.run()['state'] == 'STOPPED'
    assert rollout.tick == 2 and rollout.sim.calls.count('reset') == 1


@pytest.mark.parametrize('mode', ['student', 'stop'])
def test_planner_can_choose_self_recovery_or_stop(tmp_path, mode):
    def planner(req):
        decision = response(req['native_request'], mode)
        decision['steps'] = 1
        decision['assessment']['execution_status'] = 'failed'
        return dict(**{key: req[key] for key in ('request_id', *IDENTITY)}, decision=decision)
    runner, rollout, requests, _, _ = setup(tmp_path, planner_fn=planner)
    result = runner.run()
    assert result['state'] == ('STOPPED' if mode == 'stop' else 'COMPLETED')
    assert rollout.tick == (2 if mode == 'stop' else 7)
    assert len(requests) == 1 and rollout.counters['recovery_steps'] == 0


def test_subgoal_completion_never_overrides_failed_constraints():
    runtime = ConstraintRuntime(MonitorConfig())
    for step in (1, 2):
        result = runtime.update(frame(step, success_condition='Object held'),
                                scores(grasp_failure=.95, subgoal_complete=.95), now=100.1+step)
        assert not result['subgoal_complete']
    assert result['violations']


def test_codex_decision_proxy_accepts_high_level_plan_without_actuation(tmp_path):
    runner, _, requests, _, _ = setup(tmp_path)
    runner.run()
    proxy = DecisionOnlyRollout(requests[0], tmp_path)
    args = proxy.next_call()
    args.pop('tool')
    args['response'] = high_level_response(requests[0])['decision']
    result = proxy.execute(**args)
    assert not result['simulation_executed']
    assert proxy.response['mode'] == 'replan' and proxy.phase == 'done'
