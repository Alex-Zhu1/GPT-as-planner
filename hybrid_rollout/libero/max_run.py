"""One closed-loop Jev episode on the frozen LIBERO-MAX camera-shift case.

Physics and MAX hooks stay on the owner thread. Policy/Valen/Astra workers only
receive copied observations. Videos contain every actually executed control.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from ..robodojo.io import write_json
from ..robodojo.jev.monitor import AsyncMonitor, IDENTITY, MonitorConfig, ValenMonitor
from ..robodojo.jev.planner import HTTPPlanner
from ..robodojo.jev.runner import ReplanScheduler, automatic_response
from .contract import action_chunk
from .rollout import LiberoRollout
from .session import LiberoSession, SUITE_HORIZONS
from .max_render import install_render_validation, validate_camera_images
from .controller_options import add_controller_arguments, validate_controller_arguments, controller_settings
from .planner_health import check_planner_health
from .planner_backend import verify_backend
from .instruction_planner import EEF_INPUT_FORMAT as INSTRUCTION_FORMAT
from .compact_input import INPUT_FORMAT as LEGACY_FORMAT


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class StopOnFailure:
    """Fixed no-Astra ablation: stop only when the scheduler requests recovery."""
    def replan(self, request):
        decision = automatic_response(request['native_request'], 1, request['failure'])
        decision.pop('decision_author')
        decision.pop('evaluation_method')
        decision.update(mode='stop', reason='Astra disabled: fixed stop on confirmed failure.')
        return {**{key: request[key] for key in ('request_id', *IDENTITY)}, 'decision': decision}


class NoAstraScheduler(ReplanScheduler):
    def recover(self, failure, *, uncertainty=False):
        self.discard_inference('Fixed no-Astra stop on confirmed failure')
        self.event('fixed_recovery_stop', failure=failure, astra_called=False)
        self.transition('STOPPED', 'Astra disabled: fixed stop on confirmed failure')
        return None


class ProgressWatchdog:
    """Escalate sustained ambiguous progress evidence, not a safety violation.

    80 executed controls (4 simulation seconds) of predominantly elevated stuck
    scores merit a longer visual review. Fresh generations and gaps reset votes;
    no simulator object pose, reward, task success or perturbation label is read.
    """
    def __init__(self, window=80):
        self.window = window
        self.rows = deque()
        self.identity = None
        self.last_step = None
        self.first_step = None

    def update(self, report):
        identity = tuple(report[k] for k in IDENTITY)
        step = report['step_id']
        if identity != self.identity or report.get('gap_reset'):
            self.rows.clear()
            self.last_step = None
            self.first_step = step
            self.identity = identity
        if report.get('inspection_only') or (self.last_step is not None and step <= self.last_step):
            return None
        self.last_step = step
        self.rows.append(report)
        while self.rows and step-self.rows[0]['step_id'] > self.window:
            self.rows.popleft()
        if len(self.rows) < 2 or step-self.rows[0]['step_id'] < self.window:
            return None
        elevated = sum(r['scores']['progress_attempted'] >= .8 and
                       r['scores']['stuck'] >= .55 for r in self.rows)
        overdue = step-self.first_step >= 160 and all(
            r.get('state')=='UNCERTAIN' and not r.get('subgoal_complete') for r in self.rows)
        if elevated/len(self.rows) < .75 and not overdue:
            return None
        evidence = dict(state='PROGRESS_REVIEW', violations=[],
            reason='Persistent ambiguous stuck evidence despite attempted progress; visual review required.',
            first_step=self.rows[0]['step_id'], last_step=step,
            elevated_reports=elevated, total_reports=len(self.rows),
            trigger='unverified_progress_budget' if overdue else 'sustained_stuck_evidence',
            confirmed_failure=False, oracle_inputs=False)
        self.rows.clear()
        self.first_step = step
        return evidence


class ProgressScheduler(ReplanScheduler):
    progress_review_enabled = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.watchdog = ProgressWatchdog()

    def progress_review(self, feedback):
        return self.watchdog.update(feedback)


class MaxSession(LiberoSession):
    def __init__(self, case, scenario, arm, output, label):
        install_render_validation()
        from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
        from libero.libero import benchmark
        from libero_max.cosmos_integration import CosmosInterventionEnv
        from libero_max.env_factory import create_libero_env_with_retry
        from libero_max.pro_runtime import wrap_case_env
        from libero_max.substrate import load_case_task

        # This integration is intentionally limited to the audited physical
        # camera change. Intent replacement/cancellation need different scoring.
        if scenario['change']['operation'] != 'shift_camera':
            raise ValueError('This smoke adapter supports camera_shift only')
        self.output, self.label = Path(output), label
        self.frames = self.output / 'frames'
        self.frames.mkdir()
        task, states = load_case_task(case, benchmark)
        seed = case['policy_seed']
        def reseed(value):
            random.seed(value)
            np.random.seed(value)
        env, instruction = create_libero_env_with_retry(
            lambda: get_libero_env(task, 'cosmos', resolution=256),
            policy_seed=seed, reseed=reseed)
        env = wrap_case_env(env, case)
        env.seed(seed)
        wrapped = CosmosInterventionEnv(env, instruction, scenario, arm,
            self.output / 'max-trace.jsonl', case['task_index'], case['init_state_index'])
        horizon = SUITE_HORIZONS[case['task_suite_name']]
        wrapped.configure_episode(case['task_suite_name'], seed, 5, horizon)
        super().__init__(wrapped, states[case['init_state_index']], instruction, horizon)
        self.provenance = dict(case=case, scenario=scenario, arm=arm, seed=seed,
                               simulator='MuJoCo 3.2.6 / robosuite 1.4.0',
                               render_backend=os.environ.get('MUJOCO_GL'))
        self.steps = (self.output / 'controls.jsonl').open('x', buffering=1)
        self.outcome_written = False

    def start(self):
        self._owner()
        if self.started:
            raise ValueError('No recovery reset is permitted')
        self.started = True
        self.env.reset()
        self.raw = self.env.set_init_state(self.initial_state)
        for _ in range(self.settle_steps):
            self.raw, _, done, _ = self.env.step([0.] * 6 + [-1.])
            if done:
                raise ValueError('Initial state is already successful')
        self.record_frame()
        return self.observe()

    def observe(self, *, refresh=False):
        self._owner()
        if self.poisoned:
            raise RuntimeError('Uncertain execution cannot be resumed')
        if refresh:
            before = self.env.sim.data.time
            self.raw = self.env.backend.refresh_observation()
            if before != self.env.sim.data.time:
                self.poisoned = True
                raise RuntimeError('Observation refresh advanced physics')
        validate_camera_images(self.raw)
        observation = self.measured_observation()
        # Audit fields never enter public_frame / model prompts.
        observation['max_policy_step'] = self.tick
        observation['max_event_applied'] = self.env.runtime.applied
        observation['sim_state_sha256'] = digest(self.env.sim.get_state().flatten())
        return observation

    def record_frame(self):
        validate_camera_images(self.raw)
        canvas = Image.new('RGB', (1024, 612), '#101826')
        for i, key in enumerate(('agentview_image', 'robot0_eye_in_hand_image')):
            pixels = np.ascontiguousarray(self.raw[key][::-1, ::-1])
            canvas.paste(Image.fromarray(pixels).resize((512, 512)), (i*512, 68))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f'{self.label} | LIVE EXECUTION | control {self.tick}/{self.horizon}', fill='white')
        draw.text((12, 27), f'Native success: {self.success} | camera change applied: {self.env.runtime.applied}', fill='white')
        draw.text((12, 47), 'External camera', fill='white')
        draw.text((525, 47), 'Wrist camera', fill='white')
        draw.text((12, 592), self.instruction, fill='white')
        canvas.save(self.frames / f'{self.tick:05d}.png')

    def execute(self, actions):
        self._owner()
        actions = action_chunk(actions)
        if self.terminal or self.poisoned or np.any(np.abs(actions) > 1):
            raise ValueError('Invalid or terminal execution')
        executed = []
        try:
            for action in actions:
                before = float(self.env.sim.data.time)
                self.raw, _, done, _ = self.env.step(action.tolist())
                self.tick += 1
                # Camera shift does not change the goal: use the original native
                # success predicate, never monitor/planner text or step exhaustion.
                self.success = bool(self.env.check_success())
                self.terminal = self.success or bool(done) or self.tick >= self.horizon
                executed.append(action.copy())
                row = dict(step=self.tick, action=action.tolist(), success=self.success,
                    sim_dt=float(self.env.sim.data.time)-before, events=self.env.events,
                    sim_state_sha256=digest(self.env.sim.get_state().flatten()))
                if getattr(self, 'record_execution_timing', False):
                    row['executed_at_wall'] = time.time()
                self.steps.write(json.dumps(row)+'\n')
                self.record_frame()
                if self.tick % 10 == 0 or self.terminal:
                    print(json.dumps(dict(step=self.tick, success=self.success,
                        event_count=len(self.env.events))), flush=True)
                if self.terminal:
                    break
        except BaseException:
            self.poisoned = True
            raise
        return np.asarray(executed)

    def close(self):
        if not self.closed:
            try:
                if self.started and self.env.init_state_sha256 and not self.outcome_written:
                    self.env.record_outcome(self.success and not self.poisoned)
                    self.outcome_written = True
            finally:
                self.steps.close()
                super().close()


class MaxPolicy:
    def __init__(self, port, seed, checkpoint, control_trace=None):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy('127.0.0.1', port)
        self.client._ws.ping_timeout = 600
        self.metadata = self.client.get_server_metadata()
        if self.metadata.get('libero_max_deterministic_noise') is not True:
            raise ValueError('Deterministic MAX policy server is required')
        self.metadata.update(checkpoint=str(checkpoint), config='pi05_libero', action_horizon=10)
        self.index, self.seed = 0, seed
        self.matched_queries = None
        self.matched_until = None
        self.replay = None
        if control_trace:
            row = json.loads(Path(control_trace).read_text())
            self.replay = {q['policy_step']: q for q in row['policy_queries']}

    def release_matched_queries(self):
        # The first teacher review is the intervention boundary. From then on,
        # infer normally from the live branch, including any corrected subgoal.
        self.matched_queries = None

    def infer(self, observation, output):
        from openpi_client import image_tools
        started = time.monotonic()
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(observation['agentview'], 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(observation['wrist'], 224, 224))
        source = 'model'
        if (getattr(self, 'matched_until', None) is not None
                and observation['max_policy_step'] >= self.matched_until):
            self.release_matched_queries()
        if self.matched_queries is not None:
            prior = self.matched_queries.get(observation['max_policy_step'])
            image_hash = digest(np.concatenate((observation['agentview'], observation['wrist']), axis=1))
            if prior is None or any((
                prior['instruction'] != observation['instruction'],
                prior['sim_state_sha256'] != observation['sim_state_sha256'],
                prior['policy_image_sha256'] != image_hash)):
                raise ValueError('Matched pre-review policy replay input differs from baseline')
            actions = action_chunk(prior['actions'], horizon=10)
            source = 'matched_pre_review_replay'
        elif self.replay is not None and not observation['max_event_applied']:
            prior = self.replay.get(observation['max_policy_step'])
            if prior is None or prior['instruction'] != observation['instruction']:
                raise ValueError('Base prefix has no matching policy query/instruction')
            actions = action_chunk(prior['actions'], horizon=10)
            source = 'control_replay'
        else:
            seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{self.index}'.encode()).digest()[:4], 'big')
            result = self.client.infer({'observation/image': image, 'observation/wrist_image': wrist,
                'observation/state': observation['states'], 'prompt': observation['instruction'],
                'libero_max_noise_seed': seed})
            actions = action_chunk(result['actions'], horizon=10)
        np.savez_compressed(output, raw_actions=actions, actions=actions,
            states=observation['states'], agentview=observation['agentview'], wrist=observation['wrist'])
        prediction = dict(prediction_id=uuid.uuid4().hex, inference_index=self.index,
            inference_seconds=time.monotonic()-started, max_query_source=source)
        self.index += 1
        return actions, prediction

    def close(self):
        self.client._ws.close()


class MaxRollout(LiberoRollout):
    def _inference_packet(self, directory, *, student_instruction):
        # Replay bookkeeping belongs only in the audit trace, not Astra input.
        source = self.prediction.pop('max_query_source')
        packet = super()._inference_packet(directory, student_instruction=student_instruction)
        query = self.session.env.record_policy_query(self.actions, instruction=student_instruction,
            source=source)
        query.update(sim_state_sha256=self.obs['sim_state_sha256'],
            policy_image_sha256=digest(np.concatenate((self.obs['agentview'], self.obs['wrist']), axis=1)))
        with (self.output / 'queries.jsonl').open('a') as stream:
            stream.write(json.dumps(query)+'\n')
        return packet


def encode_video(output):
    import imageio.v2 as imageio
    output = Path(output)
    paths = sorted((output/'frames').glob('*.png'))
    controls = [json.loads(line) for line in (output/'controls.jsonl').read_text().splitlines()]
    if len(paths) != len(controls)+1:
        raise ValueError('Video must cover every executed control plus initial state')
    fps = 1/float(np.median([r['sim_dt'] for r in controls])) if controls else 20
    with imageio.get_writer(output/'execution.mp4', fps=fps, codec='libx264',
            macro_block_size=1, ffmpeg_params=['-crf','18','-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
        for path in paths:
            writer.append_data(imageio.imread(path))
    write_json(output/'video.json', dict(frames=len(paths), controls=len(controls), fps=fps,
        source='Live observations after every executed control; no re-simulation',
        model_wait_time_omitted=True, playback='1x simulation time'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--scenario', type=Path, required=True)
    p.add_argument('--arm', choices=['control','intervention'], required=True)
    p.add_argument('--astra', choices=['off','on'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--student-port', type=int, default=18840)
    p.add_argument('--valen-url', default='http://127.0.0.1:8001/predict')
    p.add_argument('--planner-url', default='http://127.0.0.1:8003/replan')
    p.add_argument('--planner-backend', choices=('codex', 'api'),
                   help='Require this backend from the planner health check')
    p.add_argument('--planner-audit', type=Path,
                   help='Record only this episode\'s shared HTTP requests and responses')
    p.add_argument('--control-trace', type=Path)
    p.add_argument('--progress-review', action='store_true',
                   help='Escalate sustained ambiguous progress evidence to Astra')
    p.add_argument('--fork-prefix', type=Path)
    p.add_argument('--fork-reference', type=Path)
    p.add_argument('--matched-policy-trace', type=Path,
                   help='Replay baseline Pi queries on exactly matched inputs until the first shared planner call')
    p.add_argument('--continuous', action='store_true', help='Run Pi without waiting for Valen at every chunk')
    add_controller_arguments(p)
    p.add_argument('--continuous-shadow', action='store_true', help='Log Valen failures without intervention for a full-budget comparator')
    p.add_argument('--matched-policy-until', type=int, help='Shadow-only continuation boundary of an already recorded Pi prefix')
    a = p.parse_args(argv)
    validate_controller_arguments(p, a)
    if a.continuous_shadow and not (a.continuous and a.astra == 'off'):
        p.error('--continuous-shadow requires --continuous --astra off')
    shadow_replay = a.continuous_shadow and a.matched_policy_until is not None
    if a.matched_policy_until is not None and not (shadow_replay and a.matched_policy_trace):
        p.error('--matched-policy-until requires a shadow comparison trace')
    if a.matched_policy_trace and not (a.fork_reference and (
            shadow_replay or (a.planner_audit and a.astra == 'on'))):
        p.error('--matched-policy-trace requires a fork reference and an audited recovery or shadow continuation')
    planner_format = INSTRUCTION_FORMAT if a.continuous else LEGACY_FORMAT
    planner_health = (check_planner_health(a.planner_url, planner_format, timeout=min(3., a.planner_timeout))
                      if a.astra == 'on' else None)
    if planner_health:
        verify_backend(planner_health, a.planner_backend)
    from libero_max.scenario import load_scenarios, validate_scenario_collection
    scenarios = load_scenarios([a.scenario])
    if len(scenarios) != 1 or validate_scenario_collection(scenarios):
        raise ValueError('Exactly one validated scenario is required')
    case = json.loads(a.case.read_text())
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'selection.json', dict(case=case, scenario=scenarios[0], arm=a.arm,
        astra=a.astra, uncertainty_policy='continue',
        confirmed_failure_without_astra='shadow log and continue' if a.continuous_shadow else 'stop',
        controller_settings=controller_settings(a), planner_input_format=planner_format,
        planner_role='instruction_eef' if a.continuous else 'legacy_control', verified_planner_health=planner_health,
        progress_review=bool(a.progress_review and a.astra == 'on'),
        continuous_monitoring=a.continuous, progress_budget=a.progress_budget,
        astra_call_budget=a.astra_call_budget, max_review_interval=a.max_review_interval,
        observation_profile='task_progress' if a.continuous else 'constraints',
        stall_steps=a.stall_steps, stall_min_reports=a.stall_min_reports,
        subgoal_budget=a.subgoal_budget, monitor_max_lag=a.monitor_max_lag,
        continuous_shadow=a.continuous_shadow, matched_policy_until=a.matched_policy_until,
        success_criterion='native original instruction success only'))
    monitor = AsyncMonitor(ValenMonitor(a.valen_url, 300,
        scope='LIBERO Panda post-action monitoring; two cameras',
        observation_profile='task_progress' if a.continuous else 'constraints'),
        MonitorConfig(max_age_s=max(900, a.planner_timeout+60),
                      max_gap_s=max(900, a.planner_timeout+60), max_step_gap=5,
                      history=6 if a.continuous else 2,
                      observation_profile='task_progress' if a.continuous else 'constraints'), 300)
    student = MaxPolicy(a.student_port, case['policy_seed'], a.checkpoint, a.control_trace)
    if a.matched_policy_trace:
        student.matched_queries = {q['policy_step']: q for q in
            (json.loads(line) for line in a.matched_policy_trace.read_text().splitlines())}
        student.matched_until = a.matched_policy_until
    def make_session():
        if a.fork_prefix:
            from .max_fork import ForkSession
            class InitializedFork(ForkSession):
                def start(self):
                    self.record_execution_timing = a.continuous
                    obs = super().start()
                    student.index = (self.tick+4)//5
                    return obs
            return InitializedFork(case, scenarios[0], a.arm, output,
                f'POST-CHANGE FORK / Astra={a.astra}', prefix=a.fork_prefix, reference=a.fork_reference)
        session = MaxSession(case, scenarios[0], a.arm, output, f'Jev Astra={a.astra} / {a.arm}')
        session.record_execution_timing = a.continuous
        return session
    rollout = MaxRollout(output, student, make_session, max_decisions=0, chunk_steps=5)
    if a.astra == 'on' and a.planner_audit:
        from .shared_planner import RecordedHTTPPlanner
        planner = RecordedHTTPPlanner(a.planner_url, a.planner_timeout, a.planner_audit)
    else:
        planner = HTTPPlanner(a.planner_url, a.planner_timeout) if a.astra == 'on' else StopOnFailure()
    scheduler_type = (ProgressScheduler if a.progress_review else ReplanScheduler) if a.astra == 'on' else NoAstraScheduler
    extra = {}
    if a.continuous:
        from .continuous import ContinuousScheduler
        scheduler_type = ContinuousScheduler
        extra = dict(astra_enabled=a.astra == 'on', progress_budget=a.progress_budget,
                     max_review_interval=a.max_review_interval,
                     subgoal_budget=a.subgoal_budget, max_monitor_lag=a.monitor_max_lag,
                     shadow_only=a.continuous_shadow, stall_steps=a.stall_steps,
                     stall_min_reports=a.stall_min_reports)
    if a.matched_policy_trace and not a.continuous:
        class MatchedReviewScheduler(scheduler_type):
            def recover(self, failure, **kwargs):
                # Finish any speculative cache lookup before switching policy
                # mode. Refreshing a paused scene can change rendered pixels;
                # generate its fresh candidate with Pi, never a cached action.
                self.discard_inference('Matched prefix ends at recovery review')
                student.release_matched_queries()
                self.event('matched_policy_replay_released', reason='Recovery review begins; no new control executed')
                return super().recover(failure, **kwargs)
        scheduler_type = MatchedReviewScheduler
    scheduler = scheduler_type(rollout, monitor, planner, chunk_steps=5, max_replans=a.astra_call_budget,
        policy_timeout=600, planner_timeout=a.planner_timeout, max_observation_refreshes=1,
        max_uncertainty_reviews=0, include_initial_context=True, uncertainty_policy='continue', **extra)
    try:
        with (output/'jev_events.jsonl').open('x', buffering=1) as events:
            scheduler.emit = lambda row: events.write(json.dumps(row)+'\n')
            summary = scheduler.run()
        print(json.dumps(summary), flush=True)
    finally:
        rollout.close()
        if not (scheduler.inference and scheduler.inference.call.thread.is_alive()):
            student.close()
    encode_video(output)
    return 0 if summary['native_result'] and summary['native_result']['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
