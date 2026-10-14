"""Run one LIBERO episode with asynchronous Valen and HTTP recovery planning."""
import argparse
import json
import os
from pathlib import Path

from ..robodojo.io import write_json
from ..robodojo.jev.monitor import AsyncMonitor, MonitorConfig, ValenMonitor
from ..robodojo.jev.planner import HTTPPlanner
from ..robodojo.jev.runner import ReplanScheduler
from .policy import Pi05Client
from .rollout import LiberoRollout
from .session import SUITE_HORIZONS, create_session


def configure_assets(checkout, output):
    root = Path(checkout).resolve() / 'libero/libero'
    paths = dict(benchmark_root=root, bddl_files=root / 'bddl_files',
                 init_states=root / 'init_files', assets=root / 'assets', datasets=root.parent / 'datasets')
    for key in ('benchmark_root', 'bddl_files', 'init_states', 'assets'):
        if not paths[key].is_dir():
            raise ValueError(f'Missing LIBERO {key}: {paths[key]}')
    # JSON is a YAML subset. Create per-run config before importing LIBERO,
    # preventing its interactive first-import prompt or changes to ~/.libero.
    destination = Path(output) / 'libero-config'
    destination.mkdir()
    write_json(destination / 'config.yaml', {k: str(v) for k, v in paths.items()})
    os.environ['LIBERO_CONFIG_PATH'] = str(destination)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libero-root', type=Path, required=True, help='LIBERO git checkout root')
    p.add_argument('--checkpoint', type=Path, required=True, help='Local pi05_libero checkpoint, never RoboDojo')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suite', choices=SUITE_HORIZONS, default='libero_spatial')
    p.add_argument('--task-id', type=int, default=0)
    p.add_argument('--initial-state-id', type=int, default=0)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--student-port', type=int, default=18830)
    p.add_argument('--valen-url', required=True)
    p.add_argument('--planner-url', required=True, help='HTTP endpoint implementing the versioned recovery contract')
    p.add_argument('--max-decisions', type=int, default=0, help='Zero runs until native suite horizon')
    p.add_argument('--chunk-steps', type=int, choices=range(1, 6), default=1)
    p.add_argument('--max-replans', type=int, default=3)
    p.add_argument('--uncertainty-policy', choices=('pause', 'continue'), default='continue',
                   help='LIBERO simulation: continue bounded student chunks on valid unknown evidence')
    p.add_argument('--max-uncertainty-reviews', type=int, choices=range(51), default=0,
                   help='Opt-in Astra review of unresolved uncertainty; only one student control or stop')
    p.add_argument('--max-observation-refreshes', type=int, choices=range(4), default=1)
    p.add_argument('--policy-timeout', type=float, default=120.)
    p.add_argument('--monitor-timeout', type=float, default=120.)
    p.add_argument('--planner-timeout', type=float, default=300.)
    p.add_argument('--max-evidence-age', type=float, default=120.)
    p.add_argument('--max-evidence-gap', type=float, default=120.)
    p.add_argument('--gl', choices=('egl', 'osmesa', 'glfw'), default=None,
                   help='Set before importing MuJoCo; default uses MUJOCO_GL or upstream platform default')
    a = p.parse_args(argv)
    if min(a.task_id, a.initial_state_id, a.max_decisions) < 0:
        p.error('Task/state IDs and decision budget must be nonnegative')
    if a.gl:
        os.environ['MUJOCO_GL'] = a.gl
        os.environ['PYOPENGL_PLATFORM'] = {'egl': 'egl', 'osmesa': 'osmesa', 'glfw': 'glx'}[a.gl]
    config = MonitorConfig(max_age_s=a.max_evidence_age, max_gap_s=a.max_evidence_gap,
                           max_step_gap=a.chunk_steps)
    monitor = AsyncMonitor(ValenMonitor(a.valen_url, a.monitor_timeout,
                           scope='LIBERO Panda post-action monitoring; two cameras'), config, a.monitor_timeout)
    planner = HTTPPlanner(a.planner_url, a.planner_timeout)
    scheduler = ReplanScheduler(None, monitor, planner, chunk_steps=a.chunk_steps,
        max_replans=a.max_replans, policy_timeout=a.policy_timeout, planner_timeout=a.planner_timeout,
        max_observation_refreshes=a.max_observation_refreshes,
        max_uncertainty_reviews=a.max_uncertainty_reviews, include_initial_context=True,
        uncertainty_policy=a.uncertainty_policy)
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    student = rollout = None
    try:
        configure_assets(a.libero_root, output)
        write_json(output / 'selection.json', dict(suite=a.suite, task_id=a.task_id,
            initial_state_id=a.initial_state_id, seed=a.seed, libero_root=str(a.libero_root.resolve()),
            mujoco_gl=os.environ.get('MUJOCO_GL', 'platform default')))
        student = Pi05Client(a.checkpoint, port=a.student_port)
        rollout = LiberoRollout(output, student,
            lambda: create_session(a.suite, a.task_id, a.initial_state_id, a.seed),
            max_decisions=a.max_decisions, chunk_steps=a.chunk_steps)
        scheduler.rollout = rollout
        with (output / 'jev_events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush()
            scheduler.emit = emit
            summary = scheduler.run()
        print(json.dumps(summary, indent=2))
        return 0 if summary['state'] == 'COMPLETED' and summary['native_result'].get('success') else 2
    finally:
        if rollout is not None:
            rollout.close()
        if student is not None:
            # Do not close a websocket under a timed-out inference worker.
            # Process exit ends the daemon worker; the trial is never resumed.
            pending = scheduler.inference and scheduler.inference.call.thread.is_alive()
            if not pending:
                student.close()


if __name__ == '__main__':
    raise SystemExit(main())
