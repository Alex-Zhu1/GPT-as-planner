"""Opt-in pi05 + asynchronous Valen + failure-triggered Astra recovery."""
import argparse
import json
from pathlib import Path

from .monitor import AsyncMonitor, MonitorConfig, ValenMonitor
from .planner import CodexPlanner, HTTPPlanner
from .runner import ReplanScheduler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--valen-url', required=True, help='Full Valen /predict endpoint; shared image filesystem required')
    planner = parser.add_mutually_exclusive_group(required=True)
    planner.add_argument('--codex', help='Use the existing pinned Astra Codex backend only on confirmed failures')
    planner.add_argument('--planner-url', help='External HTTP endpoint implementing the versioned recovery contract')
    parser.add_argument('--sim-port', type=int, default=19113)
    parser.add_argument('--student-port', type=int, default=18830)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-decisions', type=int, default=180)
    parser.add_argument('--max-replans', type=int, default=3)
    parser.add_argument('--max-observation-refreshes', type=int, default=1, choices=range(4))
    parser.add_argument('--chunk-steps', type=int, default=1, choices=range(1, 6))
    parser.add_argument('--monitor-timeout', type=float, default=120.)
    parser.add_argument('--policy-timeout', type=float, default=120.)
    parser.add_argument('--planner-timeout', type=float, default=300.)
    parser.add_argument('--max-evidence-age', type=float, default=120.)
    parser.add_argument('--max-evidence-gap', type=float, default=120.)
    parser.add_argument('--history', type=int, default=2)
    args = parser.parse_args(argv)
    # Validate all configuration before connecting to a policy or resetting a scene.
    config = MonitorConfig(max_age_s=args.max_evidence_age, max_gap_s=args.max_evidence_gap,
                           max_step_gap=args.chunk_steps, history=args.history)
    monitor = AsyncMonitor(ValenMonitor(args.valen_url, args.monitor_timeout), config, args.monitor_timeout)
    output = args.output.resolve()
    planner = (CodexPlanner(output, args.codex, args.planner_timeout) if args.codex else
               HTTPPlanner(args.planner_url, args.planner_timeout))
    scheduler = ReplanScheduler(None, monitor, planner, chunk_steps=args.chunk_steps,
        max_replans=args.max_replans, policy_timeout=args.policy_timeout, planner_timeout=args.planner_timeout,
        max_observation_refreshes=args.max_observation_refreshes)
    if args.max_decisions < 0:
        parser.error('max-decisions must be nonnegative')
    from ..pi05_server.client import Pi05Client
    from ..robodojo_server.client import RoboDojoTools
    output.mkdir(parents=True, exist_ok=False)
    student = rollout = None
    try:
        student = Pi05Client(args.student_port, args.checkpoint)
        rollout = RoboDojoTools(output, args.task, student, sim_port=args.sim_port,
                               seed=args.seed, max_decisions=args.max_decisions)
        scheduler.rollout = rollout
        with (output/'jev_events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush()
            scheduler.emit = emit
            summary = scheduler.run()
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0 if summary['state'] == 'COMPLETED' else 2
    finally:
        if hasattr(planner, 'close'):
            planner.close()
        if rollout is not None:
            rollout.close()
        if student is not None:
            student.close()


if __name__ == '__main__':
    raise SystemExit(main())
