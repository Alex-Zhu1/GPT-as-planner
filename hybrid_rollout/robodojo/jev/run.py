"""Astra language plans, Pi subgoal execution and Valen endpoint monitoring."""
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
    planner = parser.add_mutually_exclusive_group()
    planner.add_argument('--codex', help='Use the pinned Astra Codex backend for evidence-triggered reviews')
    planner.add_argument('--planner-url', help='External HTTP endpoint implementing the versioned recovery contract')
    parser.add_argument('--sim-port', type=int, default=19113)
    parser.add_argument('--student-port', type=int, default=18830)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-decisions', type=int, help='0: native episode termination; default 0 for subgoals, 180 for older policies')
    parser.add_argument('--max-replans', type=int, help='Includes initial plan; default 12 for subgoals, 0 for temporal/accumulating, 3 for legacy; 0 disables cap')
    parser.add_argument('--monitor-policy', choices=('subgoals', 'temporal', 'accumulating', 'legacy'), default='subgoals')
    parser.add_argument('--evidence-decay', type=float)
    parser.add_argument('--evidence-baseline', type=float)
    parser.add_argument('--evidence-threshold', type=float)
    parser.add_argument('--min-observations', type=int, default=3)
    parser.add_argument('--uncertainty-observations', type=int, default=12)
    parser.add_argument('--max-observation-refreshes', type=int, default=1, choices=range(4))
    parser.add_argument('--chunk-steps', type=int, choices=range(1, 26))
    parser.add_argument('--monitor-timeout', type=float, default=120.)
    parser.add_argument('--policy-timeout', type=float, default=120.)
    parser.add_argument('--planner-timeout', type=float, default=300.)
    parser.add_argument('--max-evidence-age', type=float, default=120.)
    parser.add_argument('--max-evidence-gap', type=float, default=120.)
    parser.add_argument('--history', type=int)
    parser.add_argument('--min-evidence-seconds', type=float, default=3.)
    parser.add_argument('--uncertainty-seconds', type=float, default=6.)
    parser.add_argument('--review-cooldown-seconds', type=float, default=4.)
    parser.add_argument('--max-async-seconds', type=float, default=6.)
    parser.add_argument('--subgoal-review-seconds', type=float, default=12.,
                        help='Subgoals: simulated seconds between low-frequency Astra completion reviews')
    parser.add_argument('--subgoal-review-cooldown-seconds', type=float, default=6.,
                        help='Minimum simulated seconds between non-urgent anomaly reviews')
    parser.add_argument('--planner-backend', choices=('compact', 'agent', 'api'),
                        help='compact/agent use --codex; api uses independent image model requests')
    parser.add_argument('--api-base-url', help='Model API base URL; default http://127.0.0.1:8317/v1')
    parser.add_argument('--api-key-file', type=Path, help='Local model gateway key file, not Codex auth.json')
    parser.add_argument('--astra-action-chunk', choices=('none', 'sampled', 'recent'), default='sampled',
                        help='Compact Astra input: no trajectory, current four-point sampling, or first N upcoming steps')
    parser.add_argument('--astra-action-steps', type=int, choices=range(1, 26), default=5,
                        help='Number of upcoming trajectory points for recent mode (default 5; capped by execution budget)')
    parser.add_argument('--astra-output-format', choices=('full', 'mode-specific'), default='full',
                        help='Mode-specific output omits unused edit/target fields; preserves assessment validation')
    args = parser.parse_args(argv)
    if args.planner_backend == 'api':
        if args.codex or args.planner_url:
            parser.error('--planner-backend api cannot be combined with --codex or --planner-url')
    else:
        if args.api_base_url or args.api_key_file:
            parser.error('--api-base-url and --api-key-file require --planner-backend api')
        if not (args.codex or args.planner_url):
            parser.error('Choose --codex, --planner-url, or --planner-backend api')
    temporal = args.monitor_policy == 'temporal'
    subgoals = args.monitor_policy == 'subgoals'
    if args.max_decisions is None:
        args.max_decisions = 0 if subgoals else 180
    args.chunk_steps = args.chunk_steps if args.chunk_steps is not None else (25 if temporal or subgoals else 1)
    args.history = args.history if args.history is not None else (6 if temporal or subgoals else 2)
    if subgoals and (args.astra_action_chunk != 'sampled' or args.astra_action_steps != 5 or args.astra_output_format != 'full'):
        parser.error('Subgoal planners return language plans; action-input/output options require an explicit older monitor policy')
    if subgoals and (any(getattr(args, name) is not None for name in ('evidence_decay', 'evidence_baseline', 'evidence_threshold'))
                    or args.min_observations != 3 or args.uncertainty_observations != 12
                    or args.review_cooldown_seconds != 4. or args.max_async_seconds != 6.
                    or args.max_observation_refreshes != 1):
        parser.error('Accumulation/background/refresh options require an older monitor policy; use subgoal time budgets')
    if not subgoals and (args.subgoal_review_seconds != 12. or args.subgoal_review_cooldown_seconds != 6.):
        parser.error('Subgoal review intervals require --monitor-policy subgoals')
    for key, value in dict(evidence_decay=.85 if temporal else .9,
                          evidence_baseline=.5 if temporal else .2,
                          evidence_threshold=.6 if temporal else 1.5).items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    # Validate all configuration before connecting to a policy or resetting a scene.
    values = dict(max_age_s=args.max_evidence_age, max_gap_s=args.max_evidence_gap,
                  max_step_gap=args.chunk_steps, history=args.history)
    scheduler_class = ReplanScheduler
    from .monitor import ConstraintRuntime
    runtime_factory = ConstraintRuntime
    if subgoals:
        from .subgoal_monitor import SubgoalMonitorConfig, SubgoalRuntime, SubgoalValenMonitor
        from .subgoals import SubgoalScheduler
        config = SubgoalMonitorConfig(**values, min_evidence_seconds=args.min_evidence_seconds,
            uncertainty_seconds=args.uncertainty_seconds, review_seconds=args.subgoal_review_seconds,
            review_cooldown_seconds=args.subgoal_review_cooldown_seconds)
        runtime_factory, scheduler_class = SubgoalRuntime, SubgoalScheduler
    elif args.monitor_policy in ('temporal', 'accumulating'):
        from .accumulation import AccumulationConfig, AccumulatingRuntime, AccumulatingScheduler
        config_class, extra = AccumulationConfig, {}
        runtime_factory, scheduler_class = AccumulatingRuntime, AccumulatingScheduler
        if temporal:
            from .temporal import TemporalConfig, TemporalRuntime, TemporalScheduler
            config_class, runtime_factory, scheduler_class = TemporalConfig, TemporalRuntime, TemporalScheduler
            extra = dict(min_evidence_seconds=args.min_evidence_seconds, uncertainty_seconds=args.uncertainty_seconds)
        config = config_class(**values, **extra, evidence_decay=args.evidence_decay,
            evidence_baseline=args.evidence_baseline, evidence_threshold=args.evidence_threshold,
            min_observations=args.min_observations, uncertainty_observations=args.uncertainty_observations)
    else:
        config = MonitorConfig(**values)
    monitor_class = SubgoalValenMonitor if subgoals else ValenMonitor
    monitor = AsyncMonitor(monitor_class(args.valen_url, args.monitor_timeout), config, args.monitor_timeout,
                           runtime_factory=runtime_factory)
    output = args.output.resolve()
    planner_class = CodexPlanner
    planner_options = {}
    if args.planner_backend in ('compact', 'api') or (args.planner_backend is None and (temporal or subgoals)):
        from .compact_planner import CompactCodexPlanner
        planner_class = CompactCodexPlanner
        planner_options = dict(action_chunk_mode=args.astra_action_chunk, action_chunk_steps=args.astra_action_steps,
                               output_format=args.astra_output_format)
    elif temporal or subgoals:
        parser.error('Temporal/subgoal planning requires the compact or api planner backend, or an HTTP planner')
    if (args.astra_action_chunk != 'sampled' or args.astra_action_steps != 5 or args.astra_output_format != 'full') and (
            not planner_options or (not args.codex and args.planner_backend != 'api')):
        parser.error('Astra action chunk options require the local compact or api planner backend')
    if args.planner_backend == 'api':
        from .api_planner import CompactAPIPlanner, DEFAULT_BASE_URL, DEFAULT_KEY_FILE
        planner = CompactAPIPlanner(output, args.api_base_url or DEFAULT_BASE_URL,
            args.api_key_file or DEFAULT_KEY_FILE, args.planner_timeout, **planner_options)
        selected_backend = 'api'
    else:
        planner = (planner_class(output, args.codex, args.planner_timeout, **planner_options) if args.codex else
                   HTTPPlanner(args.planner_url, args.planner_timeout))
        selected_backend = ('compact' if planner_options else 'agent') if args.codex else 'http'
    max_replans = args.max_replans if args.max_replans is not None else (12 if subgoals else 3 if args.monitor_policy == 'legacy' else 0)
    scheduler_options = dict(review_cooldown_seconds=args.review_cooldown_seconds,
                             max_async_seconds=args.max_async_seconds) if temporal else {}
    scheduler = scheduler_class(None, monitor, planner, chunk_steps=args.chunk_steps,
        max_replans=max_replans, policy_timeout=args.policy_timeout, planner_timeout=args.planner_timeout,
        max_observation_refreshes=args.max_observation_refreshes, **scheduler_options)
    if args.max_decisions < 0:
        parser.error('max-decisions must be nonnegative')
    from ..pi05_server.client import Pi05Client
    from ..robodojo_server.client import RoboDojoTools
    output.mkdir(parents=True, exist_ok=False)
    from ..io import write_json
    write_json(output/'planner_backend.json', dict(backend=selected_backend,
        api_base_url=planner.base_url if selected_backend == 'api' else None,
        planner_url=args.planner_url, monitor_policy=args.monitor_policy,
        protocol_version=5 if subgoals else None,
        options={} if subgoals or selected_backend == 'http' else planner_options))
    student = rollout = None
    try:
        student = Pi05Client(args.student_port, args.checkpoint)
        rollout = RoboDojoTools(output, args.task, student, sim_port=args.sim_port,
                               seed=args.seed, max_decisions=args.max_decisions,
                               student_step_limit=args.chunk_steps if temporal or subgoals else 15)
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
        if student is not None and not (scheduler.inference and scheduler.inference.call.thread.is_alive()):
            student.close()


if __name__ == '__main__':
    raise SystemExit(main())
