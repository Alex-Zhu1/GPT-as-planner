"""Shared controller defaults, validation and launcher forwarding (no model imports)."""
import math


OPTIONS = {
    'astra_call_budget': (int, 12, 'Total Astra calls, including instruction reviews and resumption'),
    'progress_budget': (int, 80, 'Controls without sustained progress before original-task review'),
    'subgoal_budget': (int, 80, 'Controls without sustained progress before subgoal review'),
    'max_review_interval': (int, 160, 'Cap on the execution fallback window; sustained progress resets it'),
    'monitor_max_lag': (int, 40, 'Maximum observation lag in controls before backpressure'),
    'stall_steps': (int, 30, 'Executed evidence span for low-progress review'),
    'stall_min_reports': (int, 4, 'Low-progress votes in six valid reports (effective floor 4)'),
    'planner_timeout': (float, 600., 'Wall timeout including shared service queue time'),
}


def add_controller_arguments(parser):
    for name, (kind, default, help_text) in OPTIONS.items():
        parser.add_argument('--' + name.replace('_', '-'), type=kind, default=default, help=help_text)


def validate_controller_arguments(parser, args):
    for name in ('astra_call_budget', 'stall_steps'):
        if getattr(args, name) < 1:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    for name in ('progress_budget', 'subgoal_budget', 'max_review_interval', 'monitor_max_lag'):
        if getattr(args, name) < 5:
            parser.error('--' + name.replace('_', '-') + ' must cover a five-control chunk')
    if not 2 <= args.stall_min_reports <= 6:
        parser.error('--stall-min-reports must be between 2 and 6 (effective floor 4)')
    if not math.isfinite(args.planner_timeout) or args.planner_timeout <= 0:
        parser.error('--planner-timeout must be finite and positive')


def controller_settings(args):
    return {name: getattr(args, name) for name in OPTIONS}


def controller_argv(args):
    return [value for name, setting in controller_settings(args).items()
            for value in ('--' + name.replace('_', '-'), str(setting))]
