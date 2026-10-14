"""Shared CLI selection for Codex-agent and stateless model-API planners."""
from pathlib import Path


def add_backend_arguments(parser):
    parser.add_argument('--planner-backend', choices=('codex', 'api'), default=None,
                        help='Default: existing Codex episode threads; api: independent image requests')
    parser.add_argument('--api-base-url', help='Model API base URL; default http://127.0.0.1:8317/v1')
    parser.add_argument('--api-key-file', type=Path, help='Local model API key file (never the Codex auth.json)')


def validate_backend_arguments(parser, args):
    if args.planner_backend != 'api' and (args.api_base_url or args.api_key_file):
        parser.error('--api-base-url and --api-key-file require --planner-backend api')
    if getattr(args, 'planner_url', None) and (args.api_base_url or args.api_key_file):
        parser.error('Configure API credentials on the shared planner, not with --planner-url')
    if args.planner_backend == 'api':
        args.api_base_url = args.api_base_url or 'http://127.0.0.1:8317/v1'
        args.api_key_file = (args.api_key_file or
            Path(__file__).resolve().parents[2]/'runtime/local-codex-api/api-key').absolute()


def backend_argv(args):
    if args.planner_backend != 'api':
        return []
    return ['--planner-backend', 'api', '--api-base-url', args.api_base_url,
            '--api-key-file', str(args.api_key_file)]


def verify_backend(health, requested):
    # Older health replies came exclusively from the Codex implementation.
    actual = health.get('backend', 'codex')
    if requested and actual != requested:
        raise ValueError(f'Shared planner backend is {actual}, requested {requested}')
    return actual
