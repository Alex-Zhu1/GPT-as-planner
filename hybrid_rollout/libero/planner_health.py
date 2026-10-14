"""Read-only protocol preflight shared by service reuse and episode startup."""
import json
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen


class PlannerProtocolError(RuntimeError):
    pass


def check_planner_health(endpoint, required_format, *, timeout=3.):
    parts = urlsplit(endpoint)
    if parts.scheme not in ('http', 'https') or not parts.netloc or not parts.path.endswith('/replan'):
        raise ValueError('Planner endpoint must be an HTTP(S) /replan URL')
    health_url = urlunsplit((parts.scheme, parts.netloc,
                            parts.path[:-len('/replan')] + '/health', '', ''))
    with urlopen(health_url, timeout=timeout) as response:
        health = json.load(response)
    if not isinstance(health, dict):
        raise ValueError('Planner health must be an object')
    if health.get('ready') is not True:
        raise OSError('Planner is not ready')
    formats = health.get('supported_conversation_modes')
    if formats is None:
        formats = [health.get('conversation_mode')]
    if not isinstance(formats, list) or required_format not in formats:
        raise PlannerProtocolError(f'Planner does not support {required_format}; reported formats: {formats!r}')
    if health.get('model') != 'gpt-6-astra' or health.get('effort') != 'xhigh':
        raise PlannerProtocolError('Planner must report gpt-6-astra / xhigh')
    return health
