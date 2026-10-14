"""Offline launch-chain and service compatibility tests; never start models."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from . import max_compare, max_run, planner_health, shared_astra_service
from .controller_options import controller_settings
from .instruction_planner import EEF_INPUT_FORMAT as INPUT_FORMAT
from .compact_input import INPUT_FORMAT as LEGACY_FORMAT


REPO = Path(__file__).resolve().parents[2]
OVERRIDES = dict(astra_call_budget=7, progress_budget=55, subgoal_budget=65,
                 max_review_interval=110, monitor_max_lag=25, stall_steps=20,
                 stall_min_reports=5, planner_timeout=250.5)
FLAGS = [item for name, value in OVERRIDES.items()
         for item in ('--' + name.replace('_', '-'), str(value))]


@pytest.fixture
def shell(tmp_path):
    models = tmp_path / 'model root'
    python = models / '.libero-max-venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    def invoke(*args):
        return subprocess.run(['bash', str(REPO / 'start_libero_max_jev.sh'), *map(str, args)],
                              env=dict(os.environ, JEV_MODEL_ROOT=str(models), JEV_MAX_GL='osmesa'),
                              capture_output=True, text=True)
    return invoke


@pytest.mark.parametrize('mode,extra,continuous', [
    ('--continuous', [], True), ('--dynamic', ['--continuous'], True),
    ('--dynamic', [], False), ('--run', [], False),
])
def test_shell_modes_and_parameter_forwarding(shell, tmp_path, mode, extra, continuous):
    output = tmp_path / 'new output'
    result = shell(mode, output, *extra, *FLAGS, '--port-offset', '100',
                   '--planner-url', 'http://127.0.0.1:8103/replan')
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    assert argv[:2] == ['-m', 'hybrid_rollout.libero.max_compare']
    assert ('--continuous' in argv) is continuous
    assert ('--dynamic-only' in argv) is (mode != '--run')
    assert argv[argv.index('--output') + 1] == str(output)
    for flag, value in zip(FLAGS[::2], FLAGS[1::2]):
        assert argv[argv.index(flag) + 1] == value
    assert argv[argv.index('--port-offset') + 1] == '100'
    assert argv[argv.index('--planner-url') + 1] == 'http://127.0.0.1:8103/replan'


def test_shell_fork_and_model_override(shell, tmp_path):
    prefix = tmp_path / 'recorded controls.jsonl'
    prefix.write_text('{}\n')
    result = shell('--continuous-fork', '--output', tmp_path / 'out', prefix,
                   '--model-root', tmp_path / 'model root')
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    assert argv[argv.index('--fork-prefix') + 1] == str(prefix)
    assert '--continuous' in argv


def test_shell_forwards_api_backend_and_key_path(shell, tmp_path):
    key = tmp_path / 'private key'
    result = shell('--continuous', tmp_path/'output', '--planner-backend', 'api',
                   '--api-base-url', 'http://127.0.0.1:8317/v1', '--api-key-file', key)
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    assert argv[argv.index('--planner-backend') + 1] == 'api'
    assert argv[argv.index('--api-key-file') + 1] == str(key)


def test_api_options_require_api_backend_before_launch(tmp_path):
    with pytest.raises(SystemExit):
        max_compare.main(['--output', str(tmp_path/'out'), '--api-base-url', 'http://127.0.0.1:8317/v1'])
    assert not (tmp_path/'out').exists()


def test_shared_backend_mismatch_fails_before_creating_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(max_compare, 'check_planner_health', lambda *a, **k: dict(good_health(), backend='codex'))
    with pytest.raises(ValueError, match='backend'):
        max_compare.main(['--output', str(tmp_path/'out'), '--planner-backend', 'api',
                          '--planner-url', 'http://127.0.0.1:8103/replan'])
    assert not (tmp_path/'out').exists()


def test_native_smoke_api_configuration_does_not_start_codex(tmp_path):
    from .launch_smoke import configuration
    config = configuration(tmp_path, tmp_path/'models', tmp_path/'astra',
        '/valen-python', '/pi-python', '/sim-python', 5,
        api_options=['--planner-backend', 'api', '--api-base-url', 'http://127.0.0.1:8317/v1',
                     '--api-key-file', str(tmp_path/'key')])
    service = next(s for s in config['services'] if s['name'] == 'planner')
    assert service['argv'][0] == '/sim-python'
    assert '--codex' not in service['argv']
    assert '--planner-backend' in service['argv']


@pytest.mark.parametrize('args', [
    ['--check'], ['--continuous', '--typo'], ['--continuous', '--astra-call-budget'],
    ['--continuous', '--planner-timeout', '--progress-budget', '50'],
    ['--run', '--continuous'], ['--continuous', '/tmp/a', '/tmp/b'],
    ['--continuous', '--output', '/tmp/a', '--output', '/tmp/b'],
    ['--continuous-fork', '/tmp/a'],
])
def test_shell_rejects_bad_invocations_before_exec(shell, args):
    result = shell(*args)
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.parametrize('flag,value', [('--astra-call-budget', '0'), ('--progress-budget', '4'),
    ('--monitor-max-lag', '0'), ('--stall-min-reports', '7'), ('--planner-timeout', 'nan')])
def test_invalid_parameters_fail_before_launch(monkeypatch, tmp_path, flag, value):
    monkeypatch.setattr(max_compare.signal, 'signal', lambda *a: None)
    with pytest.raises(SystemExit) as error:
        max_compare.main(['--output', str(tmp_path / 'out'), flag, value])
    assert error.value.code == 2
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('continuous,fork', [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('backend', ['codex', 'api'])
def test_comparison_configs_reach_controller_and_match_record(monkeypatch, tmp_path, continuous, fork, shared, backend):
    models = tmp_path / 'models'
    max_root = models / 'datasets/LIBERO-MAX'
    manifest = max_root / 'benchmark/lite/libero_max_lite.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'cases': [dict(
        case_id='libero_10-t0009-camera_shift-d0-p195', policy_seed=195, scenario={})]}))
    monkeypatch.setattr(max_compare.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(max_compare.subprocess, 'check_output', lambda *a, **k: 'fake-revision')
    monkeypatch.setattr(max_compare.launch, 'prepare', lambda config, destination: config)
    configs = []
    monkeypatch.setattr(max_compare.launch, 'run', lambda config, destination: configs.append(config) or 2)
    output = tmp_path / 'comparison'
    args = ['--model-root', str(models), '--output', str(output), '--dynamic-only', *FLAGS]
    if backend == 'api':
        args += ['--planner-backend', 'api']
    if shared:
        args += ['--planner-url','http://127.0.0.1:8103/replan']
        monkeypatch.setattr(max_compare, 'check_planner_health', lambda *a, **k: dict(good_health(), backend=backend))
    if continuous:
        args.append('--continuous')
    if fork:
        prefix = tmp_path / 'prefix.jsonl'
        prefix.write_text('{}\n')
        args += ['--fork-prefix', str(prefix)]
    assert max_compare.main(args) == 2
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['controller_settings'] == OVERRIDES
    assert protocol['planner_service']['backend'] == backend
    assert protocol['planner_input_format'] == (INPUT_FORMAT if continuous else LEGACY_FORMAT)
    assert protocol['confirmed_failure_without_astra'] == ('shadow log and continue' if continuous else 'fixed stop')
    assert protocol['confirmed_failure_with_astra'].startswith('up to 7 ')
    assert protocol['groups']['off']['effective_astra_call_budget'] == 0
    assert protocol['groups']['off']['continuous_shadow'] is continuous
    assert protocol['monitor_start'] == ('after recorded prefix' if fork else 'episode initial state')
    assert 'skills/astra-instruction-planner/SKILL.md' in protocol['adapter_sha256']
    assert len(configs) == 2
    if shared:
        assert all(s['name'] != 'planner' for c in configs for s in c['services'])
        assert all(8003 not in c['ports'] and 8103 not in c['ports'] for c in configs)
        assert protocol['planner_service']['owned_by_experiment'] is False
    else:
        planner = next(s for s in configs[1]['services'] if s['name'] == 'planner')
        assert float(planner['argv'][planner['argv'].index('--timeout') + 1]) == OVERRIDES['planner_timeout'] * .9
        if backend == 'api':
            assert '--codex' not in planner['argv']
            assert not any('run-with-astra' in str(v) for v in planner['argv'])
            assert planner['argv'][planner['argv'].index('--planner-backend') + 1] == 'api'
        else:
            assert '--codex' in planner['argv']
    calls = []
    def fake_episode(argv, **kwargs):
        calls.append(argv)
        destination = Path(argv[argv.index('--output') + 1])
        destination.mkdir(parents=True)
        (destination / 'result.json').write_text('{"success":false,"complete":true}')
        (destination / 'execution.mp4').touch()
        return 2
    monkeypatch.setattr(max_compare.subprocess, 'call', fake_episode)
    for mode, config in zip(('off', 'on'), configs):
        argv = [str(v).replace('{run}', str(output / ('astra_' + mode)))
                for v in config['controller']['argv'][3:]]
        assert max_compare.main(argv) == 2
    parse = argparse.ArgumentParser.parse_args
    class Parsed(Exception):
        pass
    def check_parser(parser, argv):
        parsed = parse(parser, argv)
        assert controller_settings(parsed) == OVERRIDES
        assert parsed.continuous is continuous
        assert parsed.continuous_shadow is (continuous and parsed.astra == 'off')
        assert bool(parsed.matched_policy_trace) is (continuous and fork and parsed.astra == 'on')
        assert bool(parsed.fork_reference) is (fork and parsed.astra == 'on')
        if shared and parsed.astra == 'on':
            assert parsed.planner_url == 'http://127.0.0.1:8103/replan'
        raise Parsed
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', check_parser)
    for argv in calls:
        with pytest.raises(Parsed):
            max_run.main(argv[3:])


def good_health():
    return dict(ready=True, model='gpt-6-astra', effort='xhigh', conversation_mode=INPUT_FORMAT,
                supported_conversation_modes=[INPUT_FORMAT, LEGACY_FORMAT])


@pytest.mark.parametrize('override,error', [({}, None),
    ({'supported_conversation_modes': [LEGACY_FORMAT], 'conversation_mode': LEGACY_FORMAT}, planner_health.PlannerProtocolError),
    ({'supported_conversation_modes': INPUT_FORMAT}, planner_health.PlannerProtocolError),
    ({'ready': False}, OSError), ({'model': 'other'}, planner_health.PlannerProtocolError),
    ({'effort': 'low'}, planner_health.PlannerProtocolError)])
def test_health_protocol_validation(monkeypatch, override, error):
    health = dict(good_health(), **override)
    requests = []
    def urlopen(url, timeout):
        requests.append((url, timeout))
        return io.BytesIO(json.dumps(health).encode())
    monkeypatch.setattr(planner_health, 'urlopen', urlopen)
    if error:
        with pytest.raises(error):
            planner_health.check_planner_health('http://localhost:8103/replan', INPUT_FORMAT)
    else:
        assert planner_health.check_planner_health('http://localhost:8103/replan', INPUT_FORMAT) == health
    assert requests == [('http://localhost:8103/health', 3.)]


def test_actual_server_health_satisfies_preflight_without_opening_a_socket(monkeypatch):
    from .planner_server import make_handler
    replies = []
    handler = make_handler(None)
    handler.do_GET(SimpleNamespace(path='/health', reply=lambda code, payload: replies.append((code, payload))))
    status, payload = replies[0]
    assert status == 200
    monkeypatch.setattr(planner_health, 'urlopen',
                        lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    for required in (INPUT_FORMAT, LEGACY_FORMAT):
        assert planner_health.check_planner_health('http://localhost:8103/replan', required) == payload


def test_old_server_without_capabilities_rejected_for_new_protocol(monkeypatch):
    health = dict(ready=True, model='gpt-6-astra', effort='xhigh', conversation_mode=LEGACY_FORMAT)
    monkeypatch.setattr(planner_health, 'urlopen',
                        lambda *a, **k: io.BytesIO(json.dumps(health).encode()))
    with pytest.raises(planner_health.PlannerProtocolError):
        planner_health.check_planner_health('http://localhost:8103/replan', INPUT_FORMAT)
    assert planner_health.check_planner_health('http://localhost:8103/replan', LEGACY_FORMAT) == health


@pytest.mark.parametrize('compatible', [False, True])
def test_existing_service_verified_without_restart_or_state_mutation(monkeypatch, tmp_path, compatible):
    root = tmp_path / 'service'
    root.mkdir()
    state = root / 'service.json'
    original = json.dumps(dict(pid=12345, url='http://127.0.0.1:8103/replan', evidence_root=str(tmp_path)))
    state.write_text(original)
    client = root / 'client.env'
    client.write_text('unchanged')
    signals = []
    monkeypatch.setattr(shared_astra_service.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(shared_astra_service.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not start a process'))
    def health(url, required):
        assert required == INPUT_FORMAT
        if not compatible:
            raise planner_health.PlannerProtocolError('old action-only protocol')
        return good_health()
    monkeypatch.setattr(shared_astra_service, 'check_planner_health', health)
    args = ['--service-root', str(root), '--evidence-root', str(tmp_path)]
    if compatible:
        shared_astra_service.main(args)
    else:
        with pytest.raises(RuntimeError, match='not stopped or restarted'):
            shared_astra_service.main(args)
    assert state.read_text() == original
    assert client.read_text() == 'unchanged'
    assert signals == [(12345, 0)]


def test_controller_rejects_old_service_before_simulation(monkeypatch):
    def health(url, required, **kwargs):
        assert required == INPUT_FORMAT
        raise planner_health.PlannerProtocolError('old action-only protocol')
    monkeypatch.setattr(max_run, 'check_planner_health', health)
    with pytest.raises(planner_health.PlannerProtocolError):
        max_run.main(['--case', 'unused', '--scenario', 'unused', '--arm', 'intervention',
                      '--astra', 'on', '--output', 'unused', '--checkpoint', 'unused', '--continuous'])


@pytest.mark.parametrize('compatible', [False, True])
def test_new_service_checks_protocol_and_cleans_up_only_own_process(monkeypatch, tmp_path, compatible):
    events = []
    proc = SimpleNamespace(pid=12345, poll=lambda: None, terminate=lambda: events.append('terminate'),
                           wait=lambda **k: events.append('wait'))
    monkeypatch.setattr(shared_astra_service.subprocess, 'Popen', lambda *a, **k: proc)
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, *args): pass
    monkeypatch.setattr(shared_astra_service.socket, 'socket', Socket)
    def health(*args, **kwargs):
        if not compatible:
            raise planner_health.PlannerProtocolError('old protocol')
        return good_health()
    monkeypatch.setattr(shared_astra_service, 'check_planner_health', health)
    args = ['--service-root', str(tmp_path / 'service'), '--evidence-root', str(tmp_path)]
    if compatible:
        shared_astra_service.main(args)
        state = json.loads((tmp_path / 'service/service.json').read_text())
        assert state['verified_health'] == good_health()
        assert state['required_protocol'] == INPUT_FORMAT
        assert events == []
    else:
        with pytest.raises(planner_health.PlannerProtocolError):
            shared_astra_service.main(args)
        assert events == ['terminate', 'wait']
        assert not (tmp_path / 'service/service.json').exists()
