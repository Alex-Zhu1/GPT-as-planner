"""Supervisor tests use local dummy processes, never models or simulator RPC."""
import json
import os
from pathlib import Path
import socket
import sys

import pytest

from .jev.launch import check_ports, prepare, run


def config(tmp_path, service=None, controller=None):
    return dict(startup_timeout=2, ports=[], services=[dict(
        name='service', argv=[sys.executable, '-u', '-c', service or
            'import time; print("READY", flush=True); time.sleep(60)'],
        cwd=str(tmp_path), ready_text='READY')], controller=dict(
        name='controller', argv=[sys.executable, '-c', controller or 'print("FINISHED")'],
        cwd=str(tmp_path)))


def test_launch_success_and_cleanup(tmp_path):
    pidfile = tmp_path/'pid'
    service = f'import os,time; open({str(pidfile)!r},"w").write(str(os.getpid())); print("READY",flush=True); time.sleep(60)'
    output = tmp_path/'run'
    assert run(prepare(config(tmp_path, service), output), output) == 0
    assert 'FINISHED' in (output/'controller.log').read_text()
    assert json.loads((output/'launcher_result.json').read_text())['state'] == 'controller_exited'
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_controller_failure_propagates(tmp_path):
    output = tmp_path/'run'
    assert run(prepare(config(tmp_path, controller='raise SystemExit(2)'), output), output) == 2


@pytest.mark.parametrize('service', ['raise SystemExit(4)', 'import time; time.sleep(60)'])
def test_startup_failure_never_starts_controller(tmp_path, service):
    output = tmp_path/'run'
    settings = config(tmp_path, service)
    settings['startup_timeout'] = .3
    assert run(prepare(settings, output), output) == 1
    assert not (output/'controller.log').exists()


def test_port_check_does_not_connect(tmp_path):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        listener.settimeout(.05)
        with pytest.raises(RuntimeError, match='unavailable'):
            check_ports([listener.getsockname()[1]])
        with pytest.raises(socket.timeout):
            listener.accept()


def test_output_must_be_new(tmp_path):
    with pytest.raises(FileExistsError):
        run(prepare(config(tmp_path), tmp_path), tmp_path)


def test_expansion_and_validation(tmp_path):
    output = tmp_path/'run'
    settings = config(tmp_path)
    settings['controller']['argv'].append('{run}/controller')
    assert prepare(settings, output)['controller']['argv'][-1] == str(output/'controller')
    settings['services'][0]['ready_text'] = ''
    with pytest.raises(ValueError, match='ready_text'):
        prepare(settings, output)


def test_template_requires_configuration(tmp_path):
    example = Path(__file__).parent/'jev/launch.example.json'
    with pytest.raises(ValueError, match='placeholders'):
        prepare(json.loads(example.read_text()), tmp_path/'run')
