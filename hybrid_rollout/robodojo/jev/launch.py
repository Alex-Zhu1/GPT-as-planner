"""Run isolated service environments under one local Linux/POSIX supervisor."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


REPO = Path(__file__).resolve().parents[3]


def expand(value, variables):
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace('{' + key + '}', replacement)
        return value
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def prepare(config, output):
    if os.name != 'posix':
        raise ValueError('This launcher requires Linux/POSIX process groups')
    config = expand(config, {'repo': str(REPO), 'run': str(output)})
    if 'REPLACE_WITH_' in json.dumps(config):
        raise ValueError('Replace the Valen command and ready_text placeholders first')
    timeout = config.get('startup_timeout', 900)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('startup_timeout must be finite and positive')
    services = config.get('services')
    if not isinstance(services, list) or not services:
        raise ValueError('Configure at least one managed service')
    controller = config.get('controller')
    if not isinstance(controller, dict):
        raise ValueError('Configure a controller')
    names = set()
    for item in [*services, controller]:
        name = item.get('name')
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError('Process names must be unique identifiers')
        names.add(name)
        argv = item.get('argv')
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise ValueError(f'{name}: argv must be a nonempty string array')
        if not Path(argv[0]).is_absolute() or not os.access(argv[0], os.X_OK):
            raise ValueError(f'{name}: argv[0] must be an existing absolute executable path')
        cwd = Path(item.get('cwd', str(REPO)))
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError(f'{name}: cwd must be an existing absolute directory')
        item['cwd'] = str(cwd)
        env = item.get('env', {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
            raise ValueError(f'{name}: env must map strings to strings')
    for item in services:
        marker = item.get('ready_text')
        if not isinstance(marker, str) or not marker.strip():
            raise ValueError(f'{item["name"]}: provide a real ready_text log marker')
    ports = config.get('ports', [])
    if not isinstance(ports, list) or any(type(p) is not int or not 1 <= p <= 65535 for p in ports) or len(set(ports)) != len(ports):
        raise ValueError('ports must contain distinct TCP port numbers')
    return config


def check_ports(ports):
    # Binding a temporary socket never consumes the simulator's sole connection.
    # This is a preflight check, not a reservation or a readiness probe.
    for port in ports:
        with socket.socket() as sock:
            # Match restartable HTTP servers: TIME_WAIT is not a live listener.
            # Without SO_REUSEPORT, an active listener still prevents this bind.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as error:
                raise RuntimeError(f'Port {port} is unavailable; no services started') from error


class Supervisor:
    def __init__(self, output):
        self.output = output
        self.processes = []

    def start(self, spec):
        env = os.environ.copy()
        env.update(spec.get('env', {}))
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONPATH'] = str(REPO) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        path = self.output / (spec['name'] + '.log')
        with path.open('xb') as log:
            proc = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        self.processes.append((spec['name'], proc))
        print(f'Started {spec["name"]}: PID {proc.pid}; log {path}', flush=True)
        return proc, path

    def require_alive(self):
        for name, proc in self.processes:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(f'{name} exited early with code {code}; inspect {name}.log')

    def wait_ready(self, spec, path, timeout):
        deadline = time.monotonic() + timeout
        marker = spec['ready_text']
        tail = ''
        with path.open(errors='replace') as log:
            while time.monotonic() < deadline:
                self.require_alive()
                tail += log.read(65536)
                if marker in tail:
                    print(f'Ready: {spec["name"]}', flush=True)
                    return
                tail = tail[-max(len(marker), 4096):]
                time.sleep(.1)
        raise TimeoutError(f'{spec["name"]} did not emit ready_text within {timeout}s')

    def close(self):
        # Signal only groups created by this launcher, never unrelated servers.
        groups = list(reversed(self.processes))
        for _, proc in groups:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        for _, proc in groups:
            try:
                proc.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for _, proc in groups:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def run(config, output):
    check_ports(config.get('ports', []))
    output.mkdir(parents=True, exist_ok=False)
    supervisor = Supervisor(output)
    result = {'state': 'launcher_error', 'exit_code': 1}
    try:
        for spec in config['services']:
            _, log = supervisor.start(spec)
            supervisor.wait_ready(spec, log, config.get('startup_timeout', 900))
        proc, _ = supervisor.start(config['controller'])
        while proc.poll() is None:
            # Simulator may exit normally when the controller disconnects during
            # final artifact writing. Nonzero service exits still abort the run.
            for name, service in supervisor.processes[:-1]:
                if service.poll() not in (None, 0):
                    raise RuntimeError(f'{name} failed; inspect {name}.log')
            time.sleep(.1)
        result = {'state': 'controller_exited', 'exit_code': proc.returncode}
    except KeyboardInterrupt:
        result = {'state': 'interrupted', 'exit_code': 130}
    except Exception as error:
        result['error'] = str(error)
        print(f'Launcher failed: {error}', flush=True)
    finally:
        supervisor.close()
        (output / 'launcher_result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result['exit_code'] if result['exit_code'] >= 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Validate config/ports only; no models or services')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        config = prepare(json.loads(args.config.read_text()), output)
        if output.exists():
            raise ValueError('Output directory already exists; use a new run directory')
        if args.check:
            check_ports(config.get('ports', []))
            print('Configuration and free-port checks passed; model/environment compatibility not checked.')
            return 0
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    def interrupted(*_):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(config, output)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
