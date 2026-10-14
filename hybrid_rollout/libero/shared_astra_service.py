"""Start a detached, experiment-independent Astra endpoint on the prepared host."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from .planner_health import check_planner_health
from .instruction_planner import EEF_INPUT_FORMAT as INSTRUCTION_FORMAT
from .instruction_planner import INPUT_FORMAT as INSTRUCTION_ONLY_FORMAT
from .compact_input import INPUT_FORMAT as LEGACY_FORMAT
from .planner_backend import (add_backend_arguments, validate_backend_arguments,
                              backend_argv, verify_backend)


def main(argv=None):
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8103)
    p.add_argument('--astra-runtime', type=Path, default=repo.parent.parent / 'astra-runtime')
    p.add_argument('--python', type=Path, default=repo.parent / 'model_/.libero-max-venv/bin/python')
    p.add_argument('--evidence-root', type=Path, default=repo / 'runtime')
    p.add_argument('--service-root', type=Path, default=repo / 'runtime/shared-astra')
    p.add_argument('--required-protocol', choices=[INSTRUCTION_FORMAT, INSTRUCTION_ONLY_FORMAT, LEGACY_FORMAT],
                   default=INSTRUCTION_FORMAT)
    add_backend_arguments(p)
    a = p.parse_args(argv)
    validate_backend_arguments(p, a)
    if not 1 <= a.port <= 65535:
        p.error('--port must be between 1 and 65535')
    root = a.service_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    url = f'http://127.0.0.1:{a.port}'
    # Serialize starts; the authenticated runtime separately owns the account lease.
    with (root / 'start.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = root / 'service.json'
        if state.exists():
            info = json.loads(state.read_text())
            try:
                os.kill(info['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                if info['url'] != url + '/replan':
                    raise RuntimeError('Existing service uses a different port')
                try:
                    health = check_planner_health(info['url'], a.required_protocol)
                    verify_backend(health, a.planner_backend)
                except (OSError, ValueError, RuntimeError) as error:
                    raise RuntimeError(f'Existing Astra service cannot be reused: {error}. '
                                       'It was not stopped or restarted.') from error
                if info.get('evidence_root') != str(a.evidence_root.resolve()):
                    raise RuntimeError('Existing service uses a different evidence root; it was not restarted')
                print(json.dumps(dict(info, verified_health=health,
                                      required_protocol=a.required_protocol), indent=2))
                return
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', a.port))
        command = ([str(a.python.absolute()), '-m', 'hybrid_rollout.libero.stage1_planner', *backend_argv(a)]
            if a.planner_backend == 'api' else [str(a.astra_runtime.resolve() / 'run-with-astra'),
                str(a.python.absolute()), '-m', 'hybrid_rollout.libero.stage1_planner',
                '--codex', str(a.astra_runtime.resolve() / 'codex')])
        argv = [*command,
                '--output', str(root / 'planner'),
                '--evidence-root', str(a.evidence_root.resolve()),
                '--timeout', '540', '--port', str(a.port)]
        with (root / 'server.log').open('ab') as log:
            proc = subprocess.Popen(argv, cwd=repo, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        info = dict(pid=proc.pid, url=url + '/replan', health_url=url + '/health',
                    shared=True, owned_by_experiment=False,
                    evidence_root=str(a.evidence_root.resolve()),
                    planner_log_root=str(root / 'planner'), argv=argv, backend=a.planner_backend or 'codex')
        try:
            for _ in range(60):
                if proc.poll() is not None:
                    raise RuntimeError(f'Astra startup failed; inspect {root / "server.log"}')
                try:
                    health = check_planner_health(url + '/replan', a.required_protocol, timeout=1.)
                    verify_backend(health, a.planner_backend)
                except (OSError, ValueError):
                    time.sleep(.5)
                    continue
                else:
                    info.update(verified_health=health, required_protocol=a.required_protocol)
                    state.write_text(json.dumps(info, indent=2) + '\n')
                    (root / 'client.env').write_text(
                        f'export JEV_PLANNER_URL={url}/replan\n'
                        'export JEV_PLANNER_TIMEOUT=1200\n')
                    print(json.dumps(info, indent=2))
                    return
            raise RuntimeError('Astra did not become ready with the required protocol')
        except BaseException:
            # Only clean up the process started by this invocation. Existing
            # services above are never stopped, restarted or overwritten.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise


if __name__ == '__main__':
    main()
