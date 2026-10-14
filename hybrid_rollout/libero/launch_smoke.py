"""Start a native-task LIBERO/Pi/Valen/Astra smoke on the prepared server."""
import argparse
import datetime
import json
from pathlib import Path
import uuid

from ..robodojo.jev import launch
from .planner_backend import add_backend_arguments, validate_backend_arguments, backend_argv


def configuration(repo, models, astra, valen_python, pi_python, controller_python, decisions,
                  uncertainty_reviews=10, chunk_steps=5, uncertainty_policy='continue', *, api_options=None):
    shared = {'NO_PROXY': 'localhost,127.0.0.1', 'no_proxy': 'localhost,127.0.0.1'}
    def service(name, argv, env, marker):
        return dict(name=name, argv=[str(x) for x in argv], cwd=str(repo),
                    env={**shared, **env}, ready_text=marker)
    config = dict(startup_timeout=1800, ports=[8001, 8003, 18830], services=[
        service('valen', [valen_python, '-m', 'hybrid_rollout.libero.valen_server',
            '--model', models / 'models/Valen-2B', '--port', '8001'],
            dict(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'),
            'VALEN_MODEL_AND_HTTP_READY'),
        service('pi05', [pi_python, '-m', 'hybrid_rollout.libero.pi05_server',
            '--checkpoint', models / 'models/pi05_libero', '--port', '18830',
            '--identity-output', '{run}/pi05-identity.json'],
            dict(CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                 OPENPI_DATA_HOME=str(models / '.cache/openpi'),
                 HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'), '"event": "loaded"'),
        service('planner', [*([controller_python, '-m', 'hybrid_rollout.libero.planner_server', *api_options]
            if api_options else [astra / 'run-with-astra', controller_python,
                '-m', 'hybrid_rollout.libero.planner_server', '--codex', astra / 'codex']),
            '--output', '{run}/planner', '--evidence-root', '{run}/controller',
            '--timeout', '540', '--port', '8003'], {}, 'ASTRA_LIBERO_HTTP_READY'),
    ])
    config['controller'] = dict(name='controller', cwd=str(repo), env={**shared,
        'MUJOCO_EGL_DEVICE_ID': '0', 'PYTHONPATH': str(models / 'sources/LIBERO')}, argv=[
        str(controller_python), '-m', 'hybrid_rollout.libero.smoke',
        '--libero-root', str(models / 'sources/LIBERO'),
        '--checkpoint', str(models / 'models/pi05_libero'), '--output', '{run}/controller',
        '--suite', 'libero_spatial', '--task-id', '0', '--initial-state-id', '0', '--seed', '7',
        '--student-port', '18830', '--valen-url', 'http://127.0.0.1:8001/predict',
        '--planner-url', 'http://127.0.0.1:8003/replan', '--gl', 'egl',
        '--max-decisions', str(decisions), '--chunk-steps', str(chunk_steps), '--max-replans', '3',
        '--max-uncertainty-reviews', str(uncertainty_reviews),
        '--uncertainty-policy', uncertainty_policy,
        '--policy-timeout', '600', '--monitor-timeout', '300', '--planner-timeout', '600',
        '--max-evidence-age', '600', '--max-evidence-gap', '600'])
    return config


def main(argv=None):
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, default=repo.parent / 'model_')
    p.add_argument('--astra-runtime', type=Path, default=repo.parent.parent / 'astra-runtime')
    p.add_argument('--valen-python', type=Path, default=repo.parent.parent / 'envs/jev-valen/bin/python')
    p.add_argument('--pi-python', type=Path)
    p.add_argument('--controller-python', type=Path)
    p.add_argument('--max-decisions', type=int, choices=range(0, 51), default=0,
                   help='Zero (default) runs to the native episode horizon or task success')
    p.add_argument('--max-uncertainty-reviews', type=int, choices=range(51), default=10)
    p.add_argument('--chunk-steps', type=int, choices=range(1, 6), default=5)
    p.add_argument('--uncertainty-policy', choices=('pause', 'continue'), default='continue')
    p.add_argument('--output', type=Path)
    p.add_argument('--check', action='store_true')
    add_backend_arguments(p)
    a = p.parse_args(argv)
    validate_backend_arguments(p, a)
    models, astra = a.model_root.resolve(), a.astra_runtime.resolve()
    # Keep venv executable symlinks intact: resolving bin/python loses venv identity.
    pi = a.pi_python or models / '.openpi311-venv/bin/python'
    controller = a.controller_python or models / '.libero-sim-venv/bin/python'
    config = configuration(repo, models, astra, a.valen_python.absolute(), pi.absolute(),
                           controller.absolute(), a.max_decisions, a.max_uncertainty_reviews, a.chunk_steps,
                           a.uncertainty_policy, api_options=backend_argv(a))
    tag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    output = (a.output or repo / 'runtime' / ('libero-smoke-' + tag)).absolute()
    configs = repo / 'runtime/libero-launch-configs'
    configs.mkdir(parents=True, exist_ok=True)
    path = configs / (tag + '.json')
    path.write_text(json.dumps(config, indent=2))
    print(f'Run artifacts: {output}', flush=True)
    return launch.main(['--config', str(path), '--output', str(output)] + (['--check'] if a.check else []))


if __name__ == '__main__':
    raise SystemExit(main())
