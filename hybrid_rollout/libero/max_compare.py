"""Run MAX/Jev comparisons with isolated no-Astra and Astra services."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from ..robodojo.jev import launch
from .controller_options import (add_controller_arguments, validate_controller_arguments,
                                 controller_argv, controller_settings)
from .instruction_planner import EEF_INPUT_FORMAT as INSTRUCTION_FORMAT
from .compact_input import INPUT_FORMAT as LEGACY_FORMAT
from .planner_health import check_planner_health
from .planner_backend import (add_backend_arguments, validate_backend_arguments,
                              backend_argv, verify_backend)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def pair_exit_code(results):
    """Process completion is not task-smoke success: 2 means a valid failure."""
    if set(results) != {'control', 'intervention'} or any(
            r.get('status') != 'episode_finished' for r in results.values()):
        return 1
    return 0 if all(r.get('success') is True for r in results.values()) else 2


def pair_worker(a):
    a.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for arm in (('intervention',) if a.dynamic_only else ('control', 'intervention')):
        destination = a.output/arm
        argv = [sys.executable, '-m', 'hybrid_rollout.libero.max_run', '--case', str(a.case),
            '--scenario', str(a.scenario), '--arm', arm, '--astra', a.astra,
            '--output', str(destination), '--checkpoint', str(a.model_root/'models/pi05_libero')]
        argv += ['--student-port', str(18840+a.port_offset),
                 '--valen-url', f'http://127.0.0.1:{8001+a.port_offset}/predict',
                 '--planner-url', a.planner_url or f'http://127.0.0.1:{8003+a.port_offset}/replan']
        argv += controller_argv(a)
        if a.planner_backend:
            argv += ['--planner-backend', a.planner_backend]
        if a.progress_review and a.astra == 'on':
            argv.append('--progress-review')
        if a.fork_prefix:
            argv += ['--fork-prefix', str(a.fork_prefix)]
        if a.fork_reference:
            argv += ['--fork-reference', str(a.fork_reference)]
        if a.continuous:
            argv.append('--continuous')
            if a.astra == 'on':
                argv += ['--planner-audit', str(a.output/'client-calls')]
            else:
                argv.append('--continuous-shadow')
        if a.matched_policy_trace:
            argv += ['--matched-policy-trace', str(a.matched_policy_trace)]
        if arm == 'intervention' and not a.dynamic_only:
            trace = a.output/'control/max-trace.jsonl'
            if not trace.exists():
                results[arm] = dict(status='not_run_missing_base_trace', success=None)
                break
            argv += ['--control-trace', str(trace)]
        started = time.monotonic()
        with (a.output/(arm+'.log')).open('x') as log:
            code = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT)
        result = dict(exit_code=code, wall_seconds=time.monotonic()-started, success=None)
        if (destination/'result.json').exists():
            native = json.loads((destination/'result.json').read_text())
            result.update(native=native, success=bool(native['success'] and native['complete']))
        if (destination/'jev_summary.json').exists():
            summary = json.loads((destination/'jev_summary.json').read_text())
            for key in ('state','reason','step_id','replan_count','uncertain_continuation_count'):
                result[key] = summary[key]
        if (destination/'jev_events.jsonl').exists():
            rows = [json.loads(line) for line in (destination/'jev_events.jsonl').read_text().splitlines()]
            result['monitor_reports'] = sum(r['kind']=='monitor_report' for r in rows)
            result['confirmed_failure_reports'] = sum(r['kind']=='monitor_report' and
                bool(r['report']['violations']) for r in rows)
            result['astra_requests'] = sum(r['kind']=='replan_request' for r in rows) if a.astra=='on' else 0
        result['video_exists'] = (destination/'execution.mp4').exists()
        unavailable = result.get('state') in ('PAUSED_MONITOR_UNAVAILABLE','PAUSED_POLICY_UNAVAILABLE')
        exception = bool(re.match(r'\w*(?:Error|Exception):', result.get('reason') or ''))
        valid = code in (0, 2) and result['video_exists'] and not unavailable and not exception
        result['status'] = 'episode_finished' if valid else 'infrastructure_error'
        results[arm] = result
        save(a.output/'results.json', results)
        print(json.dumps({arm:result}), flush=True)
    save(a.output/'results.json', results)
    paths = [a.output/arm/'max-trace.jsonl' for arm in ('control','intervention')]
    if all(path.exists() for path in paths):
        spec = importlib.util.spec_from_file_location('max_summary',
            a.max_root/'scripts/summarize_cosmos_paired_smoke.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            summary, terminal = module.classify_persistent_pair(*[module.load_one(path) for path in paths])
            save(a.output/'paired_summary.json', summary or terminal)
        except Exception as error:
            save(a.output/'paired_summary.json', dict(matched=False, error=f'{type(error).__name__}: {error}'))
    if a.dynamic_only:
        result = results['intervention']
        return (0 if result['success'] else 2) if result['status']=='episode_finished' else 1
    return pair_exit_code(results)


def runtime_env(repo, models, root):
    native = models/'.libero-max-native'
    return dict(PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1',
        MUJOCO_GL=os.environ.get('JEV_MAX_GL','osmesa'),
        PYOPENGL_PLATFORM=os.environ.get('JEV_MAX_GL','osmesa'), MUJOCO_EGL_DEVICE_ID='0',
        LP_NUM_THREADS='2',
        LIBERO_SOURCE_PACKAGE_ROOT=str(root/'.deps/LIBERO-plus/libero'),
        LIBERO_CONFIG_PATH=str(root/'.deps/libero-plus-config'),
        MAGICK_HOME=str(native), LD_LIBRARY_PATH=str(native/'lib')+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''),
        MAGICK_CONFIGURE_PATH=str(native/'etc/ImageMagick-6'),
        MAGICK_CODER_MODULE_PATH=str(native/'lib/ImageMagick-6.9.11/modules-Q16/coders'),
        PYTHONPATH=':'.join(map(str, [repo, root/'src', root/'scripts/libero_source_overlay',
            models/'sources/openpi/packages/openpi-client/src', root/'.deps/cosmos-policy'])),
        NO_PROXY='localhost,127.0.0.1', no_proxy='localhost,127.0.0.1')


def main(argv=None):
    # nohup/background shells can inherit SIGINT ignored. Restore explicit
    # cancellation so managed service groups still pass through cleanup.
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, default=repo.parent/'model_')
    p.add_argument('--max-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', type=Path)
    p.add_argument('--scenario', type=Path)
    p.add_argument('--pair-worker', action='store_true')
    p.add_argument('--astra', choices=['off','on'])
    p.add_argument('--dynamic-only', action='store_true')
    p.add_argument('--progress-review', action='store_true')
    p.add_argument('--fork-prefix', type=Path)
    p.add_argument('--fork-reference', type=Path)
    p.add_argument('--continuous', action='store_true')
    p.add_argument('--matched-policy-trace', type=Path)
    p.add_argument('--planner-url', help='Reuse this shared Astra /replan endpoint without owning its lifetime')
    add_backend_arguments(p)
    p.add_argument('--port-offset', type=int, default=int(os.environ.get('JEV_MAX_PORT_OFFSET','0')))
    add_controller_arguments(p)
    a = p.parse_args(argv)
    validate_controller_arguments(p, a)
    validate_backend_arguments(p, a)
    a.output = a.output.absolute()
    a.model_root = a.model_root.resolve()
    if not 0 <= a.port_offset <= 46695:
        p.error('Invalid port offset')
    a.max_root = a.max_root or a.model_root/'datasets/LIBERO-MAX'
    a.max_root = a.max_root.resolve()
    if a.fork_prefix and not a.dynamic_only:
        p.error('--fork-prefix requires --dynamic-only')
    if a.continuous and not a.dynamic_only:
        p.error('--continuous comparison requires --dynamic-only')
    if a.pair_worker:
        return pair_worker(a)
    shared_health = (check_planner_health(a.planner_url,
        INSTRUCTION_FORMAT if a.continuous else LEGACY_FORMAT, timeout=min(3., a.planner_timeout))
        if a.planner_url else None)
    actual_backend = verify_backend(shared_health, a.planner_backend) if shared_health else (a.planner_backend or 'codex')
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.max_root/'benchmark/lite/libero_max_lite.json').read_text())
    case = next(c for c in manifest['cases'] if c['case_id']=='libero_10-t0009-camera_shift-d0-p195')
    save(a.output/'case.json', case)
    save(a.output/'scenario.json', case['scenario'])
    save(a.output/'protocol.json', dict(case_id=case['case_id'], query_interval=5, action_horizon=10,
        horizon=520, policy_seed=case['policy_seed'], uncertainty_policy='continue',
        confirmed_failure_without_astra='shadow log and continue' if a.continuous else 'fixed stop',
        confirmed_failure_with_astra=(f'up to {a.astra_call_budget} instruction/bounded-EEF reviews' if a.continuous
                                      else f'up to {a.astra_call_budget} recovery calls'),
        controller_settings=controller_settings(a),
        planner_input_format=INSTRUCTION_FORMAT if a.continuous else LEGACY_FORMAT,
        planner_role='instruction_eef' if a.continuous else 'legacy_control',
        planner_service=dict(shared=bool(a.planner_url), url=a.planner_url,
                             owned_by_experiment=not bool(a.planner_url), verified_health=shared_health,
                             backend=actual_backend,
                             api_base_url=a.api_base_url if actual_backend == 'api' and not a.planner_url else None),
        planner_model_input_format='stateless_libero_review_v1' if actual_backend == 'api' else 'episode_thread',
        observation_profile='task_progress' if a.continuous else 'constraints',
        monitor_start='after recorded prefix' if a.fork_prefix else 'episode initial state',
        groups={mode:dict(astra_enabled=mode=='on', continuous_shadow=a.continuous and mode=='off',
                         effective_astra_call_budget=a.astra_call_budget if mode=='on' else 0)
                for mode in ('off','on')},
        task_success='native original instruction only', reset_during_recovery=False,
        dynamic_only=a.dynamic_only, progress_review_enabled=a.progress_review,
        continuous_monitoring=a.continuous,
        port_offset=a.port_offset,
        fork_prefix=str(a.fork_prefix) if a.fork_prefix else None,
        fork_prefix_sha256=hashlib.sha256(a.fork_prefix.read_bytes()).hexdigest() if a.fork_prefix else None,
        render_backend=os.environ.get('JEV_MAX_GL','osmesa'),
        checkpoint=str(a.model_root/'models/pi05_libero'),
        max_revision=subprocess.check_output(['git','-C',str(a.max_root),'rev-parse','HEAD'],text=True).strip(),
        adapter_sha256={name:hashlib.sha256((repo/'hybrid_rollout/libero'/name).read_bytes()).hexdigest()
            for name in ('max_run.py','max_compare.py','max_render.py','max_fork.py',
                         'continuous.py','task_context.py','controller_options.py','planner_server.py',
                         'episode_thread.py','api_planner.py','planner_backend.py','../model_api.py',
                         'instruction_planner.py','control_review.py','planner_health.py',
                         'eef.py','session.py','rollout.py','../robodojo/jev/runner.py',
                         '../robodojo/jev/plans.py','../robodojo/jev/monitor.py',
                         'skills/astra-instruction-planner/SKILL.md')},
        dynamic_pre_event_policy=('shared recorded Pi prefix; monitor and Astra start only after change' if a.fork_prefix else
                                 'independent live policy' if a.dynamic_only else
                                 'replay corresponding Base proposals; monitor remains active'),
        video='every live control; 1x simulation speed; inference/planning wait omitted',
        limitation='one fixed case; no general success-rate claim'))
    shared = dict(NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost', PYTHONFAULTHANDLER='1')
    model_env = dict(shared, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    models, root, home = a.model_root, a.max_root, repo.parent.parent
    sim = models/'.libero-max-venv/bin/python'
    def service(name, argv, env, ready):
        return dict(name=name, argv=list(map(str, argv)), env=env, ready_text=ready, cwd=str(repo))
    results = {}
    for astra in ('off','on'):
        services = [
            service('pi05', [models/'.openpi311-venv/bin/python', root/'scripts/serve_openpi_deterministic.py',
                '--port',str(18840+a.port_offset),'--checkpoint',models/'models/pi05_libero'],
                dict(model_env, OPENPI_DATA_HOME=str(models/'.cache/openpi'), XLA_PYTHON_CLIENT_PREALLOCATE='false'),
                'server listening'),
            service('valen', [home/'envs/jev-valen/bin/python','-m','hybrid_rollout.libero.valen_server',
                '--model',models/'models/Valen-2B','--port',str(8001+a.port_offset)], model_env, 'VALEN_MODEL_AND_HTTP_READY')]
        ports = [18840+a.port_offset,8001+a.port_offset]
        if astra=='on' and not a.planner_url:
            planner_command = ([sim, '-m', 'hybrid_rollout.libero.planner_server', *backend_argv(a)]
                if actual_backend == 'api' else [home/'astra-runtime/run-with-astra', sim,
                    '-m','hybrid_rollout.libero.planner_server','--codex',home/'astra-runtime/codex'])
            services.append(service('planner', [*planner_command,
                '--output','{run}/planner','--evidence-root','{run}/episodes',
                '--timeout',str(a.planner_timeout * .9),'--port',str(8003+a.port_offset)],
                shared, 'ASTRA_LIBERO_HTTP_READY'))
            ports.append(8003+a.port_offset)
        config = dict(startup_timeout=1800, ports=ports, services=services,
            controller=dict(name='controller',cwd=str(repo), env=runtime_env(repo,models,root),
                argv=list(map(str,[sim,'-m','hybrid_rollout.libero.max_compare','--pair-worker',
                    '--astra',astra,'--model-root',models,'--max-root',root,
                    '--case',a.output/'case.json','--scenario',a.output/'scenario.json',
                    '--output','{run}/episodes','--port-offset',str(a.port_offset)]))))
        config['controller']['argv'] += controller_argv(a)
        if a.planner_backend:
            config['controller']['argv'] += ['--planner-backend', a.planner_backend]
        if a.planner_url and astra == 'on':
            config['controller']['argv'] += ['--planner-url', a.planner_url]
        if a.dynamic_only:
            config['controller']['argv'].append('--dynamic-only')
        if a.progress_review:
            config['controller']['argv'].append('--progress-review')
        if a.continuous:
            config['controller']['argv'].append('--continuous')
            if astra == 'on' and a.fork_prefix:
                config['controller']['argv'] += ['--matched-policy-trace', str(
                    a.output/'astra_off/episodes/intervention/queries.jsonl')]
        if a.fork_prefix:
            config['controller']['argv'] += ['--fork-prefix', str(a.fork_prefix)]
            if astra == 'on':
                config['controller']['argv'] += ['--fork-reference', str(
                    a.output/'astra_off/episodes/intervention/branch-state.json')]
        destination = a.output/('astra_'+astra)
        save(a.output/('astra_'+astra+'-config.json'), config)
        code = launch.run(launch.prepare(config,destination),destination)
        path = destination/'episodes/results.json'
        results[astra] = dict(launcher_exit=code, results=json.loads(path.read_text()) if path.exists() else None)
        planner_dir = destination/'planner'
        results[astra]['actual_astra_invocations'] = (None if a.planner_url and astra == 'on' else
            len(list(planner_dir.rglob('invocation.json'))) if planner_dir.exists() else 0)
        results[astra]['completed_client_reviews'] = len(list((destination/'episodes/client-calls').glob('*/response.json')))
        save(a.output/'comparison.json',results)
        if code==130:
            break
    if len(results)!=2 or any(v['launcher_exit'] not in (0,2) for v in results.values()):
        return 1
    return 0 if all(v['launcher_exit']==0 for v in results.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
