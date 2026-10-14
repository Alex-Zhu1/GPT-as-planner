"""Matched single-scene Pi / Jev comparison. Task success always comes from LIBERO."""
import argparse
from copy import deepcopy
import datetime
import json
from pathlib import Path
import time
import uuid

from ..robodojo.jev import launch
from ..robodojo.io import write_json
from .launch_smoke import configuration


def main(argv=None):
    repo = Path(__file__).resolve().parents[2]
    models, home = repo.parent / 'model_', repo.parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--arms', nargs='+', choices=['pi_only', 'jev_original', 'jev_current'],
                   default=['pi_only', 'jev_original', 'jev_current'])
    a = p.parse_args(argv)
    root = (a.output or repo / 'runtime' / ('libero-comparison-' +
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])).resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = configuration(repo, models, home / 'astra-runtime', home / 'envs/jev-valen/bin/python',
        models / '.openpi311-venv/bin/python', models / '.libero-sim-venv/bin/python', 0,
        uncertainty_reviews=10, chunk_steps=5, uncertainty_policy='pause')
    protocol = dict(suite='libero_spatial', task_id=0, initial_state_id=0, simulator_seed=7,
        pi_initial_jax_seed=0, checkpoint=str(models / 'models/pi05_libero'),
        ordinary_chunk_steps=5, episode_horizon=220, max_decisions=0,
        uncertainty_review_budget=10, confirmed_failure_recovery_budget=3,
        note='Fresh Pi server per arm. Different inference/discard schedules consume RNG differently. '
             'One matched scene is diagnostic evidence, not a general success-rate benchmark.')
    write_json(root / 'protocol.json', protocol)
    results = {}
    print(f'Comparison artifacts: {root}', flush=True)
    for arm in a.arms:
        config = deepcopy(base)
        args = config['controller']['argv']
        args[2] = 'hybrid_rollout.libero.baseline' if arm == 'pi_only' else 'hybrid_rollout.libero.run'
        if arm == 'pi_only':
            keep = {'--libero-root', '--checkpoint', '--output', '--suite', '--task-id',
                    '--initial-state-id', '--seed', '--student-port', '--chunk-steps'}
            config['controller']['argv'] = args[:3] + [item for i in range(3, len(args), 2)
                if args[i] in keep for item in args[i:i+2]]
            config['services'] = [s for s in config['services'] if s['name'] == 'pi05']
            config['ports'] = [18830]
        elif arm == 'jev_original':
            args[args.index('--max-uncertainty-reviews') + 1] = '0'
        output = root / arm
        write_json(root / (arm + '-config.json'), config)
        started = time.monotonic()
        code = launch.run(launch.prepare(config, output), output)
        wall = time.monotonic()-started
        result_path = output / 'controller/result.json'
        native = json.loads(result_path.read_text()) if result_path.exists() else None
        result = dict(exit_code=code, total_wall_seconds=wall, native_result=native,
                      task_success=bool(native and native['success'] and native['complete']))
        if arm == 'pi_only':
            path = output / 'controller/baseline_summary.json'
            if path.exists(): result.update(json.loads(path.read_text()))
        else:
            path = output / 'controller/jev_summary.json'
            if path.exists():
                summary = json.loads(path.read_text())
                result.update({k: summary[k] for k in ('state','reason','replan_count',
                    'uncertainty_review_count','step_id')})
            events = output / 'controller/jev_events.jsonl'
            if events.exists():
                rows = [json.loads(x) for x in events.read_text().splitlines()]
                reports = [x['report'] for x in rows if x['kind']=='monitor_report']
                result['monitor_calls'] = len(reports)
                result['uncertain_reports'] = sum(x['state']=='UNCERTAIN' for x in reports)
                result['confirmed_failure_reports'] = sum(bool(x['violations']) for x in reports)
        results[arm] = result
        write_json(root / 'comparison.json', dict(protocol=protocol, results=results))
        print(json.dumps({arm:result}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
