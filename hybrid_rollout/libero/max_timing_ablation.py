"""Isolated timing ablations; the successful MAX execution modules are unchanged."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


class FixedStepWatchdog:
    """One scheduled review, irrespective of progress scores or generation changes."""
    def __init__(self, step=85):
        self.step, self.fired, self.rows = step, False, []

    def update(self, report):
        if report.get('inspection_only') or self.fired:
            return None
        if self.rows and report['step_id'] <= self.rows[-1]['step_id']:
            return None
        self.rows.append(report)
        if report['step_id'] < self.step:
            return None
        if report['step_id'] != self.step:
            raise RuntimeError('Fixed review missed its exact control boundary')
        self.fired = True
        rows = [r for r in self.rows if r['step_id'] >= self.step - 80]
        return dict(state='PROGRESS_REVIEW', violations=[],
            reason='Pre-scheduled step-85 visual review, independent of Valen scores. '
                   'This request does not establish lack of progress or a confirmed failure.',
            first_step=rows[0]['step_id'], last_step=self.step,
            elevated_reports=sum(r['scores']['progress_attempted'] >= .8 and
                                 r['scores']['stuck'] >= .55 for r in rows),
            total_reports=len(rows), trigger='fixed_step_85',
            confirmed_failure=False, oracle_inputs=False)


def worker(mode, arguments):
    from . import max_run
    if mode == 'fixed85':
        class FixedScheduler(max_run.ProgressScheduler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.watchdog = FixedStepWatchdog()
        # Process-local substitution; no existing source or progress rule changes.
        max_run.ProgressScheduler = FixedScheduler
    return max_run.main([*arguments, '--progress-review'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['fixed85', 'baseprogress'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    args, rest = parser.parse_known_args(argv)
    if args.worker:
        # --output belongs to the existing episode CLI in worker mode.
        return worker(args.worker, [*rest, '--output', str(args.output)])
    if rest or args.source is None or args.output is None:
        parser.error('--source and --output are required')
    from ..robodojo.jev import launch
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    template = json.loads((source/'astra_on-config.json').read_text())
    repo = Path(template['controller']['cwd'])
    for name in ('case.json', 'scenario.json'):
        (root/name).write_bytes((source/name).read_bytes())
    def save(name, value):
        (root/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    save('protocol.json', dict(source=str(source), task_success='native original instruction success only',
        modes={'fixed85': 'Dynamic: one scheduled review at 85; confirmed-failure recovery retained',
               'baseprogress': 'Base: unchanged progress watchdog'},
        query_interval=5, action_horizon=10, horizon=520, fresh_services_per_arm=True,
        limitation='Single case; Astra generation is not guaranteed deterministic. Fixed timing selected retrospectively.',
        source_hashes={name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in (
            'hybrid_rollout/libero/max_run.py', 'hybrid_rollout/robodojo/jev/runner.py',
            'hybrid_rollout/robodojo/jev/monitor.py', 'hybrid_rollout/libero/planner_server.py',
            'hybrid_rollout/libero/max_timing_ablation.py')}))
    results = {}
    for mode, arm in [('fixed85', 'intervention'), ('baseprogress', 'control')]:
        config = deepcopy(template)
        python = config['controller']['argv'][0]
        checkpoint = str(repo.parent/'model_/models/pi05_libero')
        config['controller']['argv'] = [python, '-m', 'hybrid_rollout.libero.max_timing_ablation',
            '--worker', mode, '--case', str(root/'case.json'), '--scenario', str(root/'scenario.json'),
            '--arm', arm, '--astra', 'on', '--checkpoint', checkpoint, '--output', '{run}/episodes/'+arm]
        save(mode+'-config.json', config)
        destination = root/mode
        code = launch.run(launch.prepare(config, destination), destination)
        episode = destination/'episodes'/arm
        summary = json.loads((episode/'jev_summary.json').read_text()) if (episode/'jev_summary.json').exists() else None
        results[mode] = dict(launcher_exit=code, summary=summary,
            actual_astra_invocations=len(list((destination/'planner').rglob('invocation.json'))),
            video=str(episode/'execution.mp4'), video_exists=(episode/'execution.mp4').exists())
        save('comparison.json', results)
        print(json.dumps({mode: results[mode]}), flush=True)
        if code not in (0, 2):
            return 1
    return 0 if all(r['summary']['native_result']['success'] for r in results.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
