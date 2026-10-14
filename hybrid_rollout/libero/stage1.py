"""Frozen 4-task x 3-seed x 3-arm pilot, isolated from the successful runtime."""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def fixed_step(horizon):
    return math.ceil(.2*horizon/5)*5


def eligible(mode, step, horizon, calls):
    return calls == 0 and mode == 'fixed' and step >= fixed_step(horizon)


def worker(entry, output, checkpoint):
    import numpy as np
    from .max_run import MaxPolicy, MaxRollout, MaxSession, ProgressWatchdog, encode_video
    from ..robodojo.jev.runner import ReplanScheduler, automatic_response
    from ..robodojo.jev.monitor import AsyncMonitor, MonitorConfig, ValenMonitor
    from ..robodojo.jev.planner import HTTPPlanner

    class StepPolicy(MaxPolicy):
        def infer(self, observation, path):
            # Parent hashes seed:index. Index here is the control step, never call count.
            self.index = int(observation['max_policy_step'])
            return super().infer(observation, path)

    class PilotScheduler(ReplanScheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.watchdog = ProgressWatchdog()

        def advisory(self, feedback):
            if self.replans or entry['mode'] == 'none':
                return None
            if entry['mode'] == 'jev':
                return self.watchdog.update(feedback)
            if eligible('fixed', self.rollout.tick, entry['horizon'], self.replans):
                recent = [r for r in self.reports if r['step_id'] >= self.rollout.tick-80]
                return dict(state='PROGRESS_REVIEW', violations=[], confirmed_failure=False,
                    oracle_inputs=False, trigger='fixed_20pct', first_step=recent[0]['step_id'],
                    last_step=self.rollout.tick, total_reports=len(recent),
                    reason='Scheduled review at 20 percent of the task horizon, independent of progress evidence.')
            return None

        def run(self):
            error = None
            try:
                self.rollout.start()
                self.subgoal = self.original_task = self.rollout.meta['instruction']
                self.goals.append(dict(subgoal_id='original-task',instruction=self.subgoal,success_condition=None))
                self.rollout.run.update(stage1=entry, max_astra_calls=1,
                    noise_seed='sha256(policy_seed:control_step)[:4] little endian',
                    monitor_role='Evidence only; alarms do not stop the no-Astra baseline or exhausted-budget arms',
                    success_criterion='native original instruction success and complete')
                save(self.rollout.output/'run.json',self.rollout.run)
                self.seed_context()
                while self.rollout.phase != 'done':
                    self.start_inference()
                    feedback = self.review() if self.monitor.pending is not None else None
                    advisory = self.advisory(feedback) if feedback else None
                    if advisory:
                        self.event('progress_review_trigger', evidence=advisory)
                        response = self.recover(advisory, progress=True)
                        if response is None:
                            break
                        if response['mode']=='replan':
                            continue
                    elif feedback and feedback.get('subgoal_complete') and len(self.goals)>1:
                        self.discard_inference('Completed subgoal; next instruction requires fresh inference')
                        self.event('subgoal_completed', subgoal=self.goals.popleft())
                        self.activate_goal('Advance verified recovery subgoal')
                        continue
                    else:
                        self.finish_inference()
                        response=automatic_response(self.rollout.request,5,feedback)
                    self.fresh(self.rollout.observation_packet)
                    before=self.rollout.tick
                    args=self.rollout.next_call().copy();args.pop('tool');args['response']=response
                    self.rollout.execute(**args)
                    if not 0<self.rollout.tick-before<=5:
                        raise RuntimeError('Invalid control chunk')
                    frame=self.current_frame();self.frames.append(frame)
                    if self.attempts:
                        self.attempts[-1].setdefault('executed_subgoals',[]).append(dict(
                            subgoal_id=self.subgoal_id,instruction=self.subgoal,start_step=before,end_step=self.rollout.tick))
                    self.event('executed_chunk', frame=frame,decision_author=response['decision_author'])
                    self.monitor.submit(frame)
                if self.monitor.pending is not None:
                    self.review()
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                self.event('infrastructure_error',error=error)
            finally:
                if self.inference is not None:
                    try:self.discard_inference('Episode finished')
                    except Exception as exc:error=error or f'{type(exc).__name__}: {exc}'
                native=(self.rollout.finish('stage1_stop') if self.rollout.phase!='done' and self.rollout.episode
                        else self.rollout.observation_packet.get('result'))
            summary=dict(entry=entry,native_result=native,step_id=self.rollout.tick,
                replan_count=self.replans,error=error,attempts=self.attempts,remaining_plan=list(self.goals))
            save(output/'jev_summary.json',summary)
            return summary

    output.mkdir(parents=True,exist_ok=False)
    case=entry['case'];mode=entry['mode']
    policy=StepPolicy(18840,case['policy_seed'],checkpoint)
    rollout=MaxRollout(output,policy,lambda:MaxSession(case,case['scenario'],'intervention',output,
        f"Stage1 {mode} {entry['pair_id']}"),max_decisions=0,chunk_steps=5)
    monitor=AsyncMonitor(ValenMonitor('http://127.0.0.1:8001/predict',300,
        scope='LIBERO Panda post-action monitoring; two cameras'),
        MonitorConfig(max_age_s=900,max_gap_s=900,max_step_gap=5),300)
    scheduler=PilotScheduler(rollout,monitor,HTTPPlanner('http://127.0.0.1:8003/replan',600),
        chunk_steps=5,max_replans=1,policy_timeout=600,planner_timeout=600,
        max_observation_refreshes=1,max_uncertainty_reviews=0,include_initial_context=True,uncertainty_policy='continue')
    try:
        with (output/'jev_events.jsonl').open('x',buffering=1) as stream:
            scheduler.emit=lambda d:stream.write(json.dumps(d)+'\n')
            summary=scheduler.run()
    finally:
        rollout.close();policy.close()
    if (output/'controls.jsonl').exists():encode_video(output)
    if summary['error']:return 1
    return 0 if summary['native_result'] and summary['native_result']['success'] and summary['native_result']['complete'] else 2


def batch(manifest, output, checkpoint):
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for entry in manifest['episodes']:
        destination=output/entry['episode_id']
        entryfile=output/(entry['episode_id']+'.json');save(entryfile,entry)
        # Resume skips only completed, valid results. Infrastructure failures stop for repair.
        if (destination/'jev_summary.json').exists():
            previous=json.loads((destination/'jev_summary.json').read_text())
            if previous.get('error') or not (destination/'execution.mp4').exists():
                raise RuntimeError('Existing incomplete attempt requires explicit repair: '+str(destination))
            code=0 if previous['native_result']['success'] else 2
        else:
            with (output/(entry['episode_id']+'.log')).open('x') as log:
                code=subprocess.call([sys.executable,'-m','hybrid_rollout.libero.stage1','--worker',str(entryfile),
                    '--output',str(destination),'--checkpoint',str(checkpoint)],stdout=log,stderr=subprocess.STDOUT)
        summary=json.loads((destination/'jev_summary.json').read_text()) if (destination/'jev_summary.json').exists() else None
        results.append(dict(episode_id=entry['episode_id'],exit_code=code,summary=summary))
        save(output/'results.json',results)
        print(json.dumps(dict(completed=len(results),total=36,episode=entry['episode_id'],exit_code=code,
            success=summary.get('native_result',{}).get('success') if summary else None,
            steps=summary.get('step_id') if summary else None,calls=summary.get('replan_count') if summary else None)),flush=True)
        if code not in (0,2):return 1
    return 0


def freeze(repo, source, output):
    from .session import SUITE_HORIZONS
    output.mkdir(parents=True,exist_ok=False)
    path=repo.parent/'model_/datasets/LIBERO-MAX/benchmark/max8000/libero_max_8000.json'
    data=json.loads(path.read_text())
    suites=['libero_spatial','libero_object','libero_goal','libero_10']
    selected=[];episodes=[]
    for suite in suites:
        candidates=sorted([c for c in data['cases'] if c['task_suite_name']==suite and
            c['scenario']['change_type']=='camera_shift' and c.get('substrate_variant',{}).get('benchmark')!='LIBERO-PRO'],
            key=lambda c:(c['task_index'],c['case_id']))
        selected.append(candidates[0])
    for task_index,original in enumerate(selected):
        for seed_index,seed in enumerate([7,42,195]):
            case=deepcopy(original);case['policy_seed']=seed
            pair=f't{task_index+1}-s{seed}'
            # Balanced order, pre-fixed independently of outcomes.
            modes=['none','fixed','jev'];rotation=(task_index+seed_index)%3
            for mode in modes[rotation:]+modes[:rotation]:
                episodes.append(dict(episode_id=pair+'-'+mode,pair_id=pair,mode=mode,case=case,
                    horizon=SUITE_HORIZONS[case['task_suite_name']],fixed_step=fixed_step(SUITE_HORIZONS[case['task_suite_name']])))
    manifest=dict(stage='36 episode Dynamic pilot',selection='Smallest task_index then case_id of Plus camera_shift per suite',
        seeds=[7,42,195],original_cases=selected,episodes=episodes,
        source_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        note='Seeds 7 and 42 extend the released seed-195 cases; not official Max aggregate.')
    save(output/'manifest.json',manifest)
    config=json.loads((source/'astra_on-config.json').read_text())
    for spec in config['services']:
        if spec['name']=='planner':
            spec['argv']=[v.replace('hybrid_rollout.libero.planner_server','hybrid_rollout.libero.stage1_planner') for v in spec['argv']]
    python=config['controller']['argv'][0]
    config['controller']['argv']=[python,'-m','hybrid_rollout.libero.stage1','--batch',str(output/'manifest.json'),
        '--output','{run}/episodes','--checkpoint',str(repo.parent/'model_/models/pi05_libero')]
    save(output/'launch-config.json',config)
    names=['stage1.py','stage1_planner.py','max_run.py','planner_server.py']
    save(output/'protocol.json',dict(manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
        query_interval=5,action_horizon=10,max_astra_calls=1,success='native original instruction success AND complete',
        failed_action_cost='task horizon H; normalized cost actual/H on success and 1 on failure',
        pi_noise='sha256(policy_seed:control_step) first four bytes little endian; unaffected by discarded inference',
        shared_services='One resident deterministic Pi server, Valen and Astra endpoint; fresh simulator and client per episode',
        failure_policy='Valen is observational in all arms; no automatic stop on alarms or budget exhaustion. Astra stop ends failed attempt.',
        trigger_policy='Jev uses unchanged ProgressWatchdog; fixed uses 20% H rounded upward to five controls; no additional confirmed-failure call path',
        post_correction='Continue observing even if uncertain; preserve subgoal completion checks and original native goal',
        prompt='Identical planner template and 80-control evidence selection for fixed and Jev; trigger source labelled honestly',
        hashes={n:hashlib.sha256((repo/'hybrid_rollout/libero'/n).read_bytes()).hexdigest() for n in names},
        core_hashes={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in (
            'hybrid_rollout/robodojo/jev/runner.py','hybrid_rollout/robodojo/jev/monitor.py')}))
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--batch',type=Path)
    p.add_argument('--freeze',action='store_true');p.add_argument('--source',type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();a.output=a.output.resolve()
    if a.worker:return worker(json.loads(a.worker.read_text()),a.output,a.checkpoint)
    if a.batch:return batch(json.loads(a.batch.read_text()),a.output,a.checkpoint)
    if a.freeze:
        freeze(Path(__file__).resolve().parents[2],a.source.resolve(),a.output);return 0
    p.error('Select --freeze, --batch or --worker')

if __name__=='__main__':raise SystemExit(main())
