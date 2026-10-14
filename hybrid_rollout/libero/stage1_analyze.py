"""Audit every frozen pilot case; no dropping failures or incomplete runs."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(s) for s in path.open()]
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--decode',action='store_true');a=p.parse_args()
    root=a.root.resolve();manifest=read(root/'manifest.json');protocol=read(root/'protocol.json')
    assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==protocol['manifest_sha256']
    planner_calls=collections.defaultdict(list)
    for path in (root/'run/planner').glob('*/invocation.json'):
        request=read(path.parent/'request.json');planner_calls[request['episode_id']].append(path)
    rows=[];details={};missing=[]
    for entry in manifest['episodes']:
        eid=entry['episode_id'];folder=root/'run/episodes'/eid
        if not (folder/'jev_summary.json').exists():missing.append(eid);continue
        summary=read(folder/'jev_summary.json')
        if summary['error']:raise RuntimeError(eid+': '+summary['error'])
        native=summary['native_result'];events=lines(folder/'jev_events.jsonl')
        queries=lines(folder/'queries.jsonl');controls=lines(folder/'controls.jsonl')
        trace=lines(folder/'max-trace.jsonl')[0];video=read(folder/'video.json')
        calls=planner_calls[native['episode_id']]
        assert len(calls)==summary['replan_count']<=1
        if entry['mode']=='none':assert len(calls)==0
        assert len(controls)==summary['step_id']==native['step_id']
        assert len(controls)+1==video['frames']
        success=bool(native['success'] and native['complete'])
        request_events=[e for e in events if e['kind']=='replan_request']
        trigger_events=[e for e in events if e['kind']=='progress_review_trigger']
        step=request_events[0]['step_id'] if request_events else None
        if entry['mode']=='fixed' and step is not None:assert step==entry['fixed_step']
        intervention=trace['intervention_events']
        perturb_step=intervention[0]['step'] if intervention else None
        row=dict(episode_id=eid,pair_id=entry['pair_id'],suite=entry['case']['task_suite_name'],
            task=entry['case']['task_name'],seed=entry['case']['policy_seed'],group=entry['mode'],
            success=int(success),astra_calls=len(calls),action_steps=len(controls),horizon=entry['horizon'],
            penalized_steps=len(controls) if success else entry['horizon'],
            normalized_cost=len(controls)/entry['horizon'] if success else 1.,
            call_step=step,perturbation_step=perturb_step,perturbation_applied=bool(intervention),
            call_before_perturbation=bool(step is not None and perturb_step is not None and step<perturb_step),
            trigger=trigger_events[0]['evidence']['trigger'] if trigger_events else None)
        rows.append(row)
        if a.decode:
            import imageio.v2 as imageio
            reader=imageio.get_reader(folder/'execution.mp4');count=sum(1 for _ in reader);reader.close()
            assert count==video['frames'],eid
        details[eid]=dict(row=row,native=native,video=video,decoded=a.decode,
            video_sha256=hashlib.sha256((folder/'execution.mp4').read_bytes()).hexdigest(),
            monitor_states=dict(collections.Counter(e['report']['state'] for e in events if e['kind']=='monitor_report')),
            triggers=[e['evidence'] for e in trigger_events],
            decisions=[e['response']['decision'] for e in events if e['kind']=='replan_response'],
            subgoal_completed=sum(e['kind']=='subgoal_completed' for e in events),
            queries_by_instruction=dict(collections.Counter(q['instruction'] for q in queries)))
    if missing:raise RuntimeError('Missing frozen episodes: '+str(missing))
    assert len(rows)==36 and len({r['episode_id'] for r in rows})==36
    groups={}
    for mode in ('none','fixed','jev'):
        rs=[r for r in rows if r['group']==mode];assert len(rs)==12
        successful=[r for r in rs if r['success']]
        groups[mode]=dict(n=12,successes=sum(r['success'] for r in rs),success_rate=sum(r['success'] for r in rs)/12,
            astra_calls=sum(r['astra_calls'] for r in rs),mean_calls=statistics.mean(r['astra_calls'] for r in rs),
            mean_actual_steps=statistics.mean(r['action_steps'] for r in rs),
            mean_penalized_steps=statistics.mean(r['penalized_steps'] for r in rs),
            mean_normalized_cost=statistics.mean(r['normalized_cost'] for r in rs),
            mean_steps_success_only=statistics.mean(r['action_steps'] for r in successful) if successful else None,
            perturbation_reached=sum(r['perturbation_applied'] for r in rs))
    pairs=[];wins=collections.Counter()
    for pid in sorted({r['pair_id'] for r in rows}):
        rr={r['group']:r for r in rows if r['pair_id']==pid}
        outcome='jev_only' if rr['jev']['success']>rr['fixed']['success'] else 'fixed_only' if rr['fixed']['success']>rr['jev']['success'] else 'both_success' if rr['jev']['success'] else 'both_fail'
        wins[outcome]+=1
        prefix={}
        ref=root/'run/episodes'/rr['none']['episode_id'];rq=lines(ref/'queries.jsonl');rc=lines(ref/'controls.jsonl')
        for mode in ('fixed','jev'):
            f=root/'run/episodes'/rr[mode]['episode_id'];qs=lines(f/'queries.jsonl');cs=lines(f/'controls.jsonl')
            n=min(rr[mode]['call_step'] or len(cs),len(cs),len(rc))
            prefix[mode]=dict(initial_state_match=qs[0]['sim_state_sha256']==rq[0]['sim_state_sha256'],
                initial_image_match=qs[0]['policy_image_sha256']==rq[0]['policy_image_sha256'],compared_controls=n,
                actions_exact_match=all(x['action']==y['action'] for x,y in zip(rc[:n],cs[:n])),
                max_action_abs_diff=max((max(abs(u-v) for u,v in zip(x['action'],y['action'])) for x,y in zip(rc[:n],cs[:n])),default=0))
        pairs.append(dict(pair_id=pid,outcome_jev_vs_fixed=outcome,groups=rr,prefix_audit=prefix))
    # Descriptive video selection only, never used for sample membership or scoring.
    representatives=[]
    categories=[
        [v['pair_id'] for v in pairs if v['groups']['none']['success'] and
         (not v['groups']['jev']['success'] or not v['groups']['fixed']['success'])],
        [v['pair_id'] for v in pairs if not v['groups']['none']['success'] and
         (v['groups']['jev']['success'] or v['groups']['fixed']['success'])],
        [v['pair_id'] for v in pairs if all(v['groups'][m]['success'] for m in ('none','fixed','jev')) and
         v['groups']['jev']['astra_calls'] < v['groups']['fixed']['astra_calls']]]
    for candidates in categories:
        if candidates and candidates[0] not in representatives:representatives.append(candidates[0])
    save(root/'summary.json',dict(groups=groups,paired_jev_vs_fixed=dict(wins),planned=36,completed=36,
        representative_pairs=representatives,video_selection='Post hoc descriptive: first regression, rescue, and call-saving pair. All 36 cases remain scored.',metric='Original native success AND complete; failures charged full horizon',
        limitations=['Four tasks, three seeds; exploratory only.','Policy noise tied to control step; numeric reproducibility is audited separately.',
                     'Astra may vary between requests.','Online trigger may remain unreached; all frozen episodes retained.']))
    save(root/'case-details.json',details);save(root/'paired-results.json',pairs)
    utility={}
    for mode in ('fixed','jev'):
        utility[mode]=dict(
            rescued_vs_none=sum(v['groups'][mode]['success'] and not v['groups']['none']['success'] for v in pairs),
            regressed_vs_none=sum(v['groups']['none']['success'] and not v['groups'][mode]['success'] for v in pairs),
            calls_on_baseline_success=sum(v['groups'][mode]['astra_calls'] for v in pairs if v['groups']['none']['success']),
            calls_ending_in_failure=sum(v['groups'][mode]['astra_calls'] for v in pairs if not v['groups'][mode]['success']))
    for mode in ('fixed','jev'):
        utility[mode]['trigger_counts']=dict(collections.Counter(r['trigger'] for r in rows if r['group']==mode and r['astra_calls']))
        utility[mode]['calls_before_perturbation']=sum(r['call_before_perturbation'] for r in rows if r['group']==mode)
    save(root/'intervention-utility.json',utility)
    task_summary={}
    for suite in sorted({r['suite'] for r in rows}):
        task_summary[suite]={}
        for mode in ('none','fixed','jev'):
            rs=[r for r in rows if r['suite']==suite and r['group']==mode]
            task_summary[suite][mode]=dict(successes=sum(r['success'] for r in rs),n=len(rs),
                calls=sum(r['astra_calls'] for r in rs),mean_cost=statistics.mean(r['normalized_cost'] for r in rs))
    save(root/'per-task-summary.json',task_summary)
    with (root/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(groups,indent=2));print('PAIRED',dict(wins));print('REPRESENTATIVES',representatives)

if __name__=='__main__':main()
