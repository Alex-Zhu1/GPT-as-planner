"""Offline Valen replay of a pure-Pi trajectory; no influence on execution."""
import argparse
from collections import Counter, deque
from copy import deepcopy
import json
from pathlib import Path
import time

from ..robodojo.io import write_json
from ..robodojo.jev.monitor import ConstraintRuntime, MonitorConfig, ValenMonitor


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoModel
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=True,
        dtype='auto', attn_implementation='sdpa').to('cuda').eval()
    model.make_compiler()

    class LocalMonitor(ValenMonitor):
        def post(self, payload):
            write_json(a.output / f'request_{len(reports):03d}.json', payload)
            response = model.predict(payload, execution='shared_state')
            write_json(a.output / f'response_{len(reports):03d}.json', response)
            return response

    monitor = LocalMonitor('http://127.0.0.1:8001/predict', 300,
        scope='LIBERO Panda post-action monitoring; two cameras')
    runtime = ConstraintRuntime(MonitorConfig(max_age_s=600, max_gap_s=600, max_step_gap=5))
    window, reports = deque(maxlen=2), []
    for frame in map(json.loads, (a.baseline / 'frames.jsonl').read_text().splitlines()):
        window.append(deepcopy(frame))
        if frame.get('context_only'):
            continue
        started = time.monotonic()
        values = monitor.evaluate(list(window))
        # Offline replay uses recorded observation time, never pretends these are live frames.
        report = runtime.update(frame, values, now=frame['captured_at'])
        report.update(offline_replay=True, inference_seconds=time.monotonic()-started)
        reports.append(report)
        print(json.dumps(dict(step=frame['step_id'], state=report['state'])), flush=True)
    native = json.loads((a.baseline / 'result.json').read_text())
    summary = dict(baseline_native_result=native, windows=len(reports),
        states=dict(Counter(x['state'] for x in reports)),
        first_nonclear_step=next((x['step_id'] for x in reports if x['state']!='RUNNING'), None),
        confirmed_failure_steps=[x['step_id'] for x in reports if x['violations']],
        note='Offline open-loop replay with recorded timestamps. No supplemental captures, '
             'pauses, planner calls or monitor influence. A successful episode does not '
             'label every intermediate state failure-free; these are not calibrated FPR/TPR.')
    write_json(a.output / 'reports.json', reports)
    write_json(a.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
