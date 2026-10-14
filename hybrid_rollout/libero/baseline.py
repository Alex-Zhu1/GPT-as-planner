"""Pure Pi episode using exactly the same observations, controls and native horizon."""
import argparse
import json
from pathlib import Path
import os
import time

from ..robodojo.io import write_json
from ..robodojo.jev.monitor import MonitorConfig
from ..robodojo.jev.runner import AsyncInference, automatic_response, public_frame
from .policy import Pi05Client
from .rollout import LiberoRollout
from .run import configure_assets
from .session import create_session, SUITE_HORIZONS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libero-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suite', choices=SUITE_HORIZONS, default='libero_spatial')
    p.add_argument('--task-id', type=int, default=0)
    p.add_argument('--initial-state-id', type=int, default=0)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--student-port', type=int, default=18830)
    p.add_argument('--chunk-steps', type=int, choices=range(1, 6), default=5)
    a = p.parse_args(argv)
    os.environ['MUJOCO_GL'] = os.environ['PYOPENGL_PLATFORM'] = 'egl'
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    configure_assets(a.libero_root, output)
    student = Pi05Client(a.checkpoint, port=a.student_port)
    rollout = LiberoRollout(output, student,
        lambda: create_session(a.suite, a.task_id, a.initial_state_id, a.seed),
        max_decisions=0, chunk_steps=a.chunk_steps)
    try:
        rollout.start()
        rollout.run.update(evaluation_method='pi_only', teacher_off=True, monitor_enabled=False)
        write_json(output / 'run.json', rollout.run)
        identity = dict(run_id='pi-only', episode_id=rollout.episode, plan_revision=0,
            subgoal_id='original-task', program_id=MonitorConfig(max_age_s=600,
                max_gap_s=600, max_step_gap=a.chunk_steps).program_id)
        with (output / 'frames.jsonl').open('x') as frames:
            initial = public_frame(rollout, identity, rollout.meta['instruction'])
            initial.update(context_only=True, success_condition=None)
            frames.write(json.dumps(initial) + '\n')
            while rollout.phase != 'done':
                AsyncInference(rollout, 600).finish()
                response = automatic_response(rollout.request, a.chunk_steps, None if rollout.tick == 0 else {})
                response['decision_author'] = 'pi_only'
                rollout.execute(response=response)
                frame = public_frame(rollout, identity, rollout.meta['instruction'])
                frame['success_condition'] = None
                frames.write(json.dumps(frame) + '\n')
                frames.flush()
                print(json.dumps(dict(step=rollout.tick, success=rollout.session.success)), flush=True)
        result = dict(native_result=rollout.observation_packet['result'],
            decisions=len(rollout.history), controller_wall_seconds=time.monotonic()-started)
        write_json(output / 'baseline_summary.json', result)
        return 0 if result['native_result']['success'] else 2
    finally:
        rollout.close()
        student.close()


if __name__ == '__main__':
    raise SystemExit(main())
