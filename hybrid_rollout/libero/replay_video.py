"""Render every recorded control without rerunning Pi, Valen or the planner."""
import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .run import configure_assets
from .session import create_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--comparison', type=Path)
    source_group.add_argument('--controller', type=Path, help='Replay one completed controller run')
    parser.add_argument('--libero-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['MUJOCO_GL'] = os.environ['PYOPENGL_PLATFORM'] = 'egl'
    args.output.mkdir(parents=True, exist_ok=False)
    configure_assets(args.libero_root, args.output)
    summaries = {}
    sources = [('jev_continue', args.controller)] if args.controller else [
        (arm, args.comparison / arm / 'controller')
        for arm in ('pi_only', 'jev_original', 'jev_current')]
    for arm, source in sources:
        run = json.loads((source / 'run.json').read_text())
        original = json.loads((source / 'result.json').read_text())
        provenance = run['provenance']
        session = create_session(provenance['suite'], provenance['task_id'],
                                 provenance['initial_state_id'], provenance['seed'])
        chunks = [np.load(p)['actions'] for p in sorted(source.glob('execution_*.npz'))]
        total = sum(len(x) for x in chunks)
        frames, errors, durations = [], [], []
        def frame():
            obs = session.observe()
            canvas = Image.new('RGB', (1024, 620), '#101826')
            for i, camera in enumerate(('agentview', 'wrist')):
                canvas.paste(Image.fromarray(obs[camera]).resize((512, 512)), (i*512, 70))
            draw = ImageDraw.Draw(canvas)
            draw.text((14, 10), f'{arm} | RECORDED ACTION REPLAY | 0.5x simulation speed', fill='white')
            draw.text((14, 30), f'Control {session.tick}/{total} | Native success: {session.success}', fill='white')
            draw.text((14, 52), 'External camera', fill='white')
            draw.text((526, 52), 'Wrist camera', fill='white')
            draw.text((14, 594), 'Task: pick up the black bowl between the plate and ramekin; place it on the plate.', fill='white')
            frames.append(np.asarray(canvas))
            return obs
        try:
            session.start()
            obs = frame()
            initial = np.load(source / 'observations/000/observation.npz')
            initial_error = float(np.max(np.abs(obs['states'] - initial['states'])))
            for index, chunk in enumerate(chunks):
                for action in chunk:
                    before = float(session.env.sim.data.time)
                    session.execute(np.asarray([action]))
                    durations.append(float(session.env.sim.data.time)-before)
                    obs = frame()
                saved = np.load(source / f'observations/{index+1:03d}/observation.npz')
                errors.append(float(np.max(np.abs(obs['states']-saved['states']))))
            assert session.tick == total == original['step_id']
            assert session.success == original['success']
            fps = 0.5 / float(np.median(durations))
            path = args.output / f'{arm}-action-replay.mp4'
            with imageio.get_writer(path, fps=fps, codec='libx264', macro_block_size=1,
                                    ffmpeg_params=['-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']) as writer:
                for value in frames:
                    writer.append_data(value)
            summaries[arm] = dict(video=str(path), controls=total, frames=len(frames), fps=fps,
                native_success=session.success, original_native_success=original['success'],
                original_stop_reason=original['reason'], initial_state_max_abs_difference=initial_error,
                endpoint_state_max_abs_difference=max(errors), simulation_seconds=sum(durations),
                note='Recorded controls replayed in EGL; no new policy inference. Each control rendered. '
                     'Half simulation speed; original model/planner waiting time omitted. '
                     'Initial rendering may differ from the original run.')
            print(json.dumps(summaries[arm]), flush=True)
        finally:
            session.close()
    (args.output / 'replay-validation.json').write_text(json.dumps(summaries, indent=2)+'\n')


if __name__ == '__main__':
    main()
