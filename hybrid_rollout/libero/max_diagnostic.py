"""Instrument the unchanged official MAX runner with durable control logs."""
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    from libero_max.cosmos_integration import CosmosInterventionEnv

    root = Path(os.environ['JEV_MAX_DIAGNOSTICS'])
    root.mkdir(parents=True, exist_ok=False)
    stream = (root / 'steps.jsonl').open('x', buffering=1)
    actions = (root / 'actions.jsonl').open('x', buffering=1)
    original = CosmosInterventionEnv.step

    def step(self, action):
        started = time.monotonic()
        # Write before entering native code, including the exact crashing action.
        actions.write(json.dumps(dict(total_env_steps=self.total_env_steps, action=list(action)))+'\n')
        result = original(self, action)
        if self.total_env_steps % 5 == 0:
            import numpy as np
            np.save(root/'latest-state.npy', self.sim.get_state().flatten())
        row = dict(total_env_steps=self.total_env_steps,
                   policy_steps=max(0, self.total_env_steps-self.warmup_steps),
                   success=bool(result[2]), events=self.events,
                   step_seconds=time.monotonic()-started,
                   render_stats=self._stable_camera_render_qa)
        stream.write(json.dumps(row)+'\n')
        if self.total_env_steps % 10 == 0 or result[2]:
            print(json.dumps(row), flush=True)
        return result

    CosmosInterventionEnv.step = step
    runner = Path(os.environ['JEV_MAX_ROOT']) / 'scripts/run_openpi_libero_max.py'
    sys.argv[0] = str(runner)
    try:
        runpy.run_path(str(runner), run_name='__main__')
    finally:
        stream.close()
        actions.close()


if __name__ == '__main__':
    main()
