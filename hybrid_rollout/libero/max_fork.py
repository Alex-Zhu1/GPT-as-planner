"""Reconstruct and strictly audit an identical post-change branch state.

Deterministic prefix replay retains solver warm starts, OSC controller history,
observables and MAX bookkeeping; qpos/qvel-only restore would not. The complete
MuJoCo integration state and additional controller/camera/RNG state are saved.
No branch policy or monitor runs until prefix replay and reference checks finish.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from .max_run import MaxSession, digest
from ..robodojo.io import write_json


def numeric(value, depth=0):
    if depth > 4:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [numeric(x, depth+1) for x in value]
    if isinstance(value, dict):
        return {str(k):numeric(v, depth+1) for k,v in value.items()
                if isinstance(v, (np.ndarray, np.generic, str, int, float, bool, list, tuple, dict)) or v is None}
    return None


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def assert_equal_branch(actual, expected):
    mismatches = [key for key in expected if actual.get(key) != expected[key]]
    if mismatches:
        raise ValueError('Post-change branch state mismatch: '+', '.join(mismatches))


class ForkSession(MaxSession):
    def __init__(self, *args, prefix, reference=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = Path(prefix)
        self.reference = Path(reference) if reference else None

    def integration_state(self):
        import mujoco
        sim = self.env.sim
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        values = np.empty(mujoco.mj_stateSize(sim.model._model, spec))
        mujoco.mj_getState(sim.model._model, sim.data._data, values, spec)
        return values

    def start(self):
        super().start()
        prefix = [json.loads(line) for line in self.prefix.read_text().splitlines()]
        actions, checks = [], []
        def check():
            return dict(step=self.tick, integration=digest(self.integration_state()),
                images={k:digest(self.raw[k]) for k in ('agentview_image','robot0_eye_in_hand_image')})
        checks.append(check())
        for row in prefix:
            if row['step'] != self.tick+1:
                raise ValueError('Shared prefix must be contiguous from control one')
            self.execute(np.asarray([row['action']]))
            actions.append(row['action'])
            checks.append(check())
            if self.env.runtime.applied:
                break
        if not self.env.runtime.applied or self.terminal:
            raise ValueError('Shared prefix must reach a nonterminal camera change')
        state = self.integration_state()
        model = self.env.sim.model
        cameras = {k:np.asarray(getattr(model,k)) for k in ('cam_pos','cam_quat','cam_fovy')}
        controllers = [numeric(vars(robot.controller)) for robot in self.env.robots]
        controllers_json = json.dumps(controllers, sort_keys=True, allow_nan=False)
        np.savez_compressed(self.output/'branch-state.npz', integration=state, **cameras)
        (self.output/'branch-controllers.json').write_text(controllers_json)
        rng = dict(numpy=numeric(np.random.get_state()), python=numeric(random.getstate()))
        write_json(self.output/'branch-rng.json', rng)
        metadata = dict(branch_step=self.tick, horizon=self.horizon,
            remaining_controls=self.horizon-self.tick,
            integration_sha256=digest(state), camera_sha256={k:digest(v) for k,v in cameras.items()},
            controller_sha256=sha_json(controllers), rng_sha256=sha_json(rng),
            prefix_actions_sha256=sha_json(actions), prefix_states_and_images_sha256=sha_json(checks),
            branch_images_sha256=checks[-1]['images'],
            perturbation_events_sha256=sha_json(self.env.events),
            next_policy_query_index=(self.tick+4)//5)
        write_json(self.output/'branch-state.json', metadata)
        write_json(self.output/'prefix-checks.json', checks)
        if self.reference:
            assert_equal_branch(metadata, json.loads(self.reference.read_text()))
        write_json(self.output/'branch-validation.json', dict(
            matched=True, reference=str(self.reference) if self.reference else 'reference branch',
            method='deterministic full-prefix replay + integration/controller/camera/RNG/image checks',
            no_astra_or_monitor_before_branch=True))
        self.provenance['fork'] = metadata
        return self.observe()
