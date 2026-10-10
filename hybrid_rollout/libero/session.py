"""One MuJoCo owner and one initial-state selection, with no recovery resets."""
from pathlib import Path
import hashlib
from threading import get_ident
import time

import numpy as np

from .contract import action_chunk

SUITE_HORIZONS = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300,
                  'libero_10': 520, 'libero_90': 400}
CAMERAS = {'agentview': 'agentview_image', 'wrist': 'robot0_eye_in_hand_image'}


def measured_observation(raw, instruction):
    position = np.asarray(raw['robot0_eef_pos'], dtype=np.float64)
    quat = np.asarray(raw['robot0_eef_quat'], dtype=np.float64)
    gripper = np.asarray(raw['robot0_gripper_qpos'], dtype=np.float64)
    if (position.shape != (3,) or quat.shape != (4,) or gripper.shape != (2,)
            or not np.isfinite(np.r_[position, quat, gripper]).all()
            or not np.isclose(np.linalg.norm(quat), 1., atol=1e-3)):
        raise ValueError('Invalid measured LIBERO proprioception')
    # LIBERO supplies xyzw. Follow the OpenPI LIBERO evaluation convention.
    w = np.clip(quat[3], -1., 1.)
    sin_half = np.sqrt(1. - w * w)
    axis_angle = np.zeros(3) if sin_half < 1e-8 else quat[:3] * (2. * np.arccos(w) / sin_half)
    result = dict(states=np.r_[position, axis_angle, gripper].astype(np.float32),
                  eef_position=position, eef_quaternion_xyzw=quat, instruction=instruction)
    for name, source in CAMERAS.items():
        image = np.asarray(raw[source])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB camera image')
        # Training/evaluation convention: 180 degree rotation of BOTH views.
        result[name] = np.ascontiguousarray(image[::-1, ::-1])
    result['captured_at'] = time.time()
    return result


class LiberoSession:
    def __init__(self, env, initial_state, instruction, horizon, *, settle_steps=10):
        if type(horizon) is not int or horizon < 1 or type(settle_steps) is not int or settle_steps < 0:
            raise ValueError('Invalid episode horizon or settle count')
        self.env, self.initial_state, self.instruction = env, initial_state, instruction
        self.horizon, self.settle_steps = horizon, settle_steps
        self.owner = get_ident()
        self.started = self.poisoned = self.closed = False
        self.tick, self.success, self.terminal = 0, False, False
        self.provenance = {}

    def _owner(self):
        if get_ident() != self.owner or self.closed:
            raise RuntimeError('MuJoCo must stay on the live owning thread')

    def start(self):
        self._owner()
        if self.started:
            raise ValueError('Only one initialization is allowed; no reset/rollback')
        self.started = True
        self.env.reset()
        self.raw = self.env.set_init_state(self.initial_state)
        for _ in range(self.settle_steps):
            self.raw, _, done, _ = self.env.step([0.] * 6 + [-1.])
            if done or self.env.check_success():
                raise ValueError('Episode ended during initialization; no policy success is claimed')
        if self.env.check_success():
            raise ValueError('Initial state is already successful')
        return self.observe()

    def observe(self, *, refresh=False):
        self._owner()
        if not self.started or self.poisoned:
            raise RuntimeError('Session is not observable')
        if refresh:
            # Force camera/proprio observables to update without step(), reset(), or set_state().
            before = self.env.sim.data.time
            self.env.env._update_observables(force=True)
            self.raw = self.env.env._get_observations()
            if self.env.sim.data.time != before:
                self.poisoned = True
                raise RuntimeError('Refresh advanced simulation time')
        return measured_observation(self.raw, self.instruction)

    def execute(self, actions):
        self._owner()
        values = action_chunk(actions)
        if not self.started or self.poisoned or self.terminal or np.any(np.abs(values) > 1.):
            raise ValueError('Cannot execute invalid controls or an ended session')
        trace = []
        try:
            for action in values:
                self.raw, _, done, _ = self.env.step(action.tolist())
                self.tick += 1
                self.success = bool(self.env.check_success())
                self.terminal = self.success or bool(done) or self.tick >= self.horizon
                trace.append(action.copy())
                if self.terminal:
                    break
        except BaseException:
            self.poisoned = True  # Never retry an action with uncertain execution.
            raise
        return np.asarray(trace)

    def close(self):
        if not self.closed:
            self._owner()
            self.env.close()
            self.closed = True


def create_session(suite_name, task_id, initial_state_id, seed=7):
    if suite_name not in SUITE_HORIZONS:
        raise ValueError('Unknown LIBERO suite')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    np.random.seed(seed)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    if not 0 <= task_id < suite.n_tasks:
        raise ValueError('Task index outside selected suite')
    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    if not 0 <= initial_state_id < len(states):
        raise ValueError('Initial-state index outside selected task')
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(seed)
        session = LiberoSession(env, states[initial_state_id], str(task.language), SUITE_HORIZONS[suite_name])
        state_array = np.asarray(states[initial_state_id])
        session.provenance = dict(suite=suite_name, task_id=task_id, initial_state_id=initial_state_id,
            seed=seed, bddl_path=str(bddl), bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
            initial_state_sha256=hashlib.sha256(state_array.tobytes()).hexdigest(),
            initial_state_shape=list(state_array.shape), initial_state_dtype=str(state_array.dtype))
        return session
    except BaseException:
        env.close()
        raise
