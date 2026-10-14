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
        return self.measured_observation()

    def measured_observation(self):
        """Keep Pi's trained proprioception; expose the OSC site's pose separately.

        Robosuite's robot0_eef_quat observes the hand body, whereas OSC controls
        the grip site, whose orientation can differ by a fixed gripper transform.
        """
        observation = measured_observation(self.raw, self.instruction)
        spec = self.eef_control()
        if spec['supported']:
            from scipy.spatial.transform import Rotation
            site = spec['eef_site']
            observation['eef_position'] = np.asarray(self.env.sim.data.get_site_xpos(site)).copy()
            observation['eef_quaternion_xyzw'] = Rotation.from_matrix(
                self.env.sim.data.get_site_xmat(site)).as_quat()
        return observation

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

    def eef_control(self):
        from .eef import controller_spec
        self._owner()
        try:
            robots = getattr(self.env, 'robots', None)
            if robots is None:
                robots = self.env.env.robots
            if len(robots) != 1:
                raise ValueError('EEF requires a single robot')
            return controller_spec(robots[0].controller)
        except (AttributeError, ValueError) as error:
            return dict(supported=False, reason=str(error))

    def execute_eef(self, target, proposal, steps, reference):
        from scipy.spatial.transform import Rotation
        from .eef import pose, target_action, validate_target
        self._owner()
        spec = self.eef_control()
        validate_target(target, reference, spec)
        values = action_chunk(proposal)
        if type(steps) is not int or not 1 <= steps <= min(5, len(values)):
            raise ValueError('EEF execution must select 1..5 proposal controls')
        trace, self.eef_trace = [], []
        for index in range(steps):
            # Same site read by OSC.update(); no sim.step/forward or privileged objects.
            site = spec['eef_site']
            current = dict(frame='MuJoCo world',
                position=np.asarray(self.env.sim.data.get_site_xpos(site)).tolist(),
                quaternion_xyzw=Rotation.from_matrix(
                    self.env.sim.data.get_site_xmat(site)).as_quat().tolist())
            p, r = pose(current)
            p0, r0 = pose(reference)
            if index == 0 and (np.linalg.norm(p-p0) > 1e-4 or (r*r0.inv()).magnitude() > 1e-4):
                raise ValueError('EEF observation does not match the live controller site')
            if np.linalg.norm(p-p0) > .05+1e-9 or (r*r0.inv()).magnitude() > .35+1e-9:
                raise ValueError('EEF measured motion exceeded the correction envelope')
            action = target_action(target, current, spec, values[index, 6])
            executed = self.execute(np.asarray([action]))
            trace.extend(executed)
            record = dict(step_id=self.tick, target=target, measured_before=current,
                          executed_action=executed[0].tolist())
            self.eef_trace.append(record)
            record['measured_after'] = dict(
                position=np.asarray(self.env.sim.data.get_site_xpos(site)).tolist(),
                quaternion_xyzw=Rotation.from_matrix(self.env.sim.data.get_site_xmat(site)).as_quat().tolist(),
                frame='MuJoCo world')
            # Check the last step too: a bounded target does not guarantee that
            # the physical plant stayed inside the correction envelope.
            p_after, r_after = pose(record['measured_after'])
            if np.linalg.norm(p_after-p0) > .05+1e-9 or (r_after*r0.inv()).magnitude() > .35+1e-9:
                raise ValueError('EEF measured motion exceeded the correction envelope after execution')
            if self.terminal:
                break
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
