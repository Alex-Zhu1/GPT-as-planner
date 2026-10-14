"""Single-arm geometric targets for robosuite 1.4 OSC_POSE (no physics preview).

Position is in MuJoCo world meters; xyzw quaternions describe the controlled
site. OSC uses R_goal = Exp(delta_world) @ R_measured. Never treat normalized
actions as meters or reuse a command computed before an executed control.
"""
import numpy as np
from scipy.spatial.transform import Rotation

MAX_TRANSLATION = .05
MAX_ROTATION = .35


def vector(value, size):
    a = np.asarray(value)
    if a.shape != (size,) or a.dtype.kind not in 'fi' or not np.isfinite(a).all():
        raise ValueError('Expected finite numeric EEF vector')
    return a.astype(float)


def pose(value):
    if value.get('frame') != 'MuJoCo world':
        raise ValueError('EEF frame must be MuJoCo world')
    p = vector(value['position'], 3)
    q = vector(value['quaternion_xyzw'], 4)
    if abs(np.linalg.norm(q)-1.) > 1e-4:
        raise ValueError('EEF quaternion_xyzw must be unit length')
    return p, Rotation.from_quat(q)


def controller_spec(controller):
    """Fail closed for unverified controller semantics; read configuration only."""
    if (controller.name != 'OSC_POSE' or not controller.use_delta
            or not controller.use_ori or controller.impedance_mode != 'fixed'):
        raise ValueError('EEF requires fixed-impedance delta OSC_POSE')
    if controller.position_limits is not None or controller.orientation_limits is not None:
        raise ValueError('EEF does not support additional controller pose limits')
    lo, hi = vector(controller.input_min, 6), vector(controller.input_max, 6)
    out_lo, out_hi = vector(controller.output_min, 6), vector(controller.output_max, 6)
    if (not np.array_equal(lo, -np.ones(6)) or not np.array_equal(hi, np.ones(6))
            or not np.allclose(out_lo, -out_hi, rtol=0, atol=1e-12) or np.any(out_hi <= 0)):
        raise ValueError('EEF requires symmetric normalized OSC scaling')
    if not np.isfinite(controller.control_freq) or controller.control_freq <= 0:
        raise ValueError('EEF requires a positive controller frequency')
    return dict(supported=True, frame='MuJoCo world', quaternion_order='xyzw',
                rotation_composition='Exp(delta_world) @ R_measured',
                output_max=out_hi.tolist(), eef_site=controller.eef_name,
                control_dt_s=1./float(controller.control_freq),
                max_translation_m=MAX_TRANSLATION, max_rotation_rad=MAX_ROTATION)


def validate_target(target, current, spec):
    if not spec or spec.get('supported') is not True:
        raise ValueError('EEF unsupported by this live controller')
    if not isinstance(target, dict) or set(target) != {'position', 'quaternion_xyzw', 'frame', 'gripper'}:
        raise ValueError('EEF target requires single-arm position/quaternion_xyzw/frame/gripper')
    p, r = pose(target)
    p0, r0 = pose(current)
    if np.linalg.norm(p-p0) > MAX_TRANSLATION+1e-9 or (r*r0.inv()).magnitude() > MAX_ROTATION+1e-9:
        raise ValueError('EEF target exceeds 5 cm / 0.35 rad correction bound')
    if target['gripper'] not in ('keep', 'open', 'close'):
        raise ValueError('Unknown EEF gripper command')


def target_action(target, current, spec, proposal_gripper):
    p, r = pose(target)
    measured, orientation = pose(current)
    error = np.r_[p-measured, (r*orientation.inv()).as_rotvec()]
    scale = vector(spec['output_max'], 6)
    # Scale each vector uniformly when saturated, preserving its direction.
    normalized = error / scale
    for segment in (slice(0, 3), slice(3, 6)):
        normalized[segment] /= max(1., float(np.max(np.abs(normalized[segment]))))
    grip = {'open': -1., 'close': 1.}.get(target['gripper'], float(np.clip(proposal_gripper, -1., 1.)))
    return np.r_[normalized, grip].astype(np.float32)


def command_preview(proposal, current, spec):
    """Nominal command integration, NOT predicted robot/object motion or FK."""
    p, r = pose(current)
    scale = vector(spec['output_max'], 6)
    result = []
    for index, row in enumerate(proposal):
        delta = np.clip(row[:6], -1., 1.) * scale
        p = p + delta[:3]
        r = Rotation.from_rotvec(delta[3:]) * r
        result.append(dict(index=index, position=p.tolist(), quaternion_xyzw=r.as_quat().tolist(),
                           gripper=float(np.clip(row[6], -1., 1.))))
    return dict(interpretation='nominal integrated OSC commands; not FK, achieved poses, or object prediction',
                frame='MuJoCo world', trajectory=result)
