"""LIBERO OSC_POSE controls are not RoboDojo dual-arm joint targets."""
import numpy as np

from ..robodojo.robodojo_server.gate_assessment import validate_assessment

PROFILE = 'libero_panda_osc_pose'
CONFIG = 'pi05_libero'
HORIZON = 10
ACTION_DIM = 7
MAX_EDIT = .25  # Normalized OSC inputs, NOT meters or radians.
RECOVERY_CONTRACT = (
    'Return exactly request_id, run_id, episode_id, plan_revision, subgoal_id, program_id, decision. '
    'Echo outer identities; decision.request_id must echo native_request.request_id. '
    'LIBERO Panda OSC_POSE: seven normalized controls in [-1,1], '
    'translation xyz, rotation xyz, gripper (-1 open, +1 close). '
    'These are controller inputs, not absolute poses or joint angles. No FK preview is supplied. '
    'Modes: student (retry current proposal), edit, replan, stop. Never use mode=eef or dual-arm targets. '
    'student/edit/stop need steps, reason and assessment following assessment_example in native_request. '
    'edit additionally needs edit={delta:[six finite numbers each in [-0.25,0.25]], '
    'gripper:"keep"|"open"|"close"}; delta is added to each selected proposal row, then '
    'clipped to controller bounds. An edit requires assessment.execution_status=failed '
    'or assessment.intent_status=misaligned and visible evidence. '
    'For replan use only request_id, mode, reason, subgoals (1..8 objects with unique subgoal_id, '
    'instruction, success_condition). Do not execute code or reset the scene. '
    'Only the simulator determines original-task success. Respect max_chunk_steps.'
)


def action_chunk(value, *, horizon=None):
    array = np.asarray(value)
    if (array.dtype.kind not in 'fi' or array.ndim != 2 or array.shape[1] != ACTION_DIM
            or not 1 <= array.shape[0] <= HORIZON or not np.isfinite(array).all()
            or (horizon is not None and array.shape[0] != horizon)):
        raise ValueError('Expected finite numeric LIBERO [N,7] actions (N <= 10)')
    return array.astype(np.float32)


def validate_response(response, request):
    if not isinstance(response, dict) or response.get('request_id') != request['request_id']:
        raise ValueError('Stale LIBERO decision')
    mode = response.get('mode')
    if mode not in ('student', 'edit', 'stop'):
        raise ValueError('Supported native LIBERO modes: student/edit/stop; absolute EEF is unsupported')
    required = {'request_id', 'mode', 'steps', 'reason', 'assessment'}
    allowed = required | {'decision_author', 'evaluation_method'} | ({'edit'} if mode == 'edit' else set())
    if not required <= set(response) or set(response) - allowed:
        raise ValueError('Unexpected or missing LIBERO decision fields')
    steps = response['steps']
    if type(steps) is not int or not 1 <= steps <= min(5, request['max_chunk_steps']):
        raise ValueError('Decision exceeds monitored control bound')
    if not isinstance(response['reason'], str) or not response['reason'].strip() or len(response['reason']) > 4000:
        raise ValueError('A bounded reason is required')
    validate_assessment(response, request)
    if mode == 'edit':
        edit = response.get('edit')
        if not isinstance(edit, dict) or set(edit) != {'delta', 'gripper'}:
            raise ValueError('edit requires delta and gripper')
        delta = edit['delta']
        if (not isinstance(delta, list) or len(delta) != 6 or
                any(type(x) not in (int, float) or not np.isfinite(x) or abs(x) > MAX_EDIT for x in delta)):
            raise ValueError('edit.delta must contain six finite normalized offsets bounded by 0.25')
        if edit['gripper'] not in ('keep', 'open', 'close'):
            raise ValueError('Unknown gripper edit')


def execution_actions(proposal, response):
    values = action_chunk(proposal)[:response['steps']].copy()
    if len(values) != response['steps']:
        raise ValueError('Proposal is shorter than the selected chunk')
    if response['mode'] == 'edit':
        values[:, :6] += response['edit']['delta']
        grip = response['edit']['gripper']
        if grip != 'keep':
            values[:, 6] = -1. if grip == 'open' else 1.
    # Match native OSC input clipping; record both raw proposals and sent actions.
    return np.clip(values, -1., 1.)
