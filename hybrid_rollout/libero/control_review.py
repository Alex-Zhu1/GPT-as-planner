"""Read-only control diagnostics and bounded EEF decisions; no action editing."""
from copy import deepcopy
import hashlib
import json

from .contract import action_chunk
from .eef import validate_target
from .instruction_planner import EEF_ROLE
from ..robodojo.skill.schema import _obj


CONTROL_TOOL = dict(name='read_control_diagnostics', description=(
    'Read this paused review\'s immutable Pi prediction or measured EEF/controller context. '
    'Predicted controls are not executed motion. No file writes or action edits.'),
    inputSchema=_obj(dict(review_id={'type':'string'},
        record={'type':'string','enum':['pi_proposal','eef_context']},
        offset={'type':'integer','minimum':0,'maximum':9},
        limit={'type':'integer','minimum':1,'maximum':10})))


def control_record(request):
    native = request['native_request']
    scene = request['current_scene']
    if (native['observation_id'] != scene['observation_id'] or
            native['step_id'] != scene['step_id'] or native['student_instruction'] != request['current_subgoal']):
        raise ValueError('Control diagnostics must belong to the paused scene and current instruction')
    actions = native['proposal_actions']
    action_chunk(actions)
    return dict(review_id=native['request_id'], observation_id=scene['observation_id'],
        step_id=scene['step_id'], instruction=native['student_instruction'],
        prediction_id=native['prediction_id'], actions=deepcopy(actions),
        current_eef=deepcopy(native['current_eef']), controller=deepcopy(native['eef_control']))


def record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def max_eef_steps(request):
    return max(0, min(5, request['max_chunk_steps'], request['native_request']['max_chunk_steps'],
        request.get('execution_budget', {}).get('remaining_controls', 5)))


def diagnostics(request):
    record = control_record(request)
    return dict(review_id=record['review_id'], observation_id=record['observation_id'],
        step_id=record['step_id'], prediction_id=record['prediction_id'],
        instruction=record['instruction'], sha256=record_digest(record),
        predicted_controls=len(record['actions']), prediction_executed=False,
        tool=CONTROL_TOOL['name'], records=['pi_proposal','eef_context'],
        action_space='Panda OSC_POSE normalized xyz/rotation xyz/gripper; not meters or absolute poses',
        gripper_convention='-1 open, +1 close',
        eef_supported=record['controller'].get('supported') is True and max_eef_steps(request) > 0,
        max_steps=max_eef_steps(request), max_translation_m=.05, max_rotation_rad=.35,
        eef_gripper='explicit open or close; no inherited Pi gripper command')


def record_path(folder, review_id):
    if not isinstance(review_id, str) or not 1 <= len(review_id) <= 256:
        raise ValueError('Invalid review ID')
    return folder/'control_diagnostics'/f'{hashlib.sha256(review_id.encode()).hexdigest()}.json'


def archive_control_record(folder, request):
    record = control_record(request)
    path = record_path(folder, record['review_id'])
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Control record cannot be a symlink')
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError('Control review ID reused with different data')
    else:
        with path.open('x') as stream:
            json.dump(record, stream, allow_nan=False)
    return diagnostics(request)


def read_control_record(folder, args, active):
    if set(args) != {'review_id','record','offset','limit'} or args['review_id'] != active['review_id']:
        raise ValueError('Only the active paused review can be read')
    offset, limit = args['offset'], args['limit']
    if type(offset) is not int or not 0 <= offset <= 9 or type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError('Invalid diagnostic slice')
    path = record_path(folder, args['review_id'])
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Control record cannot be a symlink')
    record = json.loads(path.read_text())
    if record_digest(record) != active['sha256']:
        raise ValueError('Control record changed after review preparation')
    reply = dict(review_id=record['review_id'], observation_id=record['observation_id'],
                 step_id=record['step_id'], sha256=active['sha256'])
    if args['record'] == 'pi_proposal':
        reply.update(prediction_id=record['prediction_id'], instruction=record['instruction'],
            prediction_executed=False, action_space=active['action_space'],
            gripper_convention=active['gripper_convention'],
            controller=record['controller'], actions=record['actions'][offset:offset+limit],
            offset=offset, total_controls=len(record['actions']))
    elif args['record'] == 'eef_context':
        if offset != 0:
            raise ValueError('EEF context requires offset zero')
        reply.update(current_eef=record['current_eef'], controller=record['controller'],
                     max_steps=active['max_steps'], gripper=active['eef_gripper'])
    else:
        raise ValueError('Unknown control diagnostic record')
    return reply


def eef_schema(request):
    if request['native_request'].get('eef_control', {}).get('supported') is not True or max_eef_steps(request) == 0:
        return None
    vec = lambda n: dict(type='array', minItems=n, maxItems=n, items={'type':'number'})
    return _obj(dict(mode={'type':'string','enum':['eef']},
        request_id={'type':'string','enum':[request['native_request']['request_id']]},
        observation_id={'type':'string','enum':[request['current_scene']['observation_id']]},
        reason={'type':'string','minLength':1,'maxLength':4000},
        basis={'type':'string','enum':['execution_failed','intent_misaligned']},
        steps={'type':'integer','minimum':1,'maximum':max_eef_steps(request)},
        target=_obj(dict(position=vec(3), quaternion_xyzw=vec(4),
            frame={'type':'string','enum':['MuJoCo world']},
            gripper={'type':'string','enum':['open','close']}))))


def validate_eef_decision(decision, request):
    schema = eef_schema(request)
    if request.get('planner_role') != EEF_ROLE or schema is None or set(decision) != set(schema['properties']):
        raise ValueError('Bounded EEF is unavailable or has unexpected fields')
    for key in ('mode','request_id','observation_id','basis'):
        if decision[key] not in schema['properties'][key]['enum']:
            raise ValueError('Stale EEF identity or invalid correction basis')
    if type(decision['steps']) is not int or not 1 <= decision['steps'] <= schema['properties']['steps']['maximum']:
        raise ValueError('EEF exceeds the monitored chunk bound')
    if not isinstance(decision['reason'], str) or not decision['reason'].strip() or len(decision['reason']) > 4000:
        raise ValueError('EEF needs a bounded visual justification')
    native = request['native_request']
    if native['observation_id'] != request['current_scene']['observation_id'] or native['step_id'] != request['current_scene']['step_id']:
        raise ValueError('EEF requires a fresh paused reference')
    validate_target(decision['target'], native['current_eef'], native['eef_control'])
    if decision['target']['gripper'] not in ('open','close'):
        raise ValueError('EEF requires explicit open or close; no inherited Pi controls')


def executable_eef(decision, native):
    """Adapt the explicit Astra diagnosis to the existing native audit contract."""
    from ..robodojo.jev.runner import automatic_response
    result = automatic_response(native, decision['steps'], {})
    result.update(mode='eef', reason=decision['reason'], target=deepcopy(decision['target']),
                  decision_author='planner', evaluation_method='pi05_valen_replan')
    assessment = result['assessment']
    assessment['execution_evidence'] = decision['reason']
    assessment['intent_evidence'] = decision['reason']
    if decision['basis'] == 'execution_failed':
        assessment['execution_status'] = 'failed'
    else:
        assessment['intent_status'] = 'misaligned'
    return result
