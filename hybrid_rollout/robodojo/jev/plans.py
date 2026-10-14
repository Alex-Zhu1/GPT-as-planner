"""Declarative high-level recovery plans alongside the native action contract."""
from copy import deepcopy
import re


def validate_decision(decision, request, max_chunk_steps, validate_native):
    if not isinstance(decision, dict):
        raise ValueError('Recovery decision must be an object')
    if decision.get('mode') != 'replan':
        allowed = {'request_id', 'mode', 'steps', 'reason', 'assessment', 'target', 'edit'}
        if set(decision) - allowed:
            raise ValueError('Unknown recovery decision fields; executable code is not accepted')
        validate_native(decision, request)
        if decision['mode'] != 'stop' and decision['steps'] > max_chunk_steps:
            raise ValueError('Planner cannot exceed the monitored chunk bound')
        return
    if set(decision) != {'request_id', 'mode', 'reason', 'subgoals'}:
        raise ValueError('High-level replan accepts only declarative subgoals')
    if decision['request_id'] != request['request_id']:
        raise ValueError('High-level replan belongs to an old proposal')
    if not isinstance(decision['reason'], str) or not decision['reason'].strip() or len(decision['reason']) > 4000:
        raise ValueError('A bounded replan explanation is required')
    goals = decision['subgoals']
    if not isinstance(goals, list) or not 1 <= len(goals) <= 8:
        raise ValueError('A high-level replan needs 1..8 subgoals')
    ids = {'original-task'}
    for goal in goals:
        if not isinstance(goal, dict) or set(goal) != {'subgoal_id', 'instruction', 'success_condition'}:
            raise ValueError('Each subgoal needs identity, instruction and observable success condition')
        for key, value in goal.items():
            if not isinstance(value, str) or not value.strip() or len(value) > 4000:
                raise ValueError('Subgoal fields must be bounded nonempty text')
        identifier = goal['subgoal_id']
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', identifier) or identifier in ids:
            raise ValueError('Subgoal IDs must be unique; original-task is reserved')
        ids.add(identifier)


def validate_recovery_decision(decision, request, validate_native):
    if request.get('planner_role') in ('instruction_only', 'instruction_eef'):
        if request.get('recovery_policy') != 'single_instruction' or not request.get('board_review_required'):
            raise ValueError('Instruction planner requires single_instruction and board review')
        if (request.get('planner_role') == 'instruction_eef' and isinstance(decision, dict)
                and decision.get('mode') == 'eef'):
            from ...libero.control_review import validate_eef_decision
            validate_eef_decision(decision, request)
            return
        if not isinstance(decision, dict) or decision.get('mode') not in (
                'keep_instruction', 'replan', 'resume_original', 'stop'):
            raise ValueError('Instruction planner cannot select robot controls')
        if decision['mode'] in ('keep_instruction', 'stop'):
            if set(decision) != {'request_id', 'mode', 'reason'}:
                raise ValueError('Instruction decision accepts only request_id, mode and reason')
            if decision['request_id'] != request['native_request']['request_id']:
                raise ValueError('Stale instruction review')
            if not isinstance(decision['reason'], str) or not decision['reason'].strip() or len(decision['reason']) > 4000:
                raise ValueError('A bounded instruction decision reason is required')
            return
    if validate_plan_operation(decision, request):
        return
    validate_decision(decision, request['native_request'], request['max_chunk_steps'], validate_native)
    if (request.get('recovery_policy') == 'single_instruction'
            and decision['mode'] == 'replan' and len(decision['subgoals']) != 1):
        raise ValueError('Single-instruction recovery requires exactly one instruction; no future queue')


def recovery_response_schema():
    from ..skill.schema import _obj, response_schema
    string = {'type': 'string', 'minLength': 1, 'maxLength': 4000}
    high_level = _obj(dict(request_id=string, mode={'type': 'string', 'enum': ['replan']},
        reason=string, subgoals={'type': 'array', 'minItems': 1, 'maxItems': 8,
            'items': _obj(dict(subgoal_id=deepcopy(string), instruction=deepcopy(string),
                               success_condition=deepcopy(string)))}))
    return {'anyOf': [response_schema(), high_level]}


def advance_schema(request):
    """Capability-gated plan operation; never a native robot command."""
    from ..skill.schema import _obj
    mode = 'resume_original' if request.get('recovery_policy') == 'single_instruction' else 'advance_subgoal'
    cap = request.get('plan_operations', {}).get(mode)
    if not cap or request.get('review_kind') == 'uncertainty':
        return None
    return _obj(dict(
        mode={'type':'string','enum':[mode]},
        request_id={'type':'string','enum':[request['native_request']['request_id']]},
        plan_revision={'type':'integer','enum':[request['plan_revision']]},
        subgoal_id={'type':'string','enum':[cap['subgoal_id']]},
        reason={'type':'string','minLength':1,'maxLength':4000},
        evidence_observation_ids={'type':'array','minItems':1,'maxItems':12,
            'items':{'type':'string','enum':cap['evidence_observation_ids']}}))


def validate_plan_operation(decision, request):
    if not isinstance(decision, dict) or decision.get('mode') not in ('advance_subgoal', 'resume_original'):
        return False
    schema = advance_schema(request)
    if (schema is None or set(decision) != set(schema['properties'])
            or decision['mode'] != schema['properties']['mode']['enum'][0]):
        raise ValueError('advance_subgoal is unavailable or has invalid fields')
    for key in ('request_id','plan_revision','subgoal_id'):
        expected = schema['properties'][key]['enum'][0]
        if type(decision[key]) is not type(expected) or decision[key] != expected:
            raise ValueError('Stale advance_subgoal identity')
    if not isinstance(decision['reason'],str) or not decision['reason'].strip() or len(decision['reason'])>4000:
        raise ValueError('advance_subgoal requires a bounded reason')
    refs=decision['evidence_observation_ids']
    allowed=schema['properties']['evidence_observation_ids']['items']['enum']
    if not isinstance(refs,list) or not 1<=len(refs)<=12 or any(not isinstance(r,str) or r not in allowed for r in refs):
        raise ValueError('advance_subgoal must cite supplied current-plan evidence')
    return True
