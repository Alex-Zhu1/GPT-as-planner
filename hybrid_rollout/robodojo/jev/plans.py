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


def recovery_response_schema():
    from ..skill.schema import _obj, response_schema
    string = {'type': 'string', 'minLength': 1, 'maxLength': 4000}
    high_level = _obj(dict(request_id=string, mode={'type': 'string', 'enum': ['replan']},
        reason=string, subgoals={'type': 'array', 'minItems': 1, 'maxItems': 8,
            'items': _obj(dict(subgoal_id=deepcopy(string), instruction=deepcopy(string),
                               success_condition=deepcopy(string)))}))
    return {'anyOf': [response_schema(), high_level]}
