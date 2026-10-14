"""Versioned language plans shared by the controller, compact and API transports."""
from copy import deepcopy
from pathlib import Path
import re

from .monitor import IDENTITY, visual_frame
from .task_instructions import ORGANIZE_TABLE, replan_candidates


PLAN_VERSION = 4
PLANNER_ROLE = 'robodojo_subgoal_planner'
PROMPT_PATH = Path(__file__).with_name('subgoal_planner.md')
PI_INSTRUCTION_CONTRACT = dict(
    max_words_per_requirement=24, max_characters_per_requirement=160,
    max_sentences_per_requirement=1, max_characters=600,
    style='Short action, target and destination; necessary orientation only. '
          'Keep verification and preservation commentary in completion_checks, reason and evidence.')
_EXTRA_COMMENTARY = re.compile(
    r'\b(?:make sure|ensure|verify|check that|remain(?:s)? stable|stable and supported|'
    r'without (?:disturbing|disrupting|moving|touching)|'
    r'(?:do not|don\x27t|avoid) (?:disturb|disturbing|disrupt|disrupting))\b', re.I)


def validate_pi_instruction(instruction, requirement_count):
    """Reject verbose model text; never silently truncate or rewrite Pi input."""
    limits = PI_INSTRUCTION_CONTRACT
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Pi instruction must be nonempty short action language')
    sentences = [part for part in re.split(r'[.!?;]+', instruction) if part.strip()]
    if (len(instruction) > min(limits['max_characters'], limits['max_characters_per_requirement'] * requirement_count)
            or len(instruction.split()) > limits['max_words_per_requirement'] * requirement_count
            or len(sentences) > limits['max_sentences_per_requirement'] * requirement_count):
        raise ValueError('Pi instruction must use at most 24 words, 160 characters and one sentence per requirement')
    if _EXTRA_COMMENTARY.search(instruction):
        raise ValueError('Keep verification and preservation commentary out of the Pi instruction')


def is_subgoal_request(request):
    return request.get('planner_role') == PLANNER_ROLE


def planning_context(task, instruction, context):
    """Public task semantics, never per-episode object truth or native scores."""
    context = deepcopy(context)
    if task == 'organize_table' and instruction == ORGANIZE_TABLE:
        context.update(task_contract='organize_table_four_items_v1', ordered_requirements=False,
            process='Arrange the mouse, keyboard, figurine and alarm clock, then return both arms.',
            instruction_discrepancy=('This installed task has four movable desk items and a static drawer. '
                'The source sentence about opening the drawer and storing miscellaneous items does not '
                'apply to this scene. Plan the four placements and arm return. Preserve the original '
                'instruction in the audit; do not invent a drawer manipulation stage.'),
            contract_source='RoboDojo/task/RoboDojo/tasks/organize_table.py:run_reward and task configuration')
    return context


def task_requirements(instruction, context):
    """Public source clauses; unknown tasks retain their complete instruction."""
    rows = replan_candidates(instruction) or [dict(subgoal_id='whole-task',
        instruction=instruction, success_condition='All requirements of the original instruction are visibly satisfied.')]
    result = [dict(requirement_id=row['subgoal_id'], instruction=row['instruction'],
                   completion_check=row['success_condition']) for row in rows]
    if context.get('task_contract') == 'organize_table_four_items_v1':
        result = result[:4]
        result[0]['completion_check'] = 'The released mouse is stable on the mouse pad and faces the intended direction.'
        result[1]['completion_check'] = 'The released keyboard is stable in the marked frame, aligned and face up.'
        result[2].update(instruction='Put the figurine upright on the stand.',
            completion_check='The released figurine is upright, stable and supported on the stand, not lying on the table.')
        result[3].update(instruction='Place the alarm clock upright on the drawer.',
            completion_check='The released alarm clock is upright and stable on top of the drawer cabinet.')
    if context.get('requires_arm_return'):
        if not any(row['requirement_id'] == 'reset-arm' for row in result):
            result.append(dict(requirement_id='reset-arm', instruction='Reset the robot arm.',
                completion_check='Both arms have released the objects and returned near their episode-start poses.'))
        else:
            result[-1]['completion_check'] = 'Both arms have released the objects and returned near their episode-start poses.'
    return result


def goal_for(requirements, identifier):
    """Compose reviewed clauses without adding model-authored motion instructions."""
    return dict(subgoal_id=identifier,
        instruction=' '.join(row['instruction'] for row in requirements),
        requirements=[row['requirement_id'] for row in requirements],
        completion_checks=[row['completion_check'] for row in requirements])


def _object(properties):
    return dict(type='object', additionalProperties=False, required=list(properties), properties=properties)


def planning_schema(request):
    rows = request['all_requirements']
    initial = request['review_kind'] == 'initial_plan'
    modes = ['plan'] if initial else ['plan', 'keep_plan', 'advance']
    goal = _object(dict(subgoal_id=dict(type='string', enum=['g1', 'g2', 'g3']),
        instruction=dict(type='string', minLength=1, maxLength=PI_INSTRUCTION_CONTRACT['max_characters'],
            description='At most 24 words, 160 characters and one sentence per covered requirement. '
                        'Action/target/destination only; no verification or preservation commentary.'),
        requirements=dict(type='array', minItems=1, maxItems=len(rows),
            items=dict(type='string', enum=[r['requirement_id'] for r in rows])),
        completion_checks=dict(type='array', minItems=1, maxItems=len(rows),
            items=dict(type='string', enum=[r['completion_check'] for r in rows]))))
    return _object(dict(decision=_object(dict(
        mode=dict(type='string', enum=modes),
        reason=dict(type='string', minLength=1, maxLength=1200),
        assessments=dict(type='array', minItems=0 if initial else len(rows), maxItems=0 if initial else len(rows),
            items=_object(dict(requirement_id=dict(type='string', enum=[r['requirement_id'] for r in rows]),
                status=dict(type='string', enum=['satisfied', 'pending', 'uncertain']),
                evidence=dict(type='string', minLength=1, maxLength=600)))),
        subgoals=dict(type='array', minItems=1 if initial else 0, maxItems=3, items=goal)))))


def validate_planning_decision(decision, request):
    if not is_subgoal_request(request) or request.get('version') != PLAN_VERSION:
        raise ValueError('Expected the version-4 subgoal planning protocol')
    if not isinstance(decision, dict) or set(decision) != {'mode', 'reason', 'assessments', 'subgoals'}:
        raise ValueError('Planner returns language goals and requirement assessments; no robot actions')
    reason = decision['reason']
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1200:
        raise ValueError('A bounded planning reason is required')
    initial = request['review_kind'] == 'initial_plan'
    rows, assessments = request['all_requirements'], decision['assessments']
    if not isinstance(assessments, list) or len(assessments) != (0 if initial else len(rows)):
        raise ValueError('A review must assess every original requirement')
    satisfied = set()
    for row, assessment in zip(rows, assessments):
        if (not isinstance(assessment, dict) or set(assessment) != {'requirement_id', 'status', 'evidence'}
                or assessment['requirement_id'] != row['requirement_id']
                or assessment['status'] not in ('satisfied', 'pending', 'uncertain')
                or not isinstance(assessment['evidence'], str) or not 0 < len(assessment['evidence'].strip()) <= 600):
            raise ValueError('Each ordered requirement assessment needs a status and visual evidence')
        if assessment['status'] == 'satisfied': satisfied.add(row['requirement_id'])
    needed = [r['requirement_id'] for r in rows if r['requirement_id'] not in satisfied]
    if 'reset-arm' in satisfied and needed:
        raise ValueError('Arm return cannot be completed while other work remains')
    goals, mode = decision['subgoals'], decision['mode']
    if mode in ('keep_plan', 'advance'):
        if initial or goals != [] or not request.get('remaining_plan'):
            raise ValueError('keep_plan/advance require an existing plan')
        if request['execution_budget']['stage_executed_controls'] <= 0:
            raise ValueError('A stage decision needs executed evidence')
        queued = {r for g in request['remaining_plan'] for r in g['requirements']}
        if not set(needed) <= queued:
            raise ValueError('A reopened requirement needs a replacement plan')
        done = set(request['remaining_plan'][0]['requirements']) <= satisfied
        if (mode == 'advance') != done:
            raise ValueError('Only Astra evidence satisfying every current-stage check permits advance')
        return
    if mode != 'plan' or not isinstance(goals, list) or not 1 <= len(goals) <= 3:
        raise ValueError('Astra must plan one to three ordered subgoals')
    by_id = {r['requirement_id']: r for r in rows}
    covered = []
    for index, goal in enumerate(goals, 1):
        if not isinstance(goal, dict) or set(goal) != {'subgoal_id', 'instruction', 'requirements', 'completion_checks'}:
            raise ValueError('Each subgoal needs exact instruction, requirement coverage and completion checks')
        coverage = goal['requirements']
        if not isinstance(coverage, list) or not coverage or any(not isinstance(r, str) or r not in by_id for r in coverage):
            raise ValueError('Empty subgoal coverage')
        if (goal['subgoal_id'] != f'g{index}' or not isinstance(goal['instruction'], str)
                or not 0 < len(goal['instruction'].strip()) <= 600
                or goal['completion_checks'] != [by_id[r]['completion_check'] for r in coverage]):
            raise ValueError('Subgoals require bounded language and exact requirement completion checks')
        validate_pi_instruction(goal['instruction'], len(coverage))
        covered.extend(coverage)
    if len(set(covered)) != len(covered) or set(covered) != set(needed):
        raise ValueError('Plan omitted or duplicated unfinished original-task requirements')
    if request.get('task_context', {}).get('ordered_requirements', True) and covered != needed:
        raise ValueError('This task requires source requirement order')
    if 'reset-arm' in covered and covered[-1] != 'reset-arm':
        raise ValueError('Arm return must remain last')


def validate_planning_response(response, request):
    if not isinstance(response, dict) or set(response) != {'request_id', *IDENTITY, 'decision'}:
        raise ValueError('Invalid subgoal plan response envelope')
    for key in ('request_id', *IDENTITY):
        if type(response[key]) is not type(request[key]) or response[key] != request[key]:
            raise ValueError('Stale subgoal plan identity')
    validate_planning_decision(response['decision'], request)
    return deepcopy(response['decision'])


def validate_planning_payload(payload, request):
    if not isinstance(payload, dict) or set(payload) != {'decision'}:
        raise ValueError('Expected exactly one subgoal planning decision')
    validate_planning_decision(payload['decision'], request)
    return payload['decision']


def planning_packet(request):
    """Attach references and recent endpoints; no proposal, reward or hidden state."""
    if request.get('version') != PLAN_VERSION:
        raise ValueError('Expected the version-4 subgoal planning protocol')
    frames = [*request.get('reference_frames', []), *request.get('evidence_frames', [])[-2:], request['current_scene']]
    seen, images, scenes = set(), [], []
    for frame in frames:
        if frame['observation_id'] in seen:
            continue
        seen.add(frame['observation_id'])
        scene = visual_frame(frame)
        scene.pop('images')
        scene['image_indices'] = []
        for im in frame['images']:
            scene['image_indices'].append(dict(index=len(images), camera=im['camera']))
            images.append(deepcopy(im))
        scenes.append(scene)
    fields = ('version', 'planner_role', 'review_kind', 'original_task', 'task_context',
              'remaining_requirements', 'completed_requirements', 'remaining_plan', 'current_subgoal',
              'review_reason', 'monitor_evidence', 'all_requirements', 'previous_assessments', 'execution_budget',
              'instruction_history')
    packet = {key: deepcopy(request[key]) for key in fields if key in request}
    packet['pi_instruction_contract'] = deepcopy(PI_INSTRUCTION_CONTRACT)
    packet['scenes'] = scenes
    return packet, images


def planning_instructions(request):
    return PROMPT_PATH.read_text(), planning_schema(request)
