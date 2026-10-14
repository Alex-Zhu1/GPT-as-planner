"""Version-5 atomic goal board and validated short language choices."""
from copy import deepcopy
from pathlib import Path
import re

from .monitor import IDENTITY, visual_frame
from .task_instructions import ORGANIZE_TABLE, replan_candidates

PLAN_VERSION = 5
PLANNER_ROLE = 'robodojo_subgoal_planner'
PROMPT_PATH = Path(__file__).with_name('subgoal_planner.md')
PI_INSTRUCTION_CONTRACT = dict(max_words=24, max_characters=160, max_sentences=1,
    selection='Choose an exact instruction_options entry for one goal; never append clauses.')
MAX_EVIDENCE_FRAMES = 8
MAX_FORMAT_REPAIRS = 2
DISPATCH_POLICY = dict(version=1, board_order='Stable requirement order, not execution order.',
    selection='Astra selects the first plan card; unfinished independent goals may be deferred.',
    ineffective='Yield to an unserved independent goal, otherwise the least recently served alternative.',
    retry='Change the last ineffective executed wording when alternatives exist; deferred wording may stay unchanged.')

class PlanningFormatError(ValueError):
    """A model answer can be corrected at the same paused scene; transport faults cannot."""
    def __init__(self, message, rejected=None):
        super().__init__(message)
        self.rejected = deepcopy(rejected)


def validate_pi_instruction(instruction, requirement_count=1):
    if requirement_count != 1:
        raise ValueError('Pi executes exactly one atomic goal, never grouped requirements')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Pi instruction must be nonempty short action language')
    if (len(instruction) > 160 or len(instruction.split()) > 24
            or len([p for p in re.split(r'[.!?;]+', instruction) if p.strip()]) != 1):
        raise ValueError('Pi instruction must use at most 24 words, 160 characters and one sentence total')


# Host-owned public language choices. These are not observations or claims that
# a recovery has been learned by Pi. Astra chooses and changes wording, not truth.
_PLACEMENT_OPTIONS = {
    'mouse-on-pad': ['Place the mouse on the mouse pad.',
        'Pick up the mouse and put it on the mouse pad.', 'Move the mouse onto the mouse pad.'],
    'keyboard-in-frame': ['Push the keyboard into the frame.',
        'Place the keyboard face up inside the marked frame.', 'Move the keyboard into the marked frame.'],
    'figurine-on-stand': ['Put the figurine upright on the stand.',
        'Pick up the fallen figurine and put it upright on the stand.',
        'Lift the fallen figurine onto the stand with its base down.',
        'Stand the figurine upright on its base on the stand.'],
    'clock-on-drawer': ['Place the alarm clock upright on the drawer.',
        'Pick up the alarm clock and put it upright on the drawer cabinet.',
        'Stand the alarm clock upright on top of the drawer.'],
    'reset-arm': ['Reset both robot arms.', 'Release the objects and return both arms to their starting poses.'],
}


def instruction_options(row):
    options = _PLACEMENT_OPTIONS.get(row['requirement_id'], [row['instruction']])
    return list(options)


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


def goal_for(requirements, identifier=None):
    if len(requirements) != 1:
        raise ValueError('One board goal per requirement; grouped instructions are not supported')
    row = requirements[0]
    return dict(subgoal_id=row['requirement_id'], instruction=instruction_options(row)[0],
                requirements=[row['requirement_id']], completion_checks=[row['completion_check']])


def _object(properties):
    return dict(type='object', additionalProperties=False, required=list(properties), properties=properties)


def planning_schema(request):
    rows = request['all_requirements']
    for row in rows:
        for text in instruction_options(row):
            validate_pi_instruction(text)
    initial = request['review_kind'] == 'initial_plan'
    # Bind a goal's instruction, ID and immutable completion check together.
    goal = dict(anyOf=[_object(dict(
        subgoal_id=dict(type='string', enum=[row['requirement_id']]),
        instruction=dict(type='string', enum=instruction_options(row)),
        requirements=dict(type='array', minItems=1, maxItems=1,
                          items=dict(type='string', enum=[row['requirement_id']])),
        completion_checks=dict(type='array', minItems=1, maxItems=1,
                               items=dict(type='string', enum=[row['completion_check']])))) for row in rows])
    return _object(dict(decision=_object(dict(
        mode=dict(type='string', enum=['plan'] if initial else ['plan', 'keep_plan', 'advance']),
        reason=dict(type='string', minLength=1, maxLength=1200),
        assessments=dict(type='array', minItems=0 if initial else len(rows), maxItems=0 if initial else len(rows),
            items=_object(dict(requirement_id=dict(type='string', enum=[r['requirement_id'] for r in rows]),
                status=dict(type='string', enum=['satisfied', 'pending', 'uncertain']),
                execution=dict(type='string', enum=['not_attempted', 'progressing', 'ineffective', 'uncertain', 'complete']),
                evidence=dict(type='string', minLength=1, maxLength=600)))),
        subgoals=dict(type='array', minItems=1 if initial else 0, maxItems=len(rows), items=goal)))))


def dispatch_candidates(request, satisfied):
    """Fair alternatives after an ineffective attempt; never infer physical failure."""
    current = request.get('remaining_plan', [])
    if not current or request.get('task_context', {}).get('ordered_requirements', True):
        return []
    candidates = [r['requirement_id'] for r in request['all_requirements']
                  if r['requirement_id'] not in {*satisfied, current[0]['subgoal_id'], 'reset-arm'}]
    history = request.get('instruction_history', [])
    last_served = {key: max((h['end_step'] for h in history
                            if key in h['requirements'] and h['executed_controls'] > 0), default=-1)
                   for key in candidates}
    oldest = min(last_served.values(), default=-1)
    return [key for key in candidates if last_served[key] == oldest]


def _validate_decision(decision, request):
    if not isinstance(decision, dict) or set(decision) != {'mode', 'reason', 'assessments', 'subgoals'}:
        raise ValueError('Return mode, reason, assessments and atomic subgoals; no robot actions')
    if not isinstance(decision['reason'], str) or not 0 < len(decision['reason'].strip()) <= 1200:
        raise ValueError('A bounded planning reason is required')
    initial = request['review_kind'] == 'initial_plan'
    rows, assessments = request['all_requirements'], decision['assessments']
    if not isinstance(assessments, list) or len(assessments) != (0 if initial else len(rows)):
        raise ValueError('A review must assess every original requirement')
    satisfied, ineffective = set(), set()
    for row, assessment in zip(rows, assessments):
        if (not isinstance(assessment, dict) or set(assessment) != {'requirement_id', 'status', 'execution', 'evidence'}
                or assessment['requirement_id'] != row['requirement_id']
                or assessment['status'] not in ('satisfied', 'pending', 'uncertain')
                or assessment['execution'] not in ('not_attempted', 'progressing', 'ineffective', 'uncertain', 'complete')
                or not isinstance(assessment['evidence'], str) or not 0 < len(assessment['evidence'].strip()) <= 600):
            raise ValueError('Each board assessment needs an ordered ID, status, execution feedback and evidence')
        if (assessment['status'] == 'satisfied') != (assessment['execution'] == 'complete'):
            raise ValueError('Only satisfied requirements have complete execution feedback')
        if assessment['execution'] == 'ineffective':
            if assessment['status'] != 'pending':
                raise ValueError('Confirmed ineffective attempts must be pending, not uncertain')
            ineffective.add(row['requirement_id'])
        if assessment['status'] == 'satisfied': satisfied.add(row['requirement_id'])
    needed = [r['requirement_id'] for r in rows if r['requirement_id'] not in satisfied]
    if 'reset-arm' in satisfied and needed:
        raise ValueError('Arm return cannot be completed while other work remains')
    goals, mode = decision['subgoals'], decision['mode']
    current = request.get('remaining_plan', [])
    by_id = {r['requirement_id']: r for r in rows}
    # Also validate retained queues: no legacy grouped or unchecked instruction
    # can bypass the contract through keep_plan/advance.
    for goal in current:
        _validate_goal(goal, by_id)
    active = current[0]['subgoal_id'] if current else None
    executed = request.get('execution_budget', {}).get('stage_executed_controls', 0) > 0
    yield_to = dispatch_candidates(request, satisfied) if active in ineffective and executed else []
    # Only actual Pi execution can associate a failed attempt with its wording.
    # Incidental manipulation of another object does not invalidate an untried card.
    failed_wording = {}
    for attempt in request.get('instruction_history', []):
        if attempt['executed_controls'] > 0:
            for key in attempt['requirements']:
                failed_wording.pop(key, None)
                reviews = attempt.get('reviews', [])
                if reviews and reviews[-1]['execution'] == 'ineffective':
                    failed_wording[key] = attempt['instruction']
    if active in ineffective and executed:
        failed_wording[active] = current[0]['instruction']

    def validate_selection(goal):
        key = goal['subgoal_id']
        if yield_to and key not in yield_to:
            raise ValueError(f'Ineffective active goal must yield: return plan selecting one of {yield_to}; retain deferred unfinished cards')
        if (len(instruction_options(by_id[key])) > 1
                and goal['instruction'] == failed_wording.get(key)):
            raise ValueError('Change the short description before retrying its ineffective executed instruction')
    if mode in ('keep_plan', 'advance'):
        if initial or goals != [] or not current:
            raise ValueError('keep_plan/advance require an existing board queue')
        if request['execution_budget']['stage_executed_controls'] <= 0:
            raise ValueError('A stage decision needs executed evidence')
        if not set(needed) <= {g['subgoal_id'] for g in current}:
            raise ValueError('A reopened requirement needs a replacement plan')
        done = current[0]['subgoal_id'] in satisfied
        if (mode == 'advance') != done:
            raise ValueError('Only Astra evidence satisfying the current goal permits advance')
        remaining = [g for g in current if g['subgoal_id'] not in satisfied]
        if remaining:
            validate_selection(remaining[0])
        return
    if mode != 'plan' or not isinstance(goals, list) or not 1 <= len(goals) <= len(rows):
        raise ValueError('Plan one atomic board card per unfinished requirement')
    for goal in goals:
        _validate_goal(goal, by_id)
    covered = [g['subgoal_id'] for g in goals]
    if len(set(covered)) != len(covered) or set(covered) != set(needed):
        raise ValueError('Plan omitted or duplicated unfinished original-task requirements')
    if request.get('task_context', {}).get('ordered_requirements', True) and covered != needed:
        raise ValueError('This task requires source requirement order')
    if 'reset-arm' in covered and covered[-1] != 'reset-arm':
        raise ValueError('Arm return must remain last')
    validate_selection(goals[0])


def _validate_goal(goal, by_id):
    if not isinstance(goal, dict) or set(goal) != {'subgoal_id', 'instruction', 'requirements', 'completion_checks'}:
        raise ValueError('Each goal needs its stable ID, instruction, requirement and completion check')
    identifier = goal['subgoal_id']
    if not isinstance(identifier, str) or identifier not in by_id:
        raise ValueError('Goal ID must be an original requirement ID')
    row = by_id[identifier]
    if goal['requirements'] != [identifier] or goal['completion_checks'] != [row['completion_check']]:
        raise ValueError('Atomic goal coverage and completion check must match its stable ID')
    validate_pi_instruction(goal['instruction'])
    if goal['instruction'] not in instruction_options(row):
        raise ValueError('Choose an exact instruction_options entry for this goal; appended or unrelated clauses are forbidden')


def validate_planning_decision(decision, request):
    if not is_subgoal_request(request) or request.get('version') != PLAN_VERSION:
        raise ValueError('Expected the version-5 goal-board protocol')
    try:
        _validate_decision(decision, request)
    except (ValueError, TypeError, KeyError) as error:
        raise PlanningFormatError(str(error), decision) from error


def validate_planning_response(response, request):
    if not isinstance(response, dict) or set(response) != {'request_id', *IDENTITY, 'decision'}:
        raise ValueError('Invalid goal-board response envelope')
    for key in ('request_id', *IDENTITY):
        if type(response[key]) is not type(request[key]) or response[key] != request[key]:
            raise ValueError('Stale goal-board response identity')
    validate_planning_decision(response['decision'], request)
    return deepcopy(response['decision'])


def validate_planning_payload(payload, request):
    if not isinstance(payload, dict) or set(payload) != {'decision'}:
        raise PlanningFormatError('Expected exactly one goal-board decision', payload)
    validate_planning_decision(payload['decision'], request)
    return payload['decision']


def select_evidence_frames(frames, limit=MAX_EVIDENCE_FRAMES):
    # Include the earliest and latest actual execution endpoints, sampling across
    # the entire review interval rather than truncating to its last seconds.
    unique = {f['observation_id']: f for f in frames}
    ordered = sorted(unique.values(), key=lambda f: f['step_id'])
    if len(ordered) <= limit: return deepcopy(ordered)
    indices = [round(i*(len(ordered)-1)/(limit-1)) for i in range(limit)]
    return [deepcopy(ordered[i]) for i in indices]


def planning_packet(request):
    if request.get('version') != PLAN_VERSION:
        raise ValueError('Expected the version-5 goal-board protocol')
    frames = [*request.get('reference_frames', []), *select_evidence_frames(request.get('evidence_frames', [])), request['current_scene']]
    seen, images, scenes = set(), [], []
    for frame in frames:
        if frame['observation_id'] in seen: continue
        seen.add(frame['observation_id'])
        scene = visual_frame(frame); scene.pop('images')
        scene['image_indices'] = []
        for im in frame['images']:
            scene['image_indices'].append(dict(index=len(images), camera=im['camera']))
            images.append(deepcopy(im))
        scenes.append(scene)
    fields = ('version', 'planner_role', 'review_kind', 'original_task', 'task_context',
              'remaining_requirements', 'completed_requirements', 'remaining_plan', 'current_subgoal',
              'review_reason', 'monitor_evidence', 'all_requirements', 'previous_assessments', 'execution_budget',
              'instruction_history', 'goal_board', 'format_repair', 'evidence_coverage')
    packet = {key: deepcopy(request[key]) for key in fields if key in request}
    packet['pi_instruction_contract'] = deepcopy(PI_INSTRUCTION_CONTRACT)
    packet['dispatch_policy'] = deepcopy(DISPATCH_POLICY)
    packet['instruction_options'] = {r['requirement_id']: instruction_options(r) for r in request['all_requirements']}
    packet['scenes'] = scenes
    return packet, images


def planning_instructions(request):
    return PROMPT_PATH.read_text(), planning_schema(request)
