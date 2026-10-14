"""Reviewed RoboDojo task templates, not a free-form language controller.

Sources and limits are documented in jev/README.md. The original episode
instruction selects the catalogue; model text cannot introduce objects, arm
assignments, digit identities or new recovery procedures.
"""
import re


ORGANIZE_TABLE = (
    'Place the mouse on the mouse pad, push the keyboard into the frame, put the figurine on the stand, '
    'place the alarm clock on the drawer, then open the drawer and put all remaining miscellaneous items inside.'
)
ARRANGE_NUMBERS = 'Arrange the numbers from left to right to form the largest possible number, and place them on the pad.'

# Only the first task is split into source clauses. Other entries retain their
# original task-level goal: there is no evidence for novel per-digit or per-arm
# commands in the source templates. Conditions are host-owned visual checks.
_CATALOGUE = {
    ORGANIZE_TABLE: (
        ('mouse-on-pad', 'Place the mouse on the mouse pad.', 'The mouse rests on the mouse pad after release.'),
        ('keyboard-in-frame', 'Push the keyboard into the frame.', 'The keyboard rests inside the frame.'),
        ('figurine-on-stand', 'Put the figurine on the stand.', 'The figurine is supported on the stand after release.'),
        ('clock-on-drawer', 'Place the alarm clock on the drawer.', 'The alarm clock rests on the drawer after release.'),
        ('open-drawer', 'Open the drawer.', 'The drawer is visibly open.'),
        ('items-in-drawer', 'Put all remaining miscellaneous items inside the drawer.',
         'The remaining miscellaneous items rest inside the drawer after release.'),
    ),
    ARRANGE_NUMBERS: (
        ('arrange-numbers', ARRANGE_NUMBERS, 'The numbers rest on the pads in the largest possible left-to-right order.'),
    ),
}
for _id, _instruction, _condition in (
    ('sort-objects', 'Sort the objects by category into the three baskets.',
     'The objects rest in the three baskets grouped by category.'),
    ('imitate-order', 'Observe the object placement order, remember it, then place the corresponding objects into the basket in the same order.',
     'The corresponding objects have been placed into the basket in the demonstrated order.'),
    ('pack-box', 'Place all the objects into the box with their front sides facing left.',
     'All objects rest inside the box with their front sides facing left.'),
    ('build-tower', 'Build a tower using the wooden blocks and wooden boards.',
     'The wooden blocks and wooden boards form a supported tower after release.'),
    ('make-kong', 'Wait for the opponent to discard a tile, then declare a kong with the matching tiles.',
     'The matching tiles form a kong with the discarded tile.'),
    ('fold-clothes', 'Fold the clothes neatly.', 'The clothes are visibly folded neatly.'),
    ('bottles-in-dustbin', 'Pick up the bottles and throw them into the dustbin, using handover when needed.',
     'The bottles are inside the dustbin.'),
):
    _CATALOGUE[_instruction] = ((_id, _instruction, _condition),)

_CATEGORY_SORT = re.compile(
    r'Put ([a-z][a-z0-9_]*) objects into the left basket, ([a-z][a-z0-9_]*) objects into the middle basket, '
    r'and ([a-z][a-z0-9_]*) objects into the right basket, then reset the robot arm\.'
)


def replan_candidates(original_task):
    """Return fresh, exact choices derived only from the original task template.

    Unknown templates disable language replanning instead of guessing a task
    decomposition. Native student/edit/EEF decisions remain available.
    """
    rows = _CATALOGUE.get(original_task, ())
    match = _CATEGORY_SORT.fullmatch(original_task)
    if match:
        rows = tuple(
            (f'sort-{side}', f'Put {category} objects into the {side} basket.',
             f'All {category} objects rest inside the {side} basket after release.')
            for category, side in zip(match.groups(), ('left', 'middle', 'right'))
        ) + (('reset-arm', 'Reset the robot arm.', 'The robot arm is back in its reset pose.'),)
    return [dict(subgoal_id=identity, instruction=instruction, success_condition=condition)
            for identity, instruction, condition in rows]


def constrain_replan_schema(schema, original_task):
    """Bind all three subgoal fields together; exactly one reviewed goal."""
    branches = schema['properties']['decision']['anyOf']
    candidates = replan_candidates(original_task)
    for branch in list(branches):
        if branch['properties']['mode']['enum'] != ['replan']:
            continue
        if not candidates:
            branches.remove(branch)
            continue
        branch['properties']['subgoals'] = dict(
            type='array', minItems=1, maxItems=1,
            items=dict(anyOf=[dict(type='object', additionalProperties=False,
                required=list(goal), properties={key: dict(type='string', enum=[value])
                                                for key, value in goal.items()})
                              for goal in candidates]))
        branch['properties']['reason']['maxLength'] = 600
    return schema


def validate_replan(decision, request):
    """Enforce the language contract even if a transport ignores the schema."""
    if not isinstance(decision, dict) or decision.get('mode') != 'replan':
        return
    if (set(decision) != {'request_id', 'mode', 'reason', 'subgoals'} or
            decision['request_id'] != request['native_request']['request_id']):
        raise ValueError('Replan requires the current proposal identity and exact decision fields')
    reason = decision['reason']
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 600:
        raise ValueError('Replan reason must be nonempty and at most 600 characters')
    goals = decision['subgoals']
    if not isinstance(goals, list) or len(goals) != 1:
        raise ValueError('RoboDojo replan requires exactly one task-template subgoal')
    if goals[0] not in replan_candidates(request['original_task']):
        raise ValueError('RoboDojo replan must select an exact task-template subgoal')
