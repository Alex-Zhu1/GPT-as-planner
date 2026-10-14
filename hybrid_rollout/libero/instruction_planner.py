"""Instruction-only Astra task briefing and public evidence boundary."""
from copy import deepcopy
from pathlib import Path


INPUT_FORMAT = 'episode_instruction_v1'
EEF_INPUT_FORMAT = 'episode_instruction_eef_v2'
EEF_ROLE = 'instruction_eef'
SKILL_PATH = Path(__file__).parent / 'skills/astra-instruction-planner/SKILL.md'


def instruction_only(request):
    return request.get('planner_role') == 'instruction_only'


def instruction_review(request):
    return request.get('planner_role') in ('instruction_only', EEF_ROLE)


QUESTIONS = {
    'completion_scores_review': 'Is the current instruction success condition actually satisfied in the current images? If so, what instruction should Pi execute next?',
    'low_progress_scores_review': 'Does the image sequence show useful progress toward the current instruction? Should Pi keep it or replace it with a more suitable current subgoal?',
    'sustained_stuck_evidence': 'Does the image sequence show repeated ineffective execution? Which current subgoal would address the observed obstacle?',
    'task_progress_review_budget': 'After this execution interval, is the original instruction still appropriate? Decide whether to keep it or focus Pi on one current subgoal.',
    'subgoal_execution_budget': 'Has the current subgoal reached its success condition? If not, is continuing it justified, or should it be replaced?',
    'ambiguous_observations_review': 'Which task-relevant facts cannot be established from the images? Is there enough evidence to keep the current instruction or choose a suitable benchmark-style task instruction?',
    'anomaly_scores_review': 'Is the reported anomaly supported by the current images? Does it invalidate the current instruction, and what current subgoal addresses it?',
}


def review_brief(request):
    failure = request.get('failure', {})
    scene = request['current_scene']
    return dict(
        trigger=failure.get('trigger', 'evidence_review'),
        question=QUESTIONS.get(failure.get('trigger'),
            'Does the current instruction still fit the observed scene? Decide what instruction Pi should execute next.'),
        trigger_is_not_a_verdict=True,
        original_task=request['original_task'],
        current_instruction=request['current_subgoal'],
        success_condition=scene.get('success_condition'),
        instruction_execution=deepcopy(request.get('instruction_execution', {})),
        required_decision=('keep_instruction, replan with one current subgoal, resume_original when offered, or stop'
            + ('; bounded eef only when control_diagnostics.eef_supported is true'
               if request.get('planner_role') == EEF_ROLE else '')),
        instruction_granularity='One short LIBERO benchmark-style task instruction: action + object + destination/relation when applicable. Natural pick-up-and-place tasks may combine verbs. Do not prescribe finger alignment, gripper controls, contact points or end-effector motions. Put the visible outcome in success_condition separately.',
    )


# Raw actions remain in controller/audit artifacts, outside model-readable records.
_PRIVATE_FIELDS = frozenset((
    'proposal_actions', 'executed_actions', 'student_eef_command_preview',
    'eef_control', 'current_eef', 'current_state', 'action_contract',
    'assessment_example', 'gate_instruction', 'proposal_path',
    'monitor_program', 'contract', 'planning_guidance',
))


def public_evidence(value):
    """Remove low-level control payloads also from on-demand episode reads."""
    if isinstance(value, dict):
        result = {k: public_evidence(v) for k, v in value.items() if k not in _PRIVATE_FIELDS}
        if 'executed_actions' in value:
            result['executed_action_count'] = len(value['executed_actions'])
        return result
    if isinstance(value, list):
        return [public_evidence(v) for v in value]
    return deepcopy(value)
