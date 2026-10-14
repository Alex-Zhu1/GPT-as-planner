"""Lossless evidence tables and host receipts for Astra's incremental input."""
from copy import deepcopy
import hashlib
import json

from ..robodojo.jev.monitor import PROJECTIONS
from ..robodojo.io import write_json


INPUT_FORMAT = 'episode_thread_delta_v2'


def compact_evidence(rows):
    """Factor metadata and answer structure; never round or interpret probabilities.

    Scores are omitted only when exactly reconstructible by the declared mapping.
    Unknown answer formats and nonmatching scores are retained verbatim.
    """
    contexts, schemas, result = [], [], []
    for row in rows:
        ref = row.get('evidence', {})
        context = dict(identity={k: v for k, v in ref.items()
                                 if k not in ('observation_id', 'step_id', 'captured_at')},
                       scope=deepcopy(row.get('scope')))
        if context not in contexts:
            contexts.append(context)
        questions, answers, derived = [], [], {}
        for name, answer in row.get('raw_answers', {}).items():
            if (isinstance(answer, dict) and answer.get('type') == 'noul'
                    and set(answer) == {'type', 'noul'}):
                questions.append(dict(name=name, encoding='noul'))
                answers.append(answer['noul'])
                derived[name] = answer['noul']
            elif (isinstance(answer, dict) and answer.get('type') == 'choice'
                  and set(answer) == {'type', 'probabilities', 'choice', 'confidence'}
                  and isinstance(answer['probabilities'], dict)):
                candidates = list(answer['probabilities'])
                questions.append(dict(name=name, encoding='choice', candidates=candidates))
                answers.append([[answer['probabilities'][k] for k in candidates],
                                answer['choice'], answer['confidence']])
                projection = PROJECTIONS.get(name)
                if projection and all(k in answer['probabilities'] for k in projection):
                    derived[name] = min(1., sum(answer['probabilities'][k] for k in projection))
            else:
                questions.append(dict(name=name, encoding='raw'))
                answers.append(deepcopy(answer))
        scores = row.get('scores', {})
        schema = dict(questions=questions, score_questions=list(scores))
        if schema not in schemas:
            schemas.append(schema)
        entry = dict(observation_id=row['observation_id'], context=contexts.index(context),
                     schema=schemas.index(schema), answers=answers)
        for key in ('step_id', 'captured_at'):
            if key in ref or key in row:
                entry[key] = ref.get(key, row.get(key))
        overrides = {k: v for k, v in scores.items() if k not in derived or v != derived[k]}
        if overrides:
            entry['score_overrides'] = deepcopy(overrides)
        if 'executed_actions' in row:
            entry['executed_action_count'] = len(row['executed_actions'])
        elif 'executed_action_count' in row:
            entry['executed_action_count'] = row['executed_action_count']
        if row.get('execution_receipt'):
            entry['execution_receipt'] = deepcopy(row['execution_receipt'])
        result.append(entry)
    return dict(contexts=contexts, schemas=schemas, score_projections=deepcopy(PROJECTIONS),
                choice_columns=['probabilities_in_candidate_order', 'choice', 'confidence']), result


def compact_image(image, attachment_index=None):
    """Map each attached image to one entry; paths stay in the transport audit."""
    result = {k: deepcopy(image[k]) for k in
              ('observation_id', 'step_id', 'camera', 'captured_at', 'plan_revision') if k in image}
    result['roles'] = list(dict.fromkeys([image['label']] +
                                        [r['role'] for r in image.get('references', [])]))
    # Usually references only repeat the row, but do not lose an alias observation.
    aliases = []
    for ref in image.get('references', []):
        if any(ref.get(k) != image.get(k) for k in
               ('observation_id', 'step_id', 'camera', 'captured_at', 'plan_revision')):
            aliases.append(deepcopy(ref))
    if aliases:
        result['aliases'] = aliases
    if attachment_index is not None:
        result['attachment_index'] = attachment_index
    return result


def execution_receipt(request):
    """Last actual execution, never a previous planner assessment or intended effect."""
    scene = request['current_scene']
    previous = request['native_request'].get('previous_result')
    if isinstance(previous, dict) and all(k in previous for k in
                                         ('start_tick', 'end_tick', 'executed_steps')):
        receipt = {k: deepcopy(previous[k]) for k in
                   ('decision', 'start_tick', 'end_tick', 'executed_steps', 'source',
                    'source_detail', 'prediction_id', 'selected_student_steps',
                    'discarded_student_steps', 'executed_gripper_closed') if k in previous}
    elif 'executed_actions' in scene and 'executed_start_step' in scene:
        receipt = dict(start_tick=scene['executed_start_step'], end_tick=scene['step_id'],
                       executed_steps=len(scene['executed_actions']),
                       source_detail=scene.get('action_source'))
    else:
        return None
    receipt.update(author='host_execution', observation_id=scene.get('observation_id'))
    if 'executed_actions' in scene:
        receipt['actions_observation_id'] = scene['observation_id']
    return receipt


def action_record_path(folder, observation_id):
    if not isinstance(observation_id, str) or not observation_id or len(observation_id) > 256:
        raise ValueError('Invalid observation ID')
    key = hashlib.sha256(observation_id.encode()).hexdigest()
    return folder / 'actions' / (key + '.json')


def archive_actions(folder, request):
    """Persist immutable action records by observation ID, separate from the prompt.

    Index every available history row, including already-delivered evidence, so
    board-window eviction and process restart do not remove access to actions.
    """
    board = request.get('task_context', {})
    rows = list(board.get('recent_history', [])) + list(board.get('new_evidence', []))
    candidates = []
    for row in rows:
        if 'executed_actions' in row:
            candidates.append(dict(observation_id=row['evidence']['observation_id'],
                                   step_id=row['evidence'].get('step_id'),
                                   executed_actions=deepcopy(row['executed_actions'])))
    frames = [request['current_scene'], *request.get('evidence_frames', [])]
    for frame in frames:
        if 'executed_actions' in frame and frame.get('observation_id'):
            candidates.append(dict(observation_id=frame['observation_id'], step_id=frame.get('step_id'),
                                   executed_actions=deepcopy(frame['executed_actions'])))
    for record in candidates:
        path = action_record_path(folder, record['observation_id'])
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Action record cannot be a symlink')
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise ValueError('Observation ID reused with different executed actions')
        else:
            path.parent.mkdir(exist_ok=True)
            write_json(path, record)


def read_actions(folder, args):
    """Bounded action-index pagination; the caller supplies no filesystem path."""
    if set(args) != {'observation_id', 'offset', 'limit'}:
        raise ValueError('Expected observation_id, offset and limit')
    offset, limit = args['offset'], args['limit']
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('Invalid action slice')
    path = action_record_path(folder, args['observation_id'])
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError('Action record cannot be a symlink')
    if not path.is_file():
        raise ValueError('No action record for this observation in this episode')
    record = json.loads(path.read_text())
    if record['observation_id'] != args['observation_id']:
        raise ValueError('Action observation ID mismatch')
    actions = record['executed_actions']
    return dict(observation_id=record['observation_id'], step_id=record['step_id'],
                executed_actions=actions[offset:offset + limit], offset=offset,
                next_offset=min(len(actions), offset + limit), total_controls=len(actions))
