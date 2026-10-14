"""Astra with attached evidence and one structured answer, without agent tool loops."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
import time
import uuid

from ..io import write_json
from ..skill.transport import StdioAppServer
from .monitor import IDENTITY
from .plans import recovery_response_schema
from .task_instructions import constrain_replan_schema, replan_candidates, validate_replan
from .subgoal_plan import (is_subgoal_request, planning_packet, planning_instructions,
                           validate_planning_payload)


def advisory_schema():
    return dict(type='object', additionalProperties=False, required=['mode', 'reason'], properties=dict(
        mode=dict(type='string', enum=['continue', 'request_recovery']),
        reason=dict(type='string', minLength=1, maxLength=1200)))


def recovery_output_schema(max_steps, output_format='full', *, original_task=None):
    """Keep native evidence validation while omitting fields unused by each mode."""
    schema = recovery_response_schema()
    native = schema['anyOf'][0]
    native['properties']['steps']['maximum'] = max_steps
    if output_format == 'mode-specific':
        branches = []
        for mode in ('student', 'edit', 'eef', 'stop'):
            branch = deepcopy(native)
            branch['properties']['mode']['enum'] = [mode]
            if mode in ('edit', 'eef'):
                branch['properties']['steps']['maximum'] = min(5, max_steps)
            for field in ('edit', 'target'):
                if (field, mode) not in (('edit', 'edit'), ('target', 'eef')):
                    del branch['properties'][field]
                    branch['required'].remove(field)
            branches.append(branch)
        schema['anyOf'] = branches + schema['anyOf'][1:]
    elif output_format != 'full':
        raise ValueError('output_format must be full or mode-specific')
    result = dict(type='object', additionalProperties=False, required=['decision'],
                  properties=dict(decision=schema))
    return constrain_replan_schema(result, original_task) if original_task is not None else result


ACTION_CHUNK_MODES = ('none', 'sampled', 'recent')


def validate_action_chunk_options(mode, steps):
    if mode not in ACTION_CHUNK_MODES:
        raise ValueError('action_chunk_mode must be none, sampled or recent')
    if type(steps) is not int or not 1 <= steps <= 25:
        raise ValueError('action_chunk_steps must be an integer in 1..25')


def compact_packet(request, *, action_chunk_mode='sampled', action_chunk_steps=5):
    """Bound image/history volume explicitly; full source remains in the audit."""
    validate_action_chunk_options(action_chunk_mode, action_chunk_steps)
    if is_subgoal_request(request):
        return planning_packet(request)
    scene = request['current_scene']
    frames = list(request.get('evidence_frames', []))
    frames = [f for f in frames if f['observation_id'] != scene['observation_id']] + [scene]
    indices = sorted(set([0, len(frames)//2, len(frames)-1]))
    selected = [frames[i] for i in indices]
    images, contexts = [], []
    for frame in selected:
        context = {k: deepcopy(frame[k]) for k in ('step_id', 'observation_id', 'current_state', 'current_eef') if k in frame}
        context['image_indices'] = []
        for im in frame['images']:
            context['image_indices'].append(dict(index=len(images), camera=im['camera']))
            images.append(dict(im))
        contexts.append(context)
    packet = dict(review_kind=request.get('review_kind'), review_reason=request.get('review_reason'),
        original_task=request['original_task'], current_subgoal=request['current_subgoal'],
        control_dt=request.get('control_dt', .04), scenes=contexts,
        background_advice=request.get('background_advice'),
        accumulated_evidence=request.get('accumulated_evidence'),
        scores=request.get('failure', {}).get('scores'),
        previous_reviews=[dict(mode=a.get('decision', {}).get('mode'),
                               reason=a.get('decision', {}).get('reason'))
                          for a in request.get('previous_attempts', [])[-2:]])
    if request.get('review_kind') != 'background':
        native = request['native_request']
        trajectory = native.get('student_eef_trajectory', [])
        packet['proposal'] = {k: deepcopy(native[k]) for k in ('request_id','step_id','current_eef',
            'student_instruction','remaining_steps','student_step_limit') if k in native}
        packet['proposal']['action_chunk_mode'] = action_chunk_mode
        if action_chunk_mode == 'sampled':
            rows = sorted(set([0, len(trajectory)//4, len(trajectory)//2, max(0,len(trajectory)-1)]))
            packet['proposal']['sampled_fk'] = [deepcopy(trajectory[i]) for i in rows] if trajectory else []
        elif action_chunk_mode == 'recent':
            limit = min(action_chunk_steps, request['max_chunk_steps'], native.get('student_step_limit', 25))
            packet['proposal']['sampled_fk'] = deepcopy(trajectory[:limit])
            packet['proposal']['action_chunk_scope'] = 'First upcoming steps of the latest pi proposal, not past executed actions'
        packet['proposal']['max_student_steps'] = request['max_chunk_steps']
        packet['proposal']['max_edit_eef_steps'] = min(5, request['max_chunk_steps'])
        packet['remaining_plan'] = request.get('remaining_plan')
        packet['replan_candidates'] = replan_candidates(request['original_task'])
    return packet, images


def compact_instructions(request, *, action_chunk_mode='sampled', output_format='full'):
    """Shared explicit task instructions and output schema for both transports."""
    if is_subgoal_request(request):
        return planning_instructions(request)
    background = request.get('review_kind') == 'background'
    instructions = (
        'You are the RoboDojo visual evidence reviewer. Return only the requested JSON; do not use tools, '
        'read files, write notes, or narrate a plan. Images are already attached in indexed order. '
        'Compare physical progress across the labeled steps, not elapsed wall time. Monitoring scores '
        'request interpretation, not a proven failure. Independently identify visible failures. '
        'An approach or incomplete grasp is not by itself a stall. Never claim whole-task success. '
        'Keep the reason concise and grounded in the images. ')
    if background:
        instructions += ('The robot continues while you review this historical snapshot. Return continue '
            'when execution is advancing or a normal phase is unfinished. Return request_recovery only '
            'for an observed failure, persistent lack of progress over this sequence, or a task mismatch '
            'requiring fresh assessment. This answer cannot execute a robot action or stop an episode.')
        schema = advisory_schema()
    else:
        from ..robodojo_server.gate_assessment import GATE_INSTRUCTION
        instructions += ('Physics is paused at the last scene. Select student, edit, eef or replan. '
            'Student can use up to max_student_steps; edit/eef up to max_edit_eef_steps. '
            'EEF positions are absolute environment-frame link6 targets within 0.05 m of current EEF; '
            'unit quaternion wxyz change <=0.35 rad; specify both arms and gripper_closed booleans. '
            'Edit deltas are relative to sampled student targets. No stop before native termination. '
            'High-level replan selects exactly ONE entry from replan_candidates; copy its subgoal_id, '
            'instruction and success_condition exactly. These are reviewed RoboDojo task templates or '
            'source clauses, not evidence that Pi learned arbitrary recovery commands. '
            'Do not write a future subgoal queue or paraphrase a candidate. Do not add colors, arm choices, '
            'motion details, preservation clauses, verification steps, or commands such as Move 2. '
            'For number arrangement retain the task-level template; never invent per-digit relocation. '
            'Select a candidate only when appropriate to the current visible scene and its prerequisites; '
            'if none fits, use the existing native decision modes instead of inventing instructions. '
            'An empty candidate list means replan is unavailable. Keep reasoning outside the Pi instruction, '
            'in reason (at most 600 characters). Original-task completion remains a separate obligation. '
            'Use the proposal request_id. Write English. '+GATE_INSTRUCTION)
        if action_chunk_mode == 'none':
            instructions += (' No predicted action trajectory is supplied. Do not invent the next student '
                'intent; mark intent_status uncertain. Ground recovery in observed execution evidence. '
                'Do not infer unseen student targets. The existing recovery evidence constraints still apply.')
        elif action_chunk_mode == 'recent':
            instructions += (' The trajectory covers only the first upcoming steps of the latest proposal. '
                'Do not infer motion beyond that prefix or treat it as previously executed evidence.')
        schema = recovery_output_schema(request['max_chunk_steps'], output_format,
                                        original_task=request['original_task'])
        if output_format == 'mode-specific':
            instructions += (' Return only fields used by the selected mode: student/stop omit edit and '
                'target; edit omits target; eef omits edit. Keep assessment evidence concise, '
                'one short sentence per field. Do not pad or pretty-print JSON.')
    return instructions, schema


class CompactCodexPlanner:
    """One reusable app-server, isolated bounded JSON turns; pinned Astra/high."""
    def __init__(self, output, codex, timeout=180., transport_factory=StdioAppServer,
                 *, action_chunk_mode='sampled', action_chunk_steps=5, output_format='full'):
        validate_action_chunk_options(action_chunk_mode, action_chunk_steps)
        self.action_chunk_mode, self.action_chunk_steps = action_chunk_mode, action_chunk_steps
        recovery_output_schema(25, output_format)
        self.output_format = output_format
        self.output, self.codex, self.timeout = Path(output), codex, timeout
        self.factory, self.transport = transport_factory, None
        self.lock, self.connection_lock = Lock(), Lock()
        self.closed = False

    def connect(self):
        with self.connection_lock:
            if self.closed:
                raise RuntimeError('Planner closed')
            if self.transport is None:
                folder = self.output/'astra_transport'/uuid.uuid4().hex
                folder.mkdir(parents=True)
                argv = [self.codex, 'app-server', '--stdio', '-c', 'model_provider="openai"',
                        '-c', 'features.shell_tool=false', '-c', 'features.view_image=false',
                        '-c', 'web_search="disabled"',
                        '-c', 'sqlite_home='+json.dumps(str(self.output/'astra_runtime_db')),
                        '-c', 'log_dir='+json.dumps(str(self.output/'astra_runtime_logs'))]
                transport = self.factory(argv, folder)
                try:
                    transport.request('initialize', dict(clientInfo=dict(name='robodojo-evidence-review',version='2'),
                                      capabilities=dict(experimentalApi=True)), self.timeout)
                    transport.notify('initialized', {})
                except BaseException:
                    transport.close()
                    raise
                self.transport = transport
            return self.transport

    def replan(self, request):
        with self.lock:
            return self._replan(request)

    def _replan(self, request):
        prepare_started = time.monotonic()
        directory = self.output/f"astra_{request['request_id']}"
        directory.mkdir(parents=True)
        write_json(directory/'request.json', request)
        packet, images = compact_packet(request, action_chunk_mode=self.action_chunk_mode,
                                        action_chunk_steps=self.action_chunk_steps)
        write_json(directory/'input.json', packet)
        write_json(directory/'images.json', images)
        background = request.get('review_kind') == 'background'
        instructions, schema = compact_instructions(request, action_chunk_mode=self.action_chunk_mode,
                                                     output_format=self.output_format)
        write_json(directory/'schema.json', schema)
        (directory/'instructions.txt').write_text(instructions)
        prepared = time.monotonic()
        transport = self.connect()
        connected = time.monotonic()
        started = time.monotonic()
        thread = transport.request('thread/start', dict(cwd=str(directory),model='gpt-6-astra',
            modelProvider='openai',config=dict(model_reasoning_effort='high'),approvalPolicy='never',
            sandbox='read-only',developerInstructions=instructions,dynamicTools=[],ephemeral=True,
            allowProviderModelFallback=False),self.timeout)
        if thread.get('model') != 'gpt-6-astra' or thread.get('reasoningEffort') != 'high':
            raise RuntimeError('Astra model or reasoning effort differs from requested gpt-6-astra/high')
        thread_id = thread['thread']['id']
        thread_ready = time.monotonic()
        prompt = json.dumps(packet,ensure_ascii=False,separators=(',',':'))
        turn = transport.request('turn/start',dict(threadId=thread_id,model='gpt-6-astra',effort='high',
            outputSchema=schema,input=[dict(type='text',text=prompt)]+
            [dict(type='localImage',path=i['path']) for i in images]),self.timeout)['turn']['id']
        turn_ready = time.monotonic()
        final = None
        first_output = None
        token_usage = None
        deadline = started+self.timeout
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Astra compact review timed out')
            event = transport.next_message(remaining)
            p = event.get('params',{})
            if 'id' in event and event.get('method'):
                transport.reply(event['id'],dict(contentItems=[dict(type='inputText',text='No tools are available. Return the decision JSON.')],success=False))
                raise RuntimeError('Unexpected tool or permission request in structured review')
            if p.get('threadId') != thread_id:
                continue
            if event.get('method') == 'item/agentMessage/delta' and p.get('turnId') == turn:
                if first_output is None:
                    first_output = time.monotonic()
            if event.get('method') == 'thread/tokenUsage/updated' and p.get('turnId') == turn:
                token_usage = p.get('tokenUsage', {}).get('last')
            if event.get('method') == 'item/completed' and p.get('turnId') == turn:
                item = p['item']
                if item.get('type') == 'agentMessage' and item.get('phase') in (None,'final_answer'):
                    final = item.get('text')
            if event.get('method') == 'turn/completed' and p['turn']['id'] == turn:
                if p['turn']['status'] != 'completed':
                    raise RuntimeError('Astra structured turn failed: '+str(p['turn'].get('error')))
                break
        decision = json.loads(final) if final else None
        if is_subgoal_request(request):
            decision = validate_planning_payload(decision, request)
        elif not background and isinstance(decision, dict):
            decision = decision.get('decision')
            validate_replan(decision, request)
        if self.closed:
            raise RuntimeError('Planner closed while request was in flight; discard result')
        response = dict(**{k:request[k] for k in ('request_id',*IDENTITY)},decision=decision)
        write_json(directory/'response.json',response)
        write_json(directory/'timing.json',dict(elapsed_seconds=time.monotonic()-started,
            prompt_characters=len(prompt),attached_images=len(images),model='gpt-6-astra',effort='high',
            backend='compact', planner_role=request.get('planner_role'), protocol_version=request.get('version'),
            action_chunk_mode='none' if is_subgoal_request(request) else self.action_chunk_mode,
            action_chunk_steps=None if is_subgoal_request(request) else self.action_chunk_steps,
            action_chunk_points=len(packet.get('proposal', {}).get('sampled_fk', [])),
            output_format='subgoal_plan_v4' if is_subgoal_request(request) else self.output_format,
            local_prepare_seconds=prepared-prepare_started,
            connect_seconds=connected-prepared, thread_start_seconds=thread_ready-started,
            turn_submit_seconds=turn_ready-thread_ready,
            first_output_seconds=first_output-started if first_output is not None else None,
            output_stream_seconds=time.monotonic()-first_output if first_output is not None else None,
            token_usage=token_usage))
        return response

    def close(self):
        with self.connection_lock:
            self.closed = True
            transport,self.transport = self.transport,None
        if transport is not None:
            transport.close()
