"""RoboDojo compact reviews over the stateless image Responses API."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import Lock
import time

from ...model_api import ResponsesClient, image_part, structured_payload
from ..io import write_json
from .compact_planner import compact_packet, compact_instructions, recovery_output_schema, validate_action_chunk_options
from .monitor import IDENTITY
from .plans import validate_decision
from .task_instructions import validate_replan
from .subgoal_plan import is_subgoal_request, validate_planning_payload, PLAN_VERSION, PlanningFormatError

MODEL, EFFORT = 'gpt-6-astra', 'high'
DEFAULT_BASE_URL = 'http://127.0.0.1:8317/v1'
DEFAULT_KEY_FILE = Path(__file__).resolve().parents[3]/'runtime/local-codex-api/api-key'


def validated_payload(payload, request):
    if is_subgoal_request(request):
        return validate_planning_payload(payload, request)
    if request.get('review_kind') == 'background':
        if (set(payload) != {'mode', 'reason'} or
                payload['mode'] not in ('continue', 'request_recovery') or
                not isinstance(payload['reason'], str) or not payload['reason'].strip() or
                len(payload['reason']) > 1200):
            raise ValueError('Background review may only continue or request fresh recovery; no stale actions')
        return payload
    if set(payload) != {'decision'}:
        raise ValueError('Expected exactly one recovery decision')
    from ..robodojo_server.client import validate_dual_response
    decision = payload['decision']
    validate_replan(decision, request)
    validate_decision(decision, request['native_request'], request['max_chunk_steps'], validate_dual_response)
    if request.get('review_kind') == 'uncertainty':
        if decision['mode'] not in ('student', 'stop') or decision['steps'] != 1:
            raise ValueError('Uncertainty review only permits a single student control or stop')
        if decision['mode'] == 'student' and decision['assessment']['intent_status'] != 'aligned':
            raise ValueError('Uncertainty continuation requires an aligned intent assessment')
    return decision


class CompactAPIPlanner:
    """Same compact evidence and decisions, without a Codex process or thread."""
    def __init__(self, output, base_url=DEFAULT_BASE_URL, key_file=DEFAULT_KEY_FILE, timeout=300., *,
                 action_chunk_mode='sampled', action_chunk_steps=5, output_format='full', client=None):
        validate_action_chunk_options(action_chunk_mode, action_chunk_steps)
        recovery_output_schema(25, output_format)
        self.output, self.base_url = Path(output).resolve(), base_url
        self.action_chunk_mode, self.action_chunk_steps = action_chunk_mode, action_chunk_steps
        self.output_format = output_format
        self.client = client if client is not None else ResponsesClient(base_url, key_file, timeout)
        self.lock, self.closed = Lock(), False

    def replan(self, request):
        with self.lock:
            if self.closed:
                raise RuntimeError('Planner closed')
            return self._replan(request)

    def _replan(self, request):
        started = time.monotonic()
        background = request.get('review_kind') == 'background'
        if request.get('version') != (PLAN_VERSION if is_subgoal_request(request) else 2 if background else 1):
            raise ValueError('Expected RoboDojo version 2 advisory or version 1 recovery')
        identity = {k: request[k] for k in ('request_id', *IDENTITY)}
        packet, images = compact_packet(request, action_chunk_mode=self.action_chunk_mode,
                                        action_chunk_steps=self.action_chunk_steps)
        instructions, schema = compact_instructions(request, action_chunk_mode=self.action_chunk_mode,
                                                     output_format=self.output_format)
        prompt = json.dumps(packet, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        content = [dict(type='input_text', text=prompt)]
        image_hashes = []
        for image in images:
            part, sha = image_part(image)
            content.append(part)
            image_hashes.append(sha)
        body = dict(model=MODEL, reasoning=dict(effort=EFFORT), instructions=instructions,
                    tools=[], store=False, stream=False, input=[dict(role='user', content=content)],
                    text=dict(format=dict(type='json_schema', name='robodojo_review', strict=True, schema=schema)))
        # One inference per request ID. The scheduler owns bounded format repair
        # with a new ID and explicit feedback; transport failures are not retried.
        name = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = self.output/f'astra_api_{name}'
        directory.mkdir(parents=True)
        timing = dict(backend='api', stateless=True, model=MODEL, effort=EFFORT,
            planner_role=request.get('planner_role'), protocol_version=request.get('version'),
            api_base_url=self.base_url, status='pending', tools=[],
            prompt_characters=len(prompt), attached_images=len(images), image_sha256=image_hashes,
            instructions_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
            action_chunk_mode='none' if is_subgoal_request(request) else self.action_chunk_mode,
            action_chunk_steps=None if is_subgoal_request(request) else self.action_chunk_steps,
            action_chunk_points=len(packet.get('proposal', {}).get('sampled_fk', [])),
            output_format='subgoal_plan_v5' if is_subgoal_request(request) else self.output_format)
        write_json(directory/'request.json', request)
        write_json(directory/'input.json', packet)
        write_json(directory/'images.json', images)
        write_json(directory/'schema.json', schema)
        (directory/'instructions.txt').write_text(instructions)
        write_json(directory/'timing.json', timing)
        prepared = time.monotonic()
        try:
            result = (self.client(body, diagnostic_path=directory/'client_events.jsonl')
                      if isinstance(self.client, ResponsesClient) else self.client(body))
            write_json(directory/'api_response.json', result)
            try:
                payload = structured_payload(result, MODEL, require_object=not is_subgoal_request(request))
            except json.JSONDecodeError as error:
                if is_subgoal_request(request):
                    raise PlanningFormatError('Return valid JSON for the goal-board schema', str(error)) from error
                raise
            decision = validated_payload(payload, request)
            if self.closed:
                raise RuntimeError('Planner closed while API request was in flight; discard result')
            response = dict(identity, decision=deepcopy(decision))
            write_json(directory/'response.json', response)
            timing.update(status='completed', token_usage=result.get('usage'), response_id=result.get('id'))
            return response
        except BaseException as error:
            timing.update(status='failed', error_type=type(error).__name__)
            if isinstance(self.client, ResponsesClient):
                timing['transport_diagnostics'] = self.client.diagnostics()
            raise
        finally:
            timing.update(local_prepare_seconds=prepared-started,
                          elapsed_seconds=time.monotonic()-prepared, total_seconds=time.monotonic()-started)
            write_json(directory/'timing.json', timing)

    def diagnostics(self):
        return self.client.diagnostics() if isinstance(self.client, ResponsesClient) else {}

    def close(self):
        # Do not block the simulator owner on an in-flight model call. Remote
        # inference may finish later, but its result cannot actuate after close.
        self.closed = True
        close = getattr(self.client, 'close', None)
        if close:
            close()
