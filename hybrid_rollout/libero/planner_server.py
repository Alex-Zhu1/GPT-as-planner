"""Loopback LIBERO recovery planner with Codex-agent and stateless image-API backends."""
import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import signal
import traceback
import uuid

from ..robodojo.io import write_json
from ..robodojo.jev.monitor import IDENTITY
from ..robodojo.jev.plans import validate_decision, validate_plan_operation, advance_schema, validate_recovery_decision
from ..robodojo.skill.schema import _obj, response_schema
from .contract import PROFILE, RECOVERY_CONTRACT, validate_response
from .task_context import board_assessment_schema, validate_board_assessment
from .visual_evidence import evidence_images, VISUAL_GUIDANCE
from .episode_thread import EpisodeThreads
from .compact_input import INPUT_FORMAT
from .instruction_planner import (instruction_review, INPUT_FORMAT as INSTRUCTION_FORMAT,
                                  EEF_INPUT_FORMAT, EEF_ROLE)
from .control_review import eef_schema
from .planner_backend import add_backend_arguments, validate_backend_arguments


def decision_schema(request):
    native = request['native_request']
    text = {'type': 'string', 'minLength': 1, 'maxLength': 4000}
    common = dict(request_id={'type': 'string', 'enum': [native['request_id']]},
                  steps={'type': 'integer', 'minimum': 1, 'maximum': request['max_chunk_steps']},
                  reason=text, assessment=response_schema()['properties']['assessment'])
    if instruction_review(request):
        ordinary = _obj(dict(request_id=common['request_id'],
                             mode={'type':'string', 'enum':['keep_instruction','stop']}, reason=text))
        replan = _obj(dict(request_id=common['request_id'], mode={'type':'string','enum':['replan']},
            reason=text, subgoals={'type':'array','minItems':1,'maxItems':1,
                # The structured-output service uses a regex engine without
                # lookaround. validate_plan separately rejects original-task.
                'items':_obj(dict(subgoal_id={'type':'string', 'pattern':'^[A-Za-z][A-Za-z0-9_-]{0,63}$',
                                            'description':'Unique recovery ID; original-task is reserved by the host.'},
                                 instruction=text, success_condition=text))}))
        variants = [ordinary, replan]
        if request.get('planner_role') == EEF_ROLE:
            bounded = eef_schema(request)
            if bounded is not None:
                variants.append(bounded)
        advance = advance_schema(request)
        if advance is not None:
            variants.append(advance)
        return _obj(dict(decision={'anyOf':variants}, board_assessment=board_assessment_schema(request)))
    ordinary = _obj(dict(common, mode={'type': 'string', 'enum': ['student', 'stop']}))
    if request.get('review_kind') == 'uncertainty':
        ordinary['properties']['steps'] = {'type': 'integer', 'enum': [1]}
        fields = {'decision': ordinary}
        if request.get('board_review_required'):
            fields['board_assessment'] = board_assessment_schema(request)
        return _obj(fields)
    edit = _obj(dict(common, mode={'type': 'string', 'enum': ['edit']}, edit=_obj(dict(
        delta={'type': 'array', 'minItems': 6, 'maxItems': 6,
               'items': {'type': 'number', 'minimum': -.25, 'maximum': .25}},
        gripper={'type': 'string', 'enum': ['keep', 'open', 'close']}))))
    replan = _obj(dict(request_id=common['request_id'], mode={'type': 'string', 'enum': ['replan']},
        reason=text, subgoals={'type': 'array', 'minItems': 1,
        'maxItems': 1 if request.get('recovery_policy') == 'single_instruction' else 8,
        'items': _obj(dict(subgoal_id={'type': 'string',
            'enum': [f'recovery-{i}' for i in range(1, 9)]},
            instruction=text, success_condition=text))}))
    variants = [ordinary, edit, replan]
    advance = advance_schema(request)
    if advance is not None:
        variants.append(advance)
    if native.get('eef_control', {}).get('supported'):
        vector3 = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
        vector4 = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 4, 'maxItems': 4}
        variants.append(_obj(dict(common, mode={'type': 'string', 'enum': ['eef']}, target=_obj(dict(
            position=vector3, quaternion_xyzw=vector4,
            frame={'type': 'string', 'enum': ['MuJoCo world']},
            gripper={'type': 'string', 'enum': ['keep', 'open', 'close']})))))
    fields = {'decision': {'anyOf': variants}}
    if request.get('board_review_required'):
        fields['board_assessment'] = board_assessment_schema(request)
    return _obj(fields)




SYSTEM_PROMPT = (
            'You are the LIBERO/Panda recovery planner. Return one declarative decision only. '
            'The host owns simulation and will validate any decision; you cannot execute actions. '
            'Use the attached real current cameras and separately labelled historical evidence. '
            'Do not infer object truth, reward, successful grasp or task success from commands alone. '
            'Pi actions are normalized single-arm OSC_POSE. EEF target poses use world meters and xyzw. '
            'There is no RoboDojo dual-arm interface or FK trajectory here. '
            'Uncertainty alone is not evidence for edit/eef. Prefer student retry when justified. '
            'Choose eef for a visibly grounded local geometric correction when supported; choose replan '
            'when the instruction or task ordering needs repair; otherwise stop. Preserve the original task. '
            'For replan, use each recovery-1 through recovery-8 ID at most once; original-task '
            'is reserved by the host. Subgoal success conditions must be visually observable; '
            'the host separately checks native task success, which is not model evidence. '
            'For review_kind=uncertainty, only student for exactly one control or stop is allowed. '
            'Judge the actual current cameras and fresh proposed next control; authorize student '
            'only if visually grounded and intent_status=aligned, otherwise stop. '
            'Do not assert that Valen became clear or that a failure was confirmed. '
            'For review_kind=progress, review may be pre-scheduled or progress-triggered; read failure.trigger. '
            'Neither scheduling rule establishes a confirmed safety failure. Use the historical images to assess '
            'repeated ineffective behavior. If the fresh student proposal repeats it, prefer a '
            'grounded eef correction for local pose errors or a short corrective subgoal for instruction errors '
            'over another unchanged retry. Explain the choice in reason. Do not equate '
            'robot motion with object/task progress. Keep the original task as the final obligation. '
            'task_context is a status board of raw Valen evidence, host measurements/execution records, '
            'and separately attributed historical Astra interpretations. Only you interpret the board. '
            'Numeric review triggers are sampling rules, not proof of task stagnation or scene facts. '
            'When board_review_required is true, return board_assessment alongside decision: interpret '
            'robot activity, task status, scene context and uncertainties, citing supplied observation IDs. '
            'Compare previous Astra judgments with new evidence and revise them when needed. '
            'For recovery_policy=single_instruction, replan supplies exactly ONE current recovery '
            'instruction in subgoals, not a future task queue. Prefer EEF for visibly grounded local '
            'pose errors. A recovery instruction persists across multiple monitored chunks. '
            'New evidence or a clearly labelled execution-budget fallback requests review. Use '
            'resume_original when the fresh scene supports returning to the original instruction; '
            'this does not assert completion. Never assume an earlier instruction remains valid. '
            'When plan_operations offers advance_subgoal, explicitly choose it only if the current '
            'subgoal is complete, citing current-plan evidence. It pops exactly one subgoal and '
            'discards the old proposal; Pi then infers for the next instruction. A board judgment '
            'or student decision alone never advances a plan. For local_correction_unverified, '
            'review the fresh scene before authorizing any further controls; uncertainty is not failure. '
            'Do not turn Valen scores or gripper commands into unquestioned facts. Historical interpretations '
            'retain their source step and may be stale. Your completed status never sets native task success. '
            'The assessment_example is a schema example, not an assessment to copy. '
            'If failure.state is SMOKE_PROBE, this is only a protocol check: no confirmed failure '
            'is asserted and your decision will not be executed. '
            + VISUAL_GUIDANCE + RECOVERY_CONTRACT
)


class AstraPlanner:
    def __init__(self, codex, output, evidence_root, timeout=300., *, transport_factory=None,
                 backend='codex', api_base_url=None, api_key_file=None, api_client=None):
        self.codex, self.output = str(codex), Path(output)
        self.evidence_root, self.timeout = Path(evidence_root), timeout
        self.output.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        if backend == 'api':
            from .api_planner import StatelessAPI
            self.threads = StatelessAPI(self.output, api_base_url, api_key_file, timeout, client=api_client)
        elif backend == 'codex':
            self.threads = EpisodeThreads(self.codex, self.output, timeout, SYSTEM_PROMPT,
                                          transport_factory=transport_factory)
        else:
            raise ValueError('Unknown planner backend')

    def close(self):
        self.threads.close()

    def replan(self, request):
        if request.get('version') != 1 or request['native_request'].get('robot_profile') != PROFILE:
            raise ValueError('Expected version 1 LIBERO recovery request')
        if type(request.get('max_chunk_steps')) is not int or not 1 <= request['max_chunk_steps'] <= 5:
            raise ValueError('Invalid monitored action bound')
        if instruction_review(request) and (request.get('recovery_policy') != 'single_instruction'
                                          or not request.get('board_review_required')):
            raise ValueError('Instruction planner requires single_instruction and board review')
        identity = {key: request[key] for key in ('request_id', *IDENTITY)}
        images = evidence_images(request, self.evidence_root)
        def validate(payload):
            expected = {'decision', 'board_assessment'} if request.get('board_review_required') else {'decision'}
            if set(payload) != expected:
                raise ValueError('Expected exactly one structured decision')
            validate_recovery_decision(payload['decision'], request, validate_response)
            response = dict(identity, decision=deepcopy(payload['decision']))
            if request.get('board_review_required'):
                validate_board_assessment(payload['board_assessment'], request)
                response['board_assessment'] = deepcopy(payload['board_assessment'])
            return response
        return self.threads.replan(request, images, decision_schema(request), validate)


def make_handler(planner):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path == '/health' else 404,
                       {'ready': True, 'model': 'gpt-6-astra', 'effort': 'xhigh',
                        'backend': getattr(planner, 'backend', 'codex'),
                        'stateless': getattr(planner, 'backend', 'codex') == 'api',
                        'model_input_format': ('stateless_libero_review_v1'
                            if getattr(planner, 'backend', 'codex') == 'api' else 'episode_thread'),
                        'conversation_mode': EEF_INPUT_FORMAT,
                        'supported_conversation_modes': [EEF_INPUT_FORMAT, INSTRUCTION_FORMAT, INPUT_FORMAT]})

        def do_POST(self):
            if self.path != '/replan':
                return self.reply(404, {'error': 'not found'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 4 * 1024 * 1024:
                    return self.reply(413, {'error': 'invalid body size'})
                self.reply(200, planner.replan(json.loads(self.rfile.read(size))))
            except Exception as error:
                traceback.print_exc()
                self.reply(500, {'error': type(error).__name__, 'detail': str(error)})
    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codex', type=Path)
    add_backend_arguments(p)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=300.)
    p.add_argument('--port', type=int, default=8003)
    a = p.parse_args(argv)
    validate_backend_arguments(p, a)
    if a.planner_backend != 'api' and not a.codex:
        p.error('--codex is required for the codex planner backend')
    if a.planner_backend == 'api' and a.codex:
        p.error('--codex cannot be combined with the api planner backend')
    planner = AstraPlanner(a.codex, a.output.resolve(), a.evidence_root.resolve(), a.timeout,
        backend=a.planner_backend or 'codex', api_base_url=a.api_base_url, api_key_file=a.api_key_file)
    def interrupted(*_):
        # Unwind a pending communicate() so its owned Codex process group is closed.
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with HTTPServer(('127.0.0.1', a.port), make_handler(planner)) as server:
            print('ASTRA_LIBERO_HTTP_READY', flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        planner.close()
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
