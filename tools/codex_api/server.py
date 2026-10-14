"""Small stateless Codex OAuth gateway; never launches the Codex agent.

Protocol reference: router-for-me/CLIProxyAPI v8.0.24 (MIT), commit
091ff27f31285ec4fd1dd5823b44f7f51fb8f48b. This implementation deliberately
omits that project's default tool injection and silent parameter removal.
"""
import argparse
import copy
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

UPSTREAM = 'https://chatgpt.com/backend-api/codex'
MAX_BODY = 16 * 1024 * 1024
RESPONSE_FIELDS = {
    'model', 'input', 'instructions', 'tools', 'tool_choice', 'parallel_tool_calls',
    'reasoning', 'text', 'include', 'service_tier', 'prompt_cache_key', 'stream', 'store',
}
CHAT_FIELDS = {
    'model', 'messages', 'tools', 'tool_choice', 'parallel_tool_calls',
    'reasoning_effort', 'response_format', 'stream', 'store',
}

_LOG_LOCK = threading.Lock()
_EVENT_TYPES = {'response.created', 'response.in_progress', 'response.completed',
                'response.failed', 'response.incomplete', 'error', 'response.output_item.added',
                'response.output_item.done', 'response.output_text.delta', 'response.output_text.done',
                'response.reasoning_summary_text.delta', 'response.reasoning_summary_text.done',
                'response.content_part.added', 'response.content_part.done'}
_TRACE_OPERATIONS = {'connection.connect_tcp', 'connection.start_tls',
                     'http11.send_request_headers', 'http11.send_request_body',
                     'http11.receive_response_headers', 'http11.receive_response_body',
                     'http11.response_closed', 'http2.send_request_headers',
                     'http2.send_request_body', 'http2.receive_response_headers',
                     'http2.receive_response_body', 'http2.response_closed'}
_UPSTREAM_REASONS = {'server_error', 'internal_error', 'rate_limit_exceeded', 'insufficient_quota',
                     'model_not_found', 'context_length_exceeded', 'max_output_tokens', 'content_filter',
                     'invalid_request_error', 'overloaded', 'service_unavailable', 'cancelled'}


def local_request_id(value):
    return value if isinstance(value, str) and len(value) == 32 and all(c in '0123456789abcdef' for c in value) else secrets.token_hex(16)


class RequestAudit:
    """Metadata only: never log headers, URLs, event contents or exception text."""
    def __init__(self, request_id):
        self.request_id = request_id
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.state = dict(stage='received', upstream_status=None, lines=0, events=0,
                          first_line_seconds=None, last_line_seconds=None,
                          first_event_seconds=None, last_event_seconds=None, last_event_type=None)
        self.done = threading.Event()
        self.peer_socket = None
        self.emit('request_start')

    def emit(self, event, **fields):
        with self.lock:
            record = dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          event=event, request_id=self.request_id,
                          elapsed_seconds=round(time.monotonic()-self.started, 6), **self.state)
            record.update(fields)
        with _LOG_LOCK:
            print(json.dumps(record), file=sys.stderr, flush=True)

    def stage(self, stage, **fields):
        with self.lock:
            self.state.update(stage=stage, **fields)
        self.emit('stage')

    def trace(self, name, info):
        operation, _, phase = name.rpartition('.')
        if operation not in _TRACE_OPERATIONS or phase not in ('started', 'complete', 'failed'):
            return
        self.stage(name)
        if phase == 'failed':
            self.emit('transport_error', exception_type=type(info.get('exception')).__name__)

    def lines(self, lines):
        for line in lines:
            with self.lock:
                elapsed = time.monotonic()-self.started
                self.state['lines'] += 1
                first = self.state['first_line_seconds'] is None
                if first: self.state['first_line_seconds'] = elapsed
                self.state['last_line_seconds'] = elapsed
            if first: self.emit('first_upstream_line')
            yield line

    def event(self, event):
        kind = event.get('type')
        with self.lock:
            elapsed = time.monotonic()-self.started
            self.state['events'] += 1
            first = self.state['first_event_seconds'] is None
            if first: self.state['first_event_seconds'] = elapsed
            self.state.update(last_event_seconds=elapsed, last_event_type=kind if kind in _EVENT_TYPES else 'other')
        if first or kind in ('response.completed', 'response.failed', 'response.incomplete', 'error'):
            if kind in ('response.failed', 'response.incomplete', 'error'):
                response = event.get('response', {})
                response = response if isinstance(response, dict) else {}
                error = event.get('error') or response.get('error') or event
                incomplete = response.get('incomplete_details', {})
                with self.lock:
                    for source, key, field in ((error, 'code', 'upstream_error_code'),
                                               (error, 'type', 'upstream_error_type'),
                                               (incomplete, 'reason', 'upstream_incomplete_reason')):
                        value = source.get(key) if isinstance(source, dict) else None
                        self.state[field] = value if isinstance(value, str) and value in _UPSTREAM_REASONS else 'unknown'
            self.emit('upstream_event')

    def check_peer(self):
        if self.peer_socket is None or self.state.get('downstream_peer_closed'):
            return
        try:
            if select.select([self.peer_socket], [], [], 0)[0] and not self.peer_socket.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                with self.lock: self.state['downstream_peer_closed'] = True
                self.emit('downstream_peer_closed')
        except (OSError, ValueError):
            pass  # The handler may have completed and closed its socket.

    def heartbeat(self):
        while not self.done.wait(30):
            self.check_peer()
            self.emit('request_pending')

    def start_heartbeat(self):
        threading.Thread(target=self.heartbeat, daemon=True).start()

    def finish(self):
        self.done.set()
        self.emit('request_end')


class APIError(Exception):
    def __init__(self, status, message, code='invalid_request'):
        super().__init__(message)
        self.status, self.code = status, code


def reject_unknown(body, fields):
    unknown = set(body) - fields
    if unknown:
        raise APIError(400, 'Unsupported parameters: ' + ', '.join(sorted(unknown)))


def normalize_response(body):
    """All transformations are explicit: no agent prompt, tools, or history."""
    if not isinstance(body, dict):
        raise APIError(400, 'Request must be a JSON object')
    reject_unknown(body, RESPONSE_FIELDS)
    if not isinstance(body.get('model'), str) or not body['model'].strip():
        raise APIError(400, 'model must be a nonempty string')
    if body.get('store', False) is not False:
        raise APIError(400, 'Only store=false is supported')
    if not isinstance(body.get('stream', False), bool):
        raise APIError(400, 'stream must be a boolean')
    out = copy.deepcopy(body)
    inputs = out.get('input')
    if isinstance(inputs, str):
        inputs = [{'role': 'user', 'content': inputs}]
    if not isinstance(inputs, list) or not inputs:
        raise APIError(400, 'input must be a nonempty string or array')
    for item in inputs:
        if not isinstance(item, dict):
            raise APIError(400, 'input items must be objects')
        if item.get('role') == 'system':
            item['role'] = 'developer'
    out['input'] = inputs
    out.setdefault('instructions', '')
    out.setdefault('tools', [])
    if not isinstance(out['instructions'], str) or not isinstance(out['tools'], list):
        raise APIError(400, 'instructions must be a string; tools must be an array')
    out.update(store=False, stream=True)
    return out


def chat_to_response(body):
    if not isinstance(body, dict):
        raise APIError(400, 'Request must be a JSON object')
    reject_unknown(body, CHAT_FIELDS)
    if body.get('stream', False) is not False:
        raise APIError(400, 'Chat streaming is not implemented; use /v1/responses with stream=true')
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        raise APIError(400, 'messages must be a nonempty array')
    inputs = []
    for message in messages:
        if not isinstance(message, dict):
            raise APIError(400, 'messages must contain objects')
        reject_unknown(message, {'role', 'content', 'tool_calls', 'tool_call_id'})
        role, content = message.get('role'), message.get('content')
        if role == 'tool':
            if not isinstance(message.get('tool_call_id'), str) or not isinstance(content, str):
                raise APIError(400, 'Tool results need tool_call_id and string content')
            inputs.append({'type': 'function_call_output', 'call_id': message['tool_call_id'], 'output': content})
            continue
        if role not in ('system', 'developer', 'user', 'assistant'):
            raise APIError(400, 'Unsupported message role')
        if content is not None:
            if isinstance(content, str):
                parts = [{'type': 'output_text' if role == 'assistant' else 'input_text', 'text': content}]
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        raise APIError(400, 'Content parts must be objects')
                    if part.get('type') == 'text' and isinstance(part.get('text'), str):
                        reject_unknown(part, {'type', 'text'})
                        parts.append({'type': 'output_text' if role == 'assistant' else 'input_text', 'text': part['text']})
                    elif part.get('type') == 'image_url' and role == 'user':
                        reject_unknown(part, {'type', 'image_url'})
                        image = part.get('image_url')
                        if not isinstance(image, dict) or not isinstance(image.get('url'), str):
                            raise APIError(400, 'image_url must contain url')
                        reject_unknown(image, {'url', 'detail'})
                        mapped = {'type': 'input_image', 'image_url': image['url']}
                        if 'detail' in image:
                            mapped['detail'] = image['detail']
                        parts.append(mapped)
                    else:
                        raise APIError(400, 'Only text and user image_url content parts are supported')
            else:
                raise APIError(400, 'content must be a string, array, or null')
            inputs.append({'role': 'developer' if role == 'system' else role, 'content': parts})
        calls = message.get('tool_calls', [])
        if not isinstance(calls, list) or (calls and role != 'assistant'):
            raise APIError(400, 'Only assistant messages may have tool_calls')
        if content is None and not calls:
            raise APIError(400, 'Message needs content or tool_calls')
        for call in calls:
            if not isinstance(call, dict) or call.get('type') != 'function':
                raise APIError(400, 'Only function tool calls are supported')
            fn = call.get('function', {})
            if not isinstance(fn, dict) or not all(isinstance(v, str) for v in (call.get('id'), fn.get('name'), fn.get('arguments'))):
                raise APIError(400, 'Invalid function tool call')
            inputs.append({'type': 'function_call', 'call_id': call['id'], 'name': fn['name'], 'arguments': fn['arguments']})
    out = {'model': body.get('model'), 'input': inputs, 'instructions': '', 'tools': []}
    tool_list = body.get('tools', [])
    if not isinstance(tool_list, list):
        raise APIError(400, 'tools must be an array')
    for tool in tool_list:
        if not isinstance(tool, dict) or tool.get('type') != 'function' or not isinstance(tool.get('function'), dict):
            raise APIError(400, 'Chat tools must be function definitions; use Responses for other tool types')
        reject_unknown(tool, {'type', 'function'})
        fn = tool['function']
        reject_unknown(fn, {'name', 'description', 'parameters', 'strict'})
        if not isinstance(fn.get('name'), str):
            raise APIError(400, 'Function requires a name')
        out['tools'].append(dict(fn, type='function'))
    for field in ('parallel_tool_calls', 'store'):
        if field in body:
            out[field] = body[field]
    if 'tool_choice' in body:
        choice = body['tool_choice']
        if isinstance(choice, dict):
            if choice.get('type') != 'function' or not isinstance(choice.get('function'), dict) or not isinstance(choice['function'].get('name'), str):
                raise APIError(400, 'Invalid tool_choice')
            choice = {'type': 'function', 'name': choice['function']['name']}
        out['tool_choice'] = choice
    if 'reasoning_effort' in body:
        out['reasoning'] = {'effort': body['reasoning_effort']}
    if 'response_format' in body:
        fmt = body['response_format']
        if not isinstance(fmt, dict):
            raise APIError(400, 'Invalid response_format')
        if fmt.get('type') == 'json_schema' and isinstance(fmt.get('json_schema'), dict):
            out['text'] = {'format': dict(fmt['json_schema'], type='json_schema')}
        elif fmt.get('type') in ('text', 'json_object'):
            out['text'] = {'format': fmt}
        else:
            raise APIError(400, 'Unsupported response_format')
    return normalize_response(out)


def response_to_chat(response):
    texts, calls = [], []
    for item in response.get('output', []):
        if item.get('type') == 'message':
            for part in item.get('content', []):
                if part.get('type') == 'output_text':
                    texts.append(part['text'])
                elif part.get('type') == 'refusal':
                    texts.append(part.get('refusal', ''))
        elif item.get('type') == 'function_call':
            calls.append({'id': item['call_id'], 'type': 'function',
                          'function': {'name': item['name'], 'arguments': item['arguments']}})
    message = {'role': 'assistant', 'content': ''.join(texts) or None}
    if calls:
        message['tool_calls'] = calls
    usage = response.get('usage') or {}
    return {'id': response.get('id'), 'object': 'chat.completion',
            'created': response.get('created_at', int(time.time())), 'model': response.get('model'),
            'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if calls else 'stop'}],
            'usage': {'prompt_tokens': usage.get('input_tokens', 0),
                      'completion_tokens': usage.get('output_tokens', 0),
                      'total_tokens': usage.get('total_tokens', 0),
                      'prompt_tokens_details': usage.get('input_tokens_details', {}),
                      'completion_tokens_details': usage.get('output_tokens_details', {})}}


def sse_events(lines):
    data = []
    for line in lines:
        if line == '':
            if data:
                joined = '\n'.join(data)
                data = []
                if joined == '[DONE]':
                    return
                yield json.loads(joined)
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    if data and '\n'.join(data) != '[DONE]':
        yield json.loads('\n'.join(data))


class Gateway:
    def __init__(self, auth_file, key, client=None):
        self.auth_file, self.key = Path(auth_file), key
        self.client = client or httpx.Client(timeout=httpx.Timeout(180, connect=20),
                                             follow_redirects=False,
                                             limits=httpx.Limits(max_connections=16, max_keepalive_connections=8))
        self.slots = threading.BoundedSemaphore(8)

    def headers(self):
        try:
            tokens = json.loads(self.auth_file.read_text())['tokens']
            token, account = tokens['access_token'], tokens['account_id']
            if not isinstance(token, str) or not token or not isinstance(account, str) or not account:
                raise ValueError()
        except (OSError, ValueError, KeyError, TypeError):
            raise APIError(503, 'Cannot read an existing Codex ChatGPT login', 'auth_unavailable') from None
        return {'Authorization': 'Bearer ' + token, 'Chatgpt-Account-Id': account,
                'User-Agent': 'local-codex-api/0.1', 'Originator': 'local_codex_api',
                'Accept': 'text/event-stream'}

    @staticmethod
    def check_status(response):
        if response.status_code == 401:
            raise APIError(401, 'Codex login expired or was rejected. Renew it with Codex, then retry; this service rereads auth.json.', 'upstream_auth_error')
        if response.status_code >= 300:
            # Do not echo upstream bodies: they can contain request data or identifiers.
            raise APIError(response.status_code if response.status_code < 600 else 502,
                           f'Codex upstream returned HTTP {response.status_code}', 'upstream_error')

    def models(self, audit=None):
        r = self.client.get(UPSTREAM + '/models', params={'client_version': '0.162.1'}, headers=self.headers(),
                            extensions={'trace': audit.trace} if audit else {})
        if audit: audit.stage('upstream_headers', upstream_status=r.status_code)
        self.check_status(r)
        data = r.json()
        models = data.get('models', data.get('data', []))
        return {'object': 'list', 'data': [{'id': m.get('slug') or m.get('id'), 'object': 'model',
                                          'created': 0, 'owned_by': 'openai'} for m in models]}

    def events(self, body, audit=None):
        if audit: audit.stage('upstream_connect_or_pool_wait')
        with self.client.stream('POST', UPSTREAM + '/responses', headers=self.headers(), json=body,
                                extensions={'trace': audit.trace} if audit else {}) as r:
            if audit: audit.stage('upstream_headers', upstream_status=r.status_code)
            self.check_status(r)
            completed = False
            if audit: audit.stage('waiting_upstream_terminal')
            lines = audit.lines(r.iter_lines()) if audit else r.iter_lines()
            for event in sse_events(lines):
                if audit: audit.event(event)
                yield event
                if event.get('type') in ('response.completed', 'response.failed', 'response.incomplete', 'error'):
                    completed = True
                    break
            if not completed:
                raise APIError(502, 'Upstream stream ended without a terminal response', 'incomplete_stream')


def collect(events):
    items = {}
    for event in events:
        kind = event.get('type')
        if kind == 'response.output_item.done':
            items[event.get('output_index', len(items))] = event['item']
        if kind in ('error', 'response.failed', 'response.incomplete'):
            raise APIError(502, 'Upstream generation failed or was incomplete', 'generation_failed')
        if kind == 'response.completed':
            response = event['response']
            if response.get('status') != 'completed':
                raise APIError(502, 'Upstream response was not completed', 'generation_failed')
            if not response.get('output') and items:
                response['output'] = [items[k] for k in sorted(items)]
            return response
    raise APIError(502, 'Missing completed response', 'incomplete_stream')


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 closes each downstream response; the shared upstream client pools connections.
    server_version = 'LocalCodexAPI/0.2'

    def log_message(self, *_):
        pass  # No prompts, responses, headers, paths, or tokens in request logs.

    def send_json(self, status, value):
        self.audit.stage('downstream_write', downstream_status=status)
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Request-Id', self.audit.request_id)
        self.end_headers()
        self.wfile.write(data)

    def send_failure(self, status, value, streaming):
        # A disconnected caller must not hide the original upstream failure.
        try:
            if streaming: self.send_event(dict(value, type='error'))
            else: self.send_json(status, value)
        except OSError as error:
            self.audit.emit('downstream_disconnected', exception_type=type(error).__name__)

    def authorize(self):
        if self.headers.get('Origin'):
            raise APIError(403, 'Browser origins are not accepted')
        supplied = self.headers.get('Authorization', '').encode('utf-8')
        expected = ('Bearer ' + self.server.gateway.key).encode('utf-8')
        if not hmac.compare_digest(supplied, expected):
            raise APIError(401, 'A local proxy API key is required', 'authentication_error')

    def run_request(self, post=False):
        streaming = False
        acquired = False
        self.audit = RequestAudit(local_request_id(self.headers.get('X-Client-Request-Id')))
        self.audit.start_heartbeat()
        self.connection.settimeout(30)
        try:
            self.authorize()
            gateway = self.server.gateway
            if not post:
                if self.path == '/health':
                    self.send_json(200, {'status': 'ok', 'stateless': True, 'agent_runtime': False,
                                         'default_instructions': '', 'default_tools': [],
                                         'auth_refresh': 'external_codex', 'pid': os.getpid(),
                                         'diagnostics_version': 1})
                elif self.path == '/v1/models':
                    self.send_json(200, gateway.models(self.audit))
                else:
                    raise APIError(404, 'Unknown endpoint')
                return
            if self.path not in ('/v1/responses', '/v1/chat/completions'):
                raise APIError(404, 'Unknown endpoint')
            if self.headers.get('Transfer-Encoding'):
                raise APIError(400, 'Use Content-Length; chunked request bodies are not supported')
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > MAX_BODY:
                raise APIError(413, 'JSON body must be between 1 byte and 16 MiB')
            self.audit.stage('reading_client_body', request_bytes=length)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise APIError(400, 'Incomplete request body')
            self.audit.peer_socket = self.connection
            body = json.loads(raw)
            chat = self.path.endswith('/chat/completions')
            upstream = chat_to_response(body) if chat else normalize_response(body)
            self.audit.stage('local_admission')
            acquired = gateway.slots.acquire(blocking=False)
            if not acquired:
                raise APIError(429, 'Local concurrency limit reached (8)', 'local_overload')
            events = gateway.events(upstream, self.audit)
            try:
                if body.get('stream', False):
                    first = next(events)  # Surface HTTP/auth failure before sending HTTP 200.
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Request-Id', self.audit.request_id)
                    self.end_headers()
                    streaming = True
                    self.send_event(first)
                    for event in events:
                        self.send_event(event)
                else:
                    response = collect(events)
                    self.send_json(200, response_to_chat(response) if chat else response)
            finally:
                events.close()
        except APIError as error:
            self.audit.emit('request_error', error_code=error.code, exception_type=type(error).__name__, http_status=error.status)
            value = {'error': {'message': str(error), 'type': error.code, 'code': error.code,
                               'request_id': self.audit.request_id}}
            self.send_failure(error.status, value, streaming)
        except (BrokenPipeError, ConnectionResetError) as error:
            self.audit.emit('downstream_disconnected', exception_type=type(error).__name__)
        except (ValueError, TypeError, KeyError) as error:
            self.audit.emit('request_error', error_code='invalid_data', exception_type=type(error).__name__)
            value = {'error': {'message': 'Invalid JSON or request/response shape', 'type': 'invalid_data'}}
            self.send_failure(400, value, streaming)
        except (httpx.HTTPError, OSError) as error:
            code = next((code for cls, code in (
                (httpx.ConnectTimeout, 'upstream_connect_timeout'), (httpx.ReadTimeout, 'upstream_read_timeout'),
                (httpx.WriteTimeout, 'upstream_write_timeout'), (httpx.PoolTimeout, 'upstream_pool_timeout'),
                (httpx.ProxyError, 'upstream_proxy_error'), (httpx.RemoteProtocolError, 'upstream_protocol_error'))
                if isinstance(error, cls)), 'connection_error')
            self.audit.emit('request_error', error_code=code, exception_type=type(error).__name__)
            value = {'error': {'message': 'Connection failed or timed out; no automatic inference retry',
                               'type': code, 'code': code, 'request_id': self.audit.request_id}}
            self.send_failure(502, value, streaming)
        except Exception as error:
            self.audit.emit('request_error', error_code='internal_error', exception_type=type(error).__name__)
            self.send_failure(500, {'error': {'type': 'internal_error', 'code': 'internal_error'}}, streaming)
        finally:
            if acquired:
                self.server.gateway.slots.release()
            self.audit.finish()

    def send_event(self, event):
        data = json.dumps(event, ensure_ascii=False)
        self.wfile.write(('data: ' + data + '\n\n').encode())
        self.wfile.flush()

    def do_GET(self):
        self.run_request()

    def do_POST(self):
        self.run_request(post=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8317)
    p.add_argument('--auth-file', type=Path, default=Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex'))) / 'auth.json')
    p.add_argument('--key-file', type=Path, required=True)
    args = p.parse_args()
    args.key_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(args.key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'w') as f:
            f.write(secrets.token_urlsafe(32) + '\n')
    if args.key_file.stat().st_mode & 0o077:
        p.error('key-file must have mode 0600 or stricter')
    key = args.key_file.read_text().strip()
    if len(key) < 24:
        p.error('key-file must contain at least 24 characters')
    gateway = Gateway(args.auth_file, key)
    gateway.headers()  # Fail at startup if no usable login exists; never print it.
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.gateway = gateway
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print(json.dumps({'event': 'ready', 'base_url': f'http://127.0.0.1:{args.port}/v1', 'pid': os.getpid()}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        gateway.client.close()


if __name__ == '__main__':
    main()
