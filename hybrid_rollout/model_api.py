"""Shared image encoding and HTTP transport for stateless visual model requests."""
import base64
import hashlib
import json
import http.client
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MAX_REQUEST_BYTES = 16 * 1024 * 1024  # Matches the local model gateway.
GATEWAY_ERROR_CODES = {
    'connection_error', 'upstream_error', 'upstream_auth_error', 'auth_unavailable',
    'incomplete_stream', 'generation_failed', 'invalid_data', 'local_overload',
    'authentication_error', 'invalid_request', 'internal_error', 'upstream_connect_timeout',
    'upstream_read_timeout', 'upstream_write_timeout', 'upstream_pool_timeout',
    'upstream_proxy_error', 'upstream_protocol_error',
}


class ModelAPIError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def safe_request_id(value):
    return value if isinstance(value, str) and len(value) == 32 and all(c in '0123456789abcdef' for c in value) else None


def image_part(image):
    data = Path(image['path']).read_bytes()
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        mime = 'image/png'
    elif data.startswith(b'\xff\xd8\xff'):
        mime = 'image/jpeg'
    elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        mime = 'image/webp'
    else:
        raise ValueError('API evidence must be PNG, JPEG, or WebP image bytes')
    return dict(type='input_image', image_url='data:' + mime + ';base64,' +
                base64.b64encode(data).decode('ascii'), detail='high'), hashlib.sha256(data).hexdigest()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class ResponsesClient:
    def __init__(self, base_url, key_file, timeout):
        parts = urlsplit(base_url)
        if (parts.scheme not in ('http', 'https') or not parts.hostname or
                parts.username or parts.password or parts.query or parts.fragment):
            raise ValueError('API base URL must be HTTP(S), without credentials, query, or fragment')
        self.base_url = base_url.rstrip('/')
        self.key_file, self.timeout = Path(key_file), timeout
        if not self.key_file.is_file() or not self.key_file.read_text().strip():
            raise ValueError('API key file is missing or empty')
        # Local gateway calls must not inherit the Codex upstream proxy.
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self._diagnostics_lock = threading.Lock()
        self._diagnostics = {}

    def diagnostics(self):
        with self._diagnostics_lock:
            return dict(self._diagnostics)

    def __call__(self, body, *, diagnostic_path=None):
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        state = dict(request_id=request_id, stage='encode_request', timeout_seconds=self.timeout)
        def record(event, **fields):
            state.update(fields)
            entry = dict(state, event=event, elapsed_seconds=time.monotonic()-started,
                         timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            with self._diagnostics_lock:
                self._diagnostics = entry
                if diagnostic_path is not None:
                    with Path(diagnostic_path).open('a') as stream:
                        stream.write(json.dumps(entry)+'\n')
                else:
                    print(json.dumps(entry), file=sys.stderr, flush=True)
            return entry
        record('client_request_start')
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        if len(data) > MAX_REQUEST_BYTES:
            raise ValueError('Review exceeds the local API 16 MiB body limit; reduce image resolution explicitly')
        request = Request(self.base_url + '/responses', data=data, headers={
            'Authorization': 'Bearer ' + self.key_file.read_text().strip(),
            'X-Client-Request-Id': request_id,
            'Content-Type': 'application/json'}, method='POST')
        try:
            record('client_http_started', stage='waiting_gateway_headers', request_bytes=len(data))
            with self.opener.open(request, timeout=self.timeout) as response:
                record('gateway_headers', stage='reading_gateway_body', http_status=response.status,
                       gateway_request_id=safe_request_id(response.headers.get('X-Request-Id')))
                raw = response.read()
                record('gateway_body_received', stage='decode_response', response_bytes=len(raw))
                result = json.loads(raw)
                record('client_request_completed', stage='completed')
                return result
        except HTTPError as error:
            record('gateway_error_headers', stage='reading_gateway_error_body', http_status=error.code)
            details = {}
            try:
                payload = json.loads(error.read(16384))
                if isinstance(payload, dict) and isinstance(payload.get('error'), dict): details = payload['error']
            except Exception as parse_error:
                record('gateway_error_body_unavailable', body_exception_type=type(parse_error).__name__)
            finally:
                error.close()
            code = details.get('code') or details.get('type')
            code = code if isinstance(code, str) and code in GATEWAY_ERROR_CODES else 'unknown'
            diagnostic = record('client_request_failed', exception_type='HTTPError', gateway_error_code=code,
                                gateway_request_id=safe_request_id(error.headers.get('X-Request-Id')) if error.headers else None)
            raise ModelAPIError(f'Model API returned HTTP {error.code} ({code}); request_id={request_id}; no automatic retry', diagnostic) from None
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as error:
            reason = error.reason if isinstance(error, URLError) else error
            diagnostic = record('client_request_failed', exception_type=type(error).__name__,
                                cause_type=type(reason).__name__, timed_out=isinstance(reason, TimeoutError))
            raise ModelAPIError(f'Model API connection failed or timed out at {state["stage"]} '
                                f'({type(reason).__name__}); request_id={request_id}; no automatic retry', diagnostic) from None
        except (ValueError, TypeError) as error:
            diagnostic = record('client_request_failed', exception_type=type(error).__name__)
            raise ModelAPIError(f'Model API invalid response; request_id={request_id}; no automatic retry', diagnostic) from None

    def close(self):
        pass


def structured_payload(response, model='gpt-6-astra'):
    if not isinstance(response, dict) or response.get('model') != model:
        raise ValueError(f'Model API returned a different model than {model}')
    if response.get('status') != 'completed':
        raise ValueError('Model API response did not complete')
    texts = []
    for item in response.get('output', []):
        if item.get('type') == 'reasoning':
            continue
        if item.get('type') != 'message':
            raise ValueError('Model API returned a tool call or unsupported output')
        for part in item.get('content', []):
            if part.get('type') != 'output_text':
                raise ValueError('Model API refused or returned non-text output')
            texts.append(part['text'])
    payload = json.loads(''.join(texts))
    if not isinstance(payload, dict):
        raise ValueError('Model API output must be a JSON object')
    return payload
