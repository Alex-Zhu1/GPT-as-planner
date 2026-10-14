"""Offline HTTP/SSE fault injection; never contacts the model upstream."""
from contextlib import contextmanager, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import socket

import httpx

from tools.codex_api.server import Gateway, Handler, RequestAudit, ThreadingHTTPServer
from hybrid_rollout.model_api import ModelAPIError, ResponsesClient


class DiagnosticsTests(unittest.TestCase):
    @contextmanager
    def server(self, upstream, timeout=1):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()) as logs:
            root = Path(directory)
            auth = root/'auth.json'
            auth.write_text(json.dumps({'tokens': {'access_token': 'SECRET_TOKEN', 'account_id': 'SECRET_ACCOUNT'}}))
            key = root/'key'; key.write_text('SECRET_LOCAL_KEY')
            http = httpx.Client(transport=httpx.MockTransport(upstream))
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            server.gateway = Gateway(auth, key.read_text(), http)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            client = ResponsesClient(f'http://127.0.0.1:{server.server_port}/v1', key, timeout)
            try:
                yield client, root/'client.jsonl', logs
            finally:
                server.shutdown(); server.server_close(); thread.join(); http.close()

    def assert_private(self, logs, trace):
        combined = logs.getvalue()+trace.read_text()
        for secret in ('SECRET_PROMPT', 'SECRET_TOKEN', 'SECRET_ACCOUNT', 'SECRET_LOCAL_KEY', 'SECRET_ERROR'):
            self.assertNotIn(secret, combined)

    def test_success_correlates_request_without_changing_payload(self):
        seen=[]
        def upstream(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, text='data: '+json.dumps({'type':'response.completed',
                'response':{'status':'completed','output':[]}})+'\n\n')
        with self.server(upstream) as (client, trace, logs):
            self.assertEqual(client({'model':'test','input':'SECRET_PROMPT','tools':[]},diagnostic_path=trace)['status'],'completed')
            entries=[json.loads(line) for line in logs.getvalue().splitlines()]
            request_id=client.diagnostics()['request_id']
            self.assertTrue(all(e['request_id']==request_id for e in entries))
            self.assertEqual(client.diagnostics()['gateway_request_id'],request_id)
            self.assertTrue(any(e.get('last_event_type')=='response.completed' for e in entries))
            self.assertEqual(seen[0]['input'][0]['content'],'SECRET_PROMPT')
            self.assertEqual(len(seen),1)
            self.assert_private(logs,trace)

    def test_distinguishes_network_failures_and_preserves_error_code(self):
        for error, code in ((httpx.ConnectTimeout,'upstream_connect_timeout'),
                            (httpx.ReadTimeout,'upstream_read_timeout'),
                            (httpx.PoolTimeout,'upstream_pool_timeout'),
                            (httpx.ProxyError,'upstream_proxy_error'),
                            (httpx.RemoteProtocolError,'upstream_protocol_error')):
            with self.subTest(code=code):
                count=[]
                def upstream(request):
                    count.append(1)
                    raise error('SECRET_ERROR')
                with self.server(upstream) as (client,trace,logs):
                    with self.assertRaises(ModelAPIError) as caught:
                        client({'model':'test','input':'SECRET_PROMPT'},diagnostic_path=trace)
                    self.assertEqual(caught.exception.diagnostics['gateway_error_code'],code)
                    self.assertIn(error.__name__,logs.getvalue())
                    self.assertEqual(len(count),1)
                    self.assert_private(logs,trace)

    def test_http_status_and_terminal_failure_and_truncation_are_distinct(self):
        for status,text,expected in (
            (502,'SECRET_ERROR','upstream_error'),
            (401,'SECRET_ERROR','upstream_auth_error'),
            (200,'data: {"type":"response.failed","error":{"message":"SECRET_ERROR"}}\n\n','generation_failed'),
            (200,': ping\n\ndata: {"type":"response.in_progress"}\n\n','incomplete_stream')):
            with self.subTest(expected=expected):
                with self.server(lambda r:httpx.Response(status,text=text)) as (client,trace,logs):
                    with self.assertRaises(ModelAPIError) as caught:
                        client({'model':'test','input':'SECRET_PROMPT'},diagnostic_path=trace)
                    self.assertEqual(caught.exception.diagnostics['gateway_error_code'],expected)
                    self.assert_private(logs,trace)

    def test_stream_activity_can_outlive_downstream_timeout_without_terminal(self):
        release=threading.Event();closed=threading.Event()
        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                yield b'data: {"type":"response.in_progress"}\n\n'
                # SSE activity need not be a completed model response. The
                # gateway buffers non-streaming output while the caller waits.
                while not release.wait(.01): yield b': keepalive\n\n'
                yield b'data: {"type":"response.completed","response":{"status":"completed","output":[]}}\n\n'
            def close(self):closed.set()
        with self.server(lambda r:httpx.Response(200,stream=Stream()),timeout=.08) as (client,trace,logs):
            try:
                with self.assertRaises(ModelAPIError) as caught:
                    client({'model':'test','input':'SECRET_PROMPT'},diagnostic_path=trace)
                diagnostic=caught.exception.diagnostics
                self.assertEqual(diagnostic['stage'],'waiting_gateway_headers')
                self.assertTrue(diagnostic['timed_out'])
                self.assertNotIn('gateway_error_code',diagnostic)
                entries=[json.loads(line) for line in logs.getvalue().splitlines()]
                self.assertTrue(any(e.get('upstream_status')==200 and e.get('last_event_type')=='response.in_progress' for e in entries))
                self.assertFalse(any(e.get('last_event_type')=='response.completed' for e in entries))
            finally:
                release.set();self.assertTrue(closed.wait(2));time.sleep(.03)
            self.assert_private(logs,trace)

    def test_trace_callback_does_not_log_network_payloads(self):
        with redirect_stderr(io.StringIO()) as logs:
            audit=RequestAudit('a'*32)
            audit.trace('connection.connect_tcp.failed',{'exception':httpx.ConnectError('SECRET_ERROR'),
                                                        'host':'SECRET_HOST','port':123})
            audit.trace('SECRET_EVENT.started',{'exception':'SECRET_ERROR'})
            audit.finish()
        self.assertIn('ConnectError',logs.getvalue())
        self.assertNotIn('SECRET_',logs.getvalue())

    def test_peer_close_and_upstream_failure_metadata_are_recorded_without_contents(self):
        local,peer=socket.socketpair()
        with redirect_stderr(io.StringIO()) as logs:
            audit=RequestAudit('b'*32);audit.peer_socket=local
            peer.close();audit.check_peer()
            audit.event({'type':'response.failed','response':{'error':{
                'code':'server_error','type':'SECRET_ERROR','message':'SECRET_ERROR'}}})
            audit.finish();local.close()
        entries=[json.loads(line) for line in logs.getvalue().splitlines()]
        self.assertTrue(entries[-1]['downstream_peer_closed'])
        self.assertEqual(entries[-1]['upstream_error_code'],'server_error')
        self.assertEqual(entries[-1]['upstream_error_type'],'unknown')
        self.assertNotIn('SECRET_',logs.getvalue())


if __name__=='__main__':unittest.main()
