import copy
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from tools.codex_api.server import (APIError, Gateway, chat_to_response, collect,
                                    normalize_response, response_to_chat, sse_events)


class ProtocolTests(unittest.TestCase):
    def test_no_prompt_or_tools_injection_and_no_input_mutation(self):
        body = {'model': 'test', 'input': [{'role': 'system', 'content': 'Only my instruction'},
                                          {'role': 'user', 'content': 'Only my observation'}]}
        original = copy.deepcopy(body)
        out = normalize_response(body)
        self.assertEqual(body, original)
        self.assertEqual(out, {'model': 'test', 'input': [
            {'role': 'developer', 'content': 'Only my instruction'},
            {'role': 'user', 'content': 'Only my observation'}],
            'instructions': '', 'tools': [], 'store': False, 'stream': True})

    def test_explicit_instructions_and_tools_are_preserved(self):
        body = {'model': 'test', 'input': 'choose', 'instructions': 'my instruction',
                'tools': [{'type': 'function', 'name': 'move', 'parameters': {'type': 'object'}}]}
        out = normalize_response(body)
        self.assertEqual(out['instructions'], body['instructions'])
        self.assertEqual(out['tools'], body['tools'])

    def test_experiment_parameters_are_not_silently_ignored(self):
        for name, value in [('temperature', 0), ('max_output_tokens', 8),
                            ('previous_response_id', 'old'), ('store', True)]:
            with self.subTest(name=name), self.assertRaises(APIError):
                normalize_response({'model': 'test', 'input': 'hello', name: value})

    def test_chat_image_and_tool_history(self):
        out = chat_to_response({'model': 'test', 'messages': [
            {'role': 'system', 'content': 'my policy'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'observation'},
                                       {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA', 'detail': 'high'}}]},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {'name': 'move', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'done'}],
            'tools': [{'type': 'function', 'function': {'name': 'move', 'parameters': {'type': 'object'}}}]})
        self.assertEqual(out['instructions'], '')
        self.assertEqual(out['input'][0]['role'], 'developer')
        self.assertEqual(out['input'][1]['content'][1],
                         {'type': 'input_image', 'image_url': 'data:image/png;base64,AA', 'detail': 'high'})
        self.assertEqual(out['input'][2]['call_id'], 'c1')
        self.assertEqual(out['input'][3]['type'], 'function_call_output')
        self.assertEqual(out['tools'], [{'type': 'function', 'name': 'move', 'parameters': {'type': 'object'}}])

    def test_sse_multiline_and_terminal_failure(self):
        events = list(sse_events([': keepalive', 'data: {"type":',
                                  'data: "response.failed"}', '', 'data: [DONE]', '']))
        self.assertEqual(events, [{'type': 'response.failed'}])
        with self.assertRaises(APIError):
            collect(iter(events))
        with self.assertRaises(APIError):
            collect(iter([{'type': 'response.output_text.delta', 'delta': 'partial'}]))

    def test_malformed_function_call_is_a_client_error(self):
        with self.assertRaises(APIError) as caught:
            chat_to_response({'model': 'test', 'messages': [{'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'c1', 'type': 'function', 'function': 'invalid'}]}]})
        self.assertEqual(caught.exception.status, 400)

    def test_completed_output_backfill_and_function_return(self):
        result = collect(iter([
            {'type': 'response.output_item.done', 'output_index': 0,
             'item': {'type': 'function_call', 'call_id': 'c1', 'name': 'move', 'arguments': '{}'}},
            {'type': 'response.completed', 'response': {'status': 'completed', 'output': []}}]))
        chat = response_to_chat(result)
        self.assertEqual(chat['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(chat['choices'][0]['message']['tool_calls'][0]['id'], 'c1')

    def test_same_gateway_has_no_history_and_rereads_external_auth(self):
        seen = []
        def upstream(request):
            seen.append((dict(request.headers), json.loads(request.content)))
            return httpx.Response(200, text='data: ' + json.dumps({
                'type': 'response.completed', 'response': {'status': 'completed', 'output': []}}) + '\n\n')
        with tempfile.TemporaryDirectory() as tmp:
            auth = Path(tmp) / 'auth.json'
            client = httpx.Client(transport=httpx.MockTransport(upstream))
            gateway = Gateway(auth, 'test-key', client)
            try:
                for token, prompt in [('first-token', 'secret A'), ('second-token', 'request B')]:
                    auth.write_text(json.dumps({'tokens': {'access_token': token, 'account_id': 'test-account'}}))
                    collect(gateway.events(normalize_response({'model': 'test', 'input': prompt})))
            finally:
                client.close()
        self.assertEqual(seen[1][0]['authorization'], 'Bearer second-token')
        self.assertNotIn('secret A', json.dumps(seen[1][1]))
        self.assertEqual(seen[1][1]['input'], [{'role': 'user', 'content': 'request B'}])


if __name__ == '__main__':
    unittest.main()
