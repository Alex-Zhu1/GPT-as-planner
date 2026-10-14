"""Explicit live smoke test: consumes three short inference requests."""
import argparse
import json
from pathlib import Path
import time

import httpx

from tools.codex_api.server import collect, sse_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8317')
    parser.add_argument('--key-file', type=Path, default=Path('runtime/local-codex-api/api-key'))
    parser.add_argument('--output', type=Path, default=Path('runtime/local-codex-api/smoke.json'))
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    evidence = {'checks': [], 'model': 'gpt-6-astra'}
    with httpx.Client(base_url=args.base_url, headers={'Authorization': 'Bearer ' + key},
                      timeout=180, trust_env=False) as client:
        health = client.get('/health')
        health.raise_for_status()
        evidence['health'] = health.json()
        assert client.get('/health', headers={'Authorization': 'Bearer wrong-key'}).status_code == 401
        assert client.get('/health', headers={'Origin': 'https://example.com'}).status_code == 403
        rejected = client.post('/v1/chat/completions', json={
            'model': 'gpt-6-astra', 'messages': [{'role': 'user', 'content': 'test'}], 'temperature': 0})
        assert rejected.status_code == 400
        evidence['checks'].extend(['local_auth', 'browser_origin_rejected', 'unsupported_parameter_rejected'])
        models = client.get('/v1/models')
        models.raise_for_status()
        evidence['models'] = [m['id'] for m in models.json()['data']]
        assert 'gpt-6-astra' in evidence['models']
        prompts = [
            ('Reply with exactly API_OK. The marker for this request is ORANGE_7429.', 'API_OK'),
            ('If any earlier user message exists in this conversation, repeat its marker. Otherwise reply with exactly NO_HISTORY.', 'NO_HISTORY'),
        ]
        for prompt, expected in prompts:
            start = time.monotonic()
            response = client.post('/v1/chat/completions', json={
                'model': 'gpt-6-astra', 'messages': [{'role': 'user', 'content': prompt}]})
            response.raise_for_status()
            result = response.json()
            text = result['choices'][0]['message']['content']
            evidence['checks'].append({'route': 'chat/completions', 'text': text,
                                       'seconds': round(time.monotonic()-start, 2), 'usage': result['usage']})
            print(json.dumps(evidence['checks'][-1]), flush=True)
            assert text.strip() == expected, text
        start = time.monotonic()
        with client.stream('POST', '/v1/responses', json={
            'model': 'gpt-6-astra', 'instructions': '', 'tools': [],
            'input': 'Reply with exactly STREAM_OK.', 'store': False, 'stream': True}) as stream:
            stream.raise_for_status()
            response = collect(sse_events(stream.iter_lines()))
        text = ''.join(p.get('text', '') for item in response.get('output', [])
                       for p in item.get('content', []) if p.get('type') == 'output_text')
        assert text.strip() == 'STREAM_OK', text
        evidence['checks'].append({'route': 'responses SSE', 'text': text,
                                   'seconds': round(time.monotonic()-start, 2),
                                   'status': response['status']})
        print(json.dumps(evidence['checks'][-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print('All smoke checks passed. Evidence: ' + str(args.output))


if __name__ == '__main__':
    main()
