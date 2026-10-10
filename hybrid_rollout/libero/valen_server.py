"""Sequential loopback /predict endpoint for a local Valen checkpoint."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import traceback


def make_handler(predict):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, result):
            data = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200 if self.path == '/health' else 404,
                       {'ready': True} if self.path == '/health' else {'error': 'not found'})

        def do_POST(self):
            if self.path != '/predict':
                return self.reply(404, {'error': 'not found'})
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 4 * 1024 * 1024:
                    return self.reply(413, {'error': 'invalid body size'})
                self.reply(200, predict(json.loads(self.rfile.read(size))))
            except Exception as error:
                traceback.print_exc()
                self.reply(500, {'error': type(error).__name__, 'detail': str(error)})
    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--port', type=int, default=8001)
    a = p.parse_args(argv)
    import torch
    from transformers import AutoModel
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(a.model, local_files_only=True, trust_remote_code=True,
        dtype='auto', attn_implementation='sdpa').to('cuda').eval()
    model.make_compiler()
    with HTTPServer(('127.0.0.1', a.port), make_handler(
            lambda request: model.predict(request, execution='shared_state'))) as server:
        print('VALEN_MODEL_AND_HTTP_READY', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
