"""A tiny stand-in for Ollama's OpenAI-compatible API, used only by tests.

It answers GET /v1/models and POST /v1/chat/completions with a fixed reply,
and records each request so tests can check what the harness sent
(model, seed, temperature).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenAIServer:
    def __init__(self, model: str = "llama3.2:3b", reply_position: str = "Vendor B"):
        self.model = model
        self.reply_position = reply_position
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send({"object": "list", "data": [{"id": outer.model, "object": "model"}]})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(req)
                content = (f"I agree the weighted score favours {outer.reply_position}.\n"
                           f'POSITION: {{"position": "{outer.reply_position}", '
                           f'"confidence": 0.8, "cites": ["a1"]}}')
                self._send({
                    "id": f"chatcmpl-{len(outer.requests)}", "object": "chat.completion",
                    "created": 0, "model": req.get("model", outer.model),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                })

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
