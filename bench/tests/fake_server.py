#!/usr/bin/env python3
"""A tiny stand-in for vLLM / mlx_lm.server, for tests: OpenAI-style streaming
/v1/chat/completions, /v1/models, /version and /metrics. Standard library only.

It accepts the protocol's server flags and ignores them, except that it remembers them and
reports a KV cache in /metrics. Each request streams `max_tokens` one-character tokens with a
short delay; requests that carry "reasoning_effort" also stream reasoning text (as a model
that cannot turn thinking off would); a request carrying "seed" is answered with HTTP 400 so
tests catch a seed leaking through."""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
SEEN = []          # request bodies, for tests that inspect them via /seen


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
        elif self.path == "/version":
            self._json(200, {"version": "0.0-fake"})
        elif self.path == "/seen":
            self._json(200, SEEN)
        elif self.path == "/metrics":
            text = ('vllm:cache_config_info{block_size="16",cache_dtype="auto",'
                    'enable_prefix_caching="True",num_gpu_blocks="1000"} 1.0\n'
                    'vllm:num_preemptions_total{model_name="fake"} 0.0\n')
            b = text.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self._json(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        SEEN.append(body)
        if "seed" in body:
            self._json(400, {"error": "seed must not be sent"})
            return
        if not body.get("stream"):
            text = "x" * min(int(body.get("max_tokens") or 16), 64)
            self._json(200, {"choices": [{"message": {"role": "assistant", "content": text},
                                          "finish_reason": "length"}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": len(text)}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        maxt = int(body.get("max_tokens") or 16)
        prompt = sum(len(m.get("content", "")) for m in body.get("messages", [])) // 4
        think = "reasoning_effort" in body
        time.sleep(0.01)
        for i in range(maxt):
            key = "reasoning" if think and i < 2 else "content"
            delta = {key: "x"}
            chunk = {"choices": [{"index": 0, "delta": delta,
                                  "finish_reason": "length" if i == maxt - 1 else None}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            time.sleep(0.001)
        if (body.get("stream_options") or {}).get("include_usage"):
            u = {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": maxt}}
            self.wfile.write(b"data: " + json.dumps(u).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="vllm")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--startup-delay", type=float, default=0.3)
    ARGS, rest = ap.parse_known_args()
    print(f"fake {ARGS.engine} server, flags {rest}", flush=True)
    print("Model loading took 1.00 GiB memory and 0.1 seconds", flush=True)
    print("GPU KV cache size: 16,000 tokens", flush=True)
    time.sleep(ARGS.startup_delay)
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
