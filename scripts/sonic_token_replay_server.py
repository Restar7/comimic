#!/usr/bin/env python3
"""Minimal /infer server that replays a recorded 64-d SONIC token stream.

Used to run the **upstream SONIC decoder** (`model_decoder.onnx`) on exactly the same HA
tokens that the SuperMimic GT replay used, in the same Isaac HA scene.  The SONIC provider
(``HA_ALLOW_SONIC=1``) only needs ``latent64_chunk``; the image/state in the request are
ignored.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


class Replay:
    def __init__(self, tokens: str, horizon: int, exec_steps: int, loop: bool):
        t = np.load(tokens).astype(np.float32)
        if t.ndim != 2 or t.shape[1] != 64:
            raise SystemExit(f"tokens must be (T,64), got {t.shape}")
        self.tokens = t
        self.horizon = int(horizon)
        self.exec_steps = int(exec_steps)
        self.loop = bool(loop)
        self.cursor = 0
        self.n_infer = 0
        self.lock = threading.Lock()

    def next_chunk(self) -> np.ndarray:
        with self.lock:
            self.n_infer += 1
            T = self.tokens.shape[0]
            if self.loop and self.cursor + self.horizon > T:
                self.cursor = 0
            start = min(self.cursor, max(0, T - 1))
            idx = np.minimum(np.arange(start, start + self.horizon), T - 1)
            chunk = self.tokens[idx]
            self.cursor += self.exec_steps
            return chunk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--exec-steps", type=int, default=20)
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    replay = Replay(a.tokens, a.horizon, a.exec_steps, a.loop)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json(200, {"ok": True, "service": "sonic-token-replay",
                             "cursor": replay.cursor, "n_infer": replay.n_infer,
                             "T": int(replay.tokens.shape[0])})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n:
                self.rfile.read(n)
            if self.path.rstrip("/") == "/reset":
                with replay.lock:
                    replay.cursor = 0
                    replay.n_infer = 0
                self._json(200, {"ok": True})
                return
            chunk = replay.next_chunk()
            self._json(200, {"latent64_chunk": chunk.tolist()})

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[replay] {a.host}:{a.port} tokens={replay.tokens.shape} horizon={a.horizon} "
          f"exec_steps={a.exec_steps} loop={a.loop}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
