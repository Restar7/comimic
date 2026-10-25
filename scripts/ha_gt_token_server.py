#!/usr/bin/env python3
"""真值 token 服务：把 HA 数据集里的 `encoder_latent` 当成 VLA 输出发出去。

用途：把"执行器（WBC）行不行"和"VLA 预测得好不好"解耦。
    VLA 服务  → 闭环摔倒：可能是 VLA token 太差，也可能是 WBC 太弱
    本服务    → 闭环还摔：WBC/场景的问题
                → 闭环能站/能走：WBC 没问题，瓶颈在 VLA

协议与 scripts/ha_vla_latent64_server.py 完全一致（`/infer`、`/reset`、`GET /`），
所以直接把闭环的 SERVER_URL 指过来即可。

用法：
    python3 scripts/ha_gt_token_server.py --npz data/ha_raw/HOI_double_desk/tw/xxx.npz --port 18081
    SERVER_URL=http://127.0.0.1:18081 bash scripts/ha_isaac_closedloop.sh
"""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


class Handler(BaseHTTPRequestHandler):
    tokens: np.ndarray = None      # (T, 64)
    cursor: int = 0
    horizon: int = 40
    n_infer: int = 0

    def log_message(self, fmt, *a):  # keep the console readable
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(200, {"ok": True, "service": "ha-gt-token", "cursor": Handler.cursor,
                         "frames": int(Handler.tokens.shape[0])})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/reset":
            Handler.cursor = 0
            Handler.n_infer = 0
            self._json(200, {"ok": True})
            return
        if path != "/infer":
            self._json(404, {"error": f"unknown path {self.path}"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n).decode())
        # touch the payload so a malformed request fails the same way the real server would
        base64.b64decode(req["observation"]["images"]["front"]["data_b64"])

        T = Handler.tokens.shape[0]
        start = Handler.cursor % T
        idx = (np.arange(Handler.horizon) + start) % T
        chunk = Handler.tokens[idx]
        Handler.cursor += Handler.horizon

        Handler.n_infer += 1
        print(f"[gt] infer #{Handler.n_infer} rows {start}..{(start + Handler.horizon) % T} "
              f"tok_mean={chunk.mean():+.3f} tok_std={chunk.std():.3f}", flush=True)
        self._json(200, {"latent64_chunk": chunk.tolist(),
                         "hand_binary_chunk": [[0.0, 0.0]] * Handler.horizon,
                         "dt_ms": 0.0})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--key", default="encoder_latent")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--horizon", type=int, default=40)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    tokens = np.asarray(z[args.key], dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 64:
        raise SystemExit(f"{args.key} has shape {tokens.shape}, expected (T, 64)")
    Handler.tokens = tokens
    Handler.horizon = int(args.horizon)
    print(f"[gt] loaded {args.npz} {args.key} {tokens.shape} "
          f"mean={tokens.mean():+.3f} std={tokens.std():.3f}", flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.verbose = False
    print(f"[gt] listening on http://127.0.0.1:{args.port}/infer", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
