#!/usr/bin/env python3
"""回放式 token 服务（闭环三臂用）：把一个 (T,64) token 数组按协议发出去。

与 `scripts/ha_vla_latent64_server.py` **协议完全一致**（`GET /`、`POST /reset`、`POST /infer`），
所以闭环只要换 `SERVER_URL`。相比旧的 `ha_gt_token_server.py`，本服务额外提供：

* `--consume`：每次 infer 前进多少帧（默认 **20** = 部署口径：sonic provider 的
  `HA_VLA_EXEC_STEPS=20`，即 2.5 Hz @ 50 Hz 重规划，一个 40 行 chunk 只执行前 20 帧）；
  若 provider 仍按旧行为全消费 40 帧，就传 `--consume 40`，否则 token 流会 2× 慢放；
* `hands14_chunk`：可选，从 npz 的 `g1_hand_joints_14` 回放（否则手部保持为零 = 张开）。
  三臂必须用**同一套手部处理**，否则比的是手不是 token。

用法：
    python3 scripts/ha_replay_token_server.py --tokens runs/oracle_tokens/xxx.npy --port 18083
    python3 scripts/ha_replay_token_server.py --npz-tokens data/npz/ha_tokens/xxx.npz --key g1_motion_token_64 \
        --hands-npz data/npz/ha_tokens/xxx.npz --port 18082 --consume 20
"""
from __future__ import annotations

import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


class Handler(BaseHTTPRequestHandler):
    tokens: np.ndarray = None       # (T,64)
    hands: np.ndarray | None = None  # (T,14) or None
    cursor: int = 0
    horizon: int = 40      # 服务端一次返回的行数（chunk 长度，部署侧固定 40）
    consume: int = 20      # 每次 infer 前进的帧数 = 重规划周期（HA_VLA_EXEC_STEPS）
    n_infer: int = 0

    def log_message(self, fmt, *a):
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(200, {"ok": True, "service": "ha-replay-token", "cursor": Handler.cursor,
                         "frames": int(Handler.tokens.shape[0]), "consume": Handler.consume})

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
        json.loads(self.rfile.read(n).decode())     # touch payload

        T = Handler.tokens.shape[0]
        start = Handler.cursor % T
        idx = (np.arange(Handler.horizon) + start) % T
        chunk = Handler.tokens[idx]
        hands = Handler.hands[idx] if Handler.hands is not None else np.zeros((Handler.horizon, 14), np.float32)
        Handler.cursor += int(Handler.consume)
        Handler.n_infer += 1
        binary = (np.abs(hands) > 1.70).astype(np.float32)
        print(f"[replay] infer #{Handler.n_infer} rows {start}..{start + Handler.horizon} "
              f"tok_mean={chunk.mean():+.3f} hand_max={float(np.abs(hands).max()):.3f}", flush=True)
        self._json(200, {"latent64_chunk": chunk.tolist(),
                         "hand_binary_chunk": binary.tolist(),
                         "hands14_chunk": hands.tolist(),
                         "dt_ms": 0.0})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=str, default=None, help="(T,64) 的 .npy")
    ap.add_argument("--npz-tokens", type=str, default=None, help="含 token 数组的 npz")
    ap.add_argument("--key", type=str, default="g1_motion_token_64")
    ap.add_argument("--hands-npz", type=str, default=None, help="含 g1_hand_joints_14 的 npz（可选）")
    ap.add_argument("--hands-key", type=str, default="g1_hand_joints_14")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--consume", type=int, default=20,
                    help="每次 infer 前进帧数；= 部署的 HA_VLA_EXEC_STEPS（默认 20）")
    ap.add_argument("--port", type=int, default=18083)
    a = ap.parse_args()

    if a.tokens:
        tok = np.load(a.tokens).astype(np.float32)
    elif a.npz_tokens:
        with np.load(a.npz_tokens) as z:
            tok = np.asarray(z[a.key], np.float32)
    else:
        raise SystemExit("需要 --tokens 或 --npz-tokens")
    hands = None
    if a.hands_npz:
        with np.load(a.hands_npz) as z:
            hands = np.asarray(z[a.hands_key], np.float32)
        m = min(len(hands), len(tok))
        hands, tok = hands[:m], tok[:m]
    Handler.tokens, Handler.hands = tok, hands
    Handler.horizon, Handler.consume = int(a.horizon), int(a.consume)
    print(f"[replay] tokens={tok.shape} hands={'none' if hands is None else hands.shape} "
          f"horizon={a.horizon} consume={a.consume} port={a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
