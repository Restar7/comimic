#!/usr/bin/env python3
"""**相位服务器**（N1'）：用机器人当前位姿决定"该回放录制命令的哪一段"。

为什么不是"直接回放"
--------------------
`scripts/ha_gt_token_server.py` 是**开环**回放：按请求序号推进 cursor。实测
（`runs/closedloop_r3/oracle2k`，2000 步）它会发散：`pos p50 117.86 cm`、`prog_end 0.124`、
净位移 0.388 m。原因是录制命令是给"录制状态轨迹"用的，机器人的状态一偏，同一条命令
就在错的位姿上执行，误差复利放大 ⇒ **开环回放测不出"执行链能跟多远"**。

本服务器改成**相位伺服**：每次请求都用机器人**当前根位姿**去参考轨迹上找最近的帧，
把该帧之后的 40 帧命令（`g1_motion_token_64`）与手指（`g1_hand_joints_14`）发回去
—— 相当于"如果 VLA 的相位估计是完美的，它能走多远"的**上限**。

位姿从哪来
-----------
HA 的 64 维 `vla_state` 里**没有位置**，所以走我们新加的通道：
`observation.state_extra6 = [dx, dy, sinψ, cosψ, t/20, v_xy]`（`SM_VLA_EXTRA6=1` 时客户端才发，
见 `action_provider_supermimic.py::_vla_state_extra6`）。DoubleDesk 的初始位姿被 pin 成
`(-3.7, -2.2, 0.8)` 且 100 条录制 clip 的起始点与它**逐位相同**（实测 max 距离 0.000 m），
所以 `世界坐标 = 起始点 + (dx, dy)` 直接可用。

用法::

    # 服务端
    python3 scripts/ha_gt_phase_server.py \
        --npz data/npz/ha_tokens/HOI_double_desk__tw__ep000000.npz --port 18086
    # 闭环（客户端要打开发相位通道）
    SM_VLA_EXTRA6=1 TAG=phase2k GPU=0 PORT=18086 MAX_STEPS=2000 \
      SRV_CMD="python3 scripts/ha_gt_phase_server.py --npz ... --port 18086" \
      bash scripts/r3_closedloop_ab.sh
"""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


class Handler(BaseHTTPRequestHandler):
    pos: np.ndarray = None        # (T,2) 参考根位置（世界系）
    tokens: np.ndarray = None     # (T,64)
    hands: np.ndarray = None      # (T,14) 可空
    horizon: int = 40
    stride: int = 20
    back: int = 50                # 允许往回找的帧数（防止一帧抖动锁死）
    ahead: int = 600              # 往前找的上限（3 s 的相位误差容忍）
    cursor: int = 0
    n_infer: int = 0
    n_extra: int = 0

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
        self._json(200, {"ok": True, "service": "ha-gt-phase",
                         "cursor": Handler.cursor, "frames": int(Handler.tokens.shape[0]),
                         "with_extra": int(Handler.n_extra)})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/reset":
            Handler.cursor = 0
            Handler.n_infer = 0
            Handler.n_extra = 0
            self._json(200, {"ok": True})
            return
        if path != "/infer":
            self._json(404, {"error": f"unknown path {self.path}"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n).decode())
        base64.b64decode(req["observation"]["images"]["front"]["data_b64"])   # 只为与真服务同构

        T = Handler.tokens.shape[0]
        e6 = req["observation"].get("state_extra6")
        if e6 is None:
            # 退化：没有位姿就按时间推进（= 旧的 GT 开环行为，会在日志里明确标出来）
            k = int(Handler.cursor) % T
            src = "TIME"
        else:
            Handler.n_extra += 1
            dx, dy = float(e6[0]), float(e6[1])
            p_now = Handler.pos[0] + np.asarray([dx, dy], dtype=np.float64)
            lo = max(0, int(Handler.cursor) - Handler.back)
            hi = min(T, int(Handler.cursor) + Handler.ahead)
            seg = Handler.pos[lo:hi]
            k = lo + int(np.argmin(np.linalg.norm(seg - p_now[None, :], axis=1)))
            src = "PHASE"
        idx = (np.arange(Handler.horizon) + k) % T
        chunk = Handler.tokens[idx]
        hands = Handler.hands[idx] if Handler.hands is not None else np.zeros((Handler.horizon, 14), np.float32)
        Handler.cursor = k + Handler.stride
        Handler.n_infer += 1

        if Handler.n_infer % 10 == 1:
            d = float(np.linalg.norm(Handler.pos[k] - p_now)) if src == "PHASE" else float("nan")
            print(f"[phase] infer #{Handler.n_infer} src={src} k={k}/{T} "
                  f"pos_err={d*100:.1f}cm tok_mean={chunk.mean():+.3f}", flush=True)
        self._json(200, {
            "latent64_chunk": chunk.tolist(),
            "hands14_chunk": hands.tolist(),
            "dt_ms": 0.0,
            "phase_frame": int(k),
            "phase_src": src,
        })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--key", default="g1_motion_token_64")
    ap.add_argument("--hand-key", default="g1_hand_joints_14")
    ap.add_argument("--port", type=int, default=18086)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--stride", type=int, default=20)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    tokens = np.asarray(z[a.key], dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 64:
        raise SystemExit(f"{a.key} shape {tokens.shape} != (T,64)")
    pos = np.asarray(z["body_pos_w"], dtype=np.float64)[:, 0, :2]
    hands = np.asarray(z[a.hand_key], dtype=np.float32) if a.hand_key in z else None
    if hands is not None and hands.shape[0] != tokens.shape[0]:
        hands = None
    Handler.pos, Handler.tokens, Handler.hands = pos, tokens, hands
    Handler.horizon, Handler.stride = int(a.horizon), int(a.stride)
    print(f"[phase] loaded {a.npz}: T={len(tokens)} ({len(tokens)/50:.1f}s) "
          f"start={np.round(pos[0], 2)} end={np.round(pos[-1], 2)} "
          f"arc={np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.2f}m "
          f"hands={'yes' if hands is not None else 'no'}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.verbose = False
    print(f"[phase] listening on http://127.0.0.1:{a.port}/infer", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
