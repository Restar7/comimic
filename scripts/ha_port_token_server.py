#!/usr/bin/env python3
"""O1（S1.3）真值 token 回放服务：按 episode 回放录制动作的 token，测**执行器上限**。

⚠️ 2026-10-26 口径更新（S0.4 / 复核附录 A.2.2）：**默认改成 label 空间**
（= npz 里的 `g1_motion_token_64` = 录制 `encoder_latent`）。端口侧车
（`data/port_tokens/`）已被证明是**错装配**（窗口/置换/anchor 三处错，与标签 corr 0.057），
只有回放"WBC 训练用的那套 token"才对得上：

* WBC 在 **label** 空间重训后 → 必须 `--token-space label`（默认）；
* 老的端口空间 WBC（`ckpt_016000*`）→ `--token-space port`，结果只能当"执行器参照"，
  不进 A1 正式表。

协议与 `scripts/ha_vla_latent64_server.py` 完全一致（`/infer`、`/reset`、`GET /`），
把闭环的 `SERVER_URL` 指过来即可：

    python3 scripts/ha_port_token_server.py --clip HOI_pp_box__yb__ep000000 --port 18082
    SERVER_URL=http://127.0.0.1:18082 SM_TOKEN_SOURCE=http bash scripts/ha_isaac_closedloop.sh

判读：VLA 被换成"录制动作的真值 token"，所以闭环还摔 = 执行器（WBC）/场景的问题；
能站能走 = 执行器没问题，瓶颈在 VLA。
"""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOKEN_DIR = ROOT / "data/port_tokens/ha_tokens"
LABEL_DIR = ROOT / "data/npz/ha_tokens"


def _load_tokens(space: str, clip: str | None, npy: Path | None) -> tuple[np.ndarray, str]:
    """返回 (tokens (T,64), 来源描述)。``label`` = npz 的 g1_motion_token_64；``port`` = 旧侧车。"""
    if npy is not None:
        return np.load(npy).astype(np.float32), str(npy)
    if not clip:
        raise SystemExit("需要 --clip（或 --npy）")
    if space == "label":
        path = LABEL_DIR / f"{clip}.npz"
        if not path.is_file():
            raise SystemExit(f"标签 token 不存在：{path}（先跑 scripts/prepare_ha_tokens_npz.py）")
        with np.load(path) as z:
            tok = np.asarray(z["g1_motion_token_64"], dtype=np.float32)
        return tok, f"{path}::g1_motion_token_64"
    path = TOKEN_DIR / f"{clip}.npy"
    if not path.is_file():
        raise SystemExit(f"端口 token 不存在：{path}（先跑 scripts/build_port_tokens.py）")
    return np.load(path).astype(np.float32), str(path)


class Handler(BaseHTTPRequestHandler):
    tokens: np.ndarray = None      # (T, 64)
    cursor: int = 0
    horizon: int = 40
    n_infer: int = 0
    at_end: str = "hold"           # hold = 停在最后一帧；wrap = 循环

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
        self._json(200, {"ok": True, "service": "ha-port-token", "cursor": int(Handler.cursor),
                         "frames": int(Handler.tokens.shape[0]), "at_end": Handler.at_end})

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
        start = int(min(Handler.cursor, max(T - 1, 0)))
        if Handler.at_end == "wrap":
            idx = (np.arange(Handler.horizon) + start) % T
        else:  # hold: 超出部分重复最后一帧（录制动作播完后就地保持）
            idx = np.minimum(np.arange(Handler.horizon) + start, T - 1)
        chunk = Handler.tokens[idx]
        Handler.cursor += Handler.horizon

        Handler.n_infer += 1
        if Handler.n_infer <= 3 or Handler.n_infer % 25 == 0:
            print(f"[port] infer #{Handler.n_infer} rows {start}..{int(idx[-1])} / {T} "
                  f"tok_mean={chunk.mean():+.3f} tok_std={chunk.std():.3f}", flush=True)
        self._json(200, {"latent64_chunk": chunk.astype(np.float32).tolist(),
                         "hand_binary_chunk": [[0.0, 0.0]] * Handler.horizon,
                         "dt_ms": 0.0})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", type=Path, default=None, help="直接给 (T,64) 的 .npy（覆盖 token-space）")
    ap.add_argument("--clip", default=None,
                    help="clip 名（<task>__<op>__ep%06d），按 --token-space 决定从哪个目录读")
    ap.add_argument("--token-space", choices=("label", "port"), default="label",
                    help="label = npz 的 g1_motion_token_64（默认，WBC 在标签空间重训后用这套）；"
                         "port = 旧端口侧车（只给老端口空间 WBC 当参照）")
    ap.add_argument("--port", type=int, default=18082)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--at-end", choices=("hold", "wrap"), default="hold")
    a = ap.parse_args()

    tokens, src = _load_tokens(a.token_space, a.clip, a.npy)
    if tokens.ndim != 2 or tokens.shape[1] != 64:
        raise SystemExit(f"{src} shape {tokens.shape}, expected (T, 64)")

    Handler.tokens = tokens
    Handler.horizon = int(a.horizon)
    Handler.at_end = a.at_end
    print(f"[AUDIT][o1] token_space={a.token_space} src={src} shape={tokens.shape} "
          f"mean={tokens.mean():+.3f} std={tokens.std():.3f} at_end={a.at_end}", flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.verbose = False
    print(f"[port] listening on http://127.0.0.1:{a.port}/infer", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
