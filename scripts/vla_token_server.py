#!/usr/bin/env python3
"""B 线（真物理闭环）用的 VLA token 服务：Isaac 每 20 步问一次，回 40×64 token + 40×14 手。

为什么需要它（2026-10-28）：
* 旧的开环回放（npy）测的是"固定 token 流"，不是闭环；闭环里 VLA 自己的误差会累积，
  那才是联合微调要纠正的东西。
* Isaac 进程里不能装我们的训练栈，所以 VLA 跑在**系统 python** 里当 HTTP 服务，
  Isaac 只发请求（provider 的 http 模式本来就是这个契约）。

请求（LeRobotVLAHttpClient）::

    POST /infer {"observation": {"images": {"front": {shape,dtype,data_b64}},
                                 "state": [64], "state46": [46]},
                 "robot_type": ..., "task": ..., "return_chunk": true}

* `state46`（**训练口径**，优先用）= 29 canonical q + 14 手 + 3 投影重力，
  键序见 `prepare_g1_tasks_npz._PROCESSOR_STATE_ORDER`；
* `state`（64）= HA 部署契约（q + dq + rot6d），我们**不用**它做推理（口径不同）。

响应::

    {"action_chunk": [[64]*40], "hands14_chunk": [[14]*40]}

用法：
    # 自检（用录制的一帧跑一遍，验证预处理/口径与训练一致）
    python3 scripts/vla_token_server.py --selftest
    # 起服务
    python3 scripts/vla_token_server.py --port 18083 --device cuda:0
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "Isaac-GR00T", ROOT / "third_party/gr00t_deps",
           ROOT / "third_party/gr00t_deps/_noflash",
           ROOT / "trackerlab-supermimic/source/trackerLab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diff_vla_wbc.action_codec import ActionCodec  # noqa: E402
from diff_vla_wbc.g1_tasks import (  # noqa: E402
    _SONIC_TAG, _STATE_KEYS, _STATE_WIDTHS, EgoViewImages, TaskFrameIndex,
    TaskObservationBuilder, load_gr00t_processor,
)
from diff_vla_wbc.gr00t_vlm import Gr00tVLM  # noqa: E402

TOKEN_DIM = 64
HAND_DIM = 14
HORIZON = 40


class VLATokenServer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.processor = load_gr00t_processor(
            Path(args.vlm_ckpt), vlm_path=Path(args.vlm_weights))
        self.codec = ActionCodec(self.processor, device=self.device)
        print(self.codec.audit(), flush=True)
        self.vlm = Gr00tVLM.from_pretrained(
            Path(args.vlm_weights), checkpoint=Path(args.vlm_ckpt),
            select_layer=16, device=self.device, action_horizon=HORIZON,
            max_state_dim=132, freeze_vlm=True, attn_implementation="sdpa")
        if args.head_ckpt:
            raw = torch.load(str(args.head_ckpt), map_location="cpu", weights_only=False)
            sd = raw.get("head", raw.get("model", raw)) if isinstance(raw, dict) else raw
            missing, unexpected = self.vlm.action_head.load_state_dict(sd, strict=False)
            print(f"[server] head override {Path(args.head_ckpt).name}: "
                  f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        else:
            print("[server] head = 底模自带（未覆盖）", flush=True)
        self.vlm.action_head.eval()
        from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: PLC0415
        self.tag = EmbodimentTag(_SONIC_TAG)
        self.lock = threading.Lock()
        self.n_calls = 0
        print(f"[server] ready: tag={_SONIC_TAG} horizon={HORIZON} "
              f"token={TOKEN_DIM} hands={HAND_DIM}", flush=True)

    # ---------------------------------------------------------------- 观测组装
    def build_observation(self, rgb: np.ndarray, state46: np.ndarray, task: str) -> dict:
        """把实时图像 + 46 维状态拼成训练时 `TaskObservationBuilder` 的那种 dict。"""
        s = np.asarray(state46, dtype=np.float32).reshape(-1)
        if s.size != sum(_STATE_WIDTHS):
            raise ValueError(f"state46 must be {sum(_STATE_WIDTHS)} dims, got {s.size}")
        obs: dict = {
            "video.ego_view": np.asarray(rgb, dtype=np.uint8)[None, None],  # (B=1,T=1,H,W,3)
            "annotation.human.task_description": [str(task or "")],
        }
        off = 0
        for key, width in zip(_STATE_KEYS, _STATE_WIDTHS):
            obs[f"state.{key}"] = np.ascontiguousarray(s[off:off + width][None])
            off += width
        return obs

    # ---------------------------------------------------------------- 推理
    @torch.no_grad()
    def _preprocess(self, obs: dict):
        from diff_vla_wbc.g1_tasks import processed_batch  # noqa: PLC0415

        class _OneFrame:
            def build(self, _frames):
                return obs

        return processed_batch(self.processor, _OneFrame(), [0])

    def infer(self, rgb: np.ndarray, state46: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
        obs = self.build_observation(rgb, state46, task)
        proc = self._preprocess(obs)
        for k in list(proc.keys()):
            if torch.is_tensor(proc[k]):
                proc[k] = proc[k].to(self.device)
        with self.lock:
            from diff_vla_wbc.fm_bc import sample_once_with_grad  # noqa: PLC0415

            head = self.vlm.action_head
            noise = torch.randn(1, int(head.action_horizon), int(head.action_dim),
                                device=self.device, dtype=head.dtype)
            bb = self.vlm.backbone(proc)
            x = sample_once_with_grad(head, bb, proc, initial_noise=noise,
                                      num_steps=int(self.args.num_steps),
                                      grad_backend="truncated")
            raw = self.codec.decode(x.detach()).float().cpu().numpy()[0]      # (T, 78)
            self.n_calls += 1
        tok = raw[:, :TOKEN_DIM]
        hands = raw[:, TOKEN_DIM:TOKEN_DIM + HAND_DIM]
        return tok.astype(np.float32), hands.astype(np.float32)


def make_handler(srv: VLATokenServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):  # 静音
            pass

        def _send(self, obj: dict, code: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") not in ("/infer", ""):
                self._send({"error": f"unknown path {self.path}"}, 404)
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                obs = payload.get("observation") or {}
                img = obs["images"]["front"]
                rgb = np.frombuffer(base64.b64decode(img["data_b64"]), dtype=img["dtype"])
                rgb = rgb.reshape(img["shape"])
                s46 = obs.get("state46")
                if s46 is None:
                    raise ValueError("payload has no observation.state46（provider 需升级；"
                                     "64 维 state 与训练口径不同，不能用来推理）")
                t0 = time.time()
                tok, hands = srv.infer(rgb, np.asarray(s46, dtype=np.float32),
                                       payload.get("task") or "")
                dt = time.time() - t0
                if srv.n_calls % 10 == 0:
                    print(f"[server] infer #{srv.n_calls} {dt*1000:.0f} ms "
                          f"tok|mean|={float(np.abs(tok).mean()):.3f}", flush=True)
                self._send({"action_chunk": tok.tolist(), "hands14_chunk": hands.tolist()})
            except Exception as exc:  # noqa: BLE001
                import traceback  # noqa: PLC0415
                traceback.print_exc()
                self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def selftest(srv: VLATokenServer) -> int:
    """用录制数据的一帧跑一遍，验证 server 侧预处理与训练一致。"""
    npz_dir = ROOT / "data/npz/ha_tokens"
    index = TaskFrameIndex(sorted(npz_dir.glob("*.npz")))
    builder = TaskObservationBuilder(index, images=EgoViewImages())
    frames = [0, 100, 200]
    obs = builder.build(frames)
    rgb = obs["video.ego_view"][0, 0]
    s46 = np.concatenate([obs[f"state.{k}"][0] for k in _STATE_KEYS])
    tok, hands = srv.infer(rgb, s46, "pick up the box from the desk")
    label = np.load(npz_dir / f"{Path(index.files[0]).stem}.npz")["g1_motion_token_64"][:3]
    print(f"[selftest] tok {tok.shape} hands {hands.shape}")
    print(f"[selftest] |tok[0] − label[0]| mean = {float(np.abs(tok[0]-label[0]).mean()):.4f} "
          f"(训练好的底模应在几十毫开量级)")
    print(f"[selftest] hands[0][:4] = {np.round(hands[0][:4], 3).tolist()} "
          f"label_hands[0][:4] = {np.round(np.load(npz_dir / f'{Path(index.files[0]).stem}.npz')['g1_hand_joints_14'][0][:4], 3).tolist()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-ckpt", type=Path,
                    default=ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--head-ckpt", type=Path, default=None,
                    help="S3 训出来的头权重（joint_*.pt）；不给就用底模自带的")
    ap.add_argument("--vlm-weights", type=Path,
                    default=ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights")
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=18083)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    srv = VLATokenServer(args)
    if args.selftest:
        return selftest(srv)
    httpd = ThreadingHTTPServer(("0.0.0.0", int(args.port)), make_handler(srv))
    print(f"[server] listening on :{args.port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
