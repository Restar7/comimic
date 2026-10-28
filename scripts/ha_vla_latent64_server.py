#!/usr/bin/env python3
"""HumanoidArena ``latent64`` VLA server — the missing link between our VLA and Isaac Lab.

HumanoidArena's simulation runtime talks to a policy over plain HTTP
(`isaaclab_twist2_g1/action_provider/lerobot_vla_http_client.py`):

    POST /infer
    {
      "observation": {
        "images": {"front": {"shape": [H,W,3], "dtype": "uint8", "data_b64": "<raw bytes>"}},
        "state":  [ ... ]            # their 64-d ``vla_state``
      },
      "robot_type": "unitree_g1_refpose_v3_1",
      "return_chunk": true,
      "task": "<instruction>"        # optional
    }

    -> {"latent64_chunk": [[64 floats], ...]}          # SONIC_VLA_ACTION_FORMAT=latent64
       (the client also accepts action_chunk / encoder_latent_chunk / latent64 / encoder_latent)

This server answers that request with **our** VLA: image + state + instruction ->
GR00T flow-matching chunk -> first 64 dims are the SONIC motion token, dims 64..78 are the
14 finger commands which are compressed to the 2-d binary the runtime expects
(``max(fingers) > 1.70`` — verified 100% agreement with HA's own ``vla_action_hand_binary``).

State conversion, because the two sides disagree:
    theirs ``vla_state`` (64) = root rot6d(6) + dof_pos(29) + dof_vel(29)
    ours   ``state_46``       = joints29 + hands14 + projected_gravity(3)
    29 joints pass through; gravity comes from the rot6d; the 14 hands are not in their
    state at all, so they are fed back from the last emitted action (``--hand-source
    feedback``) or zeros (``--hand-source zeros``).

No training happens here: the model is loaded read-only and sampled with no optimizer.

Usage::

    # protocol self-test without a model (returns zeros; proves the wiring)
    python3 scripts/ha_vla_latent64_server.py --dry-run --port 18080

    # real VLA
    python3 scripts/ha_vla_latent64_server.py \
        --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
        --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights --port 18080
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

import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The 8 state groups the checkpoint's modality config expects, in order (total 46).
#: The order must match how ``state64_to_state46`` lays the vector out
#: (``joints29 + hands14 + gravity3``), NOT the modality-config listing order --
#: ``right_arm`` comes before the hands in the joint block.
STATE_KEYS = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "projected_gravity",
)
STATE_WIDTHS = (6, 6, 3, 7, 7, 7, 7, 3)
#: ``left_hand_joints`` occupies 0:7 and ``right_hand_joints`` 7:14 of the action vector.
HAND_JOINT_SPLIT = 7
SONIC_TAG = "unitree_g1_sonic"
#: HA's own recording compression: a hand counts as "closed" when its strongest finger
#: command exceeds this.  Measured on HOI_double_desk against ``vla_action_hand_binary``:
#: 100.0% frame agreement (the finger range is +-1.74).
HAND_CLOSE_THRESHOLD = 1.70


def rot6d_to_matrix(r6: np.ndarray, layout: str = "ha_cols") -> np.ndarray:
    """6 维姿态 -> 旋转矩阵。

    **2026-10-26 更正**：HA 的 6 维姿态不是“两行/两列基向量”打包，而是
    ``quat_to_rot6d_wxyz`` 里的 ``rot[:, :2].reshape(...)``，即
    ``[R00, R01, R10, R11, R20, R21]``（前两列、row-major）。
    单位姿态在录制里就是 ``[1,0,0,1,0,0]``；按“两行”解会退化成 0 向量。

    证据：用录制 ``vla_state_root_rot6d`` 解出的重力与
    ``decoder_gravity_dir_hist``（= 由机器人真实朝向算的 R^T·[0,0,-1]）
    逐帧 **max|Δ| = 0.0**；旧写法在 tilt 帧上偏 0.5 rad。见
    ``docs/DIAG_SONIC_CLOSEDLOOP_20261026.md`` B1b。
    """
    r6 = np.asarray(r6, dtype=np.float64)
    if layout == "ha_cols":
        c0 = r6[..., [0, 2, 4]]
        c1 = r6[..., [1, 3, 5]]
    elif layout == "rows":            # 旧（错误）假设，仅保留做 A/B
        rows = np.stack([r6[..., 0:3], r6[..., 3:6]], axis=-2)
        c0, c1 = rows[..., 0], rows[..., 1]
    else:
        raise ValueError(f"unknown rot6d layout {layout!r} (ha_cols|rows)")
    c0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-9)
    c1 = c1 - (c0 * c1).sum(-1, keepdims=True) * c0
    c1 = c1 / (np.linalg.norm(c1, axis=-1, keepdims=True) + 1e-9)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=-1)          # 列 = 基向量


def gravity_from_rot6d(r6: np.ndarray, layout: str = "ha_cols") -> np.ndarray:
    """Body-frame gravity, i.e. ``R^T @ [0,0,-1]`` — the GR00T ``projected_gravity`` term."""
    m = rot6d_to_matrix(np.asarray(r6, dtype=np.float64), layout=layout)
    return np.einsum("...ji,j->...i", m, np.array([0.0, 0.0, -1.0])).astype(np.float32)


def state64_to_state46(state64: np.ndarray, hands14: np.ndarray,
                       state_order: str = "ha_v31",
                       layout: str = "train_v1") -> np.ndarray:
    """Their 64-d state + our 14-d hand feedback -> the 46-d state the checkpoint expects.

    ``state64[6:35]`` is HumanoidArena's ``vla_state_dof_pos_29``, i.e. the ``HA_V31_29``
    (interleaved) order — this is what the deployed runtime sends and what the current
    checkpoint was trained on.  ``state_order="standard"`` accepts a caller that already
    converted to the trackerlab/MuJoCo order; the 46-d vector we build always uses the
    ``joint_contract`` names, so the two can never drift silently.
    """
    s = np.asarray(state64, dtype=np.float32).reshape(-1)
    if s.size != 64:
        raise ValueError(f"expected the 64-d HA state, got {s.size}")
    joints = s[6:35]
    if state_order not in ("ha_v31", "standard"):
        raise ValueError(f"unknown --state-order {state_order!r} (ha_v31|standard)")
    hands = np.asarray(hands14, dtype=np.float32).reshape(-1)
    if hands.size != 14:
        raise ValueError(f"expected 14 hand dims, got {hands.size}")
    gravity = gravity_from_rot6d(s[0:6])
    if layout == "train_v1":
        # **训练数据真实的 46 维布局**（data/ha_sessions 的 observation.state 43 + gravity）：
        #   [ jp[0:22] | left_hand(7) | jp[22:29] | right_hand(7) | gravity ]
        # 见 data/ha_sessions/*/meta/modality.json：
        #   left_leg 0:6  right_leg 6:12  waist 12:15  left_arm 15:22
        #   left_hand 22:29  right_arm 29:36  right_hand 36:43
        # 旧实现把 29 个关节整段放在前面，等价于把 left_hand 与 right_arm 两块互换
        # （2026-10-26 用 parquet 逐帧对拍确认：旧布局 22:29 与训练差 1.74 rad）。
        return np.concatenate([joints[:22].astype(np.float32), hands[:7],
                               joints[22:].astype(np.float32), hands[7:],
                               gravity]).astype(np.float32)
    if layout == "legacy":
        return np.concatenate([joints.astype(np.float32), hands, gravity]).astype(np.float32)
    raise ValueError(f"unknown --state-layout {layout!r} (train_v1|legacy)")


class Engine:
    """Loads our VLA read-only and turns one HTTP request into a 64-d token chunk."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.last_hands14 = np.zeros(14, dtype=np.float32)
        self.model = None
        self.processor = None
        self.image_hw = (224, 224)
        if not args.dry_run:
            self._load()

    # ---- S0.3/J3：训练与服务端的 FM 步数必须一致（vla_meta.json 由 export_vla_ckpt 写入）
    def _check_fm_steps(self) -> int | None:
        """在读 6 GB 权重**之前**校验 FM 步数，不一致直接退出（fail fast）。

        原来的检查放在模型加载之后，一旦不一致要白等 ~1 min 才报错；而且旧产物
        （``vla_meta.json`` 缺失）会静默放过，J3 会悄悄复发 —— 这里只保留
        ``ALLOW_FM_STEPS_MISMATCH=1`` 这一个人为放行开关。
        """
        meta_path = Path(self.args.ckpt) / "vla_meta.json"
        train_steps = None
        if meta_path.exists():
            try:
                train_steps = int(json.loads(meta_path.read_text()).get("fm_steps", -1))
            except Exception:
                train_steps = None
        if train_steps and train_steps > 0 and train_steps != int(self.args.fm_steps):
            if os.environ.get("ALLOW_FM_STEPS_MISMATCH") != "1":
                raise SystemExit(
                    f"[fm-steps] 训练用 {train_steps} 步 Euler，服务端 --fm-steps "
                    f"{self.args.fm_steps}；二者必须一致（A/B 才允许 ALLOW_FM_STEPS_MISMATCH=1）"
                )
            print("[fm-steps][WARN] 步数不一致但已放行（A/B）", flush=True)
        elif train_steps in (None, -1):
            print(f"[fm-steps][WARN] {meta_path} 不存在或无 fm_steps：无法校验（旧 ckpt 口径）",
                  flush=True)
        return train_steps

    def _load(self) -> None:
        import torch

        from diff_vla_wbc.g1_tasks import load_gr00t_processor
        from diff_vla_wbc.gr00t_vlm import Gr00tVLM

        self._train_fm_steps = self._check_fm_steps()

        self.model = Gr00tVLM.from_pretrained(
            checkpoint=Path(self.args.ckpt),
            model_name=self.args.vlm_path,
            freeze_vlm=True,
            load_bf16=self.args.bf16,
            device=self.args.device,
        ).eval()
        self.processor = load_gr00t_processor(self.args.ckpt, vlm_path=self.args.vlm_path)
        self.codec = None
        if getattr(self.args, "decode", "codec") == "codec":
            from diff_vla_wbc.action_codec import ActionCodec  # noqa: PLC0415
            self.codec = ActionCodec(self.processor, device=self.args.device)
            print(self.codec.audit(), flush=True)
        print(f"[AUDIT][server] decode={getattr(self.args, 'decode', 'codec')} "
              f"fm_steps={self.args.fm_steps} train_fm_steps={self._train_fm_steps}", flush=True)
        # LOCKED token space: the action head target is ``action.motion_token`` ==
        # ``encoder_latent`` == SONIC encoder output computed from the **command stream**
        # (``encoder_motion_joint_pos_hist``), NOT the executed trajectory.  See
        # scripts/prepare_ha_sessions.py and scripts/build_ha_command_lib.py.
        print("[AUDIT][token] action.motion_token = encoder_latent (COMMAND stream); "
              "token_space=command_encoder_latent LOCKED", flush=True)
        self.token_space = "command_encoder_latent"
        cfg = getattr(self.processor, "config", None)
        size = getattr(cfg, "image_target_size", None) if cfg is not None else None
        if isinstance(size, (list, tuple)) and len(size) == 2:
            self.image_hw = (int(size[0]), int(size[1]))
        print(f"[server] model loaded, image_hw={self.image_hw}", flush=True)

    # ---------------------------------------------------------------- inference
    def infer(self, rgb: np.ndarray, state64: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            if self.last_hands14 is not None and self.args.hand_source == "feedback":
                hands14 = self.last_hands14.copy()
            else:
                hands14 = np.zeros(14, dtype=np.float32)
            state46 = state64_to_state46(
                state64, hands14,
                state_order=getattr(self.args, "state_order", "ha_v31"),
                layout=getattr(self.args, "state_layout", "train_v1"),
            )

            if self.args.dry_run:
                chunk64 = np.zeros((self.args.horizon, 64), dtype=np.float32)
                hands2 = np.zeros((self.args.horizon, 2), dtype=np.float32)
                return chunk64, hands2

            import torch

            from diff_vla_wbc.gr00t_fm import sample_model_with_grad

            # The processor wants (B, T=1, H, W, 3) uint8 + the 7 state groups + language.
            obs = {"video.ego_view": rgb[None, None], "annotation.human.task_description": [task]}
            offset = 0
            for key, width in zip(STATE_KEYS, STATE_WIDTHS):
                obs[f"state.{key}"] = np.ascontiguousarray(state46[offset : offset + width])[None]
                offset += width
            if offset != 46:
                raise RuntimeError(f"state_46 assembled {offset} dims")

            from gr00t.data.embodiment_tags import EmbodimentTag

            with torch.enable_grad():      # sampling requires grad; we detach right after
                batch = self.processor.process_observation(obs, EmbodimentTag(SONIC_TAG))
                if batch["state"].ndim == 2:
                    batch["state"] = batch["state"].unsqueeze(1)
                # The head emits its own padded width (``max_action_dim``, 132 for the
                # GR00T checkpoint); ``--action-dim`` (78) is the *meaningful* prefix.
                head_action_dim = int(self.model.action_head.action_dim)
                noise = torch.randn(
                    1, self.args.horizon, head_action_dim,
                    device=self.args.device, dtype=self.model.dtype,
                )
                full = sample_model_with_grad(
                    self.model, batch, initial_noise=noise, num_steps=self.args.fm_steps,
                )
            full = full.detach().to("cpu").float().numpy()[0]        # (horizon, >=78)
            if self.codec is not None:
                import torch as _t  # noqa: PLC0415
                with _t.no_grad():
                    dec = self.codec.decode(_t.from_numpy(full).to(self.codec.lo.device).unsqueeze(0)).squeeze(0).cpu().numpy()
                full = np.concatenate([dec, full[:, self.codec.dim:]], axis=-1).astype(np.float32)
            chunk64 = full[:, :64]
            fingers = full[:, 64:78]
            # HA's ``vla_action_hand_binary`` is (frames, 2) = (left, right); keep the
            # sides separate instead of collapsing both to one global max.
            left = fingers[:, :HAND_JOINT_SPLIT].max(axis=1, keepdims=True)
            right = fingers[:, HAND_JOINT_SPLIT:].max(axis=1, keepdims=True)
            hands2 = np.concatenate([left, right], axis=1)
            hands2 = (hands2 > HAND_CLOSE_THRESHOLD).astype(np.float32)
            self.last_hands14 = fingers[-1].astype(np.float32)
            self.last_hands14_chunk = fingers.astype(np.float32)
            return chunk64.astype(np.float32), hands2.astype(np.float32)


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None  # set in main()
    n_infer: int = 0

    def log_message(self, fmt, *a):  # keep the console readable
        if self.server.verbose:
            super().log_message(fmt, *a)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(200, {"ok": True, "service": "ha-latent64-vla",
                         "token_space": "command_encoder_latent"})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/reset":
            # The upstream LeRobot HTTP server exposes /reset; HumanoidArena's SONIC provider
            # posts to it between episodes and treats 404 as a hard episode failure.  We are
            # stateless per request except for the hand-feedback buffer, so resetting that is
            # enough to be protocol-compatible.
            self.engine.last_hands14 = None
            Handler.n_infer = 0
            print("[server] reset", flush=True)
            self._json(200, {"ok": True})
            return
        if path != "/infer":
            self._json(404, {"error": f"unknown path {self.path}"})
            return
        t0 = time.time()
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode())
            img = req["observation"]["images"]["front"]
            rgb = np.frombuffer(base64.b64decode(img["data_b64"]), dtype=img.get("dtype", "uint8"))
            rgb = rgb.reshape(img["shape"]).astype(np.uint8)
            state = np.asarray(req["observation"]["state"], dtype=np.float32)
            task = str(req.get("task") or "")
            chunk, hands = self.engine.infer(rgb, state, task)
            dt_ms = (time.time() - t0) * 1e3
            # One line per request: proves the sim is really driving the VLA (闭环证据) and
            # gives the latency series for ablation reporting.  Cheap: ~1 line per 40 steps.
            Handler.n_infer += 1
            # img_mean/img_std: a frozen camera shows up as a constant pair across requests,
            # which is the failure mode we hit on the 570 box (RTX sensors need a render pass).
            _h14 = getattr(self.engine, "last_hands14_chunk", None)
            print(f"[server] infer #{Handler.n_infer} dt={dt_ms:.1f}ms "
                  f"img={rgb.shape} img_mean={rgb.mean():.1f} img_std={rgb.std():.1f} "
                  # 手部统计取 engine 缓存的 14 维（Handler 里没有局部变量 fingers）
                  
                  f"tok_mean={chunk.mean():+.3f} tok_absmean={np.abs(chunk).mean():.4f} "
                  f"hand14_max={float(np.abs(_h14).max()) if _h14 is not None else float('nan'):.3f} "
                  f"hand_gt170={float((np.abs(_h14) > 1.70).mean()) if _h14 is not None else float('nan'):.3f} "
                  f"state={state.shape[0]} task={task[:40]!r}",
                  flush=True)
            self._json(200, {
                "latent64_chunk": chunk.tolist(),
                "hand_binary_chunk": hands.tolist(),
                # 14 维手指命令直通（不经 SuperMimic）；provider 用它写 finger joint target
                "hands14_chunk": np.asarray(
                    getattr(self.engine, "last_hands14_chunk", np.zeros((len(chunk), 14), np.float32))
                ).tolist(),
                "dt_ms": dt_ms,
            })
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--ckpt", type=Path, help="our VLA checkpoint (stage O output)")
    ap.add_argument("--vlm-path", type=Path, default=ROOT / "groot_repro" / "nvidia" / "Cosmos-Reason2-2B-weights")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bf16", action="store_true", help="load the backbone in bf16")
    ap.add_argument("--horizon", type=int, default=40, help="action chunk length the head was trained with")
    ap.add_argument("--action-dim", type=int, default=78, help="64 token + 7 + 7 hands")
    ap.add_argument("--fm-steps", type=int, default=4, help="flow-matching Euler steps")
    ap.add_argument("--decode", choices=("codec", "none"), default="codec",
                    help="codec = GR00T min-max 逆变换回原始 token/手指空间（官方 ckpt 必须用）；"
                         "none = 旧口径（我们自己 raw-space 微调过的 ckpt 用）")
    ap.add_argument("--state-order", choices=("ha_v31", "standard"), default="ha_v31",
                    help="order of the 29 joints inside the incoming 64-d vla_state "
                         "(ha_v31 = HumanoidArena asset order, which the current ckpt was "
                         "trained on; standard = trackerlab/MuJoCo order)")
    ap.add_argument("--state-layout", choices=("train_v1", "legacy"), default="train_v1",
                    help="46 维 state 的分块布局：train_v1 = 训练数据真实布局"
                         "（left_hand 在 22:29、right_arm 在 29:36）；legacy = 旧实现（两块互换）")
    ap.add_argument("--hand-source", choices=("feedback", "zeros"), default="feedback",
                    help="where the 46-d state's hand entries come from (their state has none)")
    ap.add_argument("--dry-run", action="store_true", help="no model: answer zeros to test the wiring")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and (args.ckpt is None or not Path(args.ckpt).is_dir()):
        raise SystemExit("--ckpt must point at our VLA checkpoint directory (or use --dry-run)")

    Handler.engine = Engine(args)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.verbose = args.verbose
    print(f"[server] listening on http://{args.host}:{args.port}/infer "
          f"(dry_run={args.dry_run}, hand_source={args.hand_source})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
