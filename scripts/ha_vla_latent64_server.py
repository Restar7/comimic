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
                       layout: str = "legacy") -> np.ndarray:
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


TOKEN_QUANT_MODES = ("none", "ste", "hard")


#: 训练数据里 state46 的**唯一真值来源**：`g1_state_46` 与 `g1_motion_token_64` 同源同序。
LAYOUT_CHECK_GLOB = "data/npz/ha_tokens/*.npz"
#: 关节/手两块在 46 维里占 0:43（重力 43:46 由 rot6d 反解，比对时跳过）。
LAYOUT_CHECK_N = 43


def assert_state_layout_matches_training_data(layout: str, state_order: str = "ha_v31",
                                              npz_glob: str = LAYOUT_CHECK_GLOB) -> None:
    """用**训练数据本身**校验 46 维布局，错的直接拒绝启动（2026-11-04）。

    背景：部署侧把 state46 拼错会让 21/46 维是垃圾（`train_v1` 把 right_arm 换成 left_hand、
    把 left_hand 换成 jp22:29），VLA 输出的 token 流会退化成"原地站桩"，而闭环**不会报错**
    —— 只会看到机器人不动。这种错只能靠对拍训练数据来锁：`g1_state_46` 就是训练用的那一份，
    我们把它拆成 (joints, hands) 再喂进 `state64_to_state46`，拼回来必须逐位相同。
    """
    import glob as _glob  # noqa: PLC0415

    files = sorted(_glob.glob(npz_glob))
    if not files:
        raise SystemExit(f"[LAYOUT][FATAL] 找不到训练数据样本 {npz_glob}，无法校验布局；"
                         f"用 --layout-check-npz 指定，或显式 --no-layout-check 跳过（不推荐）")
    z = np.load(files[0], allow_pickle=True)
    if "g1_state_46" not in z:
        raise SystemExit(f"[LAYOUT][FATAL] {files[0]} 没有 g1_state_46")
    ref = np.asarray(z["g1_state_46"], dtype=np.float32)
    bad = 0
    for t in (0, len(ref) // 2, len(ref) - 1):
        g46 = ref[t]
        joints17 = g46[:29]
        hands14 = g46[29:43]
        s64 = np.concatenate([np.zeros(6, np.float32), joints17, np.zeros(29, np.float32)])
        got = state64_to_state46(s64, hands14, state_order=state_order, layout=layout)
        err = float(np.abs(got[:LAYOUT_CHECK_N] - g46[:LAYOUT_CHECK_N]).max())
        if not (err < 1e-4):
            bad += 1
            if bad == 1:
                print(f"[LAYOUT][FATAL] layout={layout!r} 与训练数据不一致（帧 {t}，"
                      f"最大误差 {err:.4f}）", flush=True)
                print(f"  训练数据 g46[0:43] = {np.round(g46[:8], 3)} ... "
                      f"手 {np.round(g46[29:36], 3)}", flush=True)
                print(f"  拼出来 got[0:43]  = {np.round(got[:8], 3)} ... "
                      f"手 {np.round(got[29:36], 3)}", flush=True)
    if bad:
        raise SystemExit(
            f"[LAYOUT][FATAL] --state-layout {layout} 会把 46 维状态拼错 → VLA 输出会退化成"
            f"常量 token（闭环表现为机器人原地站桩）。训练数据 {Path(files[0]).name} 的正确布局是 "
            f"[joints29 | hands14 | gravity3]（= --state-layout legacy）。"
            f"见 docs/CLOSEDLOOP_INFERENCE_FIX_20261104.md")
    print(f"[LAYOUT][OK] layout={layout} 与训练数据 {Path(files[0]).name} 逐位一致（关节 29 + 手 14）",
          flush=True)


def resolve_token_quant(value) -> str:
    """校验并归一化部署侧量化口径（**委托** :func:`diff_vla_wbc.quant.resolve_deploy_mode`）。

    两个 server（本文件 + ``ha_vla_server.py``）共用同一份口径实现，避免两处漂移。
    argparse 不校验字符串默认值是否在 choices 里，所以环境变量 ``TOKEN_QUANT=bogus``
    必须在这里被挡住；共享实现抛 ``ValueError``，这里转成启动失败用的 ``SystemExit``。
    """
    from diff_vla_wbc.quant import resolve_deploy_mode  # noqa: PLC0415
    try:
        return resolve_deploy_mode(value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


class Engine:
    """Loads our VLA read-only and turns one HTTP request into a 64-d token chunk."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.last_hands14 = np.zeros(14, dtype=np.float32)
        self.model = None
        self.processor = None
        self.image_hw = (224, 224)
        self._log_req = 0
        # 部署侧 token 量化（2026-11-04）：none / ste / hard，前向行为见 diff_vla_wbc/quant.py
        self._quant_mode = str(getattr(args, "token_quant", "none") or "none").lower()
        self._quant_n = 0
        self._quant_dist = float("nan")
        if getattr(args, "log_dir", ""):
            Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            # 先做**零成本**的契约校验，再花 1 min 读权重：布局错了根本没必要求值。
            if not getattr(args, "no_layout_check", False):
                assert_state_layout_matches_training_data(
                    str(getattr(args, "state_layout", "legacy")),
                    str(getattr(args, "state_order", "ha_v31")),
                    str(getattr(args, "layout_check_npz", LAYOUT_CHECK_GLOB)))
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

    def _check_token_quant(self) -> str:
        """训练/部署的 token 量化口径必须一致（委托共享实现 ``check_meta_mode``）。"""
        from diff_vla_wbc.quant import check_meta_mode  # noqa: PLC0415
        return check_meta_mode(self.args.ckpt, getattr(self.args, "token_quant", "none"),
                               log=lambda m: print(m, flush=True))

    def _load(self) -> None:
        import torch

        from diff_vla_wbc.g1_tasks import load_gr00t_processor
        from diff_vla_wbc.gr00t_vlm import Gr00tVLM

        self._train_fm_steps = self._check_fm_steps()
        self._train_token_quant = self._check_token_quant()

        self.model = Gr00tVLM.from_pretrained(
            checkpoint=Path(self.args.ckpt),
            model_name=self.args.vlm_path,
            freeze_vlm=True,
            load_bf16=self.args.bf16,
            device=self.args.device,
            # 2026-11-03: ckpt 的 VLM 配置写的是 flash_attention_2，而 backbone 以 fp32 载入
            # → transformers 直接抛 "FlashAttention only support fp16/bf16"。强制 sdpa 即可。
            attn_implementation="sdpa",
        ).eval()
        self.processor = load_gr00t_processor(self.args.ckpt, vlm_path=self.args.vlm_path)
        self.codec = None
        if getattr(self.args, "decode", "codec") == "codec":
            from diff_vla_wbc.action_codec import ActionCodec  # noqa: PLC0415
            self.codec = ActionCodec(self.processor, device=self.args.device)
            print(self.codec.audit(), flush=True)
        print(f"[AUDIT][server] decode={getattr(self.args, 'decode', 'codec')} "
              f"fm_steps={self.args.fm_steps} train_fm_steps={self._train_fm_steps} "
              f"token_quant={self._quant_mode} train_token_quant={self._train_token_quant}",
              flush=True)
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
        if getattr(self.args, "head_ckpt", ""):
            self._load_head(self.args.head_ckpt)
        print(f"[server] model loaded, image_hw={self.image_hw}", flush=True)

    def _load_head(self, path: str) -> None:
        """S3 §3.2：把联合微调后的 action head 权重装进**已验证**的 server。

        只动 ``action_head`` 的 LoRA/输出投影，**不动** state46 构造与手部逻辑。
        流程：先按训练的同一套配置挂 LoRA（底模是普通 Linear），再 ``load_state_dict``。
        如果 LoRA 配置不对，``unexpected`` 会等于整个状态字典（432 项）——这里直接报错，
        不静默降级。
        """
        import torch  # noqa: PLC0415

        from diff_vla_wbc.lora import apply_lora, lora_summary  # noqa: PLC0415

        n = apply_lora(
            self.model.action_head,
            rank=int(self.args.lora_rank),
            alpha=None,
            targets=tuple(x for x in str(self.args.lora_targets).split(",") if x),
            also_train=tuple(x for x in str(self.args.lora_also_train).split(",") if x),
        )
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = d.get("fm") if isinstance(d, dict) and "fm" in d else d
        if not isinstance(sd, dict):
            raise SystemExit(f"[server][head] {path} 不是状态字典（也没有 'fm'）")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected:
            raise SystemExit(
                f"[server][head] unexpected={len(unexpected)}（例：{list(unexpected)[:3]}）；"
                f"LoRA 配置与训练不一致（rank={self.args.lora_rank} targets={self.args.lora_targets}）")
        lora_b = [v for k, v in sd.items() if k.endswith("lora_B")]
        print(f"[server][head] loaded {path}: fm={len(sd)} lora_layers={n} "
              f"missing(frozen)={len(missing)} unexpected=0 "
              f"lora_B_absmean={(sum(float(v.float().abs().mean()) for v in lora_b) / max(1, len(lora_b))):.6f} "
              f"step={d.get('step') if isinstance(d, dict) else '?'}", flush=True)
        print(f"[server][head] {lora_summary(self.model.action_head)}", flush=True)
        # ---- e1：把 ckpt 里的 6 个缩放带到推理端（否则训练学到的缩放部署时是 0）----
        self.state_extra_scale = None
        self.state_extra_mode = "none"
        if isinstance(d, dict) and d.get("state_extra_scale") is not None:
            _sc = torch.as_tensor(d["state_extra_scale"]).float().reshape(-1)
            if _sc.numel() != 6:
                raise SystemExit(f"[server][head] state_extra_scale 宽度 {_sc.numel()} != 6")
            self.state_extra_scale = _sc
            self.state_extra_mode = str(d.get("state_extra_mode") or "vel6")
            print(f"[server][head][state-extra] mode={self.state_extra_mode} "
                  f"scale={[round(float(x), 5) for x in _sc]}", flush=True)
        self._head_loaded = True

    # ---------------------------------------------------------------- inference
    def infer(self, rgb: np.ndarray, state64: np.ndarray, task: str,
              state_extra6=None, state46_in=None,
              hands14_in=None) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            # ★ 2026-11-03 真 bug（手部幅度只有 1/5 的根因）：
            #   客户端其实**发了真实的 46 维状态**（29 关节 + 14 手 + 重力，
            #   见 action_provider_supermimic.py:_vla_state46），但服务端一直用
            #   `state64_to_state46` 重建 —— 手部那 14 维要么是"自身上一次输出"
            #   (--hand-source feedback, 自收缩)，要么是 0。
            #   训练时用的是真实手指关节 ⇒ 部署口径不一致 ⇒ 手指几乎不动。
            #   有 state46 就直接用（尊重真实状态），否则才回退重建。
            if state46_in is not None:
                _s46 = np.asarray(state46_in, dtype=np.float32).reshape(-1)
                if _s46.size == 46:
                    state46 = _s46
                    self._n_state46_in = getattr(self, "_n_state46_in", 0) + 1
                else:
                    state46_in = None
            if state46_in is None and hands14_in is not None:
                # ★ 2026-11-04：SONIC provider 只发 64 维 vla_state（**没有手**），
                #   但又没有 supermimic 那样的完整 state46。这里用**实测手指关节**补手，
                #   比 feedback（自己喂自己，≈0）和 zeros 都更接近训练口径
                #   （训练数据 state46 的 left_hand/right_hand 是真实角度，
                #    起始帧就是闭手 [0,1,1.74,-1.57,-1.74,-1.57,-1.74]）。
                _h = np.asarray(hands14_in, dtype=np.float32).reshape(-1)
                if _h.size == 14:
                    state46 = state64_to_state46(
                        state64, _h,
                        state_order=getattr(self.args, "state_order", "ha_v31"),
                        layout=getattr(self.args, "state_layout", "legacy"))
                    self._n_hands14_in = getattr(self, "_n_hands14_in", 0) + 1
                else:
                    hands14_in = None
            if state46_in is None and hands14_in is None:
                if self.last_hands14 is not None and self.args.hand_source == "feedback":
                    hands14 = self.last_hands14.copy()
                else:
                    hands14 = np.zeros(14, dtype=np.float32)
                state46 = state64_to_state46(
                    state64, hands14,
                    state_order=getattr(self.args, "state_order", "ha_v31"),
                    layout=getattr(self.args, "state_layout", "legacy"),
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

            # 2026-10-28（本轮新增）：**多样本平均**。
            # 闭环里 flow-matching 的初始噪声是随机的，实测同一个 π₀ 单集会从 0.13 抖到 0.42 m
            # —— 噪声就是闭环误差的主要来源之一。这里对同一观测采 N 次、把解码后的 token
            # 求平均，方差按 1/√N 降，且不改变策略的均值行为（N=1 就是旧行为）。
            n_avg = max(1, int(getattr(self.args, "sample_avg", 1)))
            with torch.enable_grad():      # sampling requires grad; we detach right after
                batch = self.processor.process_observation(obs, EmbodimentTag(SONIC_TAG))
                if batch["state"].ndim == 2:
                    batch["state"] = batch["state"].unsqueeze(1)
                # ---- e1：把 6 维写进 46:52（与训练侧 `_patch_state_extra` 同一切片）----
                _mode = str(getattr(self, "state_extra_mode", "none"))
                if _mode in ("abs6", "rotlocal6"):
                    # rotlocal6 = 线上 state64[:6]，本来就是 rot6d(conj(heading(q0))*q_now)
                    # （heading-canonical，开局 identity，与训练侧 rotlocal6 逐位对拍过）。
                    # abs6 是老名字，用同一个切片（保留兼容）。
                    # 绝对朝向：线上 state64[0:6] 就是 rot6d(quat)（列块口径），直接复用
                    _e6 = torch.as_tensor(np.asarray(state64, np.float32)[:6]).reshape(-1)
                    _sc = getattr(self, "state_extra_scale", None)
                    if _sc is None:
                        raise SystemExit("[server][state-extra] 缺 state_extra_scale")
                    _st = batch["state"]
                    _st[..., 46:52] = (_e6 * _sc).to(dtype=_st.dtype, device=_st.device)
                    self._se_checksum = [round(float(v), 6) for v in _e6.tolist()]
                elif _mode != "none":
                    if state_extra6 is None:
                        raise SystemExit(
                            f"[server][state-extra] 模型是 {_mode} 口径，但请求里没有 "
                            "state_extra6 ⇒ 拒绝用 0 顶替（那就是训练/部署口径漂移）")
                    _e6 = torch.as_tensor(np.asarray(state_extra6, dtype=np.float32)).reshape(-1)
                    if _e6.numel() != 6:
                        raise SystemExit(f"[server][state-extra] state_extra6 宽度 {_e6.numel()} != 6")
                    _sc = getattr(self, "state_extra_scale", None)
                    if _sc is None:
                        raise SystemExit("[server][state-extra] 缺 state_extra_scale")
                    _st = batch["state"]
                    if int(_st.shape[-1]) < 52:
                        raise SystemExit(f"[server][state-extra] state 宽度 {_st.shape[-1]} < 52")
                    _st[..., 46:52] = (_e6 * _sc).to(dtype=_st.dtype, device=_st.device)
                    self._se_checksum = [round(float(v), 6) for v in _e6.tolist()]
                # The head emits its own padded width (``max_action_dim``, 132 for the
                # GR00T checkpoint); ``--action-dim`` (78) is the *meaningful* prefix.
                head_action_dim = int(self.model.action_head.action_dim)
            _toks, _fins = [], []
            for _si in range(n_avg):
                with torch.enable_grad():
                    noise = torch.randn(
                        1, self.args.horizon, head_action_dim,
                        device=self.args.device, dtype=self.model.dtype,
                    )
                    full = sample_model_with_grad(
                        self.model, batch, initial_noise=noise, num_steps=self.args.fm_steps,
                    )
                full = full.detach().to("cpu").float().numpy()[0]    # (horizon, >=78)
                if self.codec is not None:
                    import torch as _t  # noqa: PLC0415
                    with _t.no_grad():
                        dec = self.codec.decode(_t.from_numpy(full).to(self.codec.lo.device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    full = np.concatenate([dec, full[:, self.codec.dim:]], axis=-1).astype(np.float32)
                _toks.append(full[:, :64])
                _fins.append(full[:, 64:78])
            if n_avg > 1:
                chunk64 = np.mean(np.stack(_toks, 0), 0).astype(np.float32)
                fingers = np.mean(np.stack(_fins, 0), 0).astype(np.float32)
            else:
                chunk64, fingers = _toks[0], _fins[0]
            # ★ 2026-11-01（滞后性检验）：token 空间一阶外推 = "提前量"
            #   tok_t ← tok_t + β (tok_t − tok_{t−1})，跨请求用**未外推**的上次末 token 作 t−1，
            #   避免外推逐次复利放大。
            _beta = float(getattr(self.args, "lead_beta", 0.0) or 0.0)
            if abs(_beta) > 1e-9:
                _prev = getattr(self, "_lead_prev_tok", None)
                _prevrow = chunk64[:1] if _prev is None else _prev[None, :]
                _ext = np.concatenate([_prevrow, chunk64[:-1]], 0).astype(np.float32)
                self._lead_prev_tok = chunk64[-1].copy()
                chunk64 = (chunk64 + _beta * (chunk64 - _ext)).astype(np.float32)
                self._lead_n = getattr(self, "_lead_n", 0) + 1
                if self._lead_n % 20 == 1:
                    print(f"[lead] beta={_beta:g} req#{self._lead_n} "
                          f"|Δchunk| 均={float(np.abs(_beta*(chunk64-_ext)).mean()):.4f}", flush=True)
            # ---- 部署侧 token 量化（2026-11-04）--------------------------------
            # 位置很关键：必须在 **lead 外推之后**。外推 tok + β(tok − tok_{-1}) 一定造出
            # 格点之间的值，只在采样后量化等于没量化。
            # 注意 `self._lead_prev_tok` 存的是**未外推、未量化**的采样值（lead 的原口径，
            # 避免复利）；lead_beta=0（默认）时与本量化完全无关。
            if self._quant_mode != "none":
                from diff_vla_wbc.quant import quantize_np  # noqa: PLC0415
                # 量化前的偏离度 = 训练侧 STE 偏差在部署侧的镜像（半格 = 0.03125）
                chunk64, self._quant_dist = quantize_np(chunk64, self._quant_mode)
                self._quant_n += 1
                if self._quant_n % 20 == 1:
                    print(f"[token-quant] {self._quant_mode} req#{self._quant_n} "
                          f"pre_dist={self._quant_dist:.5f}（半格=0.03125；"
                          f"持续偏大 ⇒ 采样已跑出格点邻域）", flush=True)

            # ---- 手部改用 base 头（可选）-------------------------------------
            # 用户口径（2026-11-05）：手部不参与联合微调，直接沿用 base 的控制手。
            # 做法：**同一份 batch、关掉 LoRA** 再采一次，只取 64:78 那 14 维覆盖 fingers；
            # token 0:64 仍是 FT 头的输出。这样手部行为与 base 逐位一致（噪声独立但同分布）。
            if getattr(self.args, "hand_from_base", False) and getattr(self, "_head_loaded", False):
                from diff_vla_wbc.lora import lora_disabled  # noqa: PLC0415
                with lora_disabled(self.model.action_head), torch.enable_grad():
                    _noise0 = torch.randn(
                        1, self.args.horizon, head_action_dim,
                        device=self.args.device, dtype=self.model.dtype,
                    )
                    _full0 = sample_model_with_grad(
                        self.model, batch, initial_noise=_noise0, num_steps=self.args.fm_steps,
                    )
                _f0 = _full0.detach().to("cpu").float().numpy()[0]
                fingers = _f0[:, 64:78].astype(np.float32)
                self._n_hand_base = getattr(self, "_n_hand_base", 0) + 1
                if self._n_hand_base % 20 == 1:
                    print(f"[server][hand-from-base] req#{self._n_hand_base} "
                          f"fingers absmean={float(np.abs(fingers).mean()):.4f} "
                          f"max={float(np.abs(fingers).max()):.3f}", flush=True)

            # HA's ``vla_action_hand_binary`` is (frames, 2) = (left, right); keep the
            # sides separate instead of collapsing both to one global max.
            left = fingers[:, :HAND_JOINT_SPLIT].max(axis=1, keepdims=True)
            right = fingers[:, HAND_JOINT_SPLIT:].max(axis=1, keepdims=True)
            hands2 = np.concatenate([left, right], axis=1)
            hands2 = (hands2 > HAND_CLOSE_THRESHOLD).astype(np.float32)
            self.last_hands14 = fingers[-1].astype(np.float32)
            self.last_hands14_chunk = fingers.astype(np.float32)
            # S3 §5.3: log the exact VLA input at every re-plan so the joint trainer can start
            # the imagined rollout from the B-line state and the image/state the VLA actually saw.
            _log_dir = getattr(self.args, "log_dir", "")
            if _log_dir:
                try:
                    self._log_req += 1
                    np.savez_compressed(
                        Path(_log_dir) / f"req_{self._log_req:06d}.npz",
                        rgb=np.asarray(rgb, dtype=np.uint8),
                        state64=np.asarray(state64, dtype=np.float32),
                        state46=np.asarray(state46, dtype=np.float32),
                        hands14=np.asarray(hands14, dtype=np.float32),
                        task=np.array(task),
                        chunk64=np.asarray(chunk64, dtype=np.float32),
                        hands2=np.asarray(hands2, dtype=np.float32),
                        fingers=np.asarray(fingers, dtype=np.float32),
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[server][log] WARN save failed: {exc}", flush=True)
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
        _eng = getattr(self, "engine", None)
        self._json(200, {"ok": True, "service": "ha-latent64-vla",
                         "token_space": "command_encoder_latent",
                         # e1 握手：采集器必须看到和自己一致的口径，否则拒绝跑
                         "state_extra": str(getattr(_eng, "state_extra_mode", "none")),
                         # 采/评测侧必须看到同一个量化口径（docs/TOKEN_QUANT_STE_20261104.md）
                         "token_quant": str(getattr(_eng, "_quant_mode", "none")),
                         "token_quant_pre_dist": (None if not bool(np.isfinite(float(
                             getattr(_eng, "_quant_dist", float("nan")))))
                             else round(float(_eng._quant_dist), 6)),
                         "state_extra_scale": (None if getattr(_eng, "state_extra_scale", None) is None
                                               else [round(float(x), 6) for x in _eng.state_extra_scale])})

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
            _e6 = req["observation"].get("state_extra6")
            _s46 = req["observation"].get("state46")
            _h14in = req["observation"].get("hands14")
            chunk, hands = self.engine.infer(rgb, state, task, state_extra6=_e6,
                                             state46_in=_s46, hands14_in=_h14in)
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
                  f"tok_quant={getattr(self.engine, '_quant_mode', 'none')} "
                  f"qdist={float(getattr(self.engine, '_quant_dist', float('nan'))):.5f} "
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
                "token_quant": str(getattr(self.engine, "_quant_mode", "none")),
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
    ap.add_argument("--fm-steps", type=int, default=4,
                    help="flow-matching Euler 步数。**默认 4 = checkpoint 官方口径**"
                         "（`runs/groot_sonic_vla_ha_b64_lr1e-5/config.json` 里 "
                         "`num_inference_timesteps = 4`；GR00T N1.7 官方采样器同款）。"
                         "2026-11-05 修正：2026-11-04 曾把默认抬到 20，理由是「4 步会把 chunk 采平"
                         "⇒ 机器人站桩」——但那次 A/B 同时改了 state46 布局（train_v1→legacy），"
                         "**两个修复混在一起，归因错了**。用已修布局重测：fm=4 走 1.165 m、"
                         "fm=20 走 1.134 m（都正常），且 fm=4 单次推理 231 ms vs fm=20 555 ms。"
                         "⇒ 保持官方 4；只有在专门做采样步数消融时才显式传别的值。")
    # 默认读环境变量：这样 15 个 launcher 不用逐个改，`TOKEN_QUANT=ste bash scripts/xxx.sh` 就生效。
    # argparse 会校验字符串默认值是否在 choices 里，所以写错同样会立即报错。
    ap.add_argument("--token-quant", default=os.environ.get("TOKEN_QUANT", "none"),
                    choices=["none", "ste", "hard"],
                    help="部署侧 token 量化（见 diff_vla_wbc/quant.py）。推理没有梯度，"
                         "所以 ste 与 hard 的**前向完全一样**（都投射到 1/16 格点）；"
                         "两档都接受只是为了和训练侧 --token-quant 同名、便于校验。"
                         "由 vla_meta.json 的 token_quant 校验，放行开关 "
                         "ALLOW_TOKEN_QUANT_MISMATCH=1")
    ap.add_argument("--sample-avg", type=int, default=1,
                    help="同一观测采 N 次求平均（2026-10-28）：降低闭环采样噪声，方差按 1/√N 降。"
                         "1=旧行为；4 是推荐起点（推理成本 ×N，单次 token 只需几十 ms）")
    ap.add_argument("--decode", choices=("codec", "none"), default="codec",
                    help="codec = GR00T min-max 逆变换回原始 token/手指空间（官方 ckpt 必须用）；"
                         "none = 旧口径（我们自己 raw-space 微调过的 ckpt 用）")
    ap.add_argument("--state-order", choices=("ha_v31", "standard"), default="ha_v31",
                    help="order of the 29 joints inside the incoming 64-d vla_state "
                         "(ha_v31 = HumanoidArena asset order, which the current ckpt was "
                         "trained on; standard = trackerlab/MuJoCo order)")
    ap.add_argument("--state-layout", choices=("train_v1", "legacy"), default="legacy",
                    help="46 维 state 的分块布局。**默认 legacy = [joints29|hands14|gravity3]**，"
                         "这是 data/npz/ha_tokens 里 g1_state_46 的真实布局，也是 checkpoint 的 "
                         "dataset_statistics 分块统计能对上的那一种（left_hand 首维恒 0 只在 legacy 成立）。"
                         "train_v1 是 [jp0:22|left_hand|jp22:29|right_hand|gravity]，会把 21/46 维拼错；"
                         "启动时 `_assert_state_layout_matches_training_data()` 会硬校验并拒绝用错的布局。")
    ap.add_argument("--hand-from-base", action="store_true",
                    help="**手部指令改由 base 头出**（关掉 LoRA 再采一次，只取 64:78 这 14 维）。"
                         "2026-11-05 用户口径：手部可以不参与联合微调，直接沿用 base 的控制手，"
                         "避免 FT 把手训坏（实测 base hand14_max=1.826 vs FT 0.567，闭手阈值 1.70）")
    ap.add_argument("--hand-source", choices=("feedback", "zeros"), default="feedback",
                    help="where the 46-d state's hand entries come from (their state has none)")
    ap.add_argument("--dry-run", action="store_true", help="no model: answer zeros to test the wiring")
    ap.add_argument("--log-dir", default="",
                    help="S3 §5.3：每次 /infer 把 rgb/state64/state46/chunk 落盘到该目录")
    ap.add_argument("--head-ckpt", default="",
                    help="S3 joint FT 的 vla_*.pt（含 fm 状态字典）：只替换 action head 的 LoRA/投影，"
                         "不重新导出整份 ckpt。必须与训练时的 --lora-rank/targets 一致")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-targets", default="attn1,attn2,ff")
    ap.add_argument("--lora-also-train", default="proj_out")
    ap.add_argument("--lead-beta", type=float, default=0.0,
                    help="★ 2026-11-01（滞后性检验）：给 64 维 token 块加一阶外推的**提前量**。"
                         "tok_t ← tok_t + β(tok_t − tok_{t−1})（跨请求连续）。"
                         "β=0 旧行为；用来判断闭环误差是不是系统滞后（若是，加点 lead 就该变好）。")
    ap.add_argument("--seed", type=int, default=-1,
                    help="★ 2026-11-01（评测可复现）：给 flow-matching 的初始噪声播种。"
                         "-1=旧行为（每次进程随机，实测同一策略同一 episode 三次跑出 "
                         "0.064/0.143/0.152）⇒ 评测尺子 ±3 cm，1–2 cm 的效应量不出来。"
                         "≥0=固定序列（请求顺序固定 ⇒ 每次运行逐位可比）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    args.token_quant = resolve_token_quant(getattr(args, "token_quant", "none"))

    # ★ 2026-11-06：把**解析后的配置**打一行到 server.log。以前 `--sample-avg 4` 有没有生效
    #   只能靠"服务端起来了=argparse 认得"来推断（本轮判 P0 时真的卡在这里）。
    print(f"[server][cfg] ckpt={args.ckpt} head={args.head_ckpt or 'none'} device={args.device} "
          f"fm_steps={args.fm_steps} sample_avg={args.sample_avg} decode={args.decode} "
          f"token_quant={args.token_quant} state_layout={args.state_layout} "
          f"state_order={args.state_order} hand_from_base={args.hand_from_base} "
          f"hand_source={args.hand_source} lead_beta={args.lead_beta} seed={args.seed} "
          f"horizon={args.horizon} lora_rank={args.lora_rank}", flush=True)

    if not args.dry_run and (args.ckpt is None or not Path(args.ckpt).is_dir()):
        raise SystemExit("--ckpt must point at our VLA checkpoint directory (or use --dry-run)")

    if int(args.seed) >= 0:
        import random  # noqa: PLC0415

        import numpy as _np  # noqa: PLC0415
        import torch as _th  # noqa: PLC0415

        _th.manual_seed(int(args.seed))
        _th.cuda.manual_seed_all(int(args.seed))
        _np.random.seed(int(args.seed))
        random.seed(int(args.seed))
        print(f"[server] 采样已播种 seed={int(args.seed)}（评测可复现）", flush=True)
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
