#!/usr/bin/env python3
"""Segmented teacher-forced collection on ft318 with the SONIC executor (A' for the WM).

为什么要"分段"
--------------
`docs/MUJOCO_SONIC_JOINTFT_PLAN_20261104.md` §2.2 实测：长时回放 46–70 步就摔（Isaac 也一样），
但 **H=20 步 + 每段重置到参考** 是 0 摔。所以 A'（WM 的 teacher-forced 数据）只能这个形状：

    对每条 clip 均匀取 START_FRAME=f → 重置到 ref[f] → 闭环跑 H 步 → 记 (s, a, s', frame)

动作口径
--------
`actions` 存的是 **bundle 的动作空间**（= `q_target = default_q + action_scale·a` 的 `a`），
也就是 SONIC decoder 原始输出（HA_V31 序）**换回 STANDARD 序**之后的值。
这与 `scripts/build_real_transitions_mjlab.py` 的约定一致（"WM 吃的是归一化动作 a 本身"），
所以产物可以直接喂 `pretrain_wm.py`。

状态口径（71 维，与 `diff_vla_wbc/mjlab_joint.py` 一致）
-------------------------------------------------------
``[p(3), quat(4), q(29), v_w(3), w_b(3), qd(29)]``
（`v_w` 世界系线速度、`w_b` **体系**角速度 —— MuJoCo free joint 的 qvel[3:6] 就是体系）

用法::

    PY=third_party/SuperMimic-Reproduced/.venv/bin/python
    $PY scripts/collect_seg_ft318.py --set-index 0 --episodes 0,1 --segments 16 --seg-len 20 \
        --out runs/wm_real/ft318_seg/set0.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "groot_repro"))


def _replay_module():
    spec = importlib.util.spec_from_file_location(
        "replay_ft318_mujoco", ROOT / "scripts" / "replay_ft318_mujoco.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def state71(plant) -> np.ndarray:
    m, d = plant.model, plant.data
    import g1_sim as G

    rot = G.quat_to_rotmat(d.qpos[3:7])
    return np.concatenate([
        d.qpos[0:3], d.qpos[3:7], plant.q(), d.qvel[0:3], d.qvel[3:6], plant.dq(),
    ]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-index", type=int, default=0)
    ap.add_argument("--episodes", default="0")
    ap.add_argument("--segments", type=int, default=16, help="每条 clip 采几段")
    ap.add_argument("--seg-len", type=int, default=20, help="段长（=joint FT 的 H）")
    ap.add_argument("--gains", default="ha", choices=("ha", "deploy"))
    ap.add_argument("--decoder", type=Path, default=None)
    ap.add_argument("--decoder-impl", default="torch", choices=("torch", "onnx"))
    ap.add_argument("--action-noise-rad", type=float, default=0.0,
                    help="L1：按角度统一的高斯噪声（rad），实现为 σ/action_scale")
    ap.add_argument("--start-perturb-xy", type=float, default=0.0, help="L2：起点 xy 扰动（m）")
    ap.add_argument("--hold-jitter", type=int, default=5, help="噪声重采间隔抖动 ⇒ hold∈[H-j, H+j]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/ft318_seg/seg.pt")
    args = ap.parse_args()

    import g1_sim as G
    from sonic_loop import Decoder, SIGNALS

    rng = np.random.default_rng(args.seed)
    rp = _replay_module()
    plant = rp.Plant(rp.DEFAULT_SCENE, gains=args.gains)
    torch.set_num_threads(1)          # 多进程并行时单线程更快（避免 ORT/BLAS 线程超订）
    if args.decoder_impl == "torch":
        from diff_vla_wbc.sonic_decoder import load_decoder  # noqa: PLC0415

        class _TorchDecoder:                                   # 与 ORT 逐算子等价（3e-6）
            def __init__(self, onnx):
                self.graph = load_decoder(str(onnx)).eval()

            @torch.no_grad()
            def __call__(self, obs):
                return self.graph(torch.from_numpy(obs.reshape(1, 994).astype(np.float32)))[0].numpy()

        decoder = _TorchDecoder(str(args.decoder) if args.decoder else rp.DEFAULT_DECODER)
    else:
        decoder = Decoder(str(str(args.decoder) if args.decoder else rp.DEFAULT_DECODER))

    ha_to_std = np.array(
        __import__("joint_contract").permutation(src="ha_v31", dst="standard"), dtype=np.int64
    )
    scale = G.ACTION_SCALE.astype(np.float64)

    S, N, A, F, META = [], [], [], [], []
    for epi in [int(x) for x in args.episodes.split(",") if x.strip()]:
        ep = rp.load_episode(args.set_index, epi)
        T = ep["state"].shape[0]
        stride = max(1, (T - args.seg_len - 2) // max(args.segments, 1))
        starts = list(range(0, T - args.seg_len - 2, stride))[: args.segments]
        for f0 in starts:
            q0 = ep["body_state"][f0]
            plant.reset(q0, ep["hands_state"][f0],
                        xy=(float(rng.normal(0, args.start_perturb_xy)),
                            float(rng.normal(0, args.start_perturb_xy))))
            hist = [{
                "base_ang_vel": np.zeros(3),
                "q": (q0 - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
                "dq": np.zeros(29),
                "last_action": np.zeros(29),
                "gravity": G.projected_gravity(plant.quat()),
            } for _ in range(10)]
            hold = args.seg_len
            noise = np.zeros(29)
            for t in range(f0, min(f0 + args.seg_len, T - 1)):
                if t - f0 >= hold:
                    hold = max(2, args.seg_len + int(rng.integers(-args.hold_jitter, args.hold_jitter + 1)))
                    noise = (rng.normal(0.0, args.action_noise_rad, 29) / np.maximum(scale, 1e-6)
                             if args.action_noise_rad > 0 else np.zeros(29))
                parts = [ep["tokens"][t]]
                for name in SIGNALS:
                    parts.append(np.concatenate([f[name] for f in hist]))
                a_ha = decoder(np.concatenate(parts))
                a_std = a_ha[ha_to_std]
                a_used = a_std + noise
                q_target = G.DEFAULT_ANGLES + scale * a_used
                s_before = state71(plant)
                plant.apply_target(q_target, ep["hands_wbc"][t])
                s_after = state71(plant)
                rot = G.quat_to_rotmat(plant.quat())
                hist.append({
                    "base_ang_vel": plant.data.qvel[3:6].copy(),   # free joint: body-frame ω
                    "q": (plant.q() - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
                    "dq": plant.dq()[G.MUJOCO_TO_ISAACLAB],
                    "last_action": a_ha,
                    "gravity": G.projected_gravity(plant.quat()),
                })
                hist = hist[-10:]
                S.append(s_before)
                N.append(s_after)
                A.append(a_used.astype(np.float32))
                F.append(int(t))
                META.append({"set": args.set_index, "ep": epi, "start": int(f0)})
    if not S:
        raise SystemExit("[fail] no transitions collected")

    states = torch.tensor(np.stack(S), dtype=torch.float32)
    next_states = torch.tensor(np.stack(N), dtype=torch.float32)
    actions = torch.tensor(np.stack(A), dtype=torch.float32)
    frames = torch.tensor(F, dtype=torch.long)
    payload = {
        "states": states, "next_states": next_states, "actions": actions, "frames": frames,
        "meta": {
            "source": "ft318_seg_sonic",
            "set_index": args.set_index,
            "episodes": args.episodes,
            "segment_len": args.seg_len,
            "gains": args.gains,
            "action_noise_rad": args.action_noise_rad,
            "start_perturb_xy": args.start_perturb_xy,
            "n": int(states.shape[0]),
        },
    }
    bad = not (torch.isfinite(states).all() and torch.isfinite(actions).all())
    print(json.dumps(payload["meta"], ensure_ascii=False), flush=True)
    print(f"[shape] states {tuple(states.shape)} actions {tuple(actions.shape)} "
          f"finite={not bad}", flush=True)
    if bad:
        raise SystemExit("[fail] NaN/Inf in collected transitions")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.out))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
