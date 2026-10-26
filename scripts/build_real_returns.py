#!/usr/bin/env python3
"""用**真录制轨迹**算每帧的 return-to-go，给 critic 一个 grounded 的价值目标。

为什么（这是 S3 目前最大的漏洞）：
  之前的 critic 用"想象转移上的 GAE"训练 —— 它的目标本身来自 WM 的想象，
  WM 一偏，critic 就跟着偏，bootstrap 又把 WM 的偏差放大进策略梯度。
  但**奖励只需要状态**，而录制状态是真的 ⇒ 我们可以在真数据上算出真回报，
  critic 于是有了不依赖 WM 的监督信号。

口径（与训练逐项对齐，脚本里都会打印并写进 meta）：
  * 状态：`runs/wm_real/ha_transitions.pt` 的 `states`（按帧对齐后）；
  * 动作：同文件的 `actions`（= (q_cmd(t+1)−default_q)/scale），prev_action 取前一帧；
  * 参考：`FrameRef(f)`（motion library 的第 f 帧命令流）；
  * 奖励：`bundle.reward(...)["total"]`，**与训练完全同一个函数**，因此
    `--reward-mode / --reward-vel-scale / --posture-weight` 必须与训练一致（记录在 meta 里）；
  * 折扣：γ=0.99，逐 clip 反向累加（跨 clip 不串）。

输出：`runs/wm_real/real_returns.pt`
    frames (N,) int64 / ret (N,) float32 / reward (N,) float32 / meta (dict)

用法：
    python3 scripts/build_real_returns.py            # 默认 γ=0.99、saturating、vel_scale=1
    python3 scripts/build_real_returns.py --reward-vel-scale 0 --out runs/wm_real/real_returns_vs0.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"),
)
sys.path.insert(0, str(ROOT))


def frame_aligned(states: torch.Tensor, actions: torch.Tensor, frames: torch.Tensor,
                  n_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """转移行 → 帧对齐（每个 clip 末帧没有后继转移，用同 clip 上一帧顶替）。"""
    m = torch.full((n_frames,), -1, dtype=torch.long)
    m[frames.long()] = torch.arange(frames.numel())
    miss = (m < 0).nonzero(as_tuple=False).flatten()
    if miss.numel():
        m[miss] = m[(miss - 1).clamp(min=0)]
    if int((m < 0).sum()):
        raise SystemExit(f"[fail] {int((m < 0).sum())} 帧无法对齐")
    return states[m], actions[m]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", type=Path, default=ROOT / "runs/wm_real/ha_transitions.pt")
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--motion-packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/real_returns.pt")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--reward-mode", default="saturating", choices=("saturating", "linear"))
    ap.add_argument("--reward-vel-scale", type=float, default=1.0)
    ap.add_argument("--posture-weight", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clips", nargs="*", default=None, help="只跑这些 clip（调试用）")
    a = ap.parse_args()

    dev = torch.device(a.device)
    from diff_vla_wbc.isaac_joint import (  # noqa: PLC0415
        FrameRef, canonical_to_flat, load_isaac_bundle,
    )

    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz, packed_motion=a.motion_packed,
                               obs_mode=None, hist=None, token_dim=64)
    bundle.reward_mode = str(a.reward_mode)
    bundle.reward_vel_scale = float(a.reward_vel_scale)
    bundle.posture_weight = float(a.posture_weight)
    lib = bundle.library
    n_frames = int(lib.num_frames)
    starts = lib.clip_start.detach().cpu().long()
    lengths = lib.clip_len.detach().cpu().long()
    print(f"[rr] library 帧数={n_frames} clip 数={len(lengths)} "
          f"mode={a.reward_mode} vel_scale={a.reward_vel_scale} posture={a.posture_weight}", flush=True)

    d = torch.load(str(a.transitions), map_location="cpu", weights_only=False)
    S, A = frame_aligned(d["states"], d["actions"], d["frames"], n_frames)
    print(f"[rr] 帧对齐后 states={tuple(S.shape)} actions={tuple(A.shape)}", flush=True)

    keep = None
    if a.clips:
        names = list(getattr(lib, "names", []))
        keep = torch.zeros(n_frames, dtype=torch.bool)
        for c in a.clips:
            hit = [i for i, nm in enumerate(names) if c in str(nm)]
            for i in hit:
                keep[starts[i]:starts[i] + lengths[i]] = True
            print(f"[rr] clip {c}: {len(hit)} 段", flush=True)

    ret_all = torch.zeros(n_frames, dtype=torch.float32)
    rew_all = torch.zeros(n_frames, dtype=torch.float32)
    t0 = torch.cuda.Event(enable_timing=True) if dev.type == "cuda" else None
    t0 and t0.record()

    for ci in range(len(lengths)):
        s, n = int(starts[ci]), int(lengths[ci])
        if n <= 1:
            continue
        if keep is not None and not bool(keep[s]):
            continue
        fr = torch.arange(s, s + n, device=dev)
        rews = []
        for lo in range(0, n, a.chunk):
            f = fr[lo:lo + a.chunk]
            st = S[f.cpu()].to(dev)
            act = A[f.cpu()].to(dev)
            prev = A[(f.cpu() - 1).clamp(min=s)].to(dev)
            out = bundle.reward(st, act, prev, FrameRef(f))
            rews.append(out["total"].float())
        r = torch.cat(rews)                                     # (n,)
        # 反向折扣累加：G[f] = r[f] + γ·G[f+1]
        g = torch.zeros_like(r)
        acc = torch.zeros((), device=dev, dtype=r.dtype)
        for t in range(n - 1, -1, -1):
            acc = r[t] + a.gamma * acc
            g[t] = acc
        ret_all[s:s + n] = g.cpu()
        rew_all[s:s + n] = r.cpu()
        if ci % 100 == 0:
            print(f"[rr] clip {ci + 1}/{len(lengths)} n={n} r_mean={float(r.mean()):+.5f} "
                  f"G[0]={float(g[0]):+.4f} G[-1]={float(g[-1]):+.4f}", flush=True)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    payload = {
        "frames": torch.arange(n_frames),
        "ret": ret_all,
        "reward": rew_all,
        "meta": {
            "gamma": a.gamma, "reward_mode": a.reward_mode,
            "reward_vel_scale": a.reward_vel_scale, "posture_weight": a.posture_weight,
            "transitions": str(a.transitions), "wbc_ckpt": str(a.wbc_ckpt),
            "note": "ret[f] = Σ_{t>=f} γ^(t−f) r_t 沿**真录制轨迹**（奖励只依赖真状态，不经 WM）",
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(a.out))
    print(f"[rr] saved {a.out}  ret mean={float(ret_all.mean()):+.4f} "
          f"std={float(ret_all.std()):.4f} min={float(ret_all.min()):+.3f} max={float(ret_all.max()):+.3f}",
          flush=True)
    (a.out.with_suffix(".json")).write_text(json.dumps(payload["meta"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
