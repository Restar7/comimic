#!/usr/bin/env python3
"""真实执行漂移的**完整**度量（不是只看 root）。

为什么重写（2026-10-29，用户指出）：`build_b_self_targets.py` 的 ``realized_err`` 只算了
``‖Δroot_xyz‖``，而"执行漂移"实际有六层：root 位置 / root 朝向 / 29 关节角 / 29 关节速度 /
root 线速度 / root 角速度。只看 root 会把"身体在扭、关节在偏"的漂移漏掉，也和 reward 的权重
（body_pos/ori/vel 各 1.0，root_pos 只有 0.5）对不上。

输入是 B 线 dump（``states (T,71)`` + ``frames``，vla 臂的 ``frames`` 是**每 20 tick 一个**）
与 318 紧凑库（``ref (473876,71)``），输出每个分量的分位数。

    python3 scripts/eval_realized_drift.py \
        --dump 'runs/j1_rounds/round_1/bline/b_*.pt' \
        --ref data/library/ft318_compact.pt --out runs/j1_rounds/round_1/drift.json

状态布局（与库一致）：``[p(3), quat wxyz(4), q(29), v_w(3), w_w(3), qd(29)]``
注意库存的是 **world 系**角速度，dump 的 ``states`` 是库回放的同布局，所以两者可比；
但若与训练侧（body 系）比较，用 ``--w-body`` 开关做 ``quat_apply_inverse`` 转换。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """q * v * q⁻¹（wxyz，逐行），用于把 world 角速度转到 body 系。"""
    w, x, y, z = quat.unbind(-1)
    q_vec = torch.stack([x, y, z], -1)
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec - w.unsqueeze(-1) * t + torch.cross(q_vec, t, dim=-1)


def per_tick_frames(frames: torch.Tensor, n_tick: int) -> torch.Tensor:
    """vla 臂只在重规划时刻记 frames（600 tick → 30 个）⇒ 按段展开到每 tick。"""
    if int(frames.numel()) == n_tick:
        return frames
    n_f = int(frames.numel())
    stride = max(1, (n_tick + n_f - 1) // n_f)
    plan_of_tick = torch.clamp(torch.arange(n_tick) // stride, max=n_f - 1)
    return frames[plan_of_tick]


def quantiles(x: torch.Tensor) -> dict[str, float]:
    x = x.flatten().float()
    q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90])
    v = torch.quantile(x, q).tolist()
    return {
        "mean": float(x.mean()),
        "p10": v[0], "p25": v[1], "p50": v[2], "p75": v[3], "p90": v[4],
        "max": float(x.max()),
        "n": int(x.numel()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="glob，B 线 dump 或 eval dump")
    ap.add_argument("--ref", type=Path, default=ROOT / "data/library/ft318_compact.pt")
    ap.add_argument("--out", type=Path, default=None, help="把结果写成 json")
    args = ap.parse_args()

    lib = torch.load(str(args.ref), map_location="cpu", weights_only=False)["ref"].float()
    # ★ 2026-11-03：库里的 ref 是 **canonical 布局** `[p,quat,v,w,q,qd]`（q 在 13:42），
    #   而 A/B 线 dump 的 states 是 **扁平布局** `[p,quat,q,v,w,qd]`（q 在 7:36）。
    #   这里原来直接用 `lib` 去比 `S[:,7:36]`，等于拿关节角比 (v,w,q[0:23]) —— 
    #   `djoint` / `dlin` / `dw_body` 三项全是垃圾（实测把 0.139 rad 的真实误差报成 0.38）。
    #   统一转到扁平布局再比。
    try:
        from diff_vla_wbc.isaac_joint import canonical_to_flat  # noqa: PLC0415
        lib = canonical_to_flat(lib.unsqueeze(0)).squeeze(0)
    except Exception as _e:  # noqa: BLE001
        print(f"[drift][warn] canonical_to_flat 失败（{_e}），按旧行为直接比")
    n_lib = int(lib.shape[0])
    files = sorted(glob.glob(args.dump))
    if not files:
        raise SystemExit(f"[drift] 没匹配到 dump: {args.dump}")

    acc = {
        "root_pos_m": [], "root_ori_deg": [], "joint_rad": [],
        "joint_vel_rad_s": [], "root_lin_vel_m_s": [], "root_ang_vel_body_rad_s": [],
    }
    used = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S, F = d.get("states"), d.get("frames")
        if S is None or F is None:
            continue
        S = S.float()
        T = int(S.shape[0])
        F = per_tick_frames(F.long(), T)
        if int(F.max()) >= n_lib or int(F.min()) < 0:
            raise SystemExit(f"[drift] {Path(f).name}: frames 越界（--ref 指错库？）")
        ref = lib[F]                                              # (T,71)
        dp = (S[:, 0:3] - ref[:, 0:3]).norm(dim=-1)
        dq = S[:, 3:7] * ref[:, 3:7]
        cos_half = dq.sum(-1).abs().clamp(0.0, 1.0)
        dori = 2.0 * torch.acos(cos_half) * 180.0 / 3.141592653589793
        djoint = (S[:, 7:36] - ref[:, 7:36]).pow(2).mean(-1).sqrt()
        djoint_vel = (S[:, 42:71] - ref[:, 42:71]).pow(2).mean(-1).sqrt()
        dlin = (S[:, 36:39] - ref[:, 36:39]).norm(dim=-1)
        dw_body = quat_apply_inverse(ref[:, 3:7], S[:, 39:42]) - quat_apply_inverse(
            ref[:, 3:7], ref[:, 39:42]
        )
        dang = dw_body.norm(dim=-1)
        for k, v in (
            ("root_pos_m", dp), ("root_ori_deg", dori), ("joint_rad", djoint),
            ("joint_vel_rad_s", djoint_vel), ("root_lin_vel_m_s", dlin),
            ("root_ang_vel_body_rad_s", dang),
        ):
            acc[k].append(v)
        used += 1

    if not used:
        raise SystemExit("[drift] 没有可用 dump（需要 states + frames）")

    out = {k: quantiles(torch.cat(v)) for k, v in acc.items()}
    print(f"[drift] {used} 条 dump，{out['root_pos_m']['n']} 个 tick 时刻")
    hdr = f"{'分量':26s} {'mean':>9s} {'p50':>9s} {'p75':>9s} {'p90':>9s} {'max':>9s}"
    print(hdr)
    unit = {
        "root_pos_m": "m", "root_ori_deg": "°", "joint_rad": "rad",
        "joint_vel_rad_s": "rad/s", "root_lin_vel_m_s": "m/s",
        "root_ang_vel_body_rad_s": "rad/s",
    }
    for k, v in out.items():
        print(f"{k + ' [' + unit[k] + ']':26s} {v['mean']:9.4f} {v['p50']:9.4f} "
              f"{v['p75']:9.4f} {v['p90']:9.4f} {v['max']:9.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"n_dumps": used, **out}, indent=2))
        print(f"[drift] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
