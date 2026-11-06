#!/usr/bin/env python3
"""真实执行漂移的**完整**评测：六分量 × 分位数 × 逐集分布（可跨组对比）。

为什么重写（2026-10-30）：之前的评测只报 `pos p50/p90` 三个数，**不足以判断一次配方改动的好坏**。
一次改动可能"中位好、尾部坏"（我们已经踩过），也可能"平均好但某几集整体跑飞"（当前 R1 的情况）。
所以必须同时给出：
  1. **六个分量**：root 位置 / root 朝向 / 29 关节角 / 29 关节角速度 / root 线速度 / root 角速度
  2. **分位数**：mean / p10 / p50 / p75 / p90 / max（尾部必须看 p90+）
  3. **逐集分布**：每集自己的 p50 与 p90，再看有多少集"整体坏掉"（p50 > 阈值）
  4. **朝向按可观测性拆开**：tilt（roll/pitch，可观测）/ yaw（VLA 路径拿不到）/ yaw_rate

用法：
    python3 scripts/eval_drift_full.py --dump 'runs/xxx/b_*.pt' --tag R1
    python3 scripts/eval_drift_full.py --dump 'a/*.pt' 'b/*.pt' --tag pi0 R1     # 多组同表对比
    python3 scripts/eval_drift_full.py --dump 'a/*.pt' --out a/drift_full.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_realized_drift import per_tick_frames, quat_apply_inverse  # noqa: E402


def _up(q):
    """四元数对应的 up 向量 = R(q)·[0,0,1]（完整三维，不是只取 z）。"""
    w, x, y, z = q.unbind(-1)
    return torch.stack([2.0 * (x * z + w * y),
                        2.0 * (y * z - w * x),
                        1.0 - 2.0 * (x * x + y * y)], -1)


def _up_z(q):
    w, x, y, z = q.unbind(-1)
    return (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)


def _yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


DT = 0.02
CLIP_LAST: torch.Tensor | None = None     # 由 main 注入：每帧所在 clip 的最后一帧


def _ref_velocity(pf: torch.Tensor, ref: torch.Tensor):
    """**差分重建**参考的根线速度 / 角速度（world 系）。

    为什么必须重建：库里状态 `[36:39]`(v) 与 `[39:42]`(w) 全库恒为 0（473,876 帧实测
    `max|v| = max|w| = 0.000000`）。直接拿它们当"参考速度"去比，量到的是**机器人自己的速度**，
    不是误差 —— 2026-10-30 被用户指出，原来的 `eval_realized_drift.py` 也有同样的问题。
    """
    last = CLIP_LAST[pf] if CLIP_LAST is not None else pf
    nxt = torch.minimum(pf + 1, last)
    v = (ref[nxt, 0:3] - ref[pf, 0:3]) / DT
    q1, q2 = ref[pf, 3:7], ref[nxt, 3:7]
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    cw = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    cx = w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2
    cy = w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2
    cz = w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2
    sgn = torch.sign(cw).unsqueeze(-1)
    w_body = 2.0 * torch.stack([cx, cy, cz], -1) * sgn / DT
    # 转到 world 系（与状态布局的 w_world 对齐）
    qw, qx, qy, qz = q1.unbind(-1)
    tx = 2.0 * (qy * w_body[..., 2] - qz * w_body[..., 1])
    ty = 2.0 * (qz * w_body[..., 0] - qx * w_body[..., 2])
    tz = 2.0 * (qx * w_body[..., 1] - qy * w_body[..., 0])
    w_world = torch.stack([
        w_body[..., 0] + qw * tx + (qy * tz - qz * ty),
        w_body[..., 1] + qw * ty + (qz * tx - qx * tz),
        w_body[..., 2] + qw * tz + (qx * ty - qy * tx)], -1)
    return v, w_world


def episode_components(dump: Path, ref: torch.Tensor) -> dict:
    d = torch.load(str(dump), map_location="cpu", weights_only=False)
    S = d["states"].float()
    PF = per_tick_frames(d["frames"], S.shape[0])
    R = ref[PF]
    v_ref, w_ref = _ref_velocity(PF, ref)
    out = {
        "root_pos_m": (S[:, 0:3] - R[:, 0:3]).norm(dim=-1),
        "root_ori_deg": torch.acos(((S[:, 3:7] * R[:, 3:7]).sum(-1).abs()).clamp(max=1 - 1e-7))
                        * (360 / math.pi),
        "joint_rad": (S[:, 7:36] - R[:, 7:36]).pow(2).mean(-1).sqrt(),
        "joint_vel_rad_s": (S[:, 42:71] - R[:, 42:71]).pow(2).mean(-1).sqrt(),
        "root_lin_vel_m_s": (S[:, 36:39] - v_ref).norm(dim=-1),
        "root_ang_vel_rad_s": (S[:, 39:42] - w_ref).norm(dim=-1),
        # 朝向按可观测性拆开
        # 真正的 tilt 夹角 = 两个 **up 向量**的夹角。原来的 |tilt(S)-tilt(R)| 是差模长，
        # 两个 tilt 方向不同时会**少算**（实测偏乐观 3–5 倍）。
        "tilt_err_deg": torch.acos((_up(S[:, 3:7]) * _up(R[:, 3:7])).sum(-1).clamp(-1, 1))
                        * (180 / math.pi),
        "yaw_err_deg": (((_yaw(S[:, 3:7]) - _yaw(R[:, 3:7])) + math.pi) % (2 * math.pi) - math.pi).abs()
                       * (180 / math.pi),
        "yaw_rate_err_rad_s": (S[:, 41] - w_ref[:, 2]).abs(),   # 对**重建**参考的 yaw 角速度误差
    }
    out["_fell"] = bool((d.get("meta") or {}).get("fell", False))
    out["_max_tilt"] = float((d.get("meta") or {}).get("max_tilt_deg", 0.0) or 0.0)
    return out


def quant(x: torch.Tensor) -> dict:
    x = x.flatten().float()
    if x.numel() == 0:
        return {}
    q = torch.tensor([0.10, 0.50, 0.75, 0.90]).tolist()
    v = torch.quantile(x, torch.tensor(q)).tolist()
    return {"mean": float(x.mean()), "p10": v[0], "p50": v[1], "p75": v[2], "p90": v[3],
            "max": float(x.max()), "n": int(x.numel())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True, help="一个或多个 glob")
    ap.add_argument("--tag", nargs="*", default=None, help="每组一个名字")
    ap.add_argument("--ref", type=Path, default=ROOT / "data/library/ft318_compact_v8.pt")
    ap.add_argument("--bad-p50-cm", type=float, default=10.0, help="单集 root_pos p50 超过它算'整集坏掉'")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    files = [sorted(glob.glob(g)) for g in a.dump]
    tags = a.tag or [f"group{i}" for i in range(len(files))]
    if len(tags) != len(files):
        raise SystemExit("[fail] --tag 数量要等于 --dump 数量")
    _lib = torch.load(str(a.ref), map_location="cpu", weights_only=False)
    ref = _lib["ref"].float()
    # ★ 2026-11-03：canonical([p,quat,v,w,q,qd]) -> 扁平([p,quat,q,v,w,qd])，
    #   否则 joint_rad 拿关节角比 (v,w,q[0:23])（实测 0.139 -> 0.38 的假误差）。
    try:
        from diff_vla_wbc.isaac_joint import canonical_to_flat as _c2f  # noqa: PLC0415
        ref = _c2f(ref.unsqueeze(0)).squeeze(0)
    except Exception as _e:  # noqa: BLE001
        print(f"[drift][warn] canonical_to_flat 失败（{_e}）")

    global CLIP_LAST
    _starts, _lengths = _lib["starts"], _lib["lengths"]
    _last = torch.tensor([s + l - 1 for s, l in zip(_starts, _lengths)], dtype=torch.long)
    _clip = torch.searchsorted(torch.tensor(_starts, dtype=torch.long),
                               torch.arange(ref.shape[0], dtype=torch.long), right=True) - 1
    CLIP_LAST = _last[_clip.clamp(0, len(_starts) - 1)]

    COMP = ("root_pos_m", "root_ori_deg", "joint_rad", "joint_vel_rad_s",
            "root_lin_vel_m_s", "root_ang_vel_rad_s", "tilt_err_deg", "yaw_err_deg",
            "yaw_rate_err_rad_s")
    result: dict = {}
    for tag, fs in zip(tags, files):
        if not fs:
            print(f"[{tag}] 无数据"); continue
        eps = [episode_components(Path(f), ref) for f in fs]
        pooled = {c: torch.cat([e[c] for e in eps]) for c in COMP}
        per_ep_p50 = {c: torch.tensor([float(e[c].median()) for e in eps]) for c in COMP}
        bad = sum(1 for e in eps if float(e["root_pos_m"].median()) * 100 > a.bad_p50_cm)
        result[tag] = {"n_ep": len(eps), "n_tick": int(pooled["root_pos_m"].numel()),
                       "fell": sum(1 for e in eps if e["_fell"]),
                       "max_tilt_deg": max(e["_max_tilt"] for e in eps),
                       "bad_episodes": bad, "bad_p50_cm": a.bad_p50_cm,
                       "pooled": {c: quant(pooled[c]) for c in COMP},
                       "per_episode_p50": {c: quant(per_ep_p50[c]) for c in COMP}}
        r = result[tag]
        print(f"\n===== [{tag}] {r['n_ep']} 集 / {r['n_tick']} tick | 摔倒 {r['fell']} "
              f"| 最大倾角 {r['max_tilt_deg']:.1f}° | 整集坏掉(>p50 {a.bad_p50_cm:.0f}cm) {bad}/{r['n_ep']} =====")
        print(f"  {'分量':<22}{'mean':>9}{'p10':>8}{'p50':>8}{'p75':>8}{'p90':>9}{'max':>9}   单位")
        unit = {"root_pos_m": "cm", "root_ori_deg": "°", "joint_rad": "rad", "joint_vel_rad_s": "rad/s",
                "root_lin_vel_m_s": "m/s", "root_ang_vel_rad_s": "rad/s",
                "tilt_err_deg": "°", "yaw_err_deg": "°", "yaw_rate_err_rad_s": "rad/s"}
        scale = {"root_pos_m": 100.0, "root_lin_vel_m_s": 1.0}
        for c in COMP:
            q = r["pooled"][c]; s = scale.get(c, 1.0)
            print(f"  {c:<22}{q['mean']*s:9.3f}{q['p10']*s:8.3f}{q['p50']*s:8.3f}"
                  f"{q['p75']*s:8.3f}{q['p90']*s:9.3f}{q['max']*s:9.3f}   {unit[c]}")
        qp = r["per_episode_p50"]["root_pos_m"]
        print(f"  → 逐集 root_pos p50 的分布: p10={qp['p10']*100:.1f} p50={qp['p50']*100:.1f} "
              f"p90={qp['p90']*100:.1f} max={qp['max']*100:.1f} cm")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
