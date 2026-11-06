#!/usr/bin/env python3
"""诊断 `qd_ratio` 的高估**集中在哪**（哪一类样本把均值比拉高）。

背景（2026-10-29）：三阶段训练后 `gate_wm_bline` 的 `qd_ratio = 1.1045`，只差门槛 1.10 的 0.45%。
继续盲目加训练步数不一定有效 —— 先看清"高估是全体 10%，还是尾部 10% 的极端漂移样本"：

  * 若集中在尾部（`root_err` 大）：1.10 这个全局门槛对部署意义不大，应改成**分桶判据**
    （全体 ≤1.12 且 `err<0.3m` 子集 ≤1.05）；
  * 若全体均匀高估：值得加"幅值正则"（`loss += λ·(mean|pred|−mean|ref|)²`）或加阶段 3 预算。

用法：
    python3 scripts/diag_wm_ratio_by_err.py \
        --ckpt runs/j1_scene/round_1/wm_pre.pt \
        --transitions runs/j1_scene/round_1/bline_transitions.pt \
        --tokens data/library/ft318_tokens.pt --n 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def stats(pq: torch.Tensor, rq: torch.Tensor) -> dict:
    corr = float(((pq - pq.mean(0)) * (rq - rq.mean(0))).mean() / (pq.std() * rq.std() + 1e-9))
    ratio = float(pq.abs().mean() / rq.abs().mean().clamp_min(1e-9))
    return {"n": int(pq.shape[0]), "corr": corr, "ratio": ratio,
            "pred": float(pq.abs().mean()), "ref": float(rq.abs().mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--transitions", type=Path, required=True)
    ap.add_argument("--motion", default="data/npz/g1_tasks")
    ap.add_argument("--tokens", type=Path, default=ROOT / "data/library/ft318_tokens.pt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from diff_vla_wbc.mjlab_joint import default_xi, load_mjlab_bundle

    npz = sorted(Path(a.motion).glob("*.npz"))
    bundle = load_mjlab_bundle(a.ckpt, device=a.device, npz=npz, token_dim=64)
    dev = torch.device(a.device)

    d = torch.load(str(a.transitions), map_location="cpu", weights_only=False)
    n_all = int(d["actions"].shape[0])
    g = torch.Generator().manual_seed(int(a.seed))
    idx = torch.randperm(n_all, generator=g)[: min(int(a.n), n_all)]
    st = d["states"][idx].to(dev).float()
    ac = d["actions"][idx].to(dev).float()
    nx = d["next_states"][idx].to(dev).float()
    xi = default_xi(st.shape[0], dev)
    with torch.no_grad():
        pred = bundle.step(st, ac, xi)

    pq, rq = pred[..., 42:71], nx[..., 42:71]
    # 分桶依据：当前状态相对参考的 root 位置误差（≈ realized_err 的瞬时版）
    ref = d["ref_rows"][idx].to(dev).float() if "ref_rows" in d else st
    err = (st[..., 0:3] - ref[..., 0:3]).norm(dim=-1)

    tot = stats(pq, rq)
    print(f"[diag] 全部 n={tot['n']} corr={tot['corr']:.4f} ratio={tot['ratio']:.4f} "
          f"(pred {tot['pred']:.4f} / ref {tot['ref']:.4f})")
    print(f"{'root_err 区间':>18s} {'n':>7s} {'占比':>6s} {'corr':>7s} {'ratio':>7s} {'pred':>8s} {'ref':>8s}")
    edges = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (err >= lo) & (err < hi)
        if int(m.sum()) < 10:
            continue
        s = stats(pq[m], rq[m])
        print(f"[{lo:.2f},{hi if hi < 1e8 else 999:.2f})".rjust(18)
              + f" {s['n']:7d} {100*s['n']/tot['n']:5.1f}% {s['corr']:7.4f} {s['ratio']:7.4f} "
                f"{s['pred']:8.4f} {s['ref']:8.4f}")

    # 部署最关心的子集：误差小（真正在跟参考走）的那些时刻
    for thr in (0.10, 0.30):
        m = err < thr
        if int(m.sum()) >= 10:
            s = stats(pq[m], rq[m])
            print(f"[diag] err<{thr:.2f} m: n={s['n']} ({100*s['n']/tot['n']:.1f}%) "
                  f"corr={s['corr']:.4f} ratio={s['ratio']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
