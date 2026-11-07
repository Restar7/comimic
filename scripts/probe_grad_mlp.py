#!/usr/bin/env python3
"""**可学性**判读：单步梯度 ĝ1 = ∂r_0/∂a_0 能不能被"可观测输入"预测？

背景（2026-11-06）
----------------
四种参数化（LoRA / 反常量项 / state_extra / delta-token）全部收敛到**常向量**，
所有**线性** ridge 探针的 CV-R² 都 ≤0。但线性不可解 ≠ 不可学：∂r/∂a 里含 FK 雅可比与
huber 分段符号，本来就该是非线性的。这个脚本用**和 delta-token 同量级的两层 MLP**
做 8 折 CV，给出"给定这个容量，能不能学出来"的直接答案。

  R²_mlp 高（>0.3）⇒ 有可学结构 ⇒ 训练失败是数据量/优化/幅度的问题，继续投
  R²_mlp ≈0（且线性也 ≈0）⇒ ĝ 对可观测输入是高维噪声 ⇒ "梯度蒸馏"路线从目标上就死

用法
----
  J1_GRAD_DUMP=runs/gradump bash scripts/launch_joint_ha_v2.sh ... (STEPS=12 起)
  python scripts/probe_grad_mlp.py --dir runs/gradump
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

SETS = {
    "state": ["state"],
    "act0": ["act0"],
    "state+act0": ["state", "act0"],
    "feats": ["feats"],
    "token": ["token"],
    "ref0": ["ref0"],
    "state+act0+token": ["state", "act0", "token"],
    "state+token": ["state", "token"],
    "actH": ["actH"],
    "state+actH": ["state", "actH"],
    "state+actH+token": ["state", "actH", "token"],
    "all": ["state", "act0", "token", "feats", "ref0"],
    "all+actH": ["state", "act0", "actH", "token", "feats", "ref0"],
}


def load(d: str):
    files = sorted(glob.glob(os.path.join(d, "dump_*.npz")))
    if not files:
        raise SystemExit(f"没有 dump: {d}")
    keys = None
    chunks = {}
    for f in files:
        z = np.load(f)
        if keys is None:
            keys = [k for k in z.files if k not in ("r1",)]
        for k in keys:
            chunks.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v, 0).astype(np.float64) for k, v in chunks.items()}
    out["n_files"] = len(files)  # type: ignore[assignment]
    return out


def cv_r2_ridge(X, Y, folds=8, lam=10.0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    n = len(X)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    ss_res = ss_tot = 0.0
    rel_num = rel_den = cos_num = 0.0
    cnt = 0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ Y[tr])
        pred = X[te] @ W
        ss_res += float(((Y[te] - pred) ** 2).sum())
        ss_tot += float(((Y[te] - Y[tr].mean(0)) ** 2).sum())
        rel_num += float(np.linalg.norm(Y[te] - pred, axis=1).sum())
        rel_den += float(np.linalg.norm(Y[te], axis=1).sum())
        _a = Y[te] / (np.linalg.norm(Y[te], axis=1, keepdims=True) + 1e-12)
        _b = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
        cos_num += float((_a * _b).sum())
        cnt += len(Y[te])
    return 1.0 - ss_res / max(ss_tot, 1e-12), rel_num / max(rel_den, 1e-12), cos_num / max(cnt, 1)


def cv_r2_mlp(X, Y, folds=8, hidden=256, steps=3000, lr=1e-3, seed=0, device="cuda:0"):
    """两层 MLP（≈delta-token 容量）8 折 CV，返回 (R², 相对 L2, 平均 cos)。"""
    torch.manual_seed(seed)
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Xn = (X - Xm) / Xs
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-8
    Yn = (Y - Ym) / Ys
    n = len(Xn)
    perm = np.random.default_rng(seed).permutation(n)
    ss_res = ss_tot = 0.0
    rel_num = rel_den = 0.0
    cos_num = cos_den = 0.0
    for f in range(folds):
        te = perm[f::folds]
        tr = np.setdiff1d(perm, te)
        xtr = torch.tensor(Xn[tr], dtype=torch.float32, device=device)
        ytr = torch.tensor(Yn[tr], dtype=torch.float32, device=device)
        xte = torch.tensor(Xn[te], dtype=torch.float32, device=device)
        net = torch.nn.Sequential(
            torch.nn.Linear(Xn.shape[1], hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, Yn.shape[1]),
        ).to(device)
        # 内部再切 10% 做早停，避免"训到过拟合"被误读成"学不出来"
        ntr = len(xtr)
        vperm = torch.randperm(ntr, device=device)
        nval = max(32, ntr // 10)
        vidx, tidx = vperm[:nval], vperm[nval:]
        xva, yva = xtr[vidx], ytr[vidx]
        xtr, ytr = xtr[tidx], ytr[tidx]
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        best, best_state, bad = float("inf"), None, 0
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = ((net(xtr) - ytr) ** 2).mean()
            loss.backward()
            opt.step()
            if _ % 100 == 99:
                with torch.no_grad():
                    v = float(((net(xva) - yva) ** 2).mean())
                if v < best - 1e-6:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 4:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        with torch.no_grad():
            pred = net(xte).double().cpu().numpy()
        pred_raw = pred * Ys + Ym
        ss_res += float(((Y[te] - pred_raw) ** 2).sum())
        ss_tot += float(((Y[te] - Y[tr].mean(0)) ** 2).sum())
        rel_num += float(np.linalg.norm(Y[te] - pred_raw, axis=1).sum())
        rel_den += float(np.linalg.norm(Y[te], axis=1).sum())
        _a = Y[te] / (np.linalg.norm(Y[te], axis=1, keepdims=True) + 1e-12)
        _b = pred_raw / (np.linalg.norm(pred_raw, axis=1, keepdims=True) + 1e-12)
        cos_num += float((_a * _b).sum())
        cos_den += len(Y[te])
    return 1.0 - ss_res / max(ss_tot, 1e-12), rel_num / max(rel_den, 1e-12), cos_num / max(cos_den, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/gradump")
    ap.add_argument("--tag", default="")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--target", default="g1", help="g1(动作空间单步) / g_tok1 / g_tokH")
    args = ap.parse_args()

    d = load(args.dir)
    if args.target not in d:
        raise SystemExit(f"dump 里没有 {args.target}（有 {sorted(k for k in d if k != 'n_files')}）")
    Y = d[args.target]
    n = len(Y)
    print(f"[data] target={args.target} {args.dir} files={d['n_files']} n={n} y={Y.shape[1]} "
          f"|Y|mean={np.abs(Y).mean():.5f} per-dim-std/|mean|={Y.std(0).mean() / (np.abs(Y.mean(0)).mean() + 1e-12):.2f}")
    print(f"[data] Y 跨样本 std 占比 = {Y.std(0).sum() / (np.abs(Y).mean(0).sum() + 1e-12):.2f}")

    # 常数基线（预测每列均值）的 R² 恒为 0；这里给"零预测"的 R² 作参照
    ss0 = float((Y ** 2).sum())
    ss_tot_c = float(((Y - Y.mean(0)) ** 2).sum())
    print(f"[base] 零预测 R²={1 - ss0 / ss_tot_c:+.3f}  (≈0 表示均值≈0)")

    for name in [s for s in args.sets.split(",") if s]:
        cols = SETS.get(name)
        if cols is None:
            print(f"[skip] 未知特征集 {name}")
            continue
        if any(c not in d for c in cols):
            print(f"[skip] {name}: 缺 {[c for c in cols if c not in d]}")
            continue
        X = np.concatenate([d[c] for c in cols], 1)
        r2_lin, rel_lin, cos_lin = cv_r2_ridge(X, Y, folds=args.folds, lam=10.0)
        r2_mlp, rel, cos = cv_r2_mlp(X, Y, folds=args.folds, hidden=args.hidden,
                                     steps=args.steps, device=args.device)
        print(f"[cv] {name:<18} dim={X.shape[1]:<5} ridge r2={r2_lin:+.3f} rel={rel_lin:.3f} "
              f"cos={cos_lin:+.3f} | mlp r2={r2_mlp:+.3f} rel={rel:.3f} cos={cos:+.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
