#!/usr/bin/env python3
"""蒸馏**天花板**检验：给定缓存好的 (state, token, ĝ) 标签，学生网络到底能学到多少？

为什么要它
----------
训练时每步只给 8 个窗口（一次 20 步 WM 反传 ~0.4 s/窗口），150 步下来约 2,400 条标签，
而学生有 1,247,296 参数 —— 严重欠训练。但"欠训练"和"数据结构学不动"是两种完全不同的结论，
前者只要加标签生成量，后者必须换输入或换目标。

本脚本用**纯监督**（不跑 WM、不跑闭环、不跑 VLA）在缓存标签上训练同结构网络，
用 held-out cos 判定天花板：

    held-out cos ≥0.25 ⇒ 架构/数据能学 ⇒ 瓶颈是标签量 ⇒ 去生成 3 万条
    held-out cos ≈0.10 ⇒ 学不动 ⇒ 换输入（state−ref）或换视界

用法
----
    python scripts/probe_distill_ceiling.py --dir runs/gradtok2 --target g_tokH
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def quat_to_rot6(q: np.ndarray) -> np.ndarray:
    """wxyz quat → 前两列的 6 维表示（与部署 state64[:6] 同尺度的近似）。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-9
    w, x, y, z = w / n, x / n, y / n, z / n
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    return np.stack([r00, r10, r20, r01, r11, r21], 1).astype(np.float32)


def load(d: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(Path(d) / "dump_*.npz")))
    if not files:
        raise SystemExit(f"没有 dump: {d}")
    buf: dict[str, list] = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            buf.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, 0) for k, v in buf.items()}


def build_features(d: dict, kind: str) -> np.ndarray:
    st = d["state"].astype(np.float32)          # (N,71) = [p, quat, q, v, w, qd]
    tok = d["token"].astype(np.float32)
    q, qd = st[:, 7:36], st[:, 42:71]
    if kind == "deploy":                        # token + state64（部署口径）
        s64 = np.concatenate([quat_to_rot6(st[:, 3:7]), q, qd], 1)
        return np.concatenate([tok, s64], 1)
    if kind == "old35":                         # token + [rot6d + dof_vel]（老口径）
        return np.concatenate([tok, quat_to_rot6(st[:, 3:7]), qd], 1)
    if kind == "state71":
        return st
    if kind == "token":
        return tok
    raise SystemExit(f"未知特征集 {kind}")


def ridge_cos(X, Y, lam=10.0, folds=5):
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    n = len(X)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    cos_all = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        W = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ Y[tr])
        p = X[te] @ W
        a = Y[te] / (np.linalg.norm(Y[te], axis=1, keepdims=True) + 1e-12)
        b = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
        cos_all.append((a * b).sum(1))
    return float(np.concatenate(cos_all).mean())


def mlp_cos_split(X, Y, tr, te, seed=0, hidden=1024, depth=2, epochs=200, lr=1e-3,
                  device="cuda:0", bs=256):
    """同结构学生，**外部给定训练/验证索引**（用于 leave-cluster-out）。"""
    torch.manual_seed(seed)
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Xn = ((X - Xm) / Xs).astype(np.float32)
    Yn = (Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    xtr = torch.tensor(Xn[tr], device=device)
    ytr = torch.tensor(Yn[tr], device=device)
    xte = torch.tensor(Xn[te], device=device)
    yte = torch.tensor(Yn[te], device=device)
    layers, din = [], X.shape[1]
    for _ in range(max(1, depth)):
        layers += [torch.nn.Linear(din, hidden), torch.nn.SiLU()]
        din = hidden
    layers.append(torch.nn.Linear(din, Y.shape[1]))
    net = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ntr = len(tr)
    for _ in range(epochs):
        idx = torch.randperm(ntr, device=device)
        for i in range(0, ntr, bs):
            j = idx[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = ((net(xtr[j]) - ytr[j]) ** 2).sum(-1).mean()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        p = net(xte)
        cos = (p * yte).sum(1) / (p.norm(dim=1).clamp_min(1e-9) * yte.norm(dim=1).clamp_min(1e-9))
    return float(cos.mean())


def mlp_cos(X, Y, seed=0, hidden=1024, depth=2, epochs=200, lr=1e-3, val=0.3,
            device="cuda:0", bs=256):
    """与部署同结构（128→1024→1024→64）的学生；目标是**单位化**的 ĝ（与部署一致）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    nval = int(n * val)
    te, tr = perm[:nval], perm[nval:]
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Xn = ((X - Xm) / Xs).astype(np.float32)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)   # 单位化（部署目标=η·unit(ĝ)）
    Yn = Yn.astype(np.float32)
    xtr = torch.tensor(Xn[tr], device=device)
    ytr = torch.tensor(Yn[tr], device=device)
    xte = torch.tensor(Xn[te], device=device)
    yte = torch.tensor(Yn[te], device=device)
    layers = []
    din = X.shape[1]
    for _ in range(max(1, depth)):
        layers += [torch.nn.Linear(din, hidden), torch.nn.SiLU()]
        din = hidden
    layers.append(torch.nn.Linear(din, Y.shape[1]))
    net = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ntr = len(tr)
    for ep in range(epochs):
        idx = torch.randperm(ntr, device=device)
        for i in range(0, ntr, bs):
            j = idx[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = ((net(xtr[j]) - ytr[j]) ** 2).sum(-1).mean()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        p = net(xte)
        cos = (p * yte).sum(1) / (p.norm(dim=1).clamp_min(1e-9) * yte.norm(dim=1).clamp_min(1e-9))
    return float(cos.mean()), float(cos.std() / max(len(cos) ** 0.5, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/gradtok2")
    ap.add_argument("--target", default="g_tokH")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--split", default="random", choices=("random", "cluster"),
                    help="random=随机切分（偏乐观）；cluster=按参考轨迹聚类后留一整簇（测跨运动泛化）")
    ap.add_argument("--n-clusters", type=int, default=10)
    a = ap.parse_args()

    d = load(a.dir)
    Y = d[a.target].astype(np.float64)
    print(f"[data] {a.dir}  N={len(Y)}  target={a.target}  y_dim={Y.shape[1]}  "
          f"|y| 均值={np.abs(Y).mean():.5f}")
    out = {}
    # 聚类留出：按 ref0（窗口起点的参考状态）聚类 ⇒ 同一段运动不会被拆到训练+验证两侧
    folds = None
    if a.split == "cluster" and "ref0" in d:
        from sklearn.cluster import KMeans
        R = d["ref0"].astype(np.float64)
        R = (R - R.mean(0)) / (R.std(0) + 1e-8)
        lab = KMeans(n_clusters=int(a.n_clusters), n_init=10,
                     random_state=0).fit_predict(R)
        folds = [(np.where(np.setdiff1d(np.arange(len(lab)), np.where(lab == c)[0]) == -1)[0],)
                 for c in range(int(a.n_clusters))]  # placeholder, replaced below
        folds = []
        for c in range(int(a.n_clusters)):
            te = np.where(lab == c)[0]
            tr = np.where(lab != c)[0]
            if len(te) >= 20 and len(tr) >= 200:
                folds.append((tr, te))
        print(f"[split] cluster 留出：{len(folds)} 折（每折留一整簇，簇大小 "
              f"{[len(t) for _, t in folds] }）")
    for kind in ("deploy", "old35", "state71", "token"):
        X = build_features(d, kind).astype(np.float64)
        r_cos = ridge_cos(X, Y)
        if folds is None:
            cs = [mlp_cos(X, Y, seed=s, epochs=a.epochs, device=a.device) for s in range(a.seeds)]
            m = float(np.mean([c for c, _ in cs]))
            se = float(np.std([c for c, _ in cs]) / max(len(cs) ** 0.5, 1))
            tag = f"seeds={len(cs)}"
        else:
            per = [mlp_cos_split(X, Y, tr, te, seed=0, epochs=a.epochs, device=a.device)
                   for tr, te in folds]
            m = float(np.mean(per))
            se = float(np.std(per) / max(len(per) ** 0.5, 1))
            tag = f"folds={len(per)} min={min(per):+.3f} max={max(per):+.3f}"
        out[kind] = (r_cos, m, se, X.shape[1])
        print(f"[cv] {kind:<8} dim={X.shape[1]:<4} ridge cos={r_cos:+.3f}   "
              f"MLP held-out cos={m:+.3f} ± {se:.3f}  ({tag})", flush=True)
    print("\n[判读] MLP held-out cos ≥0.25 ⇒ 能学（瓶颈=标签量）；≈0.10 ⇒ 学不动（换输入/换视界）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
