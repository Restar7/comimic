#!/usr/bin/env python3
"""harness02：**如果学生完美学会了旧 Teacher**，它在正确目标上能得多少分？

做法：用旧标签 ``g_old``（token0 重复 + 未折扣 sum）训练 ridge / MLP，
再分别在**同一 held-out 窗口**上对 ``g_old``（学生自己的目标）与 ``g_new``（正确目标）算 cosine。
差值 = 旧 Teacher 的**口径误差**，与学生容量/样本量无关。

用法：``python3 scripts/harness02_old_label_transfer.py --dump runs/harness_02_teacher/dump_h20``
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from harness02_distill_learnability import (  # noqa: E402
    MLP, cosine_rows, fit_torch, grouped_split, load_dump, subsample, _unit,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ROOT / "runs/harness_02_teacher/dump_h20"))
    ap.add_argument("--out", default=str(ROOT / "runs/harness_02_teacher"))
    ap.add_argument("--eta", type=float, default=0.08)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", default="256")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    d = load_dump(Path(a.dump))
    state = d["state"].astype(np.float64); feats = d["feats"].astype(np.float64)
    tok0 = d["token_chunk"].astype(np.float64)[:, 0]
    X = np.concatenate([tok0, feats], 1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    U_new, ok_new = _unit(np.asarray(d["g"], np.float64))
    U_old, ok_old = _unit(np.asarray(d["g_old"], np.float64))
    groups = d["clip_ids"]
    tr_all, va, te = grouped_split(groups, seed=0)
    ok = ok_new & ok_old
    tr_all, va, te = tr_all[ok[tr_all]], va[ok[va]], te[ok[te]]
    rows = []

    def ridge_fit(Xa, Ya, lam):
        A = np.concatenate([Xa, np.ones((len(Xa), 1))], 1)
        R = np.eye(A.shape[1]) * lam; R[-1, -1] = 0.0
        return np.linalg.solve(A.T @ A + R, A.T @ Ya)

    def ridge_pred(W, Xa):
        return np.concatenate([Xa, np.ones((len(Xa), 1))], 1) @ W

    for n in [int(s) for s in a.sizes.split(",") if s.strip()]:
        tr = subsample(tr_all, n, seed=1234 + n)
        Ytr_old = a.eta * U_old[tr]
        # ridge（λ 用训练集内 5 折 CV 选，指标=对旧标签的 cos）
        best = (-9e9, None)
        order = np.arange(len(tr)); np.random.default_rng(7).shuffle(order)
        for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
            cs = []
            for k in range(5):
                va_k = order[k::5]; tr_k = np.setdiff1d(order, va_k)
                W = ridge_fit(X[tr[tr_k]], Ytr_old[tr_k], lam)
                cs.append(cosine_rows(ridge_pred(W, X[tr[va_k]]), U_old[tr[va_k]]).mean())
            if np.mean(cs) > best[0]:
                best = (float(np.mean(cs)), lam)
        W = ridge_fit(X[tr], Ytr_old, best[1])
        for split_name, idx in (("val", va), ("test", te)):
            p = ridge_pred(W, X[idx])
            rows.append({"n_train": len(tr), "model": "ridge", "split": split_name,
                         "cos_vs_old": float(cosine_rows(p, U_old[idx]).mean()),
                         "cos_vs_new": float(cosine_rows(p, U_new[idx]).mean())})
        # MLP
        preds = {"val": [], "test": []}
        for seed in range(a.seeds):
            m = MLP(X.shape[1], 256, 2, 64, 1e-2)
            (pva, pte), _ = fit_torch(m, X[tr], Ytr_old, [X[va], X[te]],
                                           epochs=a.epochs, lr=1e-3, seed=seed, device=a.device)
            preds["val"].append(pva); preds["test"].append(pte)
        for split_name, idx in (("val", va), ("test", te)):
            p = np.mean(preds[split_name], 0)
            rows.append({"n_train": len(tr), "model": "mlp2x256", "split": split_name,
                         "cos_vs_old": float(cosine_rows(p, U_old[idx]).mean()),
                         "cos_vs_new": float(cosine_rows(p, U_new[idx]).mean())})
        for r in rows:
            if r["n_train"] == len(tr):
                print(f"[old-label student] n={r['n_train']} {r['model']:<9s} {r['split']:<4s} "
                      f"cos_vs_old={r['cos_vs_old']:.4f}  cos_vs_new={r['cos_vs_new']:.4f}", flush=True)
    path = Path(a.out) / "old_label_transfer.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("[done]", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
