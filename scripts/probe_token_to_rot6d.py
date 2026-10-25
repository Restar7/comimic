#!/usr/bin/env python3
"""Probe: does ``g1_motion_token_64`` encode the reference root orientation?

Train a small MLP token(64) -> rot6d(root quat) on the HA clips and report the
held-out angular error.  If the error is small, a lightweight head can supply the
obs99 reference-orientation dims at deployment (no VLA retraining needed).

    python3 scripts/probe_token_to_rot6d.py --epochs 60
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import torch
import torch.nn as nn


def quat_to_mat(q: np.ndarray) -> np.ndarray:          # (..., 4) wxyz -> (..., 3, 3)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def mat_to_6d(R: np.ndarray) -> np.ndarray:            # first two columns
    return np.concatenate([R[..., 0], R[..., 1]], axis=-1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/npz/ha_tokens")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    print(f"[probe] clips={len(files)}")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(files))
    n_val = max(1, int(len(files) * a.val_frac))
    val_files = {files[i] for i in idx[:n_val]}

    def load(subset):
        X, Y = [], []
        for f in files:
            if (f in val_files) != subset:
                continue
            d = np.load(f)
            X.append(d["g1_motion_token_64"].astype(np.float32))
            Y.append(mat_to_6d(quat_to_mat(d["body_quat_w"][:, 0].astype(np.float32))))
        return torch.from_numpy(np.concatenate(X)), torch.from_numpy(np.concatenate(Y))

    xtr, ytr = load(False)
    xva, yva = load(True)
    print(f"[probe] train={tuple(xtr.shape)} val={tuple(xva.shape)} device={a.device}")

    net = nn.Sequential(nn.Linear(64, a.hidden), nn.SiLU(), nn.Linear(a.hidden, a.hidden), nn.SiLU(),
                        nn.Linear(a.hidden, 6)).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.MSELoss()

    def angle_err(pred, target):
        # orthonormalise 6D -> R -> geodesic angle
        def to_R(v6):
            a1, a2 = v6[:, :3], v6[:, 3:]
            b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
            b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            b3 = torch.cross(b1, b2, dim=-1)
            return torch.stack([b1, b2, b3], dim=-1)      # columns
        R1, R2 = to_R(pred), to_R(target)
        cos = ((R1.transpose(1, 2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))

    with torch.no_grad():
        base = angle_err(torch.zeros_like(yva).to(a.device), yva.to(a.device)).mean().item()
    print(f"[probe] 常数预测(0) 基线角度误差 = {base:.1f} deg")

    n = xtr.shape[0]
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            sel = perm[i:i + a.batch]
            xb, yb = xtr[sel].to(a.device), ytr[sel].to(a.device)
            pred = net(xb)
            loss = lossf(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(sel)
        if ep % 10 == 0 or ep == 1:
            with torch.no_grad():
                pr = net(xva.to(a.device))
                va = lossf(pr, yva.to(a.device)).item()
                ang = angle_err(pr, yva.to(a.device)).mean().item()
            print(f"[probe] ep {ep:>3}  train_mse {tot/n:.5f}  val_mse {va:.5f}  val_angle {ang:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
