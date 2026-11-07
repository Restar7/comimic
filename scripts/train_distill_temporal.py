#!/usr/bin/env python3
"""Temporal Δ Actor 的**离线蒸馏训练 + 强 baseline 对照**（借鉴 v5 §10 的纪律，但**不含** Critic/BPTT）。

五个臂，同一批标签、同一留出（按 `frame` 分段）：

| arm | 模型 | 输入 | 目的 |
|---|---|---|---|
| B1 | 共享常数 (1,1,64) | — | 最基础对照（"一个方向给所有情况"） |
| B2 | 逐帧位置模板 c_t (1,T,64) | 位置 t | **核心防伪对照**：只学"第几帧该修多少" |
| B3 | 扁平 MLP | token0 + state64 → 一个 64 维（广播全帧） | 上一轮最好（held-out cos 0.459） |
| **B4** | **TemporalDeltaActor** | 整段 chunk + 10×58 历史 + state64 + PE16 → 逐帧 | **本次主模型** |
| B5 | B4 + **history 打乱** | 同上但历史错位 | 判"历史是否真被用上" |

判据：**B4 > B2**（不是只记位置）且 **B4 > B3**（比扁平强）；B5 明显掉 ⇒ 历史真有用。

用法
----
    python scripts/train_distill_temporal.py --dir runs/labels_v3 \
        --arms B1,B2,B3,B4,B5 --epochs 30 --holdout-groups 2
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diff_vla_wbc.delta_token_temporal import TemporalDeltaActor  # noqa: E402

TOKEN_DIM = 64
K_EXEC = 20          # 只有前 20 帧会被执行（HA_VLA_EXEC_STEPS=20 / BPTT H=20）
HIST_BLOCKS = (("ang_vel", 3), ("jpos", 29), ("jvel", 29), ("last_action", 29), ("grav", 3))


# --------------------------------------------------------------------------- 数据
def quat_to_rot6(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-9
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (x * z + w * y), 2 * (y * z - w * x)], 1).astype(np.float32)


def state64_from_state71(state: np.ndarray) -> np.ndarray:
    import joint_contract as JC
    perm = np.asarray(JC.permutation("standard", "ha_v31"), dtype=np.int64)
    q, qd = state[:, 7:36], state[:, 42:71]
    return np.concatenate([quat_to_rot6(state[:, 3:7]), q[:, perm], qd[:, perm]], 1).astype(np.float32)


def hist58_from_930(obs930: np.ndarray) -> np.ndarray:
    """930 → (N,10,58) = [q(29) | qd(29)]，帧序 oldest→newest。"""
    off, blocks = 0, {}
    for name, d in HIST_BLOCKS:
        blocks[name] = obs930[:, off:off + 10 * d].reshape(len(obs930), 10, d)
        off += 10 * d
    return np.concatenate([blocks["jpos"], blocks["jvel"]], -1).astype(np.float32)


def load(d: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(Path(d) / "dump_*.npz")))
    if not files:
        raise SystemExit(f"没有 dump: {d}")
    keys = ("token_chunk", "obs_deploy930", "state", "g_tok_pert", "frame")
    buf: dict[str, list] = {}
    for f in files:
        z = np.load(f)
        for k in keys:
            if k in z.files:
                buf.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v, 0) for k, v in buf.items()}
    print(f"[data] {len(files)} 文件  N={len(out['token_chunk'])}")
    return out


def split_by_frame(fr: np.ndarray, n_hold: int, seed: int = 0):
    edges = np.quantile(fr, np.linspace(0, 1, 11))
    grp = np.clip(np.searchsorted(edges[1:-1], fr), 0, 9)
    te = np.where(grp < n_hold)[0]
    tr = np.where(grp >= n_hold)[0]
    return tr, te


# --------------------------------------------------------------------------- 各臂
class SharedConst(torch.nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.v = torch.nn.Parameter(torch.zeros(1, 1, TOKEN_DIM))

    def forward(self, z, hist, s64, mask=None):
        d = self.v.expand(z.shape[0], z.shape[1], TOKEN_DIM)
        return d * mask.unsqueeze(-1) if mask is not None else d


class PerFrameTemplate(torch.nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.v = torch.nn.Parameter(torch.zeros(1, T, TOKEN_DIM))

    def forward(self, z, hist, s64, mask=None):
        d = self.v.expand(z.shape[0], -1, TOKEN_DIM)
        return d * mask.unsqueeze(-1) if mask is not None else d


class FlatMLP(torch.nn.Module):
    """token0 + state64 → 一个 64 维（广播到全帧）；与部署端当前实现同构。"""

    def __init__(self, hidden=1024, init_std=4e-4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(TOKEN_DIM + 64, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, TOKEN_DIM))
        with torch.no_grad():
            self.net[-1].weight.normal_(0.0, init_std)
            self.net[-1].bias.zero_()

    def forward(self, z, hist, s64, mask=None):
        out = self.net(torch.cat([z[:, 0], s64], -1))         # (B,64)
        d = out.unsqueeze(1).expand(-1, z.shape[1], -1)
        return d * mask.unsqueeze(-1) if mask is not None else d


def build_arm(name: str, T: int) -> torch.nn.Module:
    if name == "B1":
        return SharedConst(T)
    if name == "B2":
        return PerFrameTemplate(T)
    if name == "B3":
        return FlatMLP()
    if name in ("B4", "B5"):
        return TemporalDeltaActor(cap=0.08)
    raise SystemExit(f"未知 arm {name}")


def cos_masked(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    p = pred * mask.unsqueeze(-1)
    t = y * mask.unsqueeze(-1)
    c = (p * t).sum(-1) / (p.norm(dim=-1).clamp_min(1e-9) * t.norm(dim=-1).clamp_min(1e-9))
    return float((c * mask).sum() / mask.sum().clamp_min(1e-9))


def run_arm(name, X, Y, M, tr, te, *, T, epochs, batch, lr, device, shuffle_hist=False):
    net = build_arm(name, T).to(device)
    z, hist, s64 = X["z"], X["hist"], X["s64"]
    if shuffle_hist and name in ("B4", "B5"):
        g = torch.Generator().manual_seed(12345)
        hist = hist[torch.randperm(len(hist), generator=g)]
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
    itr = torch.tensor(tr, device=device)
    ite = torch.tensor(te, device=device)
    best = -1.0
    for ep in range(int(epochs)):
        net.train()
        idx = itr[torch.randperm(len(itr), device=device)]
        for i in range(0, len(idx), int(batch)):
            j = idx[i:i + int(batch)]
            opt.zero_grad(set_to_none=True)
            pred = net(z[j], hist[j], s64[j], mask=M[j])
            loss = (((pred - Y[j]) * M[j].unsqueeze(-1)) ** 2).sum(-1).sum() / M[j].sum().clamp_min(1e-6)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 5 == 4 or ep == int(epochs) - 1:
            net.eval()
            with torch.no_grad():
                c_te = cos_masked(net(z[ite], hist[ite], s64[ite], mask=M[ite]), Y[ite], M[ite])
            best = max(best, c_te)
            print(f"   [{name}] ep {ep:3d} held-out cos={c_te:+.3f}", flush=True)
    return net, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/labels_v3")
    ap.add_argument("--arms", default="B1,B2,B3,B4,B5")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--holdout-groups", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="runs/distill_temporal")
    a = ap.parse_args()

    d = load(a.dir)
    z_np = d["token_chunk"].astype(np.float32)
    N, T, _ = z_np.shape
    G = d["g_tok_pert"].astype(np.float32)
    Gn = G / (np.linalg.norm(G, axis=-1, keepdims=True) + 1e-9)
    Y_np = (Gn * a.eta).astype(np.float32)
    mask_np = np.zeros((N, T), np.float32)
    mask_np[:, :min(K_EXEC, T)] = 1.0

    dev = torch.device(a.device)
    X = {"z": torch.tensor(z_np, device=dev),
         "hist": torch.tensor(hist58_from_930(d["obs_deploy930"]), device=dev),
         "s64": torch.tensor(state64_from_state71(d["state"].astype(np.float32)), device=dev)}
    Y = torch.tensor(Y_np, device=dev)
    M = torch.tensor(mask_np, device=dev)
    tr, te = split_by_frame(d["frame"].astype(np.float64), a.holdout_groups)
    print(f"[split] 按 frame 留出前 {a.holdout_groups}/10 段  train={len(tr)} val={len(te)}  "
          f"T={T} 只监督前 {min(K_EXEC, T)} 帧")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    nets = {}
    for name in [s for s in a.arms.split(",") if s]:
        print(f"\n=== {name} ===", flush=True)
        shuf = name == "B5"
        net, best = run_arm(name, X, Y, M, tr, te, T=T, epochs=a.epochs, batch=a.batch,
                            lr=a.lr, device=dev, shuffle_hist=shuf)
        results[name] = best
        nets[name] = net
    print("\n=== 汇总（同一批标签 / 同一留出）===")
    for k, v in results.items():
        print(f"  {k}: held-out cos = {v:+.3f}")
    (out / "arms.json").write_text(json.dumps(results, indent=2))

    if "B4" in nets:
        sd = {f"net.{i}.{p}": v.detach().cpu() for i, v in enumerate([])}  # noqa: F841
        torch.save({"state_dict": nets["B4"].state_dict(),
                    "meta": {"arch": "temporal_v5_style", "schema_version": 2,
                             "cap": 0.08, "T": int(T), "k_exec": int(min(K_EXEC, T)),
                             "heldout_cos": float(results.get("B4", float("nan"))),
                             "target": "g_tok_pert", "eta": float(a.eta),
                             "note": "需要 provider 侧执行（要 10 帧 q/qd 历史）"}},
                   out / "temporal_B4.pt")
        print(f"[ok] 写出 {out / 'temporal_B4.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
