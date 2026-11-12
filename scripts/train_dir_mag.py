#!/usr/bin/env python3
"""情况 A 的修复：**方向监督 + 独立幅度门控**（2026-11-07）。

依据（附录 A.7）：教师 δ=0.08 → 真机 0.688；学生原生 0.072 → **0.500（掷硬币）**，
压到 0.018~0.036 才 0.625；且学生 `D_i = Σ⟨∇J,Δ⟩` 的正比例 = 0.50。
⇒ 学生方向没学好、幅度又偏大。故：

* 输出参数化改成 `Δ_t = m(x)·unit(u_t)`，`m ∈ [0, 0.75·cap]`（soft-cap 可达余量）；
* 损失只用**方向**：`L = mean(1 − cos(Δ_t, unit g_t))`，不含任何幅值项；
* 幅度门控 `m` 只吃 `state64`（状态尺度信息），单独输出，便于部署时夹紧。
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "runs/labels_v3"))
    ap.add_argument("--out", type=Path, default=ROOT / "runs/dir_mag/model.pt")
    ap.add_argument("--cap", type=float, default=0.08)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--holdout-groups", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import numpy as np, torch
    from train_distill_v3 import state64_from_state71
    buf = {}
    for f in sorted(glob.glob(str(Path(a.dir) / "dump_*.npz"))):
        z = np.load(f)
        for k in ("token_chunk", "obs_deploy930", "g_tok_pert", "frame", "state"):
            buf.setdefault(k, []).append(z[k])
    D = {k: np.concatenate(v, 0) for k, v in buf.items()}
    N = len(D["frame"]); T = D["g_tok_pert"].shape[1]
    X = np.concatenate([D["token_chunk"].astype(np.float32).reshape(N, -1),
                        D["obs_deploy930"].astype(np.float32),
                        state64_from_state71(D["state"].astype(np.float32))], 1)
    s64 = X[:, -64:]
    G = torch.as_tensor(D["g_tok_pert"].astype(np.float32))
    nrm = G.norm(dim=-1)
    valid = nrm > 1e-9
    Yu = G / nrm.clamp_min(1e-12)[..., None]
    fr = D["frame"].astype(np.float64)
    edges = np.quantile(fr, np.linspace(0, 1, 11))
    grp = np.clip(np.searchsorted(edges[1:-1], fr), 0, 9)
    te = np.where(grp < a.holdout_groups)[0]; tr = np.where(grp >= a.holdout_groups)[0]
    print(f"[data] N={N} T={T} dim={X.shape[1]}  train={len(tr)} val={len(te)}  "
          f"有效帧/样本={float(valid.sum(1).float().mean()):.1f}")
    dev = torch.device(a.device)
    Xt = torch.as_tensor(X, device=dev)
    mu = Xt[tr].mean(0, keepdim=True); sd = Xt[tr].std(0, keepdim=True).clamp_min(1e-8)
    Xn = (Xt - mu) / sd
    Yt = Yu.to(dev); Vt = valid.to(dev)
    mag_cap = 0.75 * float(a.cap)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], a.hidden), torch.nn.SiLU(),
                torch.nn.Linear(a.hidden, a.hidden), torch.nn.SiLU(),
                torch.nn.Linear(a.hidden, T * 64))
            self.mag = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.SiLU(),
                                           torch.nn.Linear(64, 1))
        def forward(self, x, s):
            u = self.trunk(x).reshape(-1, T, 64)
            uh = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            m = mag_cap * torch.sigmoid(self.mag(s)).squeeze(-1)      # (B,)
            return uh * m.view(-1, 1, 1), m
    net = Net().to(dev)
    print(f"[net] params={sum(p.numel() for p in net.parameters())}  mag_cap={mag_cap}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(a.epochs))
    itr = torch.tensor(tr, device=dev); ite = torch.tensor(te, device=dev)

    def ev(idx):
        net.eval()
        with torch.no_grad():
            d, m = net(Xn[idx], Xn[idx][:, -64:])
            c = (d * Yt[idx]).sum(-1) / (d.norm(dim=-1).clamp_min(1e-9)
                                         * Yt[idx].norm(dim=-1).clamp_min(1e-9))
            v = Vt[idx]
            return float((c * v).sum() / v.sum().clamp_min(1)), float(d.norm(dim=-1)[v].mean()), float(m.mean())
    best, best_sd = -1.0, None
    for ep in range(int(a.epochs)):
        net.train()
        perm = itr[torch.randperm(len(itr), device=dev)]
        for i in range(0, len(perm), int(a.batch)):
            j = perm[i:i + int(a.batch)]
            d, m = net(Xn[j], Xn[j][:, -64:])
            c = (d * Yt[j]).sum(-1) / (d.norm(dim=-1).clamp_min(1e-9)
                                       * Yt[j].norm(dim=-1).clamp_min(1e-9))
            v = Vt[j].float()
            loss = ((1.0 - c) * v).sum() / v.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        sched.step()
        if ep % 4 == 3 or ep == int(a.epochs) - 1:
            ce, de, me = ev(ite)
            print(f"[ep {ep:3d}] loss={float(loss):.4f} held-out cos={ce:+.3f} "
                  f"‖Δ‖={de:.4f} mag={me:.4f}", flush=True)
            if ce > best:
                best = ce; best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_sd)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "meta": {"arch": "dir_mag", "T": int(T), "cap": float(a.cap),
                "mag_cap": mag_cap, "hidden": a.hidden, "input_dim": int(X.shape[1]),
                "mu": mu.cpu(), "sd": sd.cpu(), "heldout_cos": float(best),
                "feature": "token_chunk_all+obs930+state64"}}, a.out)
    print(f"[done] best held-out cos={best:+.3f}  → {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
