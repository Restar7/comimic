#!/usr/bin/env python3
"""**同一口径下的标签可学性对比**（2026-11-07）。

为什么需要：`docs/DELTA_TOKEN_TRAINING_VERDICT.md` §3.1 里
`∂r₀/∂token`(k=1) 0.472/0.500 与 `∂J₂₀/∂token`(k=20) 0.232/0.334 这两个数
**口径可能不一致**：
* k=1 的标签只有 **第 0 帧**非零（其余 39 帧精确为 0，因为 r₀ 不受未来 token 影响）；
* k=20 的标签只有**前 20 帧**非零（t≥20 的 token 影响不到 20 步回报）。

如果余弦是把 40 帧一起平均、而零标签帧贡献 cos=0，那两边就被不同的稀释比例污染，
"k=1 明显更好学"这个结论就站不住。本脚本用**同一套特征、同一套留出、同一套 mask**
把两者放到同一把尺子上。

口径
----
* 特征：`token_chunk` 全 40 帧 (2560) + `obs_deploy930` (930) + `state64` (64) = 3554 维
* 目标：`η·unit(g)`，逐帧（40 帧 × 64）
* 模型：闭式岭回归（每帧一个 64 维输出，共用同一份 Gram 矩阵）
* 留出：按 `frame` 分 10 段，留出前 `--holdout-groups` 段（与两个 trainer 一致）
* 报告：`cos_all`（40 帧平均，= 旧口径）与 `cos_info`（只在**非零标签帧**上平均）

用法::

    python3 scripts/probe_label_learnability.py --dir runs/labels_v3 --out runs/label_learnability.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKEN_DIM = 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "runs/labels_v3"))
    ap.add_argument("--holdout-groups", type=int, default=2)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--max-n", type=int, default=0, help="0=全部")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/label_learnability.json")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_distill_v3 import state64_from_state71  # noqa: PLC0415

    d_dir = Path(a.dir)
    files = sorted(glob.glob(str(d_dir / "dump_*.npz")))
    if not files:
        raise SystemExit(f"没有 dump: {d_dir}")
    buf: dict[str, list] = {}
    for f in files:
        z = np.load(f)
        for k in ("token_chunk", "obs_deploy930", "g_tok_pert", "g_tok1_pert", "frame", "state"):
            buf.setdefault(k, []).append(z[k])
    D = {k: np.concatenate(v, 0) for k, v in buf.items()}
    N = len(D["frame"])
    if a.max_n and N > a.max_n:
        sel = np.random.default_rng(0).choice(N, a.max_n, replace=False)
        D = {k: v[sel] for k, v in D.items()}
        N = len(D["frame"])
    G20 = D["g_tok_pert"].astype(np.float32)       # (N,40,64)
    G1 = D["g_tok1_pert"].astype(np.float32)
    T = G20.shape[1]
    print(f"[data] {len(files)} dumps → N={N}  T={T}")

    X = np.concatenate([
        D["token_chunk"].astype(np.float32).reshape(N, -1),
        D["obs_deploy930"].astype(np.float32),
        state64_from_state71(D["state"].astype(np.float32)),
    ], 1)
    print(f"[feat] X={X.shape}")

    fr = D["frame"].astype(np.float64)
    edges = np.quantile(fr, np.linspace(0, 1, 11))
    grp = np.clip(np.searchsorted(edges[1:-1], fr), 0, 9)
    te = np.where(grp < a.holdout_groups)[0]
    tr = np.where(grp >= a.holdout_groups)[0]
    print(f"[split] train={len(tr)} val={len(te)}")

    dev = torch.device(a.device)
    Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
    mu = Xt[tr].mean(0, keepdim=True)
    sd = Xt[tr].std(0, keepdim=True).clamp_min(1e-8)
    Xn = (Xt - mu) / sd
    Xtr = Xn[tr]
    n_tr, dim = Xtr.shape
    # 闭式岭：W = (XᵀX + λI)⁻¹ Xᵀ Y
    G = Xtr.T @ Xtr
    G += float(a.ridge) * max(1.0, float(n_tr)) * torch.eye(dim, device=dev)
    Xte = Xn[te]

    rows = []
    for name, Gt in (("k20", G20), ("k1", G1)):
        Y = torch.as_tensor(Gt, device=dev, dtype=torch.float32)     # (N,T,64)
        nrm = Y.norm(dim=-1)                                        # (N,T)
        info = nrm > 1e-9                                           # 非零标签帧
        Yu = Y / nrm.clamp_min(1e-12)[..., None]                     # unit
        Ytr_flat = Yu[tr].reshape(len(tr), -1)                       # (n_tr, T*64)
        W = torch.linalg.solve(G, Xtr.T @ Ytr_flat)                  # (dim, T*64)
        P = (Xte @ W).reshape(len(te), T, TOKEN_DIM)
        c = (P * Yu[te]).sum(-1) / (P.norm(dim=-1).clamp_min(1e-9)
                                    * Yu[te].norm(dim=-1).clamp_min(1e-9))
        m_all = float(c.mean())
        m_info = float(c[info[te]].mean())
        # 真·零标签帧上模型输出多大（越小越好；说明它学到了"这里不用改"）
        zero = ~info[te]
        out_zero = float(P[zero].norm(dim=-1).mean()) if zero.any() else float("nan")
        rows.append({
            "target": name,
            "n_inform_frames_per_sample": int(info.sum(1).float().mean()),
            "cos_all": m_all,
            "cos_info": m_info,
            "out_norm_on_zero_frames": out_zero,
        })
        print(f"[{name:>4}] 每样本有效帧 {info.sum(1).float().mean():.2f}  "
              f"cos_all={m_all:+.4f}  cos_info={m_info:+.4f}  "
              f"零帧输出范数={out_zero:.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "n": int(N), "dim": int(dim),
                                 "holdout_groups": int(a.holdout_groups)}, indent=1))
    print(f"[write] {a.out}")
    print("\n判读：旧结论'k=1 明显更好学'只有在 cos_info 上比较才成立。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
