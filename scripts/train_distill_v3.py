#!/usr/bin/env python3
"""**离线蒸馏训练 v3**：读 `runs/labels_v3` 的标签，训 Δtoken（部署同结构），全程不碰 WM/VLA/闭环。

与旧 `train_distill_offline.py` 的区别（对应 2026-11-07 的口径修正）
------------------------------------------------------------------
| 项 | 旧 | 新（本脚本） |
|---|---|---|
| 输入 token | `token`（数据集录制 token） | **`token_chunk`（VLA 本次前向的整段 chunk）** |
| 历史 | `state` 裸值 | **`obs_deploy930`（部署布局，HA_V31 + provider 块顺序）** |
| 标签 | `g_tokH`（对 T 取平均） | **`g_tok_pert` 逐帧 (T,64)**（或 `g_tok1_pert`） |
| rot6d 顺序 | 列优先（错） | 不需要：直接用部署 930 里的那份 |
| 切分 | ref0 聚类 | **按 `frame` 分段留出**（同一片段不跨训练/验证） |
| metadata | legacy | **schema v2**（含 feat 版本/teacher 语义） |

用法
----
    python scripts/train_distill_v3.py --dir runs/labels_v3 --out runs/distill_v3.pt \
        --use-chunk all --use-history flat930 --target k20 --per-frame \
        --eta 0.15 --holdout-groups 2 --epochs 40
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

TOKEN_DIM = 64
HIST_DIM = 930


def load(d: str, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(Path(d) / "dump_*.npz")))
    if not files:
        raise SystemExit(f"没有 dump: {d}")
    buf: dict[str, list] = {}
    skip: dict[str, int] = {}
    for f in files:
        z = np.load(f)
        for k in keys:
            if k in z.files:
                buf.setdefault(k, []).append(z[k])
            else:
                skip[k] = skip.get(k, 0) + 1
    out = {k: np.concatenate(v, 0) for k, v in buf.items()}
    print(f"[data] {len(files)} 文件  N={len(out['token_chunk'])}  "
          f"缺失字段={ {k: v for k, v in skip.items()} or '无'}")
    return out


def _quat_to_rot6(q: np.ndarray) -> np.ndarray:
    """wxyz → 线上口径 ``[m00,m01,m10,m11,m20,m21]``。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-9
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (x * z + w * y), 2 * (y * z - w * x)], 1).astype(np.float32)


def state64_from_state71(state: np.ndarray) -> np.ndarray:
    """dump 的 71 维（standard 序）→ 部署 state64（rot6d + dof_pos + dof_vel，ha_v31 序）。"""
    import joint_contract as JC
    perm = np.asarray(JC.permutation("standard", "ha_v31"), dtype=np.int64)
    q, qd = state[:, 7:36], state[:, 42:71]
    return np.concatenate([_quat_to_rot6(state[:, 3:7]), q[:, perm], qd[:, perm]], 1).astype(np.float32)


def build_input(d: dict, use_chunk: str, use_history: str,
                use_state64: bool = False) -> tuple[np.ndarray, int]:
    """返回 (X, token_part_dim)。``use_chunk``: first|all|prefixN；``use_history``: none|last|flat930。"""
    tok = d["token_chunk"].astype(np.float32)          # (N,T,64)
    T = tok.shape[1]
    if use_chunk == "first":
        x_tok, dim_tok = tok[:, 0], TOKEN_DIM
    elif use_chunk == "all":
        x_tok, dim_tok = tok.reshape(len(tok), -1), T * TOKEN_DIM
    elif use_chunk.startswith("prefix"):
        k = int(use_chunk[len("prefix"):])
        x_tok, dim_tok = tok[:, :k].reshape(len(tok), -1), k * TOKEN_DIM
    else:
        raise SystemExit(f"未知 --use-chunk {use_chunk}")
    parts, dim = [x_tok], dim_tok
    if use_history == "flat930":
        h = d["obs_deploy930"].astype(np.float32)
        parts.append(h)
        dim += HIST_DIM
    elif use_history == "last":
        h = d["obs_deploy930"].astype(np.float32).reshape(len(tok), 10, -1)[:, -1]
        parts.append(h)
        dim += h.shape[1]
    elif use_history != "none":
        raise SystemExit(f"未知 --use-history {use_history}")
    if use_state64:
        s64 = state64_from_state71(d["state"].astype(np.float32))
        parts.append(s64)
        dim += s64.shape[1]
    return np.concatenate(parts, 1), dim_tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/labels_v3")
    ap.add_argument("--out", default="runs/distill_v3.pt")
    ap.add_argument("--template", default="runs/gt7fix/vla_final.pt")
    ap.add_argument("--target", default="k20", choices=("k20", "k1"))
    ap.add_argument("--per-frame", action="store_true", help="输出逐帧 Δ(T,64)；否则共享 Δ(64,)")
    ap.add_argument("--use-chunk", default="all")
    ap.add_argument("--use-history", default="flat930")
    ap.add_argument("--use-state64", action="store_true",
                    help="把 dump 的 71 维 state 转成**部署 state64**（ha_v31 关节序）拼进输入；"
                         "这条是**服务端今天就能跑**的口径")
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--holdout-groups", type=int, default=2, help="按 frame 分 10 段，留出前 N 段")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    keys = ("token_chunk", "obs_deploy930", "g_tok_pert", "g_tok1_pert", "J20", "frame", "state")
    d = load(a.dir, keys)
    X, dim_tok = build_input(d, a.use_chunk, a.use_history, a.use_state64)
    G = d["g_tok_pert" if a.target == "k20" else "g_tok1_pert"].astype(np.float32)  # (N,T,64)
    N, T, _ = G.shape
    print(f"[feat] X={X.shape}（token 部分 {dim_tok}）  标签={a.target} {G.shape}")

    # 目标：逐帧 η·unit(g)；共享则先对 T 取平均再 unit
    Gn = G / (np.linalg.norm(G, axis=-1, keepdims=True) + 1e-9)
    if a.per_frame:
        Y = (Gn * a.eta).reshape(N, -1)                 # (N, T*64)
        out_dim = T * TOKEN_DIM
    else:
        gmean = G.mean(1)
        Y = (gmean / (np.linalg.norm(gmean, axis=-1, keepdims=True) + 1e-9)) * a.eta
        out_dim = TOKEN_DIM
    print(f"[target] {'逐帧' if a.per_frame else '共享'}  out_dim={out_dim}  "
          f"‖y‖ 均值={np.linalg.norm(Y.reshape(len(Y), -1, 64), axis=-1).mean():.4f}")

    # 按 frame 分段留出（同一片段不跨两侧）
    fr = d["frame"].astype(np.float64)
    edges = np.quantile(fr, np.linspace(0, 1, 11))
    grp = np.clip(np.searchsorted(edges[1:-1], fr), 0, 9)
    te = np.where(grp < a.holdout_groups)[0]
    tr = np.where(grp >= a.holdout_groups)[0]
    print(f"[split] 按 frame 10 段，留出第 0~{a.holdout_groups - 1} 段  train={len(tr)}  val={len(te)}")

    dev = torch.device(a.device)
    layers, din = [], X.shape[1]
    for _ in range(max(1, a.depth)):
        layers += [torch.nn.Linear(din, a.hidden), torch.nn.SiLU()]
        din = a.hidden
    layers.append(torch.nn.Linear(din, out_dim))
    net = torch.nn.Sequential(*layers).to(dev)
    with torch.no_grad():                       # 末层极小初始化（≈零，保证第 0 步≈底模）
        net[-1].weight.normal_(0.0, 4e-4)
        net[-1].bias.zero_()
    print(f"[net] params={sum(p.numel() for p in net.parameters())}")

    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(a.epochs))
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Xn = torch.tensor((X - Xm) / Xs, device=dev)
    Yt = torch.tensor(Y, device=dev)
    itr = torch.tensor(tr, device=dev)
    ite = torch.tensor(te, device=dev)

    def cos_of(idx: torch.Tensor) -> float:
        with torch.no_grad():
            p = net(Xn[idx])
            y = Yt[idx]
            p = p.reshape(len(idx), -1, 64)
            y = y.reshape(len(idx), -1, 64)
            c = (p * y).sum(-1) / (p.norm(dim=-1).clamp_min(1e-9) * y.norm(dim=-1).clamp_min(1e-9))
            return float(c.mean())

    best, best_state = -1.0, None
    for ep in range(int(a.epochs)):
        net.train()
        idx = itr[torch.randperm(len(itr), device=dev)]
        tot = 0.0
        for i in range(0, len(idx), int(a.batch)):
            j = idx[i:i + int(a.batch)]
            opt.zero_grad(set_to_none=True)
            p = net(Xn[j]).reshape(len(j), -1, 64)
            y = Yt[j].reshape(len(j), -1, 64)
            loss = ((p - y) ** 2).sum(-1).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        sched.step()
        if ep % 5 == 4 or ep == int(a.epochs) - 1:
            c_te, c_tr = cos_of(ite), cos_of(itr[:4096])
            print(f"[ep {ep:3d}] loss={tot / len(idx):.5f}  held-out cos={c_te:+.3f}  train cos={c_tr:+.3f}",
                  flush=True)
            if c_te > best:
                best = c_te
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    print(f"[done] best held-out cos = {best:+.3f}")

    # ---- 存成 head ckpt（沿用模板，保证 server/probe 能直接吃）----
    tmpl = Path(a.template)
    ck = torch.load(tmpl, map_location="cpu", weights_only=False) if tmpl.exists() else {}
    # 我们的净是"扁平 MLP"，直接以 delta_token_state 的两层命名落盘（与 DeltaTokenNet 同键名）
    # ★ 键名加 `net.` 前缀：`delta_token_io` / `DeltaTokenNet` 的约定是 `net.{0,2,4}.weight`
    ck["delta_token_state"] = {f"net.{k}": v for k, v in best_state.items()}
    ck["delta_token_meta"] = {
        "schema_version": 2, "arch": "mlp_v3",
        "input_dim": int(X.shape[1]), "out_dim": int(out_dim), "chunk_frames": int(T),
        "hidden": int(a.hidden), "depth": int(a.depth),
        "init_std": 4e-4, "cap": 0.0,
        "feat_dim": int(dim_tok), "token_dim": TOKEN_DIM,
        "input_features": f"token_chunk[{a.use_chunk}]+hist[{a.use_history}]",
        "input_features_version": "deploy_state64_v1",
        "feat_joint_order": "standard", "apply": "broadcast_shared_over_chunk",
        "norm_cap": "apply_norm_cap(tanh)",
        "teacher_objective": "shared_delta_discounted_v1", "teacher_reward": "discounted",
        "teacher_horizon": 20, "teacher_eta": float(a.eta), "teacher_n": 0,
        "grad_target_weight": 1.0, "trained_steps": int(a.epochs),
        "trained": "offline-distill-v3", "target": a.target,
        "per_frame": bool(a.per_frame), "use_chunk": a.use_chunk, "use_history": a.use_history,
        "heldout_cos": float(best), "n_labels": int(N),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    (out.with_suffix(".json")).write_text(json.dumps(
        {k: v for k, v in ck["delta_token_meta"].items()}, ensure_ascii=False, indent=2))
    print(f"[ok] 写出 {out}（+ .json 摘要）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
