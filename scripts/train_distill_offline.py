#!/usr/bin/env python3
"""**离线蒸馏训练**：用预生成的 ĝ 标签训 Δtoken，训练阶段完全不碰 WM / VLA / 闭环。

为什么要有它
------------
在线（每步现算 ĝ）时，一次 20 步反传 ~0.4 s/窗口 ⇒ 每步只能给 8 个标签 ⇒ 1.2M 参数的
学生严重欠训练（实测 held-out cos 只有 0.09）。而标签一旦缓存，训练就是普通监督学习
（0.02 s/步），可以多 epoch、大 batch、按参考簇做留出验收。

实测天花板（`scripts/probe_distill_ceiling.py`，3,072 条标签）：
    随机切分       cos 0.360
    跨参考簇留出   cos 0.256   ← 真正的泛化口径

用法
----
    python scripts/train_distill_offline.py --dir runs/gen30k_dump \
        --out runs/distill30k.pt --epochs 40 --holdout-clusters 2
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from diff_vla_wbc.delta_token import DeltaTokenNet  # noqa: E402


def quat_to_rot6(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-9
    w, x, y, z = w / n, x / n, y / n, z / n
    # ★ 2026-11-07 修正：元素顺序必须与**线上**一致（`state_extra.mat_to_rot6d_wire` /
    #   HA `quat_to_rotation_6d`）：``[m00, m01, m10, m11, m20, m21]``，即 R[:, :2] 行优先。
    #   之前写成了列优先 ``[m00, m10, m20, m01, m11, m21]`` ⇒ 训练/部署特征错位（静默）。
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (x * z + w * y), 2 * (y * z - w * x)], 1).astype(np.float32)


def load(d: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(Path(d) / "dump_*.npz")))
    if not files:
        raise SystemExit(f"没有 dump: {d}（{len(files)} 个文件）")
    # ★ 只取需要的键：不同批次 dump 的 `feats` 宽度不同（35 vs 64），不能无脑 concat
    keep = {"state", "token", "ref0", "g_tokH", "g_tok1"}
    buf: dict[str, list] = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k in keep:
                buf.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v, 0) for k, v in buf.items()}
    print(f"[data] {len(files)} 个文件  N={len(out['token'])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/gen30k_dump")
    ap.add_argument("--target", default="g_tokH")
    ap.add_argument("--out", default="runs/distill30k.pt")
    ap.add_argument("--template", default="runs/gt7fix/vla_final.pt",
                    help="用于拼一个格式正确的 head ckpt（LoRA + 我们的 delta_token_state）")
    ap.add_argument("--eta", type=float, default=0.08, help="目标幅度 ‖Δtoken‖（部署端直接用）")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--holdout-clusters", type=int, default=2)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = load(a.dir)
    st = d["state"].astype(np.float32)
    tok = d["token"].astype(np.float32)
    Y = d[a.target].astype(np.float32)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    X = np.concatenate([tok, quat_to_rot6(st[:, 3:7]), st[:, 7:36], st[:, 42:71]], 1)  # (N,128)
    print(f"[feat] X={X.shape} Y={Y.shape} |Y|均值={np.abs(Y).mean():.5f}")

    # ---- 按参考簇留出（跨运动泛化口径）----
    from sklearn.cluster import KMeans
    R = d["ref0"].astype(np.float64) if "ref0" in d else st.astype(np.float64)
    R = (R - R.mean(0)) / (R.std(0) + 1e-8)
    lab = KMeans(n_clusters=int(a.n_clusters), n_init=10, random_state=0).fit_predict(R)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(int(a.n_clusters))
    hold = order[: int(a.holdout_clusters)]
    te = np.where(np.isin(lab, hold))[0]
    tr = np.where(~np.isin(lab, hold))[0]
    print(f"[split] 留出簇={hold.tolist()}  train={len(tr)}  val={len(te)}")

    dev = torch.device(a.device)
    net = DeltaTokenNet(feat_dim=X.shape[1] - 64, hidden=a.hidden, depth=a.depth,
                        init_std=4e-4).to(dev)
    print(f"[net] params={net.n_params}")
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(a.epochs))
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    Xn = torch.tensor((X - Xm) / Xs, device=dev)
    Tgt = torch.tensor(Yn * float(a.eta), device=dev)      # 目标幅度 = η
    itr = torch.tensor(tr, device=dev)
    ite = torch.tensor(te, device=dev)

    best = -1.0
    for ep in range(int(a.epochs)):
        net.train()
        idx = itr[torch.randperm(len(itr), device=dev)]
        tot = 0.0
        for i in range(0, len(idx), int(a.batch)):
            j = idx[i:i + int(a.batch)]
            opt.zero_grad(set_to_none=True)
            pred = net(Xn[j][:, :64], Xn[j][:, 64:])
            loss = ((pred - Tgt[j]) ** 2).sum(-1).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        sched.step()
        net.eval()
        with torch.no_grad():
            p = net(Xn[ite][:, :64], Xn[ite][:, 64:])
            t = Tgt[ite]
            cos = (p * t).sum(1) / (p.norm(dim=1).clamp_min(1e-9) * t.norm(dim=1).clamp_min(1e-9))
            cos_in = None
            if ep % 5 == 4 or ep == int(a.epochs) - 1:
                pi = net(Xn[itr][:4096][:, :64], Xn[itr][:4096][:, 64:])
                ti = Tgt[itr][:4096]
                cos_in = float(((pi * ti).sum(1) / (pi.norm(dim=1).clamp_min(1e-9)
                                                   * ti.norm(dim=1).clamp_min(1e-9))).mean())
        m = float(cos.mean())
        print(f"[ep {ep:3d}] loss={tot/len(idx):.5f}  held-out cos={m:+.3f}"
              + (f"  train cos={cos_in:+.3f}" if cos_in is not None else ""), flush=True)
        if m > best:
            best = m
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    print(f"[done] best held-out cos = {best:+.3f}")
    # ---- 存成 head ckpt（沿用模板保证 LoRA/delta-token 格式一致）----
    tmpl = Path(a.template)
    ck = torch.load(tmpl, map_location="cpu", weights_only=False) if tmpl.exists() else {}
    ck["delta_token_state"] = best_state
    # ★ schema v2：与 `V3Trainer.delta_token_meta()` **同键**，部署侧（delta_token_io）
    #   才能不靠猜地读出输入契约；缺字段会退回 legacy 推断（有告警、但不该出现在新产物上）。
    _feat_dim = int(X.shape[1] - 64)
    _full = _feat_dim == 64
    ck["delta_token_meta"] = {
        "schema_version": 2,
        "hidden": int(a.hidden),
        "depth": int(a.depth),
        "init_std": 4e-4,
        "cap": 0.0,
        "feat_dim": _feat_dim,
        "token_dim": 64,
        "input_features": ("state64[rot6d6|dof_pos29|dof_vel29]" if _full
                           else "rot6d6+dof_vel29"),
        "input_features_version": ("deploy_state64_v1" if _full else "legacy_35_v1"),
        "feat_joint_order": "standard",       # 与训练侧 `_delta_token_features` 同序
        "apply": "broadcast_shared_over_chunk",
        "norm_cap": "apply_norm_cap(tanh)",
        "teacher_objective": "shared_delta_discounted_v1",
        "teacher_reward": "discounted",
        "teacher_horizon": 20,
        "teacher_eta": float(a.eta),
        "teacher_n": 0,
        "grad_target_weight": 1.0,
        "trained_steps": int(a.epochs),
        "trained": "offline-distill",
        "eta": float(a.eta),
        "heldout_cos": float(best),
        "n_labels": int(len(X)),
        "holdout_clusters": hold.tolist(),
        "target": str(a.target),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    print(f"[ok] 写出 {out}（含 delta_token_state/meta，可直接给 server / probe 用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
