#!/usr/bin/env python3
"""harness02：蒸馏**可学性**离线实验（吃 ``J1_TEACHER_DUMP`` 的 npz，不需要 WM/VLA）。

回答的问题
----------
1. 用**修好的** Teacher 标签，128~256 个窗口能不能过拟合？（train cos）
2. held-out（**按 clip 分组**，不是随机打散）能到多少？与基线比：
   最优常向量 / Ridge / 2×128 MLP / 2×256 MLP / 原 DeltaTokenNet(1.25M)
3. n 增大时 held-out cos 怎么走（学习曲线）？
4. 旧 Teacher 的标签与新标签差多少（``cos(g_old, g_new)``）——
   这是"之前蒸馏 cos≈0.10 到底怪谁"的直接证据。
5. H=5 与 H=20 的标签在同窗口上差多少、可学性差多少。

口径
----
* 特征 = 部署可用的 ``token0(64) ⊕ state64(64)``；**不额外加线上拿不到的量**。
* 标准化参数**只**从训练集拟合。
* 划分按 clip（episode）分组 60/20/20，同一 clip 的窗口不会跨 split。
* 指标：逐窗口 cosine（对标签方向）与 MSE（对 ``η·unit(g)``）。

用法::

    python3 scripts/harness02_distill_learnability.py \
        --dump runs/harness_02_teacher/dump_h20 --out runs/harness_02_teacher
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- data
def load_dump(dump: Path, extra_h: str = ""):
    files = sorted(dump.glob("teacher_*.npz")) if dump.is_dir() else [dump]
    if not files:
        raise SystemExit(f"[fail] {dump} 下没有 teacher_*.npz")
    chunks: dict[str, list[np.ndarray]] = {}
    keys = None
    for f in files:
        z = np.load(f)
        _k = set(z.files)
        if keys is None:
            keys = _k
        else:
            keys &= _k
        for k in z.files:
            chunks.setdefault(k, []).append(np.asarray(z[k]))
    # 时间主轴在前（frames/clip_ids: (H_ref,B)；history: (hist,B,W)）⇒ 沿 **窗口轴** 拼
    _axis = {"frames": 1, "clip_ids": 1, "history": 1}
    out = {k: np.concatenate([np.atleast_1d(x) for x in v], axis=_axis.get(k, 0))
           for k, v in chunks.items() if k in keys}
    if "clip_ids" in out:
        # clip_ids 可能是 (H_ref, B)，取第 0 行（窗口起点所在 clip）
        c = out["clip_ids"]
        out["clip_ids"] = c[0] if c.ndim == 2 else c
    if "frames" in out:
        f = out["frames"]
        out["frame0"] = f[0] if f.ndim == 2 else f
    print(f"[data] {len(files)} files / {int(out['state'].shape[0])} windows "
          f"/ keys={sorted(out.keys())}", flush=True)
    if not extra_h:
        return out
    return out


def _unit(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(g, axis=-1, keepdims=True)
    ok = np.isfinite(g).all(-1) & (n.squeeze(-1) > 1e-9)
    u = np.zeros_like(g)
    u[ok] = g[ok] / n[ok]
    return u.astype(np.float64), ok


def cosine_rows(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    npn = np.linalg.norm(pred, axis=-1).clip(1e-12)
    ntn = np.linalg.norm(tgt, axis=-1).clip(1e-12)
    return (pred * tgt).sum(-1) / (npn * ntn)


# --------------------------------------------------------------------------- models
class MLP(torch.nn.Module):
    def __init__(self, d_in: int, hidden: int, depth: int = 2, d_out: int = 64,
                 init_std: float = 1e-2):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, hidden), torch.nn.SiLU()]
            d = hidden
        last = torch.nn.Linear(d, d_out)
        torch.nn.init.normal_(last.weight, std=init_std)
        torch.nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_torch(model, Xtr, Ytr, Xev_list, *, epochs=400, lr=1e-3, bs=64, patience=60,
              wd=1e-4, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Ytr_t = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
    evs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in Xev_list]
    n = Xtr_t.shape[0]
    best, best_state, bad = -1e9, None, 0
    gen = torch.Generator(device=device).manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen, device=device)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = ((model(Xtr_t[idx]) - Ytr_t[idx]) ** 2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                # 早停指标 = **训练集** cos（与 evs 无关，evs 可能是 val/test）
                cos = float(torch.nn.functional.cosine_similarity(
                    model(Xtr_t), Ytr_t, dim=-1).mean())
            if cos > best + 1e-6:
                best, bad = cos, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad * 5 >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    outs = []
    with torch.no_grad():
        for x in evs:
            outs.append(model(x).cpu().numpy())
    return outs, best


def metrics(pred, Y_unit, eta):
    tgt = eta * Y_unit
    mse = float(((pred - tgt) ** 2).sum(-1).mean())
    cos = float(cosine_rows(pred, Y_unit).mean())
    return cos, mse


# --------------------------------------------------------------------------- split
def grouped_split(groups: np.ndarray, seed: int = 0, frac=(0.6, 0.2, 0.2)):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = max(1, int(round(frac[0] * n)))
    n_va = max(1, int(round(frac[1] * n)))
    g_tr, g_va = set(uniq[:n_tr]), set(uniq[n_tr:n_tr + n_va])
    tr = np.isin(groups, list(g_tr))
    va = np.isin(groups, list(g_va))
    te = ~(tr | va)
    return np.where(tr)[0], np.where(va)[0], np.where(te)[0]


def subsample(idx, n, seed):
    if n >= len(idx):
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=n, replace=False))


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ROOT / "runs/harness_02_teacher/dump_h20"))
    ap.add_argument("--out", default=str(ROOT / "runs/harness_02_teacher"))
    ap.add_argument("--eta", type=float, default=0.08)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", default="64,128,256")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_dump(Path(args.dump))
    eta = float(args.eta)
    state = d["state"].astype(np.float64)
    feats = d["feats"].astype(np.float64)
    tok = d["token_chunk"].astype(np.float64)
    tok0 = tok[:, 0]
    groups = d.get("clip_ids")
    if groups is None:
        fr = d.get("frame0")
        if fr is None:
            raise SystemExit("[fail] dump 里既没有 clip_ids 也没有 frames，无法做分组划分")
        groups = (fr // 200).astype(np.int64)   # 兜底：按 200 帧一个桶（不是 episode，仅应急）
        print("[warn] 没有 clip_ids ⇒ 用 frame//200 当分组（偏保守但不等价 episode）", flush=True)
    # 特征（部署可用）：token0 ⊕ state64（state64 就是 feats）
    X = {
        "token0": tok0,
        "state64": feats,
        "token0+state64": np.concatenate([tok0, feats], 1),
    }
    # 标签
    targets = {}
    if "g" in d:
        targets["H_main"] = (np.asarray(d["g"], dtype=np.float64), int(d.get("H", [20])[0]) if "H" in d else 20)
    for k in d:
        if k.startswith("g_H"):
            try:
                h = int(k[3:])
            except ValueError:
                continue
            targets[f"H{h}"] = (np.asarray(d[k], dtype=np.float64), h)
    if not targets:
        raise SystemExit("[fail] dump 里没有 g")
    print(f"[labels] {sorted(targets)}", flush=True)

    # ---------------- 新旧 Teacher 标签差异 ----------------
    ovn_rows = []
    g_new_raw, H_main = targets.get("H_main", (None, 20))
    for name, (g, h) in targets.items():
        u, ok = _unit(g)
        row = {"compare": f"{name}_vs_H_main", "H": h, "n": int(ok.sum()),
               "g_norm_mean": float(np.linalg.norm(g, axis=-1)[ok].mean())}
        if g_new_raw is not None and name != "H_main":
            u0, ok0 = _unit(g_new_raw)
            m = ok & ok0
            row["cos_vs_H_main"] = float(cosine_rows(u[m], u0[m]).mean())
            row["cos_vs_H_main_p10"] = float(np.percentile(cosine_rows(u[m], u0[m]), 10))
        ovn_rows.append(row)
    # 旧 Teacher（token0 重复 + 未折扣 sum）
    old_keys = [("g_old", "legacy_token0_undisc"), ("g_old_disc", "legacy_token0_disc"),
                ("g_sum", "new_chunk_undisc")]
    for k, label in old_keys:
        if k not in d:
            continue
        uo, oko = _unit(np.asarray(d[k], dtype=np.float64))
        un, okn = _unit(g_new_raw)
        m = oko & okn
        c = cosine_rows(uo[m], un[m])
        ovn_rows.append({
            "compare": label, "H": H_main, "n": int(m.sum()),
            "g_norm_mean": float(np.linalg.norm(d[k], axis=-1)[oko].mean()),
            "cos_vs_H_main": float(c.mean()), "cos_vs_H_main_p10": float(np.percentile(c, 10)),
            "cos_vs_H_main_p50": float(np.median(c)),
        })
        print(f"[old-vs-new] {label}: cos={c.mean():.6f} p10={np.percentile(c,10):.4f} n={int(m.sum())}",
              flush=True)
    # 旧标签当"学生目标"时，与 held-out 新标签的余弦（跨口径）
    with open(out / "teacher_old_vs_new.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in ovn_rows for k in r}))
        w.writeheader()
        w.writerows(ovn_rows)

    # ---------------- 分组划分 ----------------
    tr_idx, va_idx, te_idx = grouped_split(groups, seed=0)
    print(f"[split] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)} "
          f"clips={len(np.unique(groups))}", flush=True)
    rows = []
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    def run_targets(tname: str):
        g, h = targets[tname]
        U, ok = _unit(g)
        base_ok = ok
        tr_all = tr_idx[base_ok[tr_idx]]
        va = va_idx[base_ok[va_idx]]
        te = te_idx[base_ok[te_idx]]
        if len(va) == 0 or len(te) == 0:
            print(f"[skip] {tname}: val={len(va)} test={len(te)} 太少", flush=True)
            return
        for n in sizes:
            tr = subsample(tr_all, n, seed=1234 + n)
            if len(tr) == 0:
                continue
            Ytr, Yva, Yte = U[tr], U[va], U[te]
            # 标准化：只用训练集
            for fname, Xf in X.items():
                mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-8
                Xtr, Xva, Xte = (Xf[tr] - mu) / sd, (Xf[va] - mu) / sd, (Xf[te] - mu) / sd
                # ---- 0) best constant ----
                c = Ytr.mean(0)
                pc = np.broadcast_to(c, Ytr.shape)
                rows.append(dict(target=tname, H=h, n_train=len(tr), model="constant",
                                 features=fname,
                                 **dict(zip(("train_cos", "train_mse"),
                                            metrics(pc, Ytr, eta))),
                                 **dict(zip(("val_cos", "val_mse"), metrics(
                                     np.broadcast_to(c, Yva.shape), Yva, eta))),
                                 **dict(zip(("test_cos", "test_mse"), metrics(
                                     np.broadcast_to(c, Yte.shape), Yte, eta))),
                                 note=f"|c|={np.linalg.norm(c):.4f}"))
                # ---- 1) ridge（λ 由**训练集内**5 折 CV 选）----
                def ridge_fit(Xa, Ya, lam):
                    A = np.concatenate([Xa, np.ones((len(Xa), 1))], 1)
                    R = np.eye(A.shape[1]) * lam
                    R[-1, -1] = 0.0
                    return np.linalg.solve(A.T @ A + R, A.T @ Ya)

                def ridge_pred(W, Xa):
                    return np.concatenate([Xa, np.ones((len(Xa), 1))], 1) @ W

                best = (-1e9, None)
                kf = 5
                order = np.arange(len(tr))
                np.random.default_rng(7).shuffle(order)
                for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
                    cs = []
                    for k in range(kf):
                        va_k = order[k::kf]
                        tr_k = np.setdiff1d(order, va_k)
                        W = ridge_fit(Xtr[tr_k], Ytr[tr_k], lam)
                        cs.append(cosine_rows(ridge_pred(W, Xtr[va_k]), Ytr[va_k]).mean())
                    m = float(np.mean(cs))
                    if m > best[0]:
                        best = (m, lam)
                W = ridge_fit(Xtr, Ytr, best[1])
                rows.append(dict(target=tname, H=h, n_train=len(tr), model="ridge",
                                 features=fname,
                                 **dict(zip(("train_cos", "train_mse"), metrics(ridge_pred(W, Xtr), Ytr, eta))),
                                 **dict(zip(("val_cos", "val_mse"), metrics(ridge_pred(W, Xva), Yva, eta))),
                                 **dict(zip(("test_cos", "test_mse"), metrics(ridge_pred(W, Xte), Yte, eta))),
                                 note=f"lam={best[1]:g} cvcos={best[0]:.4f}"))
                # ---- 2) MLP / DeltaTokenNet ----
                if fname != "token0+state64":
                    continue
                for model_name, hidden, depth, init_std, lr in (
                        ("mlp2x128", 128, 2, 1e-2, 1e-3),
                        ("mlp2x256", 256, 2, 1e-2, 1e-3),
                ):
                    preds = {"train": [], "val": [], "test": []}
                    for seed in range(args.seeds):
                        m = MLP(Xtr.shape[1], hidden, depth, 64, init_std)
                        (ptr, pva, pte), _ = fit_torch(
                            m, Xtr, eta * Ytr, [Xtr, Xva, Xte],
                            epochs=args.epochs, lr=lr, seed=seed, device=args.device)
                        preds["train"].append(ptr); preds["val"].append(pva); preds["test"].append(pte)
                    for split in preds:
                        preds[split] = np.mean(preds[split], 0)
                    rows.append(dict(target=tname, H=h, n_train=len(tr), model=model_name,
                                     features=fname,
                                     **dict(zip(("train_cos", "train_mse"), metrics(preds["train"], Ytr, eta))),
                                     **dict(zip(("val_cos", "val_mse"), metrics(preds["val"], Yva, eta))),
                                     **dict(zip(("test_cos", "test_mse"), metrics(preds["test"], Yte, eta))),
                                     note=f"seeds={args.seeds}"))
                    print(f"[{tname} n={len(tr)} {model_name}] "
                          f"train_cos={metrics(preds['train'], Ytr, eta)[0]:.4f} "
                          f"val_cos={metrics(preds['val'], Yva, eta)[0]:.4f} "
                          f"test_cos={metrics(preds['test'], Yte, eta)[0]:.4f}", flush=True)
                # 原 DeltaTokenNet（生产超参：hidden 1024 / init_std 4e-4 / lr 5e-4）
                from diff_vla_wbc.delta_token import DeltaTokenNet  # noqa: PLC0415

                preds = {"train": [], "val": [], "test": []}
                for seed in range(max(1, min(args.seeds, 2))):
                    m = DeltaTokenNet(token_dim=64, feat_dim=feats.shape[1], hidden=1024,
                                      depth=2, init_std=4e-4)
                    # 复用 MLP 的 fit：把 (token,feat) 拼好当单输入
                    class _Wrap(torch.nn.Module):
                        def __init__(self, net):
                            super().__init__()
                            self.net = net

                        def forward(self, x):
                            return self.net(x[:, :64], x[:, 64:])

                    (ptr, pva, pte), _ = fit_torch(
                        _Wrap(m), Xtr, eta * Ytr, [Xtr, Xva, Xte],
                        epochs=args.epochs, lr=5e-4, seed=seed, device=args.device)
                    preds["train"].append(ptr); preds["val"].append(pva); preds["test"].append(pte)
                for split in preds:
                    preds[split] = np.mean(preds[split], 0)
                rows.append(dict(target=tname, H=h, n_train=len(tr), model="DeltaTokenNet_prod",
                                 features=fname,
                                 **dict(zip(("train_cos", "train_mse"), metrics(preds["train"], Ytr, eta))),
                                 **dict(zip(("val_cos", "val_mse"), metrics(preds["val"], Yva, eta))),
                                 **dict(zip(("test_cos", "test_mse"), metrics(preds["test"], Yte, eta))),
                                 note="hidden=1024,init_std=4e-4,lr=5e-4"))
                print(f"[{tname} n={len(tr)} DeltaTokenNet] "
                      f"train_cos={metrics(preds['train'], Ytr, eta)[0]:.4f} "
                      f"val_cos={metrics(preds['val'], Yva, eta)[0]:.4f} "
                      f"test_cos={metrics(preds['test'], Yte, eta)[0]:.4f}", flush=True)

    for tname in ["H_main"] + [k for k in targets if k != "H_main"]:
        if tname in targets:
            print(f"== target {tname} (H={targets[tname][1]}) ==", flush=True)
            run_targets(tname)

    csv_path = out / "distill_learning_curve.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} rows -> {csv_path}", flush=True)
    with open(out / "distill_summary.json", "w") as fh:
        json.dump({"dump": str(args.dump), "n_windows": int(state.shape[0]),
                   "n_clips": int(len(np.unique(groups))),
                   "split": {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)},
                   "targets": {k: int(v[1]) for k, v in targets.items()}},
                  fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
