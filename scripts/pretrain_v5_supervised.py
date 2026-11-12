#!/usr/bin/env python3
"""在 **v5 的特征口径**下做监督预训练（rotlocal6 + q−default 历史 + standard 关节序），
产出可直接给 `train_temporal_ac_bptt_v5.py --init-actor` 的初始化 ckpt。

为什么需要它（2026-11-07 冒烟发现）
----------------------------------
我们之前那版监督蒸馏用的是**自己的**特征口径（绝对 rot6d + 原始 q），
而 v5 管道喂的是 **rotlocal6 + (q−default)**。权重搬过去后条件通路吃到分布外输入，
实测 `cosZ=0.99 ⇒ FAIL_CONSTANT`（比随机初始化还差）。
本脚本按 v5 口径重做特征与监督训练，使初始化点**条件化是活的**。

特征（与 v5 文档 §3.1 一致）
---------------------------
* `Z`        = VLA 本次 chunk `(B,40,64)`（我们的 dump 里就是 `token_chunk`）
* `state64`  = `d03.state64_from(bundle, state, frame)` = `[rotlocal6 | q | qd]`（standard 序）
* `hist10`   = `[q − default | qd]`（standard 序，oldest→newest），由 `obs_deploy930` 的
               `jpos(=q−default, ha_v31)` / `jvel(=qd, ha_v31)` 两段换序得到
* 标签       = 逐帧 `η·unit(ḡ)`，只在前 `K=20` 帧监督（`mask`）

用法
----
    python scripts/pretrain_v5_supervised.py --dir runs/labels_v3 \
        --out runs/v5_sup_init/init_v5_supervised.pt --eta 0.15 --epochs 40
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
MAIN = Path(__import__("os").environ.get("COMIMIC_MAIN", str(ROOT)))
for _p in (str(MAIN), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(ROOT / "scripts"))

import diag_bptt_learnability as d03                                    # noqa: E402
from diff_vla_wbc.delta_token_io_v5 import DeltaTokenSpecV5, save_temporal_actor_v5  # noqa: E402
from diff_vla_wbc.delta_token_temporal_v5 import TemporalActorCfg, TemporalDeltaActorV5  # noqa: E402

K_EXEC = 20
HIST_BLOCKS = (("ang_vel", 3), ("jpos", 29), ("jvel", 29), ("last_action", 29), ("grav", 3))


def hist10_v5_from_930(obs930: np.ndarray) -> np.ndarray:
    """930 → `(N,10,58)`：`[q−default | qd]`，**standard 关节序**（v5 训练口径）。"""
    import joint_contract as JC

    perm = np.asarray(JC.permutation("ha_v31", "standard"), dtype=np.int64)
    off, blocks = 0, {}
    for name, d in HIST_BLOCKS:
        blocks[name] = obs930[:, off:off + 10 * d].reshape(len(obs930), 10, d)
        off += 10 * d
    q_def_std = blocks["jpos"][:, :, perm]          # 930 里 jpos 已是 (q − default)，ha_v31 序
    qd_std = blocks["jvel"][:, :, perm]
    return np.concatenate([q_def_std, qd_std], -1).astype(np.float32)


def load(d: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(Path(d) / "dump_*.npz")))
    if not files:
        raise SystemExit(f"没有 dump: {d}")
    keys = ("token_chunk", "state", "frame", "g_tok_pert", "obs_deploy930")
    buf: dict[str, list] = {}
    for f in files:
        z = np.load(f)
        for k in keys:
            if k in z.files:
                buf.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v, 0) for k, v in buf.items()}
    print(f"[data] {len(files)} 文件  N={len(out['token_chunk'])}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/labels_v3")
    ap.add_argument("--out", default="runs/v5_sup_init/init_v5_supervised.pt")
    ap.add_argument("--wbc-ckpt", default="runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--wm-ckpt", default="runs/a_prime_sonic_exc/wm_gain2.pt")
    # ★ 2026-11-07 契约修复（含 soft-cap 余量）：
    #   `apply_norm_cap` 是 ‖Δ‖ = cap·tanh(‖u‖/cap) < cap（严格小于、且边界处灵敏度→0），
    #   所以**目标不能等于 cap**。取 cap 的 75%（0.08→0.06）留出可达性与梯度余量。
    ap.add_argument("--eta", type=float, default=0.06)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--holdout-groups", type=int, default=2)
    ap.add_argument("--cap", type=float, default=0.08)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-samples", type=int, default=0, help="0=全部")
    a = ap.parse_args()

    d = load(a.dir)
    if a.max_samples and len(d["token_chunk"]) > a.max_samples:
        idx = np.random.default_rng(0).permutation(len(d["token_chunk"]))[: a.max_samples]
        d = {k: v[idx] for k, v in d.items()}
        print(f"[data] 抽样 {len(idx)}")
    N = len(d["token_chunk"])
    dev = torch.device(a.device)

    # ---- 特征：v5 口径 ----
    bundle = d03.load_bundle_for_v5(  # 复用会话 03 的装配（obs/burnin/kinematics 一致）
        wbc_ckpt=a.wbc_ckpt, wm_ckpt=a.wm_ckpt, device=dev) if hasattr(d03, "load_bundle_for_v5") else None
    if bundle is None:
        from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415
        bundle = load_isaac_bundle(
            a.wbc_ckpt, device=dev, npz=sorted((MAIN / "data/npz/ha_tokens").glob("*.npz")),
            packed_motion=str(MAIN / "runs/isaac_motion/ha_cmd/packed.pt"),
            obs_mode=None, hist=None, token_dim=64)
        _wd = torch.load(a.wm_ckpt, map_location="cpu", weights_only=False)
        bundle.wm.load_state_dict(_wd.get("wm", _wd), strict=True)

    state_t = torch.tensor(d["state"].astype(np.float32))
    frame_t = torch.tensor(d["frame"].astype(np.int64))
    s64 = torch.empty(N, 64)
    B = 4096
    rotlocal_ok = 0
    for i in range(0, N, B):
        sl = slice(i, min(i + B, N))
        s64[sl] = d03.state64_from(bundle, state_t[sl], frame_t[sl], dev).detach().cpu()
        rotlocal_ok += int(sl.stop - sl.start)
    print(f"[feat] state64(rotlocal6+q+qd) 完成 {rotlocal_ok} 条")
    hist = torch.tensor(hist10_v5_from_930(d["obs_deploy930"]))
    Z = torch.tensor(d["token_chunk"].astype(np.float32))
    G = torch.tensor(d["g_tok_pert"].astype(np.float32))
    T = Z.shape[1]
    mask = torch.zeros(N, T)
    mask[:, : min(K_EXEC, T)] = 1.0
    Y = (G / (G.norm(dim=-1, keepdim=True) + 1e-9)) * a.eta
    if float(a.eta) > 0.9 * float(a.cap):
        print(f"[warn] η={a.eta} 已接近/超过 cap={a.cap} ⇒ soft-cap 下目标不可达或被压扁，"
              f"建议 η ≤ 0.75·cap = {0.75 * float(a.cap):.3f}")
    print(f"[feat] Z={tuple(Z.shape)} hist={tuple(hist.shape)} state64={tuple(s64.shape)} "
          f"标签只监督前 {min(K_EXEC, T)} 帧")

    # ---- 按 frame 分段留出 ----
    fr = d["frame"].astype(np.float64)
    edges = np.quantile(fr, np.linspace(0, 1, 11))
    grp = np.clip(np.searchsorted(edges[1:-1], fr), 0, 9)
    te = torch.tensor(np.where(grp < a.holdout_groups)[0])
    tr = torch.tensor(np.where(grp >= a.holdout_groups)[0])
    print(f"[split] 留出前 {a.holdout_groups}/10 段  train={len(tr)} val={len(te)}")

    cfg = TemporalActorCfg(chunk_len=T, cap=float(a.cap), use_history=True)
    net = TemporalDeltaActorV5(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(a.epochs))
    Ztr, Htr, Str, Ytr, Mtr = (Z[tr].to(dev), hist[tr].to(dev), s64[tr].to(dev),
                               Y[tr].to(dev), mask[tr].to(dev))
    Zte, Hte, Ste, Yte, Mte = (Z[te].to(dev), hist[te].to(dev), s64[te].to(dev),
                               Y[te].to(dev), mask[te].to(dev))
    best, best_sd = -1.0, None

    def cond_stats(net):
        """v5 的 cond 门禁：打乱 chunk / state / hist 后输出的余弦（越小越"条件化"）。"""
        net.eval()
        with torch.no_grad():
            base = net(Zte[:128], Ste[:128], Hte[:128], Mte[:128], cap=a.cap)
            g = torch.Generator().manual_seed(7)
            p = torch.randperm(len(base), generator=g)
            cz = torch.cosine_similarity(base.reshape(-1, 64),
                                         net(Zte[:128][p], Ste[:128], Hte[:128], Mte[:128],
                                             cap=a.cap).reshape(-1, 64), dim=-1).mean()
            cs = torch.cosine_similarity(base.reshape(-1, 64),
                                         net(Zte[:128], Ste[:128][p], Hte[:128], Mte[:128],
                                             cap=a.cap).reshape(-1, 64), dim=-1).mean()
            ch = torch.cosine_similarity(base.reshape(-1, 64),
                                         net(Zte[:128], Ste[:128], Hte[:128][p], Mte[:128],
                                             cap=a.cap).reshape(-1, 64), dim=-1).mean()
        return float(cz), float(cs), float(ch)

    def cos_holdout(net) -> float:
        net.eval()
        with torch.no_grad():
            p = net(Zte, Ste, Hte, Mte, cap=a.cap)
            c = torch.cosine_similarity(p.reshape(-1, 64), Yte.reshape(-1, 64), dim=-1)
            m = Mte.reshape(-1)
            return float((c * m).sum() / m.sum().clamp_min(1e-9))

    for ep in range(int(a.epochs)):
        net.train()
        idx = torch.randperm(len(Ztr), device=dev)
        tot = 0.0
        for i in range(0, len(idx), int(a.batch)):
            j = idx[i:i + int(a.batch)]
            opt.zero_grad(set_to_none=True)
            pred = net(Ztr[j], Str[j], Htr[j], Mtr[j], cap=a.cap)
            loss = (((pred - Ytr[j]) * Mtr[j][..., None]) ** 2).sum() / Mtr[j].sum().clamp_min(1e-6)
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        if ep % 5 == 4 or ep == int(a.epochs) - 1:
            c = cos_holdout(net)
            cz, cs, ch = cond_stats(net)
            print(f"[ep {ep:3d}] loss={tot / max(1, (len(idx) + int(a.batch) - 1) // int(a.batch)):.5f} "
                  f"held-out cos={c:+.3f} | cond cosZ={cz:.3f} cosS={cs:.3f} cosH={ch:.3f}", flush=True)
            if c > best:
                best, best_sd = c, {k: v.detach().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_sd)
    cz, cs, ch = cond_stats(net)
    print(f"[done] best held-out cos={best:+.3f}  cond cosZ={cz:.3f} cosS={cs:.3f} cosH={ch:.3f}")
    if cz > 0.95:
        print("[warn] cosZ 仍 >0.95：条件化可能没学起来（初始化价值有限）")

    spec = DeltaTokenSpecV5(T=int(T), K=int(min(K_EXEC, T)), H=20, cap=float(a.cap))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_temporal_actor_v5(out, net, spec, extra={
        "init_from": str(a.dir), "pretrain": "supervised_v5_features",
        "heldout_cos": float(best), "cond_cosZ": cz, "cond_cosS": cs, "cond_cosH": ch,
        "n_labels": int(N), "eta": float(a.eta),
    })
    out.with_suffix(".json").write_text(json.dumps(
        {"out": str(out), "params": int(net.n_params), "heldout_cos": best,
         "cond_cosZ": cz, "cond_cosS": cs, "cond_cosH": ch, "n": int(N),
         "holdout_groups": int(a.holdout_groups), "eta": float(a.eta)}, indent=2))
    print(f"[ok] 写出 {out}（v5 口径监督预训练 ckpt，可直接 --init-actor）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
