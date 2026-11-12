#!/usr/bin/env python3
"""**判据：视觉里到底有没有预测 teacher 梯度的信息？**（2026-11-07）

为什么必须测这个
----------------
现有 Δ头 的输入是 `token_chunk + obs930 + state64`，方向精度（只在非零标签帧上算）约
**0.29–0.34**（`scripts/probe_label_generalization.py`），而且留族/留运动几乎一样
⇒ 瓶颈是"特征里没有信息"，不是"任务没覆盖"。

**但要小心一个混淆**：`token_chunk` 本身就是 VLA 从画面推出来的，所以视觉信息已经被
压缩进 token 了。真正的问题是：

> **DiT 直接吃到的 2048 维 backbone 特征里，还有多少是 token 里没有的？**

这决定了"把 Δ 换成 action-head LoRA"（LoRA 通过 cross-attn 直接看这份特征）能不能有增益 ——
如果这份特征对 `g` 的额外可预测性≈0，那 LoRA 只是在更贵的地方拟合同样的信息。

口径
----
* 特征三组：① 现有（token_chunk + obs930 + state64）；② 只加视觉 `bb`(2048，均值池化)；
  ③ 全部。
* 视觉特征取自**部署同一条路径**：
  `processed_batch` → `vlm.prepare_input` → `vlm.backbone` → `action_head._encode_features`
  的 `backbone_features [B,T,2048]`（与训练时 DiT cross-attend 的那份**同一层**）。
* 留出：每个任务族内部**留出 30% 的示教运动**（族见过、运动没见过 = 部署真实情形），
  训练集 = 其余全部。cos 只在非零标签帧上平均（与 `pretrain_v5_supervised.py` 同口径）。

用法::

    # 1) 抽特征（GPU）
    python3 scripts/probe_vision_information.py extract --n 2000 --out runs/vision_probe
    # 2) 比可预测性（GPU/CPU 都行）
    python3 scripts/probe_vision_information.py probe --dir runs/vision_probe
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CKPT = ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000"
LABELS = ROOT / "runs/labels_v3"
NPZ = ROOT / "data/npz/ha_tokens"
PACKED = ROOT / "runs/isaac_motion/ha_cmd/packed.pt"
FAMS = ("HOI_double_desk", "HSI_vision_navi", "HOI_pp_box",
        "HSI_open_door", "HSI_boxing", "HSI_sit_sofa")
DEPLOY = ("HSI_open_door", "HSI_sit_sofa", "HSI_vision_navi", "HOI_double_desk")


def _labels():
    import numpy as np
    buf: dict[str, list] = {}
    for f in sorted(glob.glob(str(LABELS / "dump_*.npz"))):
        z = np.load(f)
        for k in ("token_chunk", "obs_deploy930", "g_tok_pert", "frame", "state"):
            buf.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, 0) for k, v in buf.items()}


def _family_of(frame, names, starts, tot):
    import numpy as np
    mi = np.searchsorted(starts, np.clip(np.asarray(frame, np.int64), 0, tot - 1), side="right") - 1
    return mi, np.array([names[i].split("__")[0] for i in mi])


def cmd_extract(a) -> int:
    import numpy as np
    import torch

    from diff_vla_wbc.g1_tasks import (EgoViewImages, TaskFrameIndex,
                                       TaskObservationBuilder, load_gr00t_processor,
                                       processed_batch)
    from diff_vla_wbc.gr00t_vlm import Gr00tVLM

    torch.manual_seed(0)
    D = _labels()
    pk = torch.load(PACKED, map_location="cpu", weights_only=False)
    names = list(pk["names"])
    starts = np.asarray(pk["starts"])
    tot = int(np.asarray(pk["lengths"]).sum())
    mi, fam = _family_of(D["frame"], names, starts, tot)

    files = sorted(glob.glob(str(NPZ / "*.npz")))
    idx = TaskFrameIndex(files)
    lens = [int(np.load(f, mmap_mode="r")["joint_pos"].shape[0]) if False else None for f in []]
    # 校验：帧索引的 starts 必须与运动库一致，否则 frame → 图像会错位
    if idx.num_frames != tot or not np.array_equal(np.asarray(idx._starts), starts):
        raise SystemExit(f"[FATAL] 帧索引与运动库不一致：{idx.num_frames} vs {tot}，"
                         f"starts 相同={np.array_equal(np.asarray(idx._starts), starts)}")
    print(f"[index] {idx.num_episodes} 个 episode, {idx.num_frames} 帧，与运动库一致 ✓")

    # 分层抽样：每族按比例，且优先取不同运动
    rng = np.random.default_rng(0)
    per_fam = max(1, int(a.n / len(FAMS)))
    sel = []
    for f in FAMS:
        cand = np.where(fam == f)[0]
        if len(cand) == 0:
            continue
        take = min(per_fam, len(cand))
        sel.append(rng.choice(cand, take, replace=False))
    sel = np.sort(np.concatenate(sel))
    print(f"[sample] n={len(sel)} 族分布=" +
          str({f: int((fam[sel] == f).sum()) for f in FAMS if (fam[sel] == f).any()}))

    proc = load_gr00t_processor(CKPT)
    vlm = Gr00tVLM.from_pretrained(checkpoint=CKPT, device=a.device, freeze_vlm=True)
    vlm.eval()
    builder = TaskObservationBuilder(idx, EgoViewImages(cache_size=4))

    feats, done = [], 0
    order = np.argsort(fam[sel], kind="stable")          # 同族连续读，减少容器抖动
    sel = sel[order]
    for s in range(0, len(sel), a.batch):
        j = sel[s:s + a.batch]
        fr = torch.as_tensor(D["frame"][j], dtype=torch.long)
        batch = processed_batch(proc, builder, fr)
        bb_in, act_in = vlm.prepare_input(dict(batch))
        with torch.no_grad():
            bb = vlm.backbone(bb_in)
            f = vlm.action_head._encode_features(bb, act_in).backbone_features   # (B,T,2048)
            feats.append(f.mean(1).float().cpu().numpy())
        done += len(j)
        if done % (a.batch * 5) == 0 or done == len(sel):
            print(f"[extract] {done}/{len(sel)}", flush=True)
    V = np.concatenate(feats, 0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "vision_feats.npz",
                        feat=V.astype(np.float32), idx=sel.astype(np.int64),
                        family=fam[sel], frame=D["frame"][sel].astype(np.int64),
                        motion=mi[sel].astype(np.int64))
    print(f"[write] {out / 'vision_feats.npz'}  feat={V.shape}")
    return 0


def cmd_probe(a) -> int:
    import numpy as np
    import torch

    from train_distill_v3 import state64_from_state71

    d = Path(a.dir)
    z = np.load(d / "vision_feats.npz")
    sel, V, fam, motion = z["idx"], z["feat"].astype(np.float32), z["family"], z["motion"]
    D = _labels()
    N = len(sel)
    G = torch.as_tensor(D["g_tok_pert"][sel].astype(np.float32))
    T = G.shape[1]
    nrm = G.norm(dim=-1)
    info = nrm > 1e-9
    Yu = G / nrm.clamp_min(1e-12)[..., None]
    base = np.concatenate([
        D["token_chunk"][sel].astype(np.float32).reshape(N, -1),
        D["obs_deploy930"][sel].astype(np.float32),
        state64_from_state71(D["state"][sel].astype(np.float32)),
    ], 1)
    print(f"[probe] n={N} 视频特征 dim={V.shape[1]} 基础特征 dim={base.shape[1]}")
    print(f"[probe] 族分布={ {f: int((fam == f).sum()) for f in DEPLOY if (fam == f).any()} }")

    dev = torch.device(a.device)

    def fit_eval(X, names_tr):
        ci_list = []
        for f in DEPLOY:
            m = np.where(fam == f)[0]
            if len(m) < 60:
                continue
            mids = np.unique(motion[m])
            rng = np.random.default_rng(0)
            mids = mids.copy()
            rng.shuffle(mids)
            te_m = set(mids[: max(1, int(0.3 * len(mids)))].tolist())
            te = m[np.isin(motion[m], list(te_m))]
            tr = np.setdiff1d(np.arange(N), te)
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            A = torch.as_tensor((X[tr] - mu) / sd, device=dev)
            B = torch.as_tensor((X[te] - mu) / sd, device=dev)
            Gg = A.T @ A + 1e-2 * max(1, len(tr)) * torch.eye(A.shape[1], device=dev)
            W = torch.linalg.solve(Gg, A.T @ Yu[tr].to(dev).reshape(len(tr), -1))
            P = (B @ W).reshape(len(te), T, 64)
            c = (P * Yu[te].to(dev)).sum(-1) / (P.norm(dim=-1).clamp_min(1e-9)
                                                * Yu[te].to(dev).norm(dim=-1).clamp_min(1e-9))
            ci = float(c[info[te].to(dev)].mean())
            ci_list.append((f, len(te), ci))
            print(f"  {names_tr:26s} 留运动 {f:20s} n_te={len(te):4d}  cos={ci:+.3f}")
        return ci_list

    def pca_apply(Vtr, Vte, k, seed=0):
        """在训练集上拟合 PCA，投影到 k 维（视觉特征 2048 维 / 样本 ~1500 ⇒ 必须降维）。"""
        mu = Vtr.mean(0, keepdims=True)
        X = torch.as_tensor(Vtr - mu, device=dev, dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        Wk = Vh[:k].T
        return (torch.as_tensor(Vte - mu, device=dev, dtype=torch.float32) @ Wk).cpu().numpy()

    print("\n[1] 现有特征（token+obs930+state64）")
    r1 = fit_eval(base, "base")
    print("\n[2] 只加视觉（PCA 降到 32 维）")
    r2 = fit_eval(pca_apply(V, V, 32) if False else V, "vision-raw")   # 原始 2048，仅作参照
    print("\n[3] 现有 + 视觉（PCA，含打乱对照）")
    rnd = np.random.default_rng(0).permutation(N)                     # 打乱对照：同样容量、无信息
    r3 = {}
    for k in (16, 32, 64):
        Vp = pca_apply(V, V, k)
        r3[k] = (fit_eval(np.concatenate([base, Vp], 1), f"base+vis_pca{k}"),
                 fit_eval(np.concatenate([base, Vp[rnd]], 1), f"base+shuf_pca{k}"))

    def summarize(rows):
        return float(np.mean([c for _, _, c in rows])) if rows else float("nan")
    s1, s2 = summarize(r1), summarize(r2)
    print(f"\n{'特征':<32}{'平均 cos':>10}")
    print(f"{'现有（base）':<32}{s1:>10.3f}")
    print(f"{'仅视觉（原始 2048）':<32}{s2:>10.3f}")
    out = {"base": r1, "vision_raw": r2, "n": int(N), "pca": {}}
    print(f"{'现有+视觉(k)':<32}{'vis':>9}{'shuffled 对照':>15}{'增量':>9}")
    for k, (rv, rs) in r3.items():
        sv, ss = summarize(rv), summarize(rs)
        out["pca"][str(k)] = {"vis": rv, "shuf": rs, "mean_vis": sv, "mean_shuf": ss}
        print(f"{'k=' + str(k):<32}{sv:>9.3f}{ss:>15.3f}{sv - s1:>+9.3f}")
    print("\n⇒ 判据：`vis` 要**同时**高于 `shuffled 对照` 且增量 ≥ +0.05，才算视觉里有额外信息。")
    (Path(a.dir) / "vision_probe.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--n", type=int, default=2000)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--out", default="runs/vision_probe")
    e.set_defaults(fn=cmd_extract)
    p = sub.add_parser("probe")
    p.add_argument("--dir", default="runs/vision_probe")
    p.add_argument("--device", default="cuda:0")
    p.set_defaults(fn=cmd_probe)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
