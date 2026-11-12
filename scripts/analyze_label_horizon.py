#!/usr/bin/env python3
"""**判据：Teacher 标签的视界 H 该取多少？**（2026-11-07，回答"5/8/10 哪个合适"）

问题
----
部署要执行 20 步（K=20 的想象窗口），但标签视界 H 是可选的：

* H 大 ⇒ 标签直接包含未来（不需要 critic），但梯度要穿过 20 步 WM，统计变差（可学性 0.33）；
* H 小 ⇒ 标签好学好信（k=1 可学性 0.50、真机幅值 1.14×），但**丢掉尾巴**，要靠 critic 用
  `V(s_H)` 补 `γ^H ∂V/∂token` 那一项。

所以 H 的选择是一个可测的权衡，判据是三个量（本脚本算的）：

1. **方向保真** `cos(g_H, g_K)`：H 视界梯度是否仍指向 K 步最优方向。
2. **幅度份额** `‖g_H‖ / ‖g_K‖`：critic 需要补的那部分有多大（1 − 比值）。
3. **可用性**：无效帧（‖g‖≈0）占比、逐帧范数分布 —— 这决定"逐帧监督"是否有意义。

预注册判据（先写死，避免事后找理由）
------------------------------------
* **① 方向保真**：`p50 cos(g_H, g_20) ≥ 0.85`
* **② critic 负担**：`p50 ‖g_H‖/‖g_20‖ ≥ 0.60`（即 critic 只需补 ≤40%）
* 在同时满足 ①② 的 H 里取**最大**的（最好学）；若都不满足 ⇒ 留在 H=20，走"修损失"路线。

用法::

    python3 scripts/analyze_label_horizon.py --dump "runs/harness_02_teacher/dump_h20/teacher_*.npz" \
        --hs 2 3 5 8 10 15 20 --batch 64 --out runs/label_horizon.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ROOT / "runs/harness_02_teacher/dump_h20/teacher_*.npz"))
    ap.add_argument("--ckpt", default=str(ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt"))
    ap.add_argument("--hs", nargs="+", type=int, default=[2, 3, 5, 8, 10, 15, 20])
    ap.add_argument("--horizon", type=int, default=20, help="想象窗口 K（= 部署的 20 步）")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/label_horizon.json")
    a = ap.parse_args()

    import numpy as np
    import torch

    from diff_vla_wbc import teacher_probe as tp
    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle

    files = sorted(glob.glob(a.dump))
    if not files:
        raise SystemExit(f"没有匹配 {a.dump}")
    buf: dict[str, list] = {}
    for f in files:
        d = np.load(f)
        for k in ("token_chunk", "state", "xi", "prev_action", "history", "obs_hist",
                  "frames", "clip_ids"):
            buf.setdefault(k, []).append(d[k])
    D = {k: np.concatenate(v, axis=1 if k in ("history", "frames", "clip_ids") else 0)
         for k, v in buf.items()}
    N = D["token_chunk"].shape[0]
    K = int(D["frames"].shape[0])
    print(f"[data] {len(files)} dumps → N={N} 窗口, K={K} 帧, "
          f"history{D['history'].shape} frames{D['frames'].shape}")

    dev = torch.device(a.device)
    bundle = load_isaac_bundle(a.ckpt, device=dev,
                               npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)

    tok_all = torch.as_tensor(D["token_chunk"], dtype=torch.float32)
    st_all = torch.as_tensor(D["state"], dtype=torch.float32)
    xi_all = torch.as_tensor(D["xi"], dtype=torch.float32)
    prev_all = torch.as_tensor(D["prev_action"], dtype=torch.float32)
    hist_all = torch.as_tensor(D["history"], dtype=torch.float32)     # (T,N,93)
    oh_all = torch.as_tensor(D["obs_hist"], dtype=torch.float32)
    fr_all = torch.as_tensor(D["frames"], dtype=torch.long)           # (K,N)

    res: dict[int, dict] = {}
    G: dict[int, list] = {h: [] for h in a.hs}
    JJ: dict[int, list] = {h: [] for h in a.hs}
    for s in range(0, N, a.batch):
        e = min(N, s + a.batch)
        sl = slice(s, e)
        n = e - s
        refs = [FrameRef(fr_all[t, sl].to(dev)) for t in range(K)]
        print(f"[grad] {s}..{e} ", end="", flush=True)
        for h in a.hs:
            out = tp.tokenwise_reward_grad(
                bundle, tok_all[sl].to(dev), state=st_all[sl].to(dev), references=refs,
                xi=xi_all[sl].to(dev), prev_action=prev_all[sl].to(dev),
                history=hist_all[:, sl].to(dev), obs_hist=None,
                horizon=int(a.horizon), gamma=0.99,
                reward_steps=(0 if h >= a.horizon else int(h)),
            )
            # ★ 必须用 out.g（= ∂J/∂token，J 已按 reward_steps 截断）；
            #   out.g_undisc 是**全视界未折扣**回报的梯度，与 reward_steps 无关 ⇒ 各 H 恒等。
            G[h].append(out.g.detach().cpu().float())    # (n,T,64)
            JJ[h].append(out.J.detach().cpu().float())
        print("ok", flush=True)

    for h in a.hs:
        G[h] = torch.cat(G[h])       # (N,T,64)
        JJ[h] = torch.cat(JJ[h])     # (N,)

    if int(a.horizon) not in res and a.horizon in G:
        pass
    ref_h = max(a.hs)                # 默认拿最大的 H 当"全长视界"参照
    gr = G[ref_h]
    nr = gr.norm(dim=-1)             # (N,T)
    out_rows = []
    print(f"\n参照 = H={ref_h}（K={a.horizon}）")
    hdr = (f"{'H':>4}{'‖g_H‖p50':>11}{'比值p50':>9}{'cos p50':>9}{'cos mean':>10}"
           f"{'cos p10':>9}{'cos≥0.85':>10}{'零帧占比':>9}{'J_H/J_K':>9}")
    print(hdr)
    print("-" * len(hdr))
    for h in a.hs:
        g = G[h]
        n = g.norm(dim=-1)
        good = (n > 1e-9) & (nr > 1e-9)
        c = ((g * gr).sum(-1) / (n.clamp_min(1e-12) * nr.clamp_min(1e-12)))
        cv = c[good]
        ratio = (n / nr.clamp_min(1e-12))[good]
        row = {
            "H": int(h),
            "g_p50": float(n[n > 0].median()) if (n > 0).any() else 0.0,
            "ratio_p50": float(ratio.median()) if good.any() else 0.0,
            "cos_p50": float(cv.median()) if good.any() else 0.0,
            "cos_mean": float(cv.mean()) if good.any() else 0.0,
            "cos_p10": float(np.quantile(cv.numpy(), 0.1)) if good.any() else 0.0,
            "frac_ge_085": float((cv >= 0.85).float().mean()) if good.any() else 0.0,
            "zero_frac": float((n <= 1e-9).float().mean()),
            "J_ratio": float((JJ[h] / JJ[ref_h].clamp_min(1e-9)).median()),
            "per_frame_zero": (n <= 1e-9).float().mean(0).tolist(),
            "per_frame_g_p50": n.median(0).values.tolist(),
        }
        out_rows.append(row)
        print(f"{h:>4}{row['g_p50']:>11.4f}{row['ratio_p50']:>9.3f}{row['cos_p50']:>+9.3f}"
              f"{row['cos_mean']:>+10.3f}{row['cos_p10']:>+9.3f}{row['frac_ge_085']:>10.2f}"
              f"{row['zero_frac']:>9.2f}{row['J_ratio']:>9.3f}")

    print("\n判据：① cos p50 ≥ 0.85  ② ‖g_H‖/‖g_K‖ p50 ≥ 0.60 ⇒ 取满足两者的**最大** H")
    ok = [r for r in out_rows if r["cos_p50"] >= 0.85 and r["ratio_p50"] >= 0.60]
    pick = max((r["H"] for r in ok), default=None)
    print(f"⇒ 结论：{'H=' + str(pick) if pick else '没有 H 同时满足 ①② ⇒ 留在 H=20，先修损失/上 critic'}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": out_rows, "pick": pick, "K": int(a.horizon),
                                 "n_windows": int(N)}, indent=1))
    print(f"[write] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
