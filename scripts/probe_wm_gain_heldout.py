#!/usr/bin/env python3
"""**直接测产品 WM 的动作增益**：训练时见过的配对 vs 完全没见过的配对（留出集）。

为什么这个测法比"两模型转移率"更直接
--------------------------------------
`wm_gain2.pt` 训练时**只用了 A 半任务（DoubleDesk/PPBox）的配对数据**；
B 半任务（OpenDoor/SitSofa/VisionNavi）的配对它**从没见过** ⇒ 天然是留出集。

于是可以直接问它：
    同一个状态、两个不同动作 → 它预测的"下一状态差"有多接近真值？
  * 留出集误差 ≈ 训练集误差 ⇒ **增益泛化**，产品 WM 可用；
  * 留出集误差 ≫ 训练集误差 ⇒ 增益不泛化，产品 WM 不可用（和转移率 0.13 一致）。

用法::
    python3 scripts/probe_wm_gain_heldout.py --ckpt runs/a_prime_sonic_exc/wm_gain2.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs/a_prime_sonic_exc/wm_gain2.pt")
    ap.add_argument("--pairs-train", type=Path, default=ROOT / "runs/paired_probe2/gain_pairs.pt")
    ap.add_argument("--pairs-heldout", type=Path, default=ROOT / "runs/paired_probe_b/gain_pairs.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle
    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    b = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                          packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                          obs_mode=None, hist=None, token_dim=64)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    b.wm.load_state_dict(ck.get("wm", ck), strict=True)
    b.wm.requires_grad_(False); b.wm.eval()
    print(f"[gain] WM = {a.ckpt.name}", flush=True)

    blocks = {"pos(m)": (0, 3), "quat": (3, 7), "q(rad)": (7, 36),
              "linvel": (36, 39), "angvel": (39, 42), "qd(rad/s)": (42, 71)}

    def eval_pairs(path: Path, name: str):
        p = torch.load(path, map_location="cpu", weights_only=False)
        S, A, S1 = p["s"].float(), p["a"].float(), p["s1"].float()
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(S.shape[0], generator=g)[: a.n]
        S, A, S1 = S[idx].to(dev), A[idx].to(dev), S1[idx].to(dev)
        M, K = A.shape[0], A.shape[1]
        with torch.no_grad():
            s_rep = S[:, None, :].expand(M, K, S.shape[-1]).reshape(M * K, -1)
            pred = b.step(s_rep, A.reshape(M * K, -1), b.default_xi(M * K, dev)).reshape(M, K, -1)
            d_pred = pred - pred[:, :1]
            d_true = S1 - S1[:, :1]
        out = {}
        print(f"  [{name}] 组 {M} × {K}", flush=True)
        for k, (i, j) in blocks.items():
            e = (d_pred[..., i:j] - d_true[..., i:j]).abs().mean()
            r = d_true[..., i:j].abs().mean().clamp_min(1e-6)
            out[k] = float(e / r)
            print(f"     {k:<10} 归一化误差 {float(e/r):.3f}", flush=True)
        return out

    tr = eval_pairs(a.pairs_train, "训练集（A 半，见过）")
    print()
    ho = eval_pairs(a.pairs_heldout, "**留出集（B 半，没见过）**")
    print()
    print("[gain] 泛化比 = 留出误差 / 训练误差（≈1 好，≫1 说明增益不泛化）：")
    for k in blocks:
        print(f"     {k:<10} {ho[k] / max(tr[k], 1e-9):.2f}")
    ratios = [ho[k] / max(tr[k], 1e-9) for k in blocks]
    mean_ratio = sum(ratios) / len(ratios)
    print(f"[gain] ★ 平均泛化比 {mean_ratio:.2f}  "
          f"{'✅ 增益泛化' if mean_ratio < 1.5 else ('⚠ 部分泛化' if mean_ratio < 3 else '❌ 不泛化')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
