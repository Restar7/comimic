#!/usr/bin/env python3
"""WM 刷新的**可复现 A/B**（2026-11-03）

问题：`wm_refresh.eval_wm_real` 的评测集是 `randperm(n)[:batch]` —— 采样取决于 n。
A' 数据量一变（547,714 → 548,226 行），"before" 那一列就换了评测集：同一条基线
`WBC_A24000_wmvel_cmd.pt` 两轮实测 qd_corr 0.421 / 0.363（差 0.06），
比我们想度量的改进幅度还大 ⇒ 跨轮次比较 "0.421→0.700 vs 0.363→0.512" **没有意义**。

本脚本用**同一份数据、同一个评测集（固定 seed + 大 batch）** 并排评多个 WM ckpt。
注意 `eval_wm_real` 是单步（t → t+1）评测，batch 要大才稳（默认 65536）。

用法：
    python scripts/eval_wm_refresh_ab.py \
        --data runs/a_prime_seg20/filtered.pt \
        --ckpts runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
                runs/a_prime_seg20/wm_pre.pt \
                runs/a_prime_seg20/wm_pre_full.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def load_bundle(ckpt: Path, device):
    """复用 pretrain_wm 的加载口径（Isaac/trackerlab ckpt 或旧 mjlab ckpt）。"""
    from diff_vla_wbc.isaac_joint import is_isaac_checkpoint, load_isaac_bundle

    if is_isaac_checkpoint(ckpt):
        npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
        return load_isaac_bundle(
            ckpt, device=device, npz=npz,
            packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
            obs_mode=None, hist=None, token_dim=64,
        ), "isaac"
    from diff_vla_wbc.mjlab_joint import load_mjlab_bundle  # noqa: PLC0415

    return load_mjlab_bundle(ckpt, device=device), "mjlab"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "runs/a_prime_seg20/filtered.pt")
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from diff_vla_wbc.wm_refresh import eval_wm_real

    data = torch.load(a.data, map_location="cpu", weights_only=False)
    n = int(data["states"].shape[0])
    print(f"[ab] data={a.data.name} rows={n} batch={a.batch} seed={a.seed}", flush=True)

    rows = []
    for ck in a.ckpts:
        ck = Path(ck)
        if not ck.exists():
            print(f"[ab] 跳过（不存在）{ck}")
            continue
        bundle, backend = load_bundle(ck, torch.device(a.device))
        m = eval_wm_real(data, bundle.wm, batch=a.batch, device=a.device,
                         seed=a.seed, step_fn=bundle.step)
        m["ckpt"] = ck.name
        m["backend"] = backend
        rows.append(m)
        print(f"[ab] {ck.name:34s} qd_corr={m['qd_corr']:.4f} qd_ratio={m['qd_ratio']:.4f} "
              f"q_mse={m['q_mse']:.3e} pos_mse={m['pos_mse']:.3e}", flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"data": str(a.data), "rows_total": n, "batch": a.batch,
                                     "seed": a.seed, "results": rows}, indent=2))
        print(f"[ab] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
