#!/usr/bin/env python3
"""对拍两个世界模型在**同一批 HA 真转移**上的速度通道质量。

背景：v1 的标定 WM（`ckpt_013500_physint.pt`）在 BONES-SEED 上报过 `qd_corr 0.90`，
但它是在 `bones_seed_compact.pt` 域上标定的；我们现在用的是 HA 命令流域。
所以"移植 v1 的 WM 速度通道"这件事必须先验证：**换到 HA 真转移上它还准不准**。

做法（同一仪器、同一批样本，全部 no-grad）：
  1. 载入 C6h bundle → 取它的 WM；
  2. 在 `runs/wm_real/ha_transitions.pt` 上 `eval_wm_real`（pos/quat/q MSE、qd_corr、qd_ratio）；
  3. 把 v1 的 `wm` 权重灌进同一个模块 → 再测一次；
  4. 可选：对两者各做 N 步真转移刷新后再测（看谁受益更多）。

输出一行 JSON + 一张小表；不改任何训练代码。

用法：
    python3 scripts/probe_wm_graft.py
    python3 scripts/probe_wm_graft.py --refresh-steps 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--v1-ckpt", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500_physint.pt")
    ap.add_argument("--transitions", type=Path, default=ROOT / "runs/wm_real/ha_transitions.pt")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--refresh-steps", type=int, default=0,
                    help=">0 时对每个 WM 各做这么多步真转移刷新后再测（会改内存里的副本）")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)

    from diff_vla_wbc.isaac_joint import load_isaac_bundle           # noqa: PLC0415
    from diff_vla_wbc.rollout import freeze_world_model              # noqa: PLC0415
    from diff_vla_wbc.wm_refresh import eval_wm_real, refresh_wm     # noqa: PLC0415

    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    data = torch.load(str(a.transitions), map_location="cpu", weights_only=False)
    print(f"[graft] HA 真转移 {tuple(data['states'].shape)}", flush=True)

    def measure(tag: str) -> dict:
        bundle.wm.eval()
        m = eval_wm_real(data, bundle.wm, batch=a.batch, device=dev, seed=0)
        print(f"[graft] {tag:16s} pos={m['pos_mse']:.2e} quat={m['quat_mse']:.2e} "
              f"q={m['q_mse']:.2e} qd_corr={m['qd_corr']:+.4f} qd_ratio={m['qd_ratio']:.3f}", flush=True)
        return m

    out: dict[str, dict] = {}
    out["c6h"] = measure("C6h (现用)")

    # ---- 移植 v1 的 WM ----
    v1 = torch.load(str(a.v1_ckpt), map_location="cpu", weights_only=False)
    sd = v1["wm"]
    cur = bundle.wm.state_dict()
    miss = [k for k in cur if k not in sd]
    shape_bad = [k for k in cur if k in sd and cur[k].shape != sd[k].shape]
    if miss or shape_bad:
        print(f"[graft] ❌ 不兼容 miss={miss[:4]} shape_bad={shape_bad[:4]}", flush=True)
        return 2
    orig = {k: v.detach().clone() for k, v in cur.items()}
    bundle.wm.load_state_dict(sd, strict=True)
    print(f"[graft] 已灌入 v1 WM（{a.v1_ckpt.name}，{len(sd)} 个张量）", flush=True)
    out["v1"] = measure("v1 (移植)")

    if a.refresh_steps > 0:
        for tag, restore in (("c6h", True), ("v1", False)):
            if restore:
                bundle.wm.load_state_dict(orig, strict=True)
            m = refresh_wm(bundle.wm, data, steps=a.refresh_steps, batch=2048, lr=1e-4,
                           vel_weight=5.0, kl_anchor=1e-3, device=dev, log=lambda s: None)
            freeze_world_model(bundle.wm)
            bundle.wm.eval()
            mm = eval_wm_real(data, bundle.wm, batch=a.batch, device=dev, seed=0)
            out[f"{tag}_refresh{a.refresh_steps}"] = mm
            print(f"[graft] {tag}+refresh{a.refresh_steps} pos={mm['pos_mse']:.2e} "
                  f"q={mm['q_mse']:.2e} qd_corr={mm['qd_corr']:+.4f} qd_ratio={mm['qd_ratio']:.3f} "
                  f"(refresh: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if isinstance(v, (int, float))})})",
                  flush=True)
            bundle.wm.load_state_dict(sd if tag == "v1" else orig, strict=True)
            bundle.wm.eval()

    print("[graft] RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
