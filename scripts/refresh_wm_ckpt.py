#!/usr/bin/env python3
"""用真转移刷新一个 ckpt 里的 WM，并另存为新 ckpt（不改原文件）。

用途：验证"WM 保鲜到底是有益还是有害"。
我们之前唯一的判据是 `eval_wm_real`（在 ha_transitions 上算 qd_corr），
而 C6h 的 WM 是按 `eval_wm_velocity.py`（ha_cmd 库 + label token + policy rollout）标定的。
两个仪器不一致 ⇒ 无法判断刷新有没有把标定洗掉。这个脚本把刷新结果存成新 ckpt，
之后就能用**同一个仪器**（eval_wm_velocity）前后对比。

用法：
    python3 scripts/refresh_wm_ckpt.py --steps 200 --out /tmp/C6h_ref200.pt
    python3 scripts/eval_wm_velocity.py --ckpt /tmp/C6h_ref200.pt --packed runs/isaac_motion/ha_cmd/packed.pt --tokens label --tag C6h_ref200
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
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--transitions", type=Path, default=ROOT / "runs/wm_real/ha_transitions.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-anchor", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("/tmp/C6h_refresh.pt"))
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = torch.device(a.device)

    from diff_vla_wbc.isaac_joint import load_isaac_bundle       # noqa: PLC0415
    from diff_vla_wbc.wm_refresh import eval_wm_real, refresh_wm  # noqa: PLC0415

    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.ckpt, device=dev, npz=npz,
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    data = torch.load(str(a.transitions), map_location="cpu", weights_only=False)
    before = eval_wm_real(data, bundle.wm, batch=8192, device=dev, seed=0)
    print(f"[refresh] before  {json.dumps({k: round(v, 5) for k, v in before.items()})}", flush=True)
    m = refresh_wm(bundle.wm, data, steps=a.steps, lr=a.lr, kl_anchor=a.kl_anchor,
                   device=dev, log=lambda s: print("[refresh]", s, flush=True))
    after = eval_wm_real(data, bundle.wm, batch=8192, device=dev, seed=0)
    print(f"[refresh] after   {json.dumps({k: round(v, 5) for k, v in after.items()})}", flush=True)
    print(f"[refresh] inner   {json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()})}", flush=True)

    ck = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    ck["wm"] = {k: v.detach().cpu().clone() for k, v in bundle.wm.state_dict().items()}
    ck.setdefault("provenance", {})["wm_refreshed_from"] = str(a.ckpt)
    ck["provenance"]["wm_refresh_steps"] = int(a.steps)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(a.out))
    print(f"[refresh] saved {a.out}  ({a.out.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
