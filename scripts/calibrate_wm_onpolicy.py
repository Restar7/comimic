#!/usr/bin/env python3
"""用 **on-policy 真转移**（我们自己的 WBC 在 Isaac 里跑出来的）重标定 WM。

与 `refresh_wm_ckpt.py` 的区别：
  * 数据换成 `scripts/isaac_g1_collect.py` + `scripts/compute_collect_reward.py` 产出的
    on-policy 转移（**我们 WBC 的动作 + 我们 WBC 走出的状态**）；
  * 按 **C6 的标定协议**（600–800 步、lr 1e-4、kl_anchor 1e-3），而不是 in-loop 的 200 步；
  * 输出新 ckpt，供 `eval_wm_velocity.py` 用**同一仪器**验收（目标 qd_corr ≥ 0.85）。

反面案例（必须避免）：用 `runs/wm_real/ha_transitions.pt`（HA sonic 的执行轨迹）刷新，
在正确仪器上 `qd_corr 0.752 → 0.589` —— 域不一致会**洗坏**标定。

用法：
    python3 scripts/calibrate_wm_onpolicy.py \
        --inputs 'runs/wm_real/onpolicy_gt_*.pt' \
        --out runs/wm_real/C6h_onpolicy.pt --steps 800
    # 验收（同一仪器）：
    python3 scripts/eval_wm_velocity.py --ckpt runs/wm_real/C6h_onpolicy.pt \
        --packed runs/isaac_motion/ha_cmd/packed.pt --tokens label \
        --samples 32 --batch 32 --tag C6h_onpolicy
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True,
                    help="glob，例如 'runs/wm_real/onpolicy_gt_*.pt'（需含 states/actions/next_states）")
    ap.add_argument("--base-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/C6h_onpolicy.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-anchor", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle      # noqa: PLC0415
    from diff_vla_wbc.wm_refresh import eval_wm_real, refresh_wm  # noqa: PLC0415

    dev = torch.device(a.device)
    files = sorted(glob.glob(a.inputs))
    if not files:
        raise SystemExit(f"[calib] 没有匹配 {a.inputs}")
    S, A, S2 = [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S.append(d["states"].float()); A.append(d["actions"].float())
        S2.append(d["next_states"].float())
        print(f"[calib] + {Path(f).name}: n={d['states'].shape[0]} clip={d.get('clip')}", flush=True)
    data = {"states": torch.cat(S), "actions": torch.cat(A), "next_states": torch.cat(S2)}
    n = int(data["states"].shape[0])
    print(f"[calib] 合计 {n} 条 on-policy 转移（{len(files)} 个文件）", flush=True)

    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.base_ckpt, device=dev, npz=npz,
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    before = eval_wm_real(data, bundle.wm, batch=8192, device=dev, seed=0)
    print(f"[calib] before {json.dumps({k: round(v, 5) for k, v in before.items()})}", flush=True)

    m = refresh_wm(bundle.wm, data, steps=int(a.steps), batch=int(a.batch), lr=float(a.lr),
                   kl_anchor=float(a.kl_anchor), device=dev,
                   log=lambda s: print("[calib]", s, flush=True))
    after = eval_wm_real(data, bundle.wm, batch=8192, device=dev, seed=0)
    print(f"[calib] after  {json.dumps({k: round(v, 5) for k, v in after.items()})}", flush=True)

    ck = torch.load(str(a.base_ckpt), map_location="cpu", weights_only=False)
    ck["wm"] = {k: v.detach().cpu().clone() for k, v in bundle.wm.state_dict().items()}
    prov = ck.setdefault("provenance", {})
    prov["wm_from"] = str(a.base_ckpt)
    prov["wm_calibrated_on"] = "on-policy Isaac rollouts of our own WBC"
    prov["wm_calib_inputs"] = files
    prov["wm_calib_steps"] = int(a.steps)
    prov["wm_calib_before"] = {k: float(v) for k, v in before.items()}
    prov["wm_calib_after"] = {k: float(v) for k, v in after.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(a.out))
    print(f"[calib] saved {a.out} ({a.out.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"[calib] 下一步验收（同一仪器）：\n"
          f"  python3 scripts/eval_wm_velocity.py --ckpt {a.out} "
          f"--packed runs/isaac_motion/ha_cmd/packed.pt --tokens label "
          f"--samples 32 --batch 32 --tag onpolicy", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
