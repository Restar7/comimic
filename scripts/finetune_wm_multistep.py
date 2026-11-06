#!/usr/bin/env python3
"""**只续训 WM 的多步（k）损失**：从现成 ckpt 出发，只跑 k 步自回归。

为什么单独一个脚本：
    `pretrain_wm --resume-stage2` 只能从"阶段2 之后"续跑（会丢掉已有的 2b 进度）。
    而实测 2b 的 loss 在 1500 步时仍以每 250 步 ~14% 的速度下降（4.15→1.58），
    所以"再续训一小段、把裁判练得更准"是有收益的。这里从**最终 ckpt** 出发继续压 k 步误差。

用法：
    python3 scripts/finetune_wm_multistep.py \
        --init-wm runs/a_prime_sonic_leash/wm_ssm_s0.pt \
        --windows runs/a_prime_sonic_leash/roll_windows.pt \
        --steps 3000 --lr 1.5e-5 --batch 2048 --k 20 \
        --out runs/a_prime_sonic_leash/wm_ssm_s0_more.pt
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--init-wm", type=Path, required=True)
    ap.add_argument("--windows", type=Path, default=ROOT / "runs/a_prime_sonic_leash/roll_windows.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--mag-weight", type=float, default=10.0)
    ap.add_argument("--inc-weight", type=float, default=0.0,
                    help="动作可归因增量损失的权重（0=旧口径；修复 ∂s/∂a 增益用 1–3）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle
    from diff_vla_wbc.wm_refresh import train_multistep

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    ck = torch.load(a.init_wm, map_location="cpu", weights_only=False)
    bundle.wm.load_state_dict(ck.get("wm", ck), strict=True)
    print(f"[more] WM 初值来自 {a.init_wm}", flush=True)

    w = torch.load(a.windows, map_location="cpu", weights_only=False)
    S, A = w["states"].float().to(dev), w["actions"].float().to(dev)
    print(f"[more] 窗口 {tuple(S.shape)}（k={a.k}）steps={a.steps} batch={a.batch} lr={a.lr}", flush=True)
    t0 = time.time()
    out = train_multistep(bundle.wm, S, A, steps=int(a.steps), batch=int(a.batch),
                          lr=float(a.lr), k=int(a.k), mag_weight=float(a.mag_weight),
                          inc_weight=float(a.inc_weight),
                          device=dev, step_fn=bundle.step,
                          xi_fn=(lambda b: bundle.default_xi(b, dev)),
                          seed=int(a.seed), log=lambda m: print(m, flush=True))
    print(f"[more] 完成 {out} | {time.time()-t0:.0f}s", flush=True)
    bundle.save_checkpoint(a.out)
    print(f"[more] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
