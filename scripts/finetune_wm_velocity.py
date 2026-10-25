#!/usr/bin/env python3
"""Offline fine-tune of the world model's *velocity* channel (no IsaacLab, no retrain from scratch).

Why: with reference actions the WM's position/quaternion/joint prediction is already exact
(1e-4 / 0.035 / 0.030), but its velocity output is action-insensitive and ~2.5x biased
(|qd| 1.24 at a=0 vs reference 0.48, corr 0.09).  The WM was trained one-step-ahead on PhysX
rollouts where the state's velocities are dominated by the exploration noise, so the velocity
head regressed to a mean field.  Positions are fine -> we only need to recalibrate the 35
velocity dims, which is a supervised regression on reference transitions.

Data:  reference transitions (S_t, a_t = dq_t, S_{t+1}) from the packed HA motion library --
       exactly the distribution where the position head is already correct.
Loss:  position/orientation/joint (kept, so nothing regresses) + lambda * normalised velocity
       MSE on (v, w, qd).
Check: corr(pred_qd, ref_qd) and |pred_qd|/|ref_qd| must go from (0.08, 2.5x) to (>0.9, ~1.0).

Usage:
    python3 scripts/finetune_wm_velocity.py --steps 600 --batch 4096
    python3 scripts/finetune_wm_velocity.py --ckpt .../ckpt_013500.pt --out .../ckpt_013500_wmvel.pt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "trackerlab-supermimic/source/trackerLab"))
from diff_vla_wbc.isaac_joint import load_isaac_bundle, canonical_to_state  # noqa: E402
from trackerLab.kinematics.supermimic.rotations import quat_to_mat, so3_log  # noqa: E402


def geodesic(R1, R2):
    return so3_log(R1.transpose(-1, -2) @ R2).square().mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_tokens/packed.pt")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500_wmvel.pt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vel-lambda", type=float, default=5.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-mode", choices=["dq", "dq_over_scale"], default="dq",
                    help="dq: a=Δq（本工程当前口径）；dq_over_scale: a=Δq/action_scale（原始 code 口径）")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    bundle = load_isaac_bundle(a.ckpt, device=dev, npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
                               packed_motion=a.packed, packed_dir=None, obs_mode=None, hist=10, token_dim=64)
    motion = bundle.motion
    wm = bundle.wm
    default_q = bundle.default_joint_pos
    ref = motion.ref
    T = ref.shape[0]
    clip_lo = motion.clip_start[motion.clip_id[torch.arange(T, device=ref.device)]]
    clip_hi = motion.clip_end[motion.clip_id[torch.arange(T, device=ref.device)]]

    def sample(n):
        idx = torch.randint(0, T - 1, (n,), device=ref.device)
        hi = clip_hi[idx]
        idx = torch.minimum(idx, hi - 1)
        nxt = idx + 1
        rows = ref[idx].float().to(dev), ref[nxt].float().to(dev)
        S = canonical_to_state(rows[0])
        S1 = canonical_to_state(rows[1])
        dq = rows[1][..., 13:13 + 29] - default_q
        if a.action_mode == "dq_over_scale":
            dq = dq / bundle.action_scale
        xi = torch.zeros(n, 33, device=dev)
        return S, dq, S1

    def velocity_stats(n=8192):
        S, _, S1 = sample(n)
        return S1[2].std(0).mean(), S1[3].std(0).mean(), S1[5].std(0).mean()

    sv, sw, sq = velocity_stats()
    wv, ww, wq = 1.0 / sv**2, 1.0 / sw**2, 1.0 / sq**2
    print(f"[stats] ref velocity std: v={float(sv):.4f} w={float(sw):.4f} qd={float(sq):.4f}")

    def evaluate(n=8192):
        S, dq, S1 = sample(n)
        with torch.no_grad():
            out = wm.step(S, dq, torch.zeros(n, 33, device=dev))
        err = out[5] - S1[5]
        corr = float(((out[5] - out[5].mean()) * (S1[5] - S1[5].mean())).mean()
                     / (out[5].std() * S1[5].std() + 1e-9))
        return {"qd_corr": corr,
                "qd_ratio": float(out[5].abs().mean() / (S1[5].abs().mean() + 1e-9)),
                "pos_mse": float((out[0] - S1[0]).square().mean()),
                "joint_mse": float((out[4] - S1[4]).square().mean()),
                "qd_mse": float(err.square().mean())}

    print("[before] " + json.dumps({k: round(v, 4) for k, v in evaluate().items()}))

    params = list(wm.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    for it in range(1, a.steps + 1):
        S, dq, S1 = sample(a.batch)
        out = wm.step(S, dq, torch.zeros(a.batch, 33, device=dev))
        loss_pos = (out[0] - S1[0]).square().mean() + geodesic(out[1], S1[1]) + (out[4] - S1[4]).square().mean()
        loss_vel = (wv * (out[2] - S1[2]).square().mean()
                    + ww * (out[3] - S1[3]).square().mean()
                    + wq * (out[5] - S1[5]).square().mean())
        loss = loss_pos + a.vel_lambda * loss_vel
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % max(1, a.steps // 6) == 0:
            print(f"  [{it:>4}] " + json.dumps({k: round(v, 4) for k, v in evaluate(2048).items()}), flush=True)

    after = evaluate()
    print("[after]  " + json.dumps({k: round(v, 4) for k, v in after.items()}))
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ck = dict(ck)
    ck["wm"] = {k: v.detach().cpu() for k, v in wm.state_dict().items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out)
    print(f"[saved] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
