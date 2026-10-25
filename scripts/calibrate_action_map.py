#!/usr/bin/env python3
"""Calibrate the reference-joint-target -> WM action map empirically.

The WM's action input is not `q_target - q_default` up to a per-joint scale (measured:
inverted a* has |a*|/|dq| ~ 5 and corr(a*, dq) ~ 0.23), so no closed-form guess converged.
This fits `a = dq @ W^T + b` by (1) inverting the WM for the action that hits the reference
next state, (2) least-squares. Output: runs/ablations/action_map.pt
"""
import argparse, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diff_vla_wbc.isaac_joint import load_isaac_bundle
from diff_vla_wbc.data import MotionBatchProvider

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/ablations/action_map.pt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    files = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt",
                               device=dev, npz=files, packed_motion=ROOT / "runs/isaac_motion/ha_tokens/packed.pt",
                               packed_dir=None, obs_mode=None, hist=10, token_dim=64)
    lib = bundle.library
    prov = MotionBatchProvider(bundle, lib, batch_size=a.samples, horizon=40, token_source=None,
                               token_dim=64, token_horizon=40, seed=12345, hand_dir=None)
    b = prov.sample()
    motion = bundle.motion
    def dq(frames):
        r = motion.ref[frames.long()].float()
        return r[..., 13:13+29] - bundle.default_joint_pos - b.xi[..., 1:1+29]
    x = dq(b.frames + 1)                      # 参考：下一帧相对默认位姿的偏移
    target = b.teacher_states[:, 0]           # 下一帧参考状态
    act = torch.zeros_like(x).requires_grad_(True)
    opt = torch.optim.Adam([act], lr=0.2)
    for _ in range(a.iters):
        nxt = bundle.step(b.start_state, act, b.xi)
        loss = (((nxt[..., 0:7] - target[..., 0:7]) ** 2).mean()
                + ((nxt[..., 7:36] - target[..., 7:36]) ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    star = act.detach()
    print(f"[invert] pos/quat/joint loss={float(loss):.6f}  |a*|={float(star.abs().mean()):.3f} |dq|={float(x.abs().mean()):.3f}")
    X = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1)          # (N,30)
    W = torch.linalg.lstsq(X, star).solution                                    # (30,29)
    pred = X @ W
    ss_res = float(((star - pred) ** 2).mean()); ss_tot = float(((star - star.mean(0)) ** 2).mean())
    print(f"[fit] R2 = {1 - ss_res / ss_tot:.4f}   rel_err = {(ss_res ** 0.5) / float(star.abs().mean()):.4f}")
    # 用拟合映射在**参考动作**轨迹上验证
    A, bb = W[:29].T, W[29]
    acts = torch.stack([dq(b.frames + t + 1) @ A.T + bb for t in range(8)], 1)
    from diff_vla_wbc.rollout import _call_robot_bodies, _state_error
    state, xi, a_prev = b.start_state, b.xi, b.prev_action
    states, rews = [state], []
    for t in range(8):
        ref = b.references[t]; act_t = acts[:, t]
        nxt = bundle.step(state, act_t, xi); bodies = _call_robot_bodies(bundle, nxt)
        terms = bundle.reward(nxt, act_t, a_prev, ref, bodies)
        rews.append(terms["total"]); states.append(nxt); a_prev = act_t; state = nxt
    st = torch.stack(states)[1:9]
    tgt = b.teacher_states[:, :8].permute(1, 0, 2)
    print(f"[verify] mapped ref-actions: reward={float(torch.stack(rews).mean()):+.5f} "
          f"state_term={float(_state_error(st, tgt)):.4f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"A": A.cpu(), "b": bb.cpu()}, a.out)
    print(f"[saved] {a.out}")

if __name__ == "__main__":
    raise SystemExit(main())
