#!/usr/bin/env python3
"""Acceptance test: with the corrected reference-action convention (a = dq, no /action_scale),
do the *recorded* tokens finally track the reference in the offline bundle?"""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter
from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource
from diff_vla_wbc.isaac_joint import (load_isaac_bundle, canonical_to_state)
from diff_vla_wbc.rollout import (compute_joint_loss, imagine_with_tokens)
from diff_vla_wbc.train import JointTrainerConfig

dev = torch.device("cuda")
files = sorted(Path("data/npz/ha_tokens").glob("*.npz"))
bundle = load_isaac_bundle(Path("trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt"),
                           device=dev, npz=files, packed_motion=Path("runs/isaac_motion/ha_tokens/packed.pt"),
                           packed_dir=None, obs_mode=None, hist=10, token_dim=64)
lib = bundle.library
ad = MotionGoalAdapter(AdapterConfig(token_dim=64, token_horizon=40, hidden=256,
                                     transformer_layers=2, transformer_heads=4)).to(dev)
rec = RecordedTokenSource(Path("data/npz/ha_tokens"), lib, token_horizon=40, token_dim=64, device=dev)
prov = MotionBatchProvider(bundle, lib, batch_size=2, horizon=40, token_source=None,
                           token_dim=64, token_horizon=40, seed=12345, hand_dir=None)
b = prov.sample()
H = 8
cfg = JointTrainerConfig(horizon=H)
motion = bundle.motion
HIST = 10

def corrected_history(frames):
    """Same recipe as IsaacBundle.reference_start but a = dq (no /action_scale)."""
    offsets = torch.arange(HIST, device=dev) - (HIST - 1)
    clip_first = motion.clip_start[motion.clip_id[frames]]
    hf = (frames[:, None] + offsets[None, :]).clamp(min=clip_first[:, None])
    rows = motion.ref[hf.reshape(-1).to(motion.ref.device)].float().reshape(frames.shape[0], HIST, -1)
    q_ref = rows[..., 13:13+29]
    bias = b.xi[..., 1:1+29]
    a_ref = q_ref - bundle.default_joint_pos - bias[:, None]           # <-- 不再除以 action_scale
    a_prev = torch.cat((torch.zeros_like(a_ref[:, :1]), a_ref[:, :-1]), dim=1)
    xi_hist = b.xi[:, None].expand(frames.shape[0], HIST, -1)
    rows_flat = rows.reshape(-1, rows.shape[-1])
    obs = bundle.mdp.obs(canonical_to_state(rows_flat), a_prev.reshape(-1, 29),
                         xi_hist.reshape(-1, xi_hist.shape[-1]), hf.reshape(-1)).reshape(
        frames.shape[0], HIST, -1)
    from diff_vla_wbc.isaac_joint import flatten_term_major
    return obs.permute(1, 0, 2).contiguous(), a_prev[:, -1]

hist_fix, prev_fix = corrected_history(b.frames)
print(f"[check] prev_action |orig|={float(b.prev_action.abs().mean()):.3f} "
      f"|fixed|={float(prev_fix.abs().mean()):.3f}")
tok = rec({"frames": b.frames})
for name, hist, prev in [("orig (a/scale)", b.history, b.prev_action),
                         ("fixed (a=dq)", hist_fix, prev_fix)]:
    with torch.no_grad():
        ro = imagine_with_tokens(bundle, tokens=tok, state=b.start_state, obs_hist=None,
                                 history=hist, xi=b.xi, references=b.references,
                                 prev_action=prev, horizon=H, obs_noise=0.0)
        _, terms = compute_joint_loss(ro, teacher_states=b.teacher_states, gamma=cfg.gamma, weights=cfg.weights)
    print(f"[recorded token | {name}] reward={float(ro.rewards.mean()):+.5f} "
          f"state_term={float(terms['state']):.4f}")
