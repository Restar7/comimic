#!/usr/bin/env python3
"""Scan the frame offset between the token chunk and the reference/state window."""
import argparse, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter
from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource
from diff_vla_wbc.isaac_joint import load_isaac_bundle
from diff_vla_wbc.rollout import compute_joint_loss, imagine_with_tokens
from diff_vla_wbc.train import JointTrainerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--motion", type=Path, default=ROOT/"data/npz/ha_tokens")
ap.add_argument("--packed", type=Path, default=ROOT/"runs/isaac_motion/ha_tokens/packed.pt")
ap.add_argument("--supermimic-ckpt", type=Path, default=ROOT/"trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt")
ap.add_argument("--km", type=int, default=20, help="max |shift|")
ap.add_argument("--horizon", type=int, default=8)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = torch.device(a.device)
files = sorted(a.motion.glob("*.npz"))
bundle = load_isaac_bundle(a.supermimic_ckpt, device=dev, npz=files, packed_motion=a.packed,
                           packed_dir=None, obs_mode=None, hist=10, token_dim=64)
lib = bundle.library
ad = MotionGoalAdapter(AdapterConfig(token_dim=64, token_horizon=40, hidden=256,
                                     transformer_layers=2, transformer_heads=4)).to(dev)
rec = RecordedTokenSource(a.motion, lib, token_horizon=40, token_dim=64, device=dev)
prov = MotionBatchProvider(bundle, lib, batch_size=2, horizon=40, token_source=None,
                           token_dim=64, token_horizon=40, seed=12345, hand_dir=None)
b = prov.sample()
cfg = JointTrainerConfig(horizon=a.horizon)

def score(tok):
    with torch.no_grad():
        ro = imagine_with_tokens(bundle, tokens=tok, state=b.start_state, obs_hist=b.obs_hist,
                                 history=b.history, xi=b.xi, references=b.references,
                                 prev_action=b.prev_action, horizon=a.horizon, obs_noise=0.0)
        _, t = compute_joint_loss(ro, teacher_states=b.teacher_states, gamma=cfg.gamma, weights=cfg.weights)
    return float(ro.rewards.mean()), float(t.get("state", torch.zeros(())))

print(f"{'shift':>6} {'reward':>9} {'state_term':>11}")
best = None
for k in range(-a.km, a.km + 1):
    fr = torch.clamp(b.frames + k, min=0)
    tok = rec({"frames": fr})
    r, s = score(tok)
    flag = ""
    if best is None or r > best[0]:
        best = (r, s, k); flag = " *"
    print(f"{k:>6} {r:>9.5f} {s:>11.4f}{flag}")
print(f"\nbest: shift={best[2]} reward={best[0]:.5f} state={best[1]:.4f}")
print(f"reference: trained-VLA 0.0742/0.99 ; mjlab recorded 0.096/0.183")
