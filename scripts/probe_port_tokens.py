#!/usr/bin/env python3
"""Is the dataset's 64-d token the same space as the WBC's 64-d token?

``scripts/rls/supermimic/check_sonic_token.py`` reports

    [npz] window f+0..f+9: max |dToken| = 1.062e+00
    [FAIL] SONIC token port disagrees — do not train

i.e. rebuilding the token from (joint_pos, joint_vel, root_quat) through the SONIC port
does *not* reproduce the ``g1_motion_token_64`` stored in our HA npz.  Both are 64-d and
both come from ``model_encoder.onnx``, but they are different points of the space.

This script closes the loop for the joint finetune:

  1. rebuild the token with the **port** (what the WBC consumes at train time) for the
     exact eval windows the trainer uses;
  2. report MSE(port token, dataset token) -> should be ~0.5, the level ``token_mse``
     drifts to during joint finetuning;
  3. feed the port token into ``imagine_with_tokens`` -> if this is the token the WBC
     actually understands, reward/state_term should be near the reference ceiling
     (~0.09 / ~0.3) instead of 0.017 / 22.4 for the dataset token.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "trackerlab-supermimic/source/trackerLab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter  # noqa: E402
from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource  # noqa: E402
from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: E402
from diff_vla_wbc.rollout import compute_joint_loss, imagine_with_tokens  # noqa: E402
from diff_vla_wbc.train import JointTrainerConfig  # noqa: E402
from trackerLab.algorithms.supermimic.sonic_token import (  # noqa: E402
    SonicTokenBuilder, default_encoder_path,
)


class ShimLib:
    """Minimal shim: the SONIC port only needs q / qd / root_quat / last_frame_of."""

    def __init__(self, jp: torch.Tensor, jv: torch.Tensor, quat: torch.Tensor):
        self.q = jp
        self.qd = jv
        self._quat = quat
        self.n = jp.shape[0]

    def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.full_like(frame, self.n - 1)

    def root_quat(self, frames: torch.Tensor) -> torch.Tensor:
        return self._quat[frames]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_tokens/packed.pt")
    ap.add_argument("--supermimic-ckpt", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    files = sorted(args.motion.glob("*.npz"))
    bundle = load_isaac_bundle(
        args.supermimic_ckpt, device=device, npz=files, packed_motion=args.packed,
        packed_dir=None, obs_mode=None, hist=args.hist, token_dim=64,
    )
    library = bundle.library
    adapter = MotionGoalAdapter(AdapterConfig(
        token_dim=64, token_horizon=40, hidden=256, transformer_layers=2, transformer_heads=4,
    )).to(device)
    recorded = RecordedTokenSource(
        args.motion, library, token_horizon=adapter.cfg.token_horizon,
        token_dim=adapter.cfg.token_dim, device=device,
    )
    provider = MotionBatchProvider(
        bundle, library, batch_size=args.batch_size, horizon=40,
        token_source=None, token_dim=adapter.cfg.token_dim,
        token_horizon=adapter.cfg.token_horizon, seed=args.seed, hand_dir=None,
    )
    b = provider.sample()
    cfg = JointTrainerConfig(horizon=args.horizon)
    enc_path = default_encoder_path()
    print(f"[port] encoder = {enc_path}", flush=True)

    starts = [int(c.start) for c in library.clips]
    names = [c.name for c in library.clips]
    horizon = int(adapter.cfg.token_horizon)

    def port_tokens(frames) -> torch.Tensor:
        out = torch.zeros(len(frames), horizon, 64)
        for i, f in enumerate(frames):
            ci = max(bisect.bisect_right(starts, int(f)) - 1, 0)
            local = int(f) - starts[ci]
            with np.load(args.motion / f"{names[ci]}.npz") as d:
                jp = torch.from_numpy(np.asarray(d["joint_pos"], dtype=np.float32))
                jv = torch.from_numpy(np.asarray(d["joint_vel"], dtype=np.float32))
                quat = torch.from_numpy(np.asarray(d["body_quat_w"], dtype=np.float32)[:, 0])
            lib = ShimLib(jp, jv, quat)
            builder = SonicTokenBuilder(lib, "cpu", num_future=10)
            idx = torch.clamp(torch.arange(local, local + horizon), max=len(jp) - 1)
            tok = builder(idx, robot_quat=quat[idx].repeat_interleave(1, dim=0))
            out[i] = tok.float()
        return out.to(device)

    frames = b.frames.detach().cpu().tolist()
    ds_tok = recorded({"frames": b.frames})
    port_tok = port_tokens(frames)
    mse = float(((port_tok - ds_tok) ** 2).mean())
    print(f"[cmp] MSE(port token, dataset token) = {mse:.5f}   "
          f"(joint finetune plateaus at token_mse ~0.53)", flush=True)
    print(f"[cmp] port  std={float(port_tok.std()):.4f}  dataset std={float(ds_tok.std()):.4f}",
          flush=True)
    print(f"[cmp] 逐元素最大差 = {float((port_tok - ds_tok).abs().max()):.4f}", flush=True)

    def score(tok: torch.Tensor, label: str) -> None:
        with torch.no_grad():
            ro = imagine_with_tokens(
                bundle, tokens=tok, state=b.start_state, obs_hist=b.obs_hist, history=b.history,
                xi=b.xi, references=b.references, prev_action=b.prev_action,
                horizon=args.horizon, obs_noise=0.0,
            )
            _, terms = compute_joint_loss(
                ro, teacher_states=b.teacher_states, gamma=cfg.gamma, weights=cfg.weights,
            )
        print(json.dumps({
            "tokens": label,
            "rollout_reward": round(float(ro.rewards.mean()), 5),
            "state_term": round(float(terms.get("state", torch.zeros(()))), 4),
            "survival": round(float(ro.survival[-1].mean()), 5),
        }), flush=True)

    score(ds_tok, "dataset g1_motion_token_64")
    score(port_tok, "SONIC port rebuild (what WBC trained on)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
