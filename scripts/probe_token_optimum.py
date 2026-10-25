#!/usr/bin/env python3
"""Where does THIS bundle's reward actually want the token to be?

No VLA, no trainer: take the fixed eval windows (same provider/seed the trainer uses),
initialise the 64-d token chunk and do plain gradient ascent on ``rollout_reward``.
Then compare the optimum against the recorded token:

  * optimum close to the recording  -> the reward route is sane; the VLA run drifting to
    token_mse 0.53 would be a *training* bug (gradient path / optimiser / loss);
  * optimum far from the recording (mse ~0.5) and reward ~0.075 -> the landscape itself
    prefers off-manifold tokens, i.e. the joint finetune is faithfully finding a spurious
    optimum of this bundle (model-based reward hacking / convention mismatch), not a bug
    in ``train_joint.py``.

Two starts (recorded and noise) show whether the basin is unique.

Usage:
    python3 scripts/probe_token_optimum.py --steps 300
    python3 scripts/probe_token_optimum.py --steps 300 --start noise --lr 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter  # noqa: E402
from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource  # noqa: E402
from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: E402
from diff_vla_wbc.rollout import compute_joint_loss, imagine_with_tokens  # noqa: E402
from diff_vla_wbc.train import JointTrainerConfig  # noqa: E402


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
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--start", choices=["recorded", "noise"], default="recorded")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/ablations/probe_token_optimum.json")
    args = ap.parse_args()

    torch.manual_seed(0)
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
    rec = recorded({"frames": b.frames})
    cfg = JointTrainerConfig(horizon=args.horizon)

    if args.start == "recorded":
        tok0 = rec.clone()
    else:
        tok0 = torch.randn_like(rec)
    tok = tok0.clone().requires_grad_(True)
    opt = torch.optim.Adam([tok], lr=args.lr)

    def evaluate(t: torch.Tensor) -> dict:
        with torch.no_grad():
            ro = imagine_with_tokens(
                bundle, tokens=t, state=b.start_state, obs_hist=b.obs_hist, history=b.history,
                xi=b.xi, references=b.references, prev_action=b.prev_action,
                horizon=args.horizon, obs_noise=0.0,
            )
            _, terms = compute_joint_loss(
                ro, teacher_states=b.teacher_states, gamma=cfg.gamma, weights=cfg.weights,
            )
        return {
            "reward": float(ro.rewards.mean()),
            "state_term": float(terms.get("state", torch.zeros(()))),
            "token_mse": float(((t - rec) ** 2).mean()),
            "token_std": float(t.std()),
        }

    print(f"[start] {json.dumps({k: round(v, 5) for k, v in evaluate(tok.detach()).items()})}", flush=True)
    hist = []
    for i in range(1, args.steps + 1):
        ro = imagine_with_tokens(
            bundle, tokens=tok, state=b.start_state, obs_hist=b.obs_hist, history=b.history,
            xi=b.xi, references=b.references, prev_action=b.prev_action,
            horizon=args.horizon, obs_noise=0.0,
        )
        loss = -ro.rewards.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % max(1, args.steps // 10) == 0 or i == 1:
            m = evaluate(tok.detach())
            m["iter"] = i
            hist.append(m)
            print(f"[{i:>4}] " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                             for k, v in m.items()}), flush=True)

    final = evaluate(tok.detach())
    out = {
        "bundle": str(args.supermimic_ckpt), "start": args.start, "steps": args.steps,
        "lr": args.lr, "horizon": args.horizon, "seed": args.seed,
        "final": final, "history": hist,
        "token_dump": str(args.out.with_suffix(".pt")),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    torch.save({"tokens": tok.detach().cpu(), "recorded": rec.cpu(), "frames": b.frames.cpu()},
               args.out.with_suffix(".pt"))
    print("[final] " + json.dumps({k: round(v, 5) for k, v in final.items()}) + f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
