#!/usr/bin/env python3
"""Probe the *shape* of the Isaac-bundle reward landscape in token space.

Why: the pure-HA joint finetune improves ``rollout_reward`` (0.014 -> 0.074) while
``token_mse`` against the recorded token explodes (0.069 -> 0.53).  Two very different
explanations produce that pattern:

  (a) the reward landscape genuinely prefers tokens far away from the recorded ones
      (reward hacking / off-manifold exploitation), or
  (b) the recorded-token ceiling of *this* bundle is itself ~0.075, i.e. the WM simply
      cannot be driven to the reference states and 0.074 already is the maximum.

This script answers it without touching the running trainer (bundle only: ~150 MB of
GPU memory, no VLA):

  1. feed the **recorded** token of the very same fixed eval windows -> true ceiling
     (cross-check against the 0.0994 recorded in ``docs/ABLATION_A4_RUN.md``);
  2. scale the recorded token by alpha in [0, 3] and re-measure reward / state_term;
     if reward *increases* with alpha while tracking quality stays flat, the gradient
     that joint finetuning follows is "make the token bigger", not "make it correct";
  3. add isotropic noise of matched magnitude as a control (drift vs scale).

Everything below reuses the exact objects the trainer uses (same bundle loader, same
``RecordedTokenSource``, same ``MotionBatchProvider`` seed=12345, same
``imagine_with_tokens`` / ``compute_joint_loss``).

Usage:
    python3 scripts/probe_reward_landscape.py --device cuda
    python3 scripts/probe_reward_landscape.py --alphas 0,0.5,1,1.5,2,3 --eval-batches 4
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

PREFIX_STEPS = 8  # same fixed prefix the trainer reports as @8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_tokens/packed.pt")
    ap.add_argument("--supermimic-ckpt", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=2)
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345,
                    help="同 train_joint 的评测提供器种子，保证是同一批窗口")
    ap.add_argument("--alphas", default="0,0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/ablations/probe_reward_landscape.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    files = sorted(args.motion.glob("*.npz"))
    if not files:
        raise SystemExit(f"no npz under {args.motion}")

    print(f"[probe] bundle <- {args.supermimic_ckpt}", flush=True)
    bundle = load_isaac_bundle(
        args.supermimic_ckpt, device=device, npz=files,
        packed_motion=args.packed, packed_dir=None,
        obs_mode=None, hist=args.hist, token_dim=64,
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
    batches = [provider.sample() for _ in range(args.eval_batches)]
    cfg = JointTrainerConfig(horizon=args.horizon)
    print(f"[probe] {len(batches)} fixed eval batches (batch_size={args.batch_size}, "
          f"rollout horizon={args.horizon})", flush=True)

    rows: list[dict] = []

    def measure(label: str, token_fn) -> None:
        tm, st, rw, rw8, sv = [], [], [], [], []
        for i, b in enumerate(batches):
            rec = recorded({"frames": b.frames})
            tok = token_fn(rec, i)
            n = min(tok.shape[1], rec.shape[1])
            tm.append(float(((tok[:, :n] - rec[:, :n]) ** 2).mean()))
            with torch.no_grad():
                ro = imagine_with_tokens(
                    bundle, tokens=tok, state=b.start_state, obs_hist=b.obs_hist,
                    history=b.history, xi=b.xi, references=b.references,
                    prev_action=b.prev_action, horizon=args.horizon, obs_noise=0.0,
                )
                _, terms = compute_joint_loss(
                    ro, teacher_states=b.teacher_states, gamma=cfg.gamma, weights=cfg.weights,
                )
            st.append(float(terms.get("state", torch.zeros(()))))
            rw.append(float(ro.rewards.mean()))
            pref = min(PREFIX_STEPS, ro.rewards.shape[0])
            rw8.append(float(ro.rewards[:pref].mean()))
            sv.append(float(ro.survival[-1].mean()))
        m = {
            "label": label,
            "token_mse": sum(tm) / len(tm),
            "state_term": sum(st) / len(st),
            "rollout_reward": sum(rw) / len(rw),
            "rollout_reward@8": sum(rw8) / len(rw8),
            "survival": sum(sv) / len(sv),
        }
        rows.append(m)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}),
              flush=True)

    # 1) 录制 token = 上界参照（alpha = 1）
    for a in [float(x) for x in args.alphas.split(",") if x.strip()]:
        measure(f"recorded*{a:g}", lambda rec, _i, a=a: rec * a)

    # 2) 同量级噪声（drift 控制组）：scale=1，噪声 std 与录制 token 同量级
    for sigma in (0.1, 0.3, 0.5):
        measure(f"recorded+noise{sigma:g}",
                lambda rec, i, s=sigma: rec + s * torch.randn_like(rec))

    out = {
        "bundle": str(args.supermimic_ckpt),
        "packed": str(args.packed),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "eval_batches": len(batches),
        "horizon": args.horizon,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"[probe] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
