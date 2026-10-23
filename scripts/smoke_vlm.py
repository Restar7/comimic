#!/usr/bin/env python3
"""Smoke-test the top-level VLM: image + instruction -> FM action chunk -> goal.

Loads the real Cosmos-Reason2-2B backbone (local weights only), builds the GR00T
N1.7 flow-matching head, and walks the top of the joint graph:

    RGB + instruction + state -> backbone -> head (Euler, grad enabled)
        -> tokens [B,40,64] -> MotionGoalAdapter -> goal [B,10,68]

Gradients are checked on the head and (optionally) the VLM.  This is a *wiring*
test with real VLM weights: the action head itself is randomly initialised
unless ``--gr00t-ckpt`` points at a GR00T checkpoint.

Usage:
    python scripts/smoke_vlm.py --device cuda:0
    python scripts/smoke_vlm.py --device cpu --select-layer 2 --max-state-dim 43
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
from diff_vla_wbc.gr00t_vlm import (  # noqa: E402
    Gr00tVLM,
    describe_vlm,
    resolve_vlm_source,
    vlm_available,
)


def synthetic_image(size: int = 256) -> torch.Tensor:
    """A deterministic gradient-ish image so repeated runs match."""
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, size), torch.linspace(0.0, 1.0, size), indexing="ij"
    )
    return torch.stack((x, y, (x + y) / 2.0), dim=0).unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", type=Path, default=None, help="local VLM directory (default: auto-resolve)")
    ap.add_argument("--gr00t-ckpt", type=Path, default=None, help="optional full GR00T checkpoint")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--select-layer", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=2, help="Euler steps for the sampled chunk")
    ap.add_argument("--max-state-dim", type=int, default=43, help="G1 state width")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--instruction", default="Pick up the carton and hand it over.")
    ap.add_argument("--train-vlm", action="store_true", help="also unfreeze the VLM backbone")
    ap.add_argument("--flash-attention", action="store_true")
    args = ap.parse_args()

    if not vlm_available(args.vlm) and args.vlm is None:
        print(json.dumps({"error": "no local Cosmos-Reason2-2B weights found"}), flush=True)
        return 2
    print(json.dumps({"resolved_vlm": resolve_vlm_source(args.vlm)}), flush=True)

    vlm = Gr00tVLM.from_pretrained(
        args.vlm,
        checkpoint=args.gr00t_ckpt,
        select_layer=args.select_layer,
        device=args.device,
        action_horizon=args.horizon,
        max_state_dim=args.max_state_dim,
        freeze_vlm=not args.train_vlm,
        use_flash_attention=args.flash_attention,
    )
    print(json.dumps(describe_vlm(vlm), indent=2), flush=True)

    state = torch.zeros(1, args.max_state_dim)
    batch = vlm.build_batch(synthetic_image(), [args.instruction], state)
    print(
        json.dumps({key: tuple(value.shape) for key, value in batch.items() if torch.is_tensor(value)}),
        flush=True,
    )

    # Run the differentiable FM sampler explicitly so num_steps stays visible.
    from diff_vla_wbc.gr00t_fm import sample_model_with_grad

    tokens = sample_model_with_grad(vlm, batch, num_steps=args.num_steps)
    print(
        json.dumps(
            {
                "tokens": tuple(tokens.shape),
                "tokens_dtype": str(tokens.dtype),
                "requires_grad": bool(tokens.requires_grad),
            }
        ),
        flush=True,
    )

    # The adapter keeps its goals in fp32 on purpose (`tokens.float()` inside
    # `MotionGoalAdapter.forward`), so only the device moves.
    adapter = MotionGoalAdapter(
        AdapterConfig(token_dim=int(tokens.shape[-1]), token_horizon=int(tokens.shape[1]))
    ).to(tokens.device)
    goal = adapter(tokens)
    print(json.dumps({"goal": tuple(goal.shape), "goal_dtype": str(goal.dtype)}), flush=True)

    loss = tokens.float().square().mean() + goal.float().square().mean()
    loss.backward()
    head_grad = sum(
        float(p.grad.abs().sum()) for p in vlm.action_head.parameters() if p.grad is not None
    )
    vlm_grad = sum(float(p.grad.abs().sum()) for p in vlm.backbone.parameters() if p.grad is not None)
    print(
        json.dumps(
            {
                "loss": float(loss.detach()),
                "action_head_grad_abs_sum": head_grad,
                "backbone_grad_abs_sum": vlm_grad,
                "vlm_frozen": not args.train_vlm,
            }
        ),
        flush=True,
    )
    ok = head_grad > 0.0 and (vlm_grad > 0.0 or not args.train_vlm)
    print("SMOKE OK" if ok else "SMOKE FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
