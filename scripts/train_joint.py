#!/usr/bin/env python3
"""Joint training / gradient smoke CLI for the differentiable framework.

The default run uses random FM tokens and the tiny real GR00T action head is
optional (``--real-gr00t-head``).  For a real experiment, plug a
``Gr00tTokenSource`` wrapping a loaded ``Gr00tN1d7`` and a token source that
returns the paired tokens for each motion window; the rest of the loop is
unchanged.
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
from diff_vla_wbc.data import MotionBatchProvider, sample_reference_windows  # noqa: E402
from diff_vla_wbc.gr00t_fm import sample_model_with_grad  # noqa: E402
from diff_vla_wbc.supermimic_backend import (  # noqa: E402
    SuperMimicBundle,
    build_g1_assets,
    load_motion_library,
)
from diff_vla_wbc.train import JointTrainer, JointTrainerConfig  # noqa: E402


def collect_npz(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.npz")))
        elif p.suffix == ".npz":
            out.append(p)
        else:
            raise SystemExit(f"not an npz file or directory: {p}")
    if not out:
        raise SystemExit("no .npz motion clips found")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/joint_smoke"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--small", action="store_true", help="tiny SuperMimic for CPU smoke runs")
    ap.add_argument("--supermimic-ckpt", type=Path, default=None)
    ap.add_argument("--adapter-lr", type=float, default=1e-4)
    ap.add_argument("--policy-lr", type=float, default=1e-5)
    ap.add_argument("--fm-lr", type=float, default=1e-5)
    ap.add_argument("--pretrain-wm", type=int, default=0)
    ap.add_argument("--pretrain-wm-batch", type=int, default=8)
    ap.add_argument("--pretrain-wm-window", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--real-gr00t-head", action="store_true",
                    help="use the real GR00T action head with a tiny DiT (gradient smoke)")
    ap.add_argument("--log-every", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    assets = build_g1_assets(device=device)
    if args.supermimic_ckpt:
        bundle = SuperMimicBundle.from_checkpoint(args.supermimic_ckpt, assets)
    else:
        bundle = SuperMimicBundle.random(assets, small=args.small)
    files = collect_npz(args.motion)
    library = load_motion_library(files, assets, device=device)
    print(f"[info] {library.summary()}")

    adapter = MotionGoalAdapter(AdapterConfig(
        token_dim=64,
        token_horizon=12 if args.real_gr00t_head else 40,
        hidden=128 if args.small else 256,
        transformer_layers=1 if args.small else 2,
        transformer_heads=4,
    )).to(device)

    fm_sampler = None
    fm_params = None
    if args.real_gr00t_head:
        from diff_vla_wbc.gr00t_tiny import TinyGr00tModel, make_tiny_action_head, tiny_model_inputs

        head = make_tiny_action_head()
        model = TinyGr00tModel(head).to(device)
        inputs = tiny_model_inputs(args.batch_size, device=device)
        fixed_noise = torch.randn(args.batch_size, 12, 64, device=device)

        def fm_sampler(meta):  # noqa: ANN001
            return sample_model_with_grad(model, inputs, initial_noise=fixed_noise, num_steps=3)

        fm_params = [p for p in head.parameters() if p.requires_grad]
        print(f"[info] real GR00T action head params: {sum(p.numel() for p in head.parameters()):,}")

    provider = MotionBatchProvider(
        bundle,
        library,
        batch_size=args.batch_size,
        horizon=args.horizon,
        token_source=fm_sampler,
        token_dim=64,
        token_horizon=adapter.cfg.token_horizon,
        seed=args.seed,
    )
    cfg = JointTrainerConfig(
        horizon=args.horizon,
        adapter_lr=args.adapter_lr,
        policy_lr=args.policy_lr,
        fm_lr=args.fm_lr,
    )
    trainer = JointTrainer(
        bundle, adapter, fm_sampler=fm_sampler, fm_parameters=fm_params, cfg=cfg, device=device
    )

    if args.pretrain_wm:
        for i in range(args.pretrain_wm):
            windows = sample_reference_windows(
                bundle, library, batch_size=args.pretrain_wm_batch,
                window=args.pretrain_wm_window, seed=args.seed + i,
            )
            m = trainer.train_world_model(windows)
            print(f"[wm {i + 1}/{args.pretrain_wm}] " + json.dumps({k: round(v, 6) for k, v in m.items()}))

    for step in range(1, args.steps + 1):
        batch = provider.sample()
        metrics = trainer.train_step(batch)
        if step % args.log_every == 0 or step == 1:
            print(f"[step {step}/{args.steps}] " + json.dumps({k: round(v, 6) for k, v in metrics.items()}))

    args.out.mkdir(parents=True, exist_ok=True)
    if args.save:
        path = args.out / "joint_latest.pt"
        trainer.save(path)
        print(f"[saved] {path}")
    print("[done] differentiable FM -> adapter -> WBC -> WM -> task loss step completed")


if __name__ == "__main__":
    main()
