#!/usr/bin/env python3
"""Joint training / gradient smoke CLI for the differentiable framework.

Three token sources, most synthetic first:

* default: random FM tokens (fast CPU smoke of the whole graph);
* ``--real-gr00t-head``: the real ``Gr00tN1d7ActionHead`` with a tiny DiT and
  random backbone features;
* ``--vlm``: the **real top-level VLM** -- Cosmos-Reason2-2B (local weights) ->
  GR00T N1.7 flow-matching head -> ``[B, 40, 64]`` chunk, with gradients reaching
  the head (and the VLM with ``--train-vlm``).  There is no paired RGB data in
  this repository, so the observation is a deterministic synthetic image unless
  ``--vlm-image`` is given.
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


def load_vlm_image(path: Path | None, size: int = 256):
    """Observation image for the VLM path (deterministic synthetic if unset)."""
    from PIL import Image

    if path is not None:
        return Image.open(path).convert("RGB")
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, size), torch.linspace(0.0, 1.0, size), indexing="ij"
    )
    rgb = torch.stack((x, y, (x + y) / 2.0), dim=-1).numpy()
    return Image.fromarray((rgb * 255.0).round().astype("uint8"))


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
    ap.add_argument("--recorded-tokens", type=Path, default=None, metavar="DIR",
                    help="use the tokens recorded in DIR/*.npz (g1_motion_token_64, real ft318) "
                         "instead of a VLM/FM sampler — the 'real data' token source")
    ap.add_argument("--real-gr00t-head", action="store_true",
                    help="use the real GR00T action head with a tiny DiT (gradient smoke)")
    vlm = ap.add_argument_group("top-level VLM (Cosmos-Reason2-2B)")
    vlm.add_argument("--vlm", action="store_true",
                     help="use the real Cosmos-Reason2-2B VLM + GR00T head as the token source")
    vlm.add_argument("--vlm-path", type=Path, default=None,
                     help="local VLM directory (default: auto-resolve, see diff_vla_wbc.gr00t_vlm)")
    vlm.add_argument("--vlm-ckpt", type=Path, default=None,
                     help="optional full GR00T checkpoint (backbone + action head)")
    vlm.add_argument("--vlm-select-layer", type=int, default=16,
                     help="LLM layers kept by the backbone (GR00T N1.7 truncation)")
    vlm.add_argument("--vlm-steps", type=int, default=2,
                     help="Euler steps of the differentiable flow-matching sampler")
    vlm.add_argument("--vlm-image", type=Path, default=None,
                     help="observation image (default: deterministic synthetic RGB)")
    vlm.add_argument("--vlm-instruction", default="Pick up the carton and hand it over.")
    vlm.add_argument("--vlm-state-dim", type=int, default=43, help="raw robot state width")
    vlm.add_argument("--vlm-embodiment-id", type=int, default=0,
                     help="embodiment id fed to the action head; a GR00T checkpoint needs the "
                          "id it was trained with (e.g. 11 = unitree_g1_sonic)")
    vlm.add_argument("--vlm-dataset", action="store_true",
                     help="feed per-sample REAL observations: each step reads the ego_view frame, "
                          "46-dim state and instruction belonging to the sampled motion window, "
                          "preprocessed by the checkpoint's own Gr00tN1d7Processor. Requires "
                          "--vlm-ckpt and a --motion directory built by prepare_g1_tasks_npz.py")
    vlm.add_argument("--train-vlm", action="store_true",
                     help="also unfreeze the VLM backbone (otherwise only the action head trains)")
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
    if args.recorded_tokens is not None:
        from diff_vla_wbc.data import RecordedTokenSource

        fm_sampler = RecordedTokenSource(
            args.recorded_tokens,
            library,
            token_horizon=adapter.cfg.token_horizon,
            token_dim=adapter.cfg.token_dim,
            device=device,
        )
        fm_params = []
        print(
            f"[info] token source: recorded g1_motion_token_64 from {args.recorded_tokens} "
            f"(real ft318 tokens, horizon={adapter.cfg.token_horizon})"
        )

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

    if args.vlm:
        if args.real_gr00t_head:
            raise SystemExit("--vlm and --real-gr00t-head are mutually exclusive")
        from diff_vla_wbc.gr00t_vlm import Gr00tVLM, describe_vlm, resolve_vlm_source, vlm_available

        if not vlm_available(args.vlm_path) and args.vlm_path is None and args.vlm_ckpt is None:
            raise SystemExit(
                "no local Cosmos-Reason2-2B weights found; pass --vlm-path, set "
                "$GR00T_VLM_PATH, or run scripts/link_vlm_weights.py"
            )
        print(f"[info] top-level VLM source: {resolve_vlm_source(args.vlm_path)}")
        vlm_model = Gr00tVLM.from_pretrained(
            args.vlm_path,
            checkpoint=args.vlm_ckpt,
            select_layer=args.vlm_select_layer,
            device=device,
            action_horizon=adapter.cfg.token_horizon,
            max_state_dim=args.vlm_state_dim,
            freeze_vlm=not args.train_vlm,
        )
        print("[info] top-level VLM: " + json.dumps(describe_vlm(vlm_model), default=str))

        # The head emits its own action width (``Gr00tN1d7Config.max_action_dim``:
        # 64 for the standalone default, 132 for a GR00T checkpoint).  GR00T packs
        # an embodiment's action vector into that width following modality-key
        # order, so for ``unitree_g1_sonic`` columns 0:64 are the motion token and
        # the rest is unsupervised padding.  Sample at the head's width, then slice.
        head_action_dim = int(vlm_model.action_head.action_dim)

        def _sample(inputs) -> torch.Tensor:
            noise = torch.randn(
                int(inputs["input_ids"].shape[0]),
                adapter.cfg.token_horizon,
                head_action_dim,
                device=device,
                dtype=vlm_model.dtype,
            )
            tokens = sample_model_with_grad(
                vlm_model, inputs, initial_noise=noise, num_steps=args.vlm_steps
            )
            return tokens[..., : adapter.cfg.token_dim]

        if args.vlm_dataset:
            if not args.vlm_ckpt:
                raise SystemExit(
                    "--vlm-dataset needs --vlm-ckpt: the 46-dim state layout, percentile "
                    "normalisation and image transform all come from the checkpoint's processor"
                )
            from diff_vla_wbc.g1_tasks import (
                TaskFrameIndex,
                TaskObservationBuilder,
                load_gr00t_processor,
                processed_batch,
            )

            frame_index = TaskFrameIndex(files)
            if frame_index.num_frames != library.num_frames:
                raise SystemExit(
                    f"npz index has {frame_index.num_frames} frames but the motion library has "
                    f"{library.num_frames}; rebuild it with scripts/prepare_g1_tasks_npz.py"
                )
            builder = TaskObservationBuilder(frame_index)
            processor = load_gr00t_processor(args.vlm_ckpt, vlm_path=args.vlm_path)
            print(
                f"[info] real observations: {frame_index.num_episodes} episodes / "
                f"{frame_index.num_frames} frames, processor={type(processor).__name__}"
            )

            def fm_sampler(meta):  # noqa: ANN001
                return _sample(processed_batch(processor, builder, meta["frames"]))

        else:
            observations = [load_vlm_image(args.vlm_image)] * args.batch_size
            instructions = [args.vlm_instruction] * args.batch_size
            vlm_state = torch.zeros(args.batch_size, args.vlm_state_dim)
            vlm_batch = vlm_model.build_batch(
                observations, instructions, vlm_state, embodiment_id=args.vlm_embodiment_id
            )
            print(
                "[warn] --vlm without --vlm-dataset reuses ONE synthetic image every step; "
                "fine-tuning the head this way collapses its visual conditioning. Use "
                "--vlm-dataset for anything beyond a gradient smoke test."
            )

            def fm_sampler(meta):  # noqa: ANN001
                return _sample(vlm_batch)

        fm_params = [p for p in vlm_model.parameters() if p.requires_grad]
        print(f"[info] trainable VLM/head params: {sum(p.numel() for p in fm_params):,}")

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
