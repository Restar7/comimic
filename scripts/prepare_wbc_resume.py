#!/usr/bin/env python3
"""Prepare a SuperMimic checkpoint for continued training on a *different* motion library.

Why this is needed
------------------
``Trainer.load`` restores the command sampler's per-bin statistics verbatim
(``supermimic/training/loop.py``):

    self.command.bin_return.copy_(ck["bin_return"])   # shape = (library.num_bins,)
    self.command.bin_memory.copy_(ck["bin_memory"])

``num_bins`` is a property of the **reference library** (``commands.py`` builds them with
``bin_frames=50``).  Resuming the BONES-SEED checkpoint (1,013,724 bins) on our 318-clip
library (9,621 bins) therefore crashes with a size mismatch before the first step.

This tool rewrites exactly those two keys (plus the accounting counters) and leaves every
model / optimizer tensor untouched.  The source checkpoint is never modified.

Usage
-----
    .venv/bin/python scripts/prepare_wbc_resume.py \\
        --src     logs/e9_bones_seed_token_h200x2/ckpt_0024000.pt \\
        --library /path/to/data/library/ft318_compact.pt \\
        --tokens  /path/to/data/library/ft318_tokens.pt \\
        --out     /path/to/runs/wbc_resume/e9_0024000_ft318.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

#: Must match ``MultiClipMotionCommandCfg.bin_frames`` (envs/mdp/commands.py).
BIN_FRAMES = 50
#: ``MultiClipMotionCommandCfg.g_max`` — initial optimism of the sampler (commands.py:70).
G_MAX = 10.0


def num_bins(lengths) -> int:
    """Same formula as ``MotionLibrary.num_bins`` (motion_library.py:160-163).

    ``bins_per_clip = (clip_len + bin_frames - 1) // bin_frames``  — i.e. ``ceil(L / 50)``,
    **not** ``ceil((L-1)/50)`` (that is ``merge_libraries.py``'s coarser estimate and
    under-counts by 1 for every clip whose length is not a multiple of 50).
    """
    return sum((int(L) + BIN_FRAMES - 1) // BIN_FRAMES for L in lengths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="checkpoint to continue from (never modified)")
    ap.add_argument("--library", type=Path, required=True, help="new motion library (.pt from build_library.py)")
    ap.add_argument("--tokens", type=Path, default=None, help="recorded token table for the new library (optional)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-step", action="store_true", help="keep iteration/num_updates (default: reset to 0)")
    ap.add_argument("--total", type=int, default=None, help="record the new --schedule.total-iterations in the ckpt config")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force")
    for p in (args.src, args.library):
        if not p.is_file():
            raise SystemExit(f"missing: {p}")

    lib = torch.load(str(args.library), map_location="cpu", weights_only=False)
    if "lengths" not in lib:
        raise SystemExit(f"{args.library}: missing 'lengths' (not a packed library)")
    nb_new = num_bins(lib["lengths"])
    n_frames = int(lib["ref"].shape[0]) if "ref" in lib else -1

    ck = torch.load(str(args.src), map_location="cpu", weights_only=False)
    for key in ("policy", "wm", "bin_return", "bin_memory"):
        if key not in ck:
            raise SystemExit(f"{args.src}: not a training checkpoint (missing {key!r})")

    cfg = ck.get("config") or {}
    token_dim = int((cfg.get("policy") or {}).get("token_dim", 0) or 0)
    if token_dim <= 0:
        raise SystemExit(
            f"{args.src}: policy.token_dim={token_dim} — this checkpoint is the 680-d goal route, "
            "not the SONIC token route; refusing to prepare it."
        )
    spec = (cfg.get("world_model") or {}).get("state_spec") or {}
    if not spec.get("explicit_integration", True):
        print("[warn] world_model.state_spec.explicit_integration is False in the source config; "
              "the WM will be trained without the dt coupling", flush=True)

    nb_old = int(ck["bin_return"].numel())
    ck["bin_return"] = torch.full((nb_new,), G_MAX, dtype=ck["bin_return"].dtype)
    ck["bin_memory"] = torch.full((nb_new,), 1.0 / nb_new, dtype=ck["bin_memory"].dtype)

    it_old = int(ck.get("iteration", 0))
    up_old = int(ck.get("num_updates", 0))
    if not args.keep_step:
        ck["iteration"] = 0
        ck["num_updates"] = 0

    # make the checkpoint self-describing for anyone resuming from it later
    if cfg:
        cfg.setdefault("data", {})
        cfg["data"]["library"] = str(args.library.resolve())
        if args.tokens is not None:
            cfg["data"]["tokens"] = str(args.tokens.resolve())
        if args.total is not None:
            cfg.setdefault("schedule", {})
            cfg["schedule"]["total_iterations"] = int(args.total)
        ck["config"] = cfg

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(args.out))

    print(
        json.dumps(
            {
                "src": str(args.src.resolve()),
                "out": str(args.out.resolve()),
                "library": str(args.library.resolve()),
                "library_frames": n_frames,
                "tokens": str(args.tokens.resolve()) if args.tokens else None,
                "bins_old": nb_old,
                "bins_new": nb_new,
                "iteration_old": it_old,
                "iteration_new": int(ck["iteration"]),
                "num_updates_old": up_old,
                "num_updates_new": int(ck["num_updates"]),
                "policy_token_dim": token_dim,
                "out_bytes": args.out.stat().st_size,
            },
            indent=1,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
