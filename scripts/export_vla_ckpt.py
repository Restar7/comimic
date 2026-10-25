#!/usr/bin/env python3
"""Fold a ``vla_<step>.pt`` update file back into a GR00T checkpoint directory.

``scripts/train_joint.py --save-vla-every N`` writes

    vla_0004000.pt = {"step": 4000, "fm": {name: fp32 tensor, ...}}

i.e. **only the parameters that were being optimised** (the action head, plus the
VLM backbone layers if ``--train-vlm``).  ``Gr00tPolicy`` however wants a normal
offline checkpoint directory (``model-XXXXX-of-00003.safetensors`` + index +
processor configs).  This script rebuilds one:

    base checkpoint dir  +  vla_<step>.pt  ->  merged checkpoint dir

Only shards that actually contain an overridden tensor are re-written; every
other shard and every config file is symlinked, so the merge costs one shard of
disk (~2.5 GB), not the full 12.6 GB.

Examples
--------
    # our 4000-step differentiable finetune of the SONIC 20k VLA
    python3 scripts/export_vla_ckpt.py \\
        --base runs/ckpt_vla20k \\
        --vla  runs/joint_diff_ft318/vla_004000.pt \\
        --out  runs/ckpt_vla_ft4000

    # re-check an export without rewriting it
    python3 scripts/export_vla_ckpt.py --out runs/ckpt_vla_ft4000 --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch

INDEX_NAME = "model.safetensors.index.json"

# Files the policy loader / processor reads; copied (symlinked) verbatim.
SIDE_FILES = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
    "experiment_cfg",
    "processor",
    "wandb_config.json",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _tensor_digest(t: torch.Tensor, limit: int = 4096) -> str:
    flat = t.detach().to(torch.float32).reshape(-1)[:limit]
    return hashlib.sha1(flat.numpy().tobytes()).hexdigest()[:16]


def load_update(vla_path: Path) -> tuple[int | None, dict[str, torch.Tensor]]:
    ck = torch.load(vla_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "fm" not in ck:
        raise SystemExit(
            f"{vla_path}: expected a dict with a 'fm' entry (what --save-vla-every writes), "
            f"got keys {list(ck) if isinstance(ck, dict) else type(ck)}"
        )
    fm = ck["fm"]
    if not fm:
        raise SystemExit(f"{vla_path}: 'fm' is empty")
    return ck.get("step"), {k: v for k, v in fm.items()}


def read_index(base: Path) -> dict:
    path = base / INDEX_NAME
    if not path.exists():
        raise SystemExit(f"{path} not found: {base} is not a sharded safetensors checkpoint")
    return json.loads(path.read_text())


def merge_shard(src: Path, dst: Path, overrides: dict[str, torch.Tensor]) -> int:
    """Copy one shard, replacing the tensors in ``overrides``.  Returns #replaced."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    out: dict[str, torch.Tensor] = {}
    replaced = 0
    with safe_open(str(src), framework="pt") as f:
        for key in f.keys():
            if key in overrides:
                new = overrides[key]
                old = f.get_tensor(key)
                if tuple(new.shape) != tuple(old.shape):
                    raise SystemExit(f"{key}: shape {tuple(new.shape)} != base {tuple(old.shape)}")
                out[key] = new.to(old.dtype).contiguous()
                replaced += 1
            else:
                out[key] = f.get_tensor(key)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    save_file(out, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, dst)
    return replaced


def verify(out: Path, overrides: dict[str, torch.Tensor]) -> None:
    """Reload the merged shards and confirm every override landed byte-exactly."""
    from safetensors import safe_open

    index = read_index(out)
    weight_map = index["weight_map"]
    bad = 0
    seen = 0
    per_shard: dict[str, list[str]] = {}
    for key in overrides:
        per_shard.setdefault(weight_map[key], []).append(key)
    for shard, keys in per_shard.items():
        with safe_open(str(out / shard), framework="pt") as f:
            for key in keys:
                got = f.get_tensor(key)
                want = overrides[key].to(got.dtype)
                if not torch.equal(got, want):
                    bad += 1
                    md = (got.float() - want.float()).abs().max().item()
                    _log(f"[verify] MISMATCH {key} ({md:.3e})")
                seen += 1
    if bad:
        raise SystemExit(f"[verify] {bad}/{seen} tensors differ after merge")
    _log(f"[verify] OK: {seen}/{len(overrides)} tensors match the update file")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, help="base checkpoint dir (source of the non-updated tensors)")
    ap.add_argument("--vla", type=Path, help="vla_<step>.pt written by --save-vla-every")
    ap.add_argument("--out", type=Path, help="destination checkpoint dir")
    ap.add_argument("--force", action="store_true", help="overwrite non-empty --out")
    ap.add_argument("--verify-only", action="store_true", help="only re-run the round-trip check")
    ap.add_argument("--link-side-files", action="store_true", default=True,
                    help="symlink configs instead of copying them (default)")
    args = ap.parse_args()

    if args.verify_only:
        if not args.out or not args.vla:
            raise SystemExit("--verify-only needs --out and --vla")
        _, overrides = load_update(args.vla)
        verify(args.out, overrides)
        return

    if not (args.base and args.vla and args.out):
        raise SystemExit("--base, --vla and --out are required (or use --verify-only)")

    step, overrides = load_update(args.vla)
    index = read_index(args.base)
    weight_map: dict[str, str] = index["weight_map"]

    missing = [k for k in overrides if k not in weight_map]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(overrides)} tensors of {args.vla.name} are absent from the base "
            f"index, e.g. {missing[:3]} — was the base checkpoint built with a different config?"
        )

    shards = sorted(set(weight_map.values()))
    per_shard: dict[str, dict[str, torch.Tensor]] = {s: {} for s in shards}
    for key, t in overrides.items():
        per_shard[weight_map[key]][key] = t
    touched = {s: v for s, v in per_shard.items() if v}

    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        raise SystemExit(f"{args.out} exists and is not empty; pass --force to overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    _log(f"[info] base   = {args.base}")
    _log(f"[info] update = {args.vla}  (step={step}, {len(overrides)} tensors)")
    _log(f"[info] out    = {args.out}")
    _log(f"[info] shards to rewrite: {sorted(touched)} / {len(shards)}")

    for shard in shards:
        src = args.base / shard
        dst = args.out / shard
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if shard in touched:
            n = merge_shard(src, dst, touched[shard])
            _log(f"[merge] {shard}: replaced {n} tensors -> {dst.stat().st_size / 2**30:.2f} GiB")
        else:
            dst.symlink_to(src.resolve())

    (args.out / INDEX_NAME).symlink_to((args.base / INDEX_NAME).resolve())
    for name in SIDE_FILES:
        src = args.base / name
        dst = args.out / name
        if not src.exists() and not src.is_symlink():
            continue
        if dst.exists() or dst.is_symlink():
            continue
        if args.link_side_files:
            dst.symlink_to(src.resolve())
        elif src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    report = {
        "base": str(args.base.resolve()),
        "update": str(args.vla.resolve()),
        "step": step,
        "num_tensors": len(overrides),
        "rewritten_shards": {s: sorted(v) for s, v in touched.items()},
        "digests": {k: _tensor_digest(t) for k, t in list(overrides.items())[:16]},
    }
    (args.out / "vla_merge_report.json").write_text(json.dumps(report, indent=2))
    _log(f"[done] wrote {args.out / 'vla_merge_report.json'}")

    verify(args.out, overrides)


if __name__ == "__main__":
    sys.exit(main())
