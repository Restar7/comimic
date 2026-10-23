#!/usr/bin/env python3
"""Materialise a local, offline-loadable Cosmos-Reason2-2B view inside the repo.

GR00T checkpoints written on the deployment box hard-code an absolute
``model_name`` for the (gated) VLM.  Pointing ``model_name`` at a *local*
directory that already contains the weights removes the download, the Hugging
Face token and the gated-repo error in one step.

Defaults:

* ``--source``: auto-detected with
  :func:`diff_vla_wbc.gr00t_vlm.resolve_vlm_source` -- ``$GR00T_VLM_PATH``, an
  existing checkout copy, then a local Hugging Face cache snapshot (the
  workspace keeps it in ``<root>/cache/huggingface/hub``);
* ``--dest``: ``groot_repro/nvidia/Cosmos-Reason2-2B`` (git-ignored, so the
  multi-GB weights never enter the repository).

Every file is *symlinked*, never copied, and the previous contents of ``--dest``
are replaced so the config and the weights always describe the same model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.gr00t_vlm import LOCAL_VLM_DIR, VLM_REPO_ID, resolve_vlm_source  # noqa: E402

_SKIP = {".cache", ".git", ".hfd", "blobs"}


def snapshot_files(source: Path) -> list[Path]:
    return [p for p in sorted(source.iterdir()) if p.name not in _SKIP and not p.is_dir()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=None, help="VLM directory or HF cache snapshot")
    ap.add_argument("--dest", type=Path, default=LOCAL_VLM_DIR, help="local view to create")
    ap.add_argument("--force", action="store_true", help="replace an existing non-symlink dest")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.source is None:
        source_str, origin = resolve_vlm_source()
        source = Path(source_str)
        if origin == "repo-id":
            print(
                f"no local {VLM_REPO_ID} found; pass --source or set $GR00T_VLM_PATH",
                file=sys.stderr,
            )
            return 2
        print(f"auto-resolved source ({origin}): {source}")
    else:
        source = args.source
        origin = "argument"

    if not source.is_dir():
        print(f"source is not a directory: {source}", file=sys.stderr)
        return 2
    files = snapshot_files(source)
    if not any(p.suffix == ".safetensors" or p.suffix == ".bin" for p in files):
        print(f"source has no weights (*.safetensors): {source}", file=sys.stderr)
        return 2

    dest = args.dest
    if os.path.realpath(dest) == os.path.realpath(source):
        print(f"dest already is the source: {dest}")
        return 0
    if dest.is_dir():
        hand_copied = [
            p.name
            for p in dest.iterdir()
            if p.is_file() and not p.is_symlink() and p.name not in ("SOURCE.txt",)
        ]
        if hand_copied and not args.force:
            print(
                f"{dest} holds real files (e.g. {hand_copied[:3]}); re-run with --force to replace them",
                file=sys.stderr,
            )
            return 3

    print(f"linking {len(files)} file(s): {source} -> {dest}")
    if args.dry_run:
        for path in files:
            print(f"  {path.name} -> {dest / path.name}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    for entry in list(dest.iterdir()):
        if entry.is_file() or entry.is_symlink():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry)
    for path in files:
        target = dest / path.name
        os.symlink(path, target)
    (dest / "SOURCE.txt").write_text(
        json.dumps({"source": str(source), "origin": origin, "repo_id": VLM_REPO_ID}, indent=2) + "\n"
    )
    total = sum(p.stat().st_size for p in files)  # follows symlinks into the HF cache blobs
    print(f"done: {dest} ({total / 1e9:.2f} GB, {len(files)} files)")
    print("now `python scripts/smoke_vlm.py --device cuda:0` loads it without network access")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
