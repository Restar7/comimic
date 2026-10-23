#!/usr/bin/env python
"""Build a local, offline-loadable view of a GR00T checkpoint.

The training images wrote absolute Orin paths into `config.json`
(`model_name`) and `processor_config.json`
(`processor_kwargs.model_name`).  This script symlinks every file of the
checkpoint unchanged, then rewrites only those two JSON files so they point at
the local Cosmos-Reason2 config stub.  No weight is copied or modified, so the
SHA256 of the shards still matches the deployment's checksum files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

STUB = "/home/wusichen/groot_repro/nvidia/Cosmos-Reason2-2B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    src, dst = Path(args.source), Path(args.dest)
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        target = dst / entry.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if entry.name in ("config.json", "processor_config.json"):
            continue
        os.symlink(entry.resolve(), target)

    cfg = json.loads((src / "config.json").read_text())
    cfg["model_name"] = STUB
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    proc = json.loads((src / "processor_config.json").read_text())
    proc["processor_kwargs"]["model_name"] = STUB
    (dst / "processor_config.json").write_text(json.dumps(proc, indent=2))
    print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
