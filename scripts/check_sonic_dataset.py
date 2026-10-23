#!/usr/bin/env python3
"""Preflight check for the pinned SONIC fine-tuning dataset.

Runs the same verification the tests use (episode counts, frame counts, token and
hand columns) and optionally re-hashes every parquet file against the frozen
fingerprints in ``data/sonic_datasets.json``.  Exits non-zero on any mismatch, so
it can be dropped into a job launcher / CI before training.

    python scripts/check_sonic_dataset.py
    python scripts/check_sonic_dataset.py --fingerprints
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc import sonic_dataset as sd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, help="manifest set name (default: active)")
    ap.add_argument("--fingerprints", action="store_true", help="re-hash every parquet file")
    args = ap.parse_args()

    manifest = sd.load_manifest()
    name = args.set or sd.active_set_name(manifest)
    set_cfg = manifest["sets"][name]
    root = Path(set_cfg["root"])
    print(f"set        : {name}")
    print(f"root       : {root}")
    print(f"episodes   : {set_cfg['totals']['episodes']} (manifest)")
    print(f"frames     : {set_cfg['totals']['frames']} (manifest)")
    print(f"tasks      : {sd.task_counts(name, manifest)}")

    report = sd.verify(name, manifest)
    failures = list(report["errors"])
    for item in report["datasets"]:
        status = "ok" if "error" not in item and item["episodes"] == item["expected"] else "FAIL"
        print(
            f"  [{status:4s}] {item['task_id']:38s} {item['path']:52s} "
            f"{item.get('episodes', '?'):>3} eps / {item.get('frames', '?'):>7} frames"
        )

    if args.fingerprints:
        for entry in sd.dataset_entries(name, manifest):
            files = sd.episode_files(root / entry["path"])
            h = hashlib.sha256()
            for f in files:
                h.update(str(f.relative_to(root)).encode())
                h.update(str(f.stat().st_size).encode())
            got = h.hexdigest()
            if got != entry["fingerprint_sha256"]:
                failures.append(f"{entry['path']}: fingerprint {got[:12]} != {entry['fingerprint_sha256'][:12]}")
        print("fingerprints: checked")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1

    print(
        f"\nOK: {report['episodes']} episodes, {report['frames']} frames, "
        f"{report['hours_at_50fps']} h @50fps, tasks={len(report['tasks'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
