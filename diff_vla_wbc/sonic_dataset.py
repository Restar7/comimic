"""Pinned SONIC four-task fine-tuning dataset (LeRobot v2.1) resolver.

The dataset was frozen on 2026-10-23 after failed episodes were removed:
``flowbody/FlowBody/dataset/groot_sonic_vla_ft318_frozen_20261023`` holds four
task groups and nine LeRobot sub-datasets, 318 episodes in total (task_1 = 89,
task_2 = 70, task_3 = 65, task_4 = 94), replaying at 50 fps.

The choice, the per-dataset episode/frame counts, task instructions and a
SHA-256 fingerprint over every parquet file are frozen in
``data/sonic_datasets.json`` so no script has to hard-code a machine path
(the older ``groot_repro`` scripts point at ``/home/wusichen/SONIC_DATA/...``).

Every sub-dataset has 31 columns, including::

    action.motion_token        [64]   SONIC FSQ motion token (float grid, step 1/16)
    action.wbc                 [43]   29 body + 14 hand joint targets
    teleop.left_hand_joints    [7]    left hand joint angles (rad)
    teleop.right_hand_joints   [7]    right hand joint angles (mirrored signs)

Usage::

    from diff_vla_wbc.sonic_dataset import (
        active_datasets, dataset_entries, episode_files, task_counts, verify,
    )

    verify()        # -> {"ok": True, "episodes": 318, "tasks": {...}, ...}
    task_counts()   # -> {"task_1_meeting_room_milk_basket": 89, ...}
    for ds in active_datasets():
        for parquet in episode_files(ds):
            ...
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "sonic_datasets.json"


def load_manifest(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else MANIFEST_PATH
    if not p.is_file():
        raise FileNotFoundError(f"pinned SONIC manifest not found: {p}")
    manifest = json.loads(p.read_text())
    if "sets" not in manifest or "active" not in manifest:
        raise ValueError(f"malformed SONIC manifest: {p}")
    if manifest["active"] not in manifest["sets"]:
        raise ValueError(f"active set {manifest['active']!r} missing from manifest sets")
    return manifest


def active_set_name(manifest: dict | None = None) -> str:
    return (manifest or load_manifest())["active"]


def active_set(manifest: dict | None = None) -> dict:
    manifest = manifest or load_manifest()
    return manifest["sets"][manifest["active"]]


def dataset_entries(set_name: str | None = None, manifest: dict | None = None) -> list[dict]:
    """Manifest entries of a set (default: active), with ``path`` relative to its root."""
    manifest = manifest or load_manifest()
    name = set_name or manifest["active"]
    if name not in manifest["sets"]:
        raise KeyError(f"unknown set {name!r}; available: {sorted(manifest['sets'])}")
    return manifest["sets"][name]["datasets"]


def _root(set_name: str | None = None, manifest: dict | None = None) -> Path:
    manifest = manifest or load_manifest()
    name = set_name or manifest["active"]
    if name not in manifest["sets"]:
        raise KeyError(f"unknown set {name!r}; available: {sorted(manifest['sets'])}")
    return Path(manifest["sets"][name]["root"])


def dataset_paths(set_name: str | None = None, manifest: dict | None = None) -> list[Path]:
    """Absolute paths of every sub-dataset in a set (default: the pinned active set)."""
    root = _root(set_name, manifest)
    return [root / e["path"] for e in dataset_entries(set_name, manifest)]


def active_datasets(manifest: dict | None = None) -> list[Path]:
    return dataset_paths(None, manifest)


def task_counts(set_name: str | None = None, manifest: dict | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in dataset_entries(set_name, manifest):
        counts[e["task_id"]] = counts.get(e["task_id"], 0) + int(e["episodes"])
    return counts


def instructions(set_name: str | None = None, manifest: dict | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for e in dataset_entries(set_name, manifest):
        if e.get("instruction"):
            out[e["task_id"]] = e["instruction"]
    return out


def episode_files(dataset: str | Path) -> list[Path]:
    """Sorted ``episode_*.parquet`` files of one sub-dataset."""
    ds = Path(dataset)
    files = sorted(Path(p) for p in glob.glob(str(ds / "data" / "chunk-*" / "episode_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no episode parquet under {ds}")
    return files


def verify(set_name: str | None = None, manifest: dict | None = None) -> dict:
    """Check the pinned set on disk: presence, episode counts, token/hand columns."""
    import pyarrow.parquet as pq

    manifest = manifest or load_manifest()
    name = set_name or manifest["active"]
    root = _root(name, manifest)
    report: dict = {"set": name, "root": str(root), "datasets": [], "ok": True, "errors": []}
    total_eps = total_frames = 0
    for entry in dataset_entries(name, manifest):
        ds = root / entry["path"]
        item = {"path": entry["path"], "task_id": entry["task_id"], "exists": ds.is_dir()}
        try:
            files = episode_files(ds)
            names = [f.name for f in pq.ParquetFile(files[0]).schema_arrow]
            item.update(
                episodes=len(files),
                expected=entry["episodes"],
                frames=sum(pq.ParquetFile(f).metadata.num_rows for f in files),
                has_motion_token="action.motion_token" in names,
                has_hand_joints=all(f"teleop.{s}_hand_joints" in names for s in ("left", "right")),
            )
            if item["episodes"] != entry["episodes"]:
                report["errors"].append(f"{entry['path']}: {item['episodes']} episodes != {entry['episodes']}")
            if item["frames"] != entry["frames"]:
                report["errors"].append(f"{entry['path']}: {item['frames']} frames != {entry['frames']}")
            if not (item["has_motion_token"] and item["has_hand_joints"]):
                report["errors"].append(f"{entry['path']}: missing motion token / hand columns")
            total_eps += item["episodes"]
            total_frames += item["frames"]
        except Exception as exc:  # noqa: BLE001 - report instead of raising
            item["error"] = repr(exc)
            report["errors"].append(f"{entry['path']}: {exc!r}")
        report["datasets"].append(item)
    report["episodes"] = total_eps
    report["frames"] = total_frames
    report["hours_at_50fps"] = round(total_frames / 50 / 3600, 3)
    report["tasks"] = task_counts(name, manifest)
    report["ok"] = not report["errors"]
    return report


def fingerprint_set(set_name: str | None = None, manifest: dict | None = None) -> str:
    """SHA-256 over (relative path, file size) of every parquet in the set."""
    import hashlib

    manifest = manifest or load_manifest()
    name = set_name or manifest["active"]
    root = _root(name, manifest)
    h = hashlib.sha256()
    for ds in dataset_paths(name, manifest):
        for f in episode_files(ds):
            h.update(str(f.relative_to(root)).encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()


__all__ = [
    "MANIFEST_PATH",
    "active_datasets",
    "active_set",
    "active_set_name",
    "dataset_entries",
    "dataset_paths",
    "episode_files",
    "fingerprint_set",
    "instructions",
    "load_manifest",
    "task_counts",
    "verify",
]
