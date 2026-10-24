"""Precomputed typed-slot store (one HDF5 file per episode).

Phase A is frozen, so its output can be computed once per camera frame instead of
inside the Phase B training loop.  The store records the identity of the encoder
that produced it: a slot file from a different Phase A checkpoint may never be
mixed into one training run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np

from ..ac.slots import SLOT_DIM, TOTAL_SLOTS


SLOTS_SCHEMA = "mosaic.n17.typed_slots.v1"
REQUIRED_ATTRS = (
    "schema",
    "phase_a_sha256",
    "dense_dim",
    "video_key",
    "fps",
    "episode_index",
    "stride",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_slots_h5(
    path: str | os.PathLike[str],
    *,
    episode_index: int,
    slots: np.ndarray,
    valid: np.ndarray | None = None,
    meta: dict,
) -> str:
    """Write ``[T,14,384]`` slots for one episode; returns the file sha256."""

    array = np.asarray(slots, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (TOTAL_SLOTS, SLOT_DIM):
        raise ValueError(f"slots must be [T,{TOTAL_SLOTS},{SLOT_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("slots must be finite")
    if valid is None:
        valid = np.ones(array.shape[0], dtype=bool)
    mask = np.asarray(valid).astype(bool)
    if mask.shape != (array.shape[0],):
        raise ValueError(f"valid must be [T], got {mask.shape}")
    missing = [
        key
        for key in ("phase_a_sha256", "dense_dim", "video_key", "fps", "stride")
        if key not in meta
    ]
    if missing:
        raise ValueError(f"slot metadata is missing {missing}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("slots", data=array, compression="gzip", compression_opts=4)
        handle.create_dataset("valid", data=mask)
        attrs = dict(meta)
        attrs.update(
            {
                "schema": SLOTS_SCHEMA,
                "episode_index": int(episode_index),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        for key, value in attrs.items():
            handle.attrs[key] = value
    os.replace(temporary, target)
    return _sha256_file(target)


def read_slots_h5(path: str | os.PathLike[str]) -> dict:
    """Read one slot file and validate its schema/identity metadata."""

    with h5py.File(Path(path), "r") as handle:
        missing = [key for key in REQUIRED_ATTRS if key not in handle.attrs]
        if missing:
            raise ValueError(f"{path} is missing slot attributes {missing}")
        if str(handle.attrs["schema"]) != SLOTS_SCHEMA:
            raise ValueError(f"unsupported slot schema {handle.attrs['schema']!r}")
        slots = np.asarray(handle["slots"][...], dtype=np.float32)
        valid = np.asarray(handle["valid"][...], dtype=bool)
        attrs = {key: handle.attrs[key] for key in handle.attrs}
    if slots.ndim != 3 or slots.shape[1:] != (TOTAL_SLOTS, SLOT_DIM):
        raise ValueError(f"{path} slots must be [T,{TOTAL_SLOTS},{SLOT_DIM}], got {slots.shape}")
    if valid.shape != (slots.shape[0],):
        raise ValueError(f"{path} valid mask must be [T], got {valid.shape}")
    return {"slots": slots, "valid": valid, "attrs": attrs}


__all__ = ["REQUIRED_ATTRS", "SLOTS_SCHEMA", "read_slots_h5", "write_slots_h5"]
