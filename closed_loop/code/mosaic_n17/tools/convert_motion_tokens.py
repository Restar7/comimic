#!/usr/bin/env python3
"""Convert FlowBody 49D demonstrations into SONIC motion tokens (G1 mode).

The frozen release encoder is used as a black box; only its 1762-dim input is
assembled here.  The contract below is pinned by the release C++ header and was
validated frame-by-frame against the official recorded ``action.motion_token``
stream (rebuilding the input from measured SONIC state reproduces the recorded
token's decoded targets to the recorded-token floor):

* ``q``: ``actions_49d`` body targets ``leg15 + arm14`` in MuJoCo order, mapped
  to IsaacLab slots with ``mujoco_to_isaaclab`` and kept **absolute** (the
  training motion lib reorders but never subtracts ``default_angles``);
* ``dq``: 50 Hz finite difference of that reference ``q``;
* ``anchor``: ``inv(measured_base) * reference_root`` at ``t + 5k``, first two
  rotation-matrix columns flattened row-major (not the stacked-column layout);
* future window ``t, t+5, ..., t+45``; a token is valid only when the complete
  window exists, invalid frames stay NaN and are counted in the report.

For ``simple`` the action stream is the teleop command (the smooth reference);
the measured ``states46`` only supply the base orientation.  For ``leverb`` the
action stream equals the measured states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # direct `python3 .../tools/convert_motion_tokens.py`
    sys.path.insert(0, str(_REPO_ROOT))

from mosaic_n17.contracts.state46 import joint_order_sha256  # noqa: E402
from mosaic_n17.tools.motion_token_encoder import (  # noqa: E402
    FUTURE_FRAMES,
    FUTURE_STEP,
    MUJOCO_TO_ISAACLAB,
    assemble_g1_input,
    make_session,
    relative_orientation_6d,
)


SCHEMA = "mosaic.n17.motion_token_encoder_reference.v2"
LOOKAHEAD = (FUTURE_FRAMES - 1) * FUTURE_STEP
FPS = 50.0
BODY_JOINTS = 29
TOKEN_ABS_BOUND = 1.25

# Canonical arm-first 49D layout: arm14 + hand14 + leg15 + rootXYZ3 + rootRPY3.
ARM_SLICE = slice(0, 14)
LEG_SLICE = slice(28, 43)
ROOT_RPY_SLICE = slice(46, 49)


def rpy_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    """``[roll, pitch, yaw]`` -> ``[w, x, y, z]`` with ``Rz(yaw) Ry(pitch) Rx(roll)``."""

    roll, pitch, yaw = (np.asarray(rpy, dtype=np.float64) * 0.5).T
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    quat = np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("root RPY produced a zero quaternion")
    return quat / norm


def body_reference(actions_49d: np.ndarray) -> np.ndarray:
    """``[T,49]`` arm-first actions -> ``[T,29]`` body targets in MuJoCo order."""

    actions = np.asarray(actions_49d, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 49:
        raise ValueError(f"actions_49d must be [T,49], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("actions_49d must be finite")
    return np.concatenate((actions[:, LEG_SLICE], actions[:, ARM_SLICE]), axis=1)


def encode_sequence(
    session,
    actions_49d: np.ndarray,
    states46: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Encode every valid 50 Hz tick of one demonstration -> ``([T,64], valid)``."""

    actions = np.asarray(actions_49d, dtype=np.float64)
    states = np.asarray(states46, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 46:
        raise ValueError(f"states46 must be [T,46], got {states.shape}")
    if states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"actions_49d and states46 frame counts differ: {actions.shape[0]} vs {states.shape[0]}"
        )
    if not np.isfinite(states).all():
        raise ValueError("states46 must be finite")
    frames = actions.shape[0]
    if frames <= LOOKAHEAD:
        raise ValueError(f"need more than {LOOKAHEAD} frames for one token window, got {frames}")

    q_mujoco = body_reference(actions)
    dq_mujoco = np.zeros_like(q_mujoco)
    dq_mujoco[1:] = (q_mujoco[1:] - q_mujoco[:-1]) * FPS
    q_isaaclab = q_mujoco[:, list(MUJOCO_TO_ISAACLAB)]
    dq_isaaclab = dq_mujoco[:, list(MUJOCO_TO_ISAACLAB)]
    reference_quat = rpy_to_quat_wxyz(actions[:, ROOT_RPY_SLICE])
    base_quat = rpy_to_quat_wxyz(states[:, 0:3])
    offsets = FUTURE_STEP * np.arange(FUTURE_FRAMES)
    valid = frames - LOOKAHEAD
    tokens = np.full((frames, 64), np.nan, dtype=np.float32)
    input_name = session.get_inputs()[0].name
    for start in range(valid):
        window = start + offsets
        anchor = relative_orientation_6d(reference_quat[window], base_quat[start])
        encoder_input = assemble_g1_input(q_isaaclab[window], dq_isaaclab[window], anchor)
        token = np.asarray(session.run(None, {input_name: encoder_input})[0][0], dtype=np.float64)
        if token.shape != (64,) or not np.isfinite(token).all():
            raise ValueError(f"encoder token at t={start} must be finite [64]")
        if float(np.abs(token).max()) > TOKEN_ABS_BOUND:
            raise ValueError(
                f"encoder token at t={start} exceeds |x|<={TOKEN_ABS_BOUND}: "
                f"{float(np.abs(token).max()):.3f}"
            )
        tokens[start] = token
    return tokens, valid


def sequence_digest(actions_49d: np.ndarray, states46: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(actions_49d, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(states46, dtype=np.float64).tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-h5", type=Path, required=True)
    parser.add_argument("--encoder-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="leverb")
    parser.add_argument("--group", default="sequences", help="H5 group holding the records")
    parser.add_argument("--limit", type=int, default=0, help="0 means every record")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    session = make_session(args.encoder_onnx)
    encoder_sha = sha256_file(args.encoder_onnx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict = {
        "schema": SCHEMA,
        "source_h5": str(args.mixed_h5),
        "group": args.group,
        "prefix": args.prefix,
        "encoder_sha256": encoder_sha,
        "joint_order_sha256": joint_order_sha256(),
        "encoder_input_layout": (
            "sonic-g1-1762: selector@0, onehot@1..3, q@4, dq@294, root_z@584, anchor6d@601"
        ),
        "q_source": "actions_49d body targets (leg15+arm14, MuJoCo order, absolute)",
        "anchor": "inv(measured base quat) * reference root quat, first two columns row-major",
        "future_window": {"frames": FUTURE_FRAMES, "step": FUTURE_STEP, "lookahead": LOOKAHEAD},
        "records": 0,
        "encoded_sequences": 0,
        "valid_tokens": 0,
        "sequences": {},
    }
    token_min = np.inf
    token_max = -np.inf

    with h5py.File(args.mixed_h5, "r") as source, h5py.File(args.output, "w") as sink:
        if args.group not in source:
            raise KeyError(f"{args.mixed_h5}: group {args.group!r} not found")
        records = source[args.group]
        keys = [key for key in records if key.startswith(args.prefix)]
        if args.limit:
            keys = keys[: args.limit]
        if not keys:
            raise ValueError(f"no {args.group} record starts with prefix {args.prefix!r}")
        tokens_group = sink.create_group("tokens")
        tokens_group.attrs["schema"] = SCHEMA
        tokens_group.attrs["source"] = f"{args.mixed_h5}::{args.group}"
        tokens_group.attrs["encoder_sha256"] = encoder_sha
        tokens_group.attrs["layout"] = report["encoder_input_layout"]
        tokens_group.attrs["q_source"] = report["q_source"]
        tokens_group.attrs["anchor"] = report["anchor"]
        tokens_group.attrs["deduplicated"] = "sha256(actions_49d+states46)"
        encoded: dict[str, str] = {}
        for key in keys:
            entry = records[key]
            actions = entry["actions_49d"][...].astype(np.float64)
            states = entry["states46"][...].astype(np.float64)
            digest = sequence_digest(actions, states)
            source_key = encoded.get(digest, key)
            if source_key == key:
                tokens, valid = encode_sequence(session, actions, states)
                encoded[digest] = key
                report["encoded_sequences"] += 1
                finite = tokens[np.isfinite(tokens)]
                token_min = min(token_min, float(finite.min()))
                token_max = max(token_max, float(finite.max()))
            else:
                tokens = np.asarray(tokens_group[source_key][...], dtype=np.float32)
                valid = int(tokens_group[source_key].attrs["valid_frames"])
            dataset = tokens_group.create_dataset(key, data=tokens, compression="gzip")
            dataset.attrs["frames"] = int(tokens.shape[0])
            dataset.attrs["valid_frames"] = int(valid)
            dataset.attrs["copy_of"] = source_key
            report["records"] += 1
            report["valid_tokens"] += int(valid)
            report["sequences"][key] = {
                "frames": int(tokens.shape[0]),
                "valid_frames": int(valid),
                "copy_of": None if source_key == key else source_key,
            }
    report["seconds"] = round(time.perf_counter() - started, 3)
    report["token_min"] = token_min
    report["token_max"] = token_max
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "records": report["records"],
                "encoded_sequences": report["encoded_sequences"],
                "valid_tokens": report["valid_tokens"],
                "seconds": report["seconds"],
                "output": str(args.output),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
