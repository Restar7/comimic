#!/usr/bin/env python3
"""Convert FlowBody 49-D predictions to a SONIC body + Dex3 hand motion PKL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import DOF_AXIS

ACTION_DIM = 49
ARM_SLICE = slice(0, 14)
HAND_SLICE = slice(14, 28)

# MuJoCo body order: left leg, right leg, waist, left arm, right arm.
STANDING_BODY_DOF = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

# FlowBody stores each hand as index0, index1, middle0, middle1,
# thumb0, thumb1, thumb2. Its dataset values use positive flexion magnitudes;
# the G1 URDF uses negative left-hand and positive right-hand axes for
# index/middle flexion.
HAND_FLEXION_SIGN = np.array(
    [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)
HAND_JOINT_NAMES = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]


def build_motion_entry(
    actions: np.ndarray,
    action_mask: np.ndarray | None,
    *,
    fps: float,
    root_height: float,
) -> tuple[dict[str, np.ndarray | float], list[int]]:
    """Build a 29-body + 14-hand motion entry from executed FlowBody frames."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"actions must be [T,{ACTION_DIM}], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("actions contain non-finite values")

    if action_mask is None:
        mask = np.ones_like(actions, dtype=bool)
    else:
        mask = np.asarray(action_mask, dtype=bool)
        if mask.shape != actions.shape:
            raise ValueError(f"action_mask must match actions {actions.shape}, got {mask.shape}")

    if not mask[:, ARM_SLICE].all():
        raise ValueError("all arm14 dimensions must be supervised for SONIC upper-body tracking")

    frame_count = actions.shape[0]
    body_dof = np.zeros((frame_count, 29), dtype=np.float32)
    body_dof[:, :15] = STANDING_BODY_DOF
    body_dof[:, 15:] = actions[:, ARM_SLICE]

    hand_mask = mask[:, HAND_SLICE]
    hand_dof = np.where(hand_mask, actions[:, HAND_SLICE], 0.0).astype(np.float32)
    hand_dof *= HAND_FLEXION_SIGN[None, :]
    active_hand_dims = np.flatnonzero(hand_mask.any(axis=0)).astype(int).tolist()

    pose_aa = np.zeros((frame_count, 30, 3), dtype=np.float32)
    pose_aa[:, 1:, :] = DOF_AXIS[None, :, :] * body_dof[:, :, None]
    root_trans = np.zeros((frame_count, 3), dtype=np.float32)
    root_trans[:, 2] = float(root_height)
    root_rot = np.zeros((frame_count, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0

    entry = {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa,
        # MotionLibRobot performs body FK from pose_aa and extracts indices
        # 29:43 as the separately actuated continuous hand reference.
        "dof": np.concatenate([body_dof, hand_dof], axis=-1),
        "root_rot": root_rot,
        "smpl_joints": np.zeros((frame_count, 24, 3), dtype=np.float32),
        "fps": float(fps),
    }
    return entry, active_hand_dims


def convert(args: argparse.Namespace) -> tuple[Path, Path]:
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    predictions = payload[args.prediction_key]
    if predictions.ndim != 3 or predictions.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"{args.prediction_key} must be [N,H,{ACTION_DIM}], got {tuple(predictions.shape)}"
        )

    actions = predictions[:, args.horizon_index].float().numpy()
    mask_tensor = payload.get(args.mask_key)
    action_mask = None
    if mask_tensor is not None:
        if mask_tensor.shape != predictions.shape:
            raise ValueError(
                f"{args.mask_key} must match {args.prediction_key}, got "
                f"{tuple(mask_tensor.shape)} vs {tuple(predictions.shape)}"
            )
        action_mask = mask_tensor[:, args.horizon_index].bool().numpy()

    entry, active_hand_dims = build_motion_entry(
        actions,
        action_mask,
        fps=args.fps,
        root_height=args.root_height,
    )
    motion_name = args.motion_name or output.stem
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({motion_name: entry}, output, compress=3)

    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "format": "sonic_motion_lib_body29_hand14_pkl",
        "motion_name": motion_name,
        "source": str(source),
        "selection": f"{args.prediction_key}[:, {args.horizon_index}, :]",
        "fps": float(args.fps),
        "num_frames": int(actions.shape[0]),
        "duration_seconds": float(actions.shape[0] / args.fps),
        "body_control": "SONIC policy 29 DOF; standing lower body plus FlowBody arm14 reference",
        "hand_control": "FlowBody hand14 direct continuous reference; unsupervised dimensions zeroed",
        "active_hand_dims": active_hand_dims,
        "hand_joint_names": HAND_JOINT_NAMES,
        "hand_sign_conversion": HAND_FLEXION_SIGN.tolist(),
        "root_trans_offset": [0.0, 0.0, float(args.root_height)],
        "root_rot_xyzw": [0.0, 0.0, 0.0, 1.0],
        "smpl_joints": "zeros placeholder; evaluation must force use_encoder=g1",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="FlowBody actions .pt")
    parser.add_argument("--output", required=True, help="Output motion-lib .pkl")
    parser.add_argument("--motion-name", default=None)
    parser.add_argument("--prediction-key", default="pred_actions")
    parser.add_argument("--mask-key", default="action_mask")
    parser.add_argument("--horizon-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--root-height", type=float, default=0.76)
    return parser


def main() -> None:
    output, manifest = convert(build_parser().parse_args())
    print(output)
    print(manifest)


if __name__ == "__main__":
    main()
