#!/usr/bin/env python3
"""Hand-interface mapping between our 14-D continuous hands and HumanoidArena's 2-D binary grip.

Our VLA emits ``g1_hand_joints_14`` (7 joints per hand, continuous).  HumanoidArena's
evaluation contract uses ``vla_action[38:40]`` — **2 binary left/right gripper commands**.
Anything that crosses between the two needs an explicit, documented mapping, otherwise the
grasping tasks (grap_cup / pp_box / double_desk) are judged under a different hand semantics
than the one the policy was trained for.

Definitions
-----------
* ``continuous``: ``(B, 14)`` or ``(..., 14)`` joint angles, order ``[left(7), right(7)]``
  (same order as ``g1_hand_joints_14`` / ``hand_action_left|right`` in the datasets).
* ``binary``: ``(..., 2)`` in ``[0, 1]``; ``1`` = closed/grasp, ``0`` = open.
* A hand is "closed" when its **mean absolute flexion** exceeds ``threshold``.  The datasets'
  open/grasp poses differ in sign and magnitude per task, so the threshold is applied to the
  per-hand mean and should be calibrated from a task's own recorded npz (open vs grasp frames)
  — the same idea as ``--hand-calib-npz`` in the closed-loop harness.

Calibration helper: ``calibrate_from_npz`` derives ``(open_value, grasp_value)`` per hand from a
recorded clip and returns the midpoint as the threshold, so every task gets its own operating
point instead of a global constant.

Usage
-----
    python3 scripts/hand_interface.py --calibrate data/npz/g1_tasks/task_1_*.npz
    python3 scripts/hand_interface.py --self-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def per_hand_flexion(continuous: np.ndarray) -> np.ndarray:
    """``(..., 14)`` -> ``(..., 2)`` mean absolute flexion per hand."""
    c = np.asarray(continuous, dtype=np.float64)
    if c.shape[-1] != 14:
        raise ValueError(f"expected 14 hand dims (7+7), got {c.shape[-1]}")
    return np.stack((np.abs(c[..., :7]).mean(-1), np.abs(c[..., 7:]).mean(-1)), axis=-1)


def calibrate_from_npz(path: str | Path) -> dict[str, float]:
    """Derive per-hand ``(open, grasp, threshold)`` from one recorded clip."""
    z = np.load(path, allow_pickle=True)
    key = "g1_hand_joints_14" if "g1_hand_joints_14" in z.files else "hand_action_left"
    if key == "g1_hand_joints_14":
        hands = np.asarray(z[key], dtype=np.float64)
    else:  # HA raw layout: two (T,7) arrays
        hands = np.concatenate((np.asarray(z["hand_action_left"], dtype=np.float64),
                                np.asarray(z["hand_action_right"], dtype=np.float64)), -1)
    f = per_hand_flexion(hands)                      # (T,2)
    open_pose = np.percentile(f, 5, axis=0)          # quietest 5% = open
    grasp_pose = np.percentile(f, 95, axis=0)        # strongest 5% = grasp
    thr = (open_pose + grasp_pose) / 2
    # a hand that never moves (single-arm task, or an idle channel) gives a degenerate
    # threshold of ~0; fall back to a small absolute value and flag it
    inactive = (grasp_pose - open_pose) < 0.05
    thr = np.where(inactive, open_pose + 0.05, thr)
    return {
        "open_left": float(open_pose[0]), "open_right": float(open_pose[1]),
        "grasp_left": float(grasp_pose[0]), "grasp_right": float(grasp_pose[1]),
        "threshold_left": float(thr[0]), "threshold_right": float(thr[1]),
        "frames": int(f.shape[0]),
        "inactive_left": bool((grasp_pose - open_pose)[0] < 0.05),
        "inactive_right": bool((grasp_pose - open_pose)[1] < 0.05),
    }


def to_binary(continuous: np.ndarray, threshold: float | tuple[float, float] = 0.1) -> np.ndarray:
    """``(...,14)`` continuous -> ``(...,2)`` binary grip commands."""
    thr = np.asarray(threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr = np.array([thr, thr])
    return (per_hand_flexion(continuous) > thr).astype(np.float32)


def to_continuous(binary: np.ndarray, open_pose: np.ndarray, grasp_pose: np.ndarray) -> np.ndarray:
    """``(...,2)`` binary -> ``(...,14)`` continuous by lerping between recorded poses.

    This is what a *deployment* path needs when the environment only speaks binary but the
    robot expects joint targets: ``grasp ? grasp_pose : open_pose``.
    """
    b = np.asarray(binary, dtype=np.float64)
    if b.shape[-1] != 2:
        raise ValueError(f"expected 2 binary dims, got {b.shape[-1]}")
    op = np.asarray(open_pose, dtype=np.float64).reshape(14)
    gp = np.asarray(grasp_pose, dtype=np.float64).reshape(14)
    left = b[..., 0:1] * gp[:7] + (1 - b[..., 0:1]) * op[:7]
    right = b[..., 1:2] * gp[7:] + (1 - b[..., 1:2]) * op[7:] if b.shape[-1] > 1 else left[..., 7:]
    del right
    left_b = b[..., 0:1].repeat(7, -1)
    right_b = b[..., 1:2].repeat(7, -1)
    return np.concatenate((left_b * gp[:7] + (1 - left_b) * op[:7],
                           right_b * gp[7:] + (1 - right_b) * op[7:]), axis=-1)


def self_test() -> None:
    rng = np.random.default_rng(0)
    open_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    grasp_c = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                        0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
    assert to_binary(open_c, 0.2).tolist() == [0.0, 0.0]
    assert to_binary(grasp_c, 0.2).tolist() == [1.0, 1.0]
    back = to_continuous(np.array([1.0, 0.0]), open_c, grasp_c)
    assert np.allclose(back[:7], grasp_c[:7]) and np.allclose(back[7:], open_c[7:])
    batch = rng.normal(0, 0.2, (5, 14))
    assert to_binary(batch, (0.1, 0.3)).shape == (5, 2)
    print("[ok] hand_interface self-test passed (binary/continuous round-trip)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", type=Path, help="recorded npz used to derive the per-hand threshold")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override; otherwise the calibrated midpoint is used")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test or (args.calibrate is None and args.threshold is None):
        self_test()
        return
    if args.calibrate is not None:
        cal = calibrate_from_npz(args.calibrate)
        for k, v in cal.items():
            print(f"  {k:16s} {v}")
        thr = (cal["threshold_left"], cal["threshold_right"])
        print(f"\n建议 --threshold 用 ({thr[0]:.4f}, {thr[1]:.4f})（左/右手中位 flexion）")
        return
    print(f"threshold = {args.threshold}")


if __name__ == "__main__":
    main()
