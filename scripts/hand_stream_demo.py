#!/usr/bin/env python3
"""Run the 43-DoF G1 with the body held at its default pose and the hand channel
sweeping through named poses.

This is the SuperMimic-side counterpart of SONIC's "external hand stream"
(``gear_sonic/envs/wrapper/manager_env_wrapper.py:246``): the body keeps its
29-DoF control loop, the hands are driven by a separate 14-D channel.  It needs
no trained checkpoint and no hand motion data — it is the wiring test for the
decoupling described in ``docs/HAND_STREAM.md``.

Example:
    python scripts/hand_stream_demo.py --poses open grasp open --hold 0.6
    python scripts/hand_stream_demo.py --poses fist --seconds 2 --save runs/hand_stream/fist.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.compat import G1_KD, G1_KP, g1_default_joint_pos  # noqa: E402
from diff_vla_wbc.hand_stream import (  # noqa: E402
    HAND_DIM,
    HAND_JOINT_NAMES,
    HAND_POSES,
    G1BodyIndex,
    HandStream,
    HandStreamCfg,
    load_hand_model,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, default=None, help="43-DoF G1 MJCF")
    ap.add_argument("--poses", nargs="+", default=["open", "grasp", "open"],
                    choices=sorted(HAND_POSES), help="pose sequence")
    ap.add_argument("--hold", type=float, default=0.6, help="seconds per pose")
    ap.add_argument("--seconds", type=float, default=None, help="total seconds (overrides --hold)")
    ap.add_argument("--dt", type=float, default=0.02, help="control period (s)")
    ap.add_argument("--substeps", type=int, default=10, help="physics steps per control step")
    ap.add_argument("--hand-kp", type=float, default=12.0)
    ap.add_argument("--hand-kd", type=float, default=0.5)
    ap.add_argument("--slew", type=float, default=8.0, help="pose blend rate (rad/s)")
    ap.add_argument("--save", type=Path, default=None, help="write trajectory .npy here")
    args = ap.parse_args()

    import mujoco

    model = load_hand_model(args.xml)
    model.opt.timestep = args.dt / args.substeps
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)

    body = G1BodyIndex(model)
    hand = HandStream(
        model,
        HandStreamCfg(pose=args.poses[0], kp=args.hand_kp, kd=args.hand_kd, slew_rate=args.slew),
    )
    print("[hand_stream] " + ", ".join(f"{k}={v}" for k, v in hand.info().items()))

    q0 = g1_default_joint_pos().numpy().astype(np.float64)
    kp = G1_KP.numpy().astype(np.float64)
    kd = G1_KD.numpy().astype(np.float64)
    body.write_state(data, q0, np.zeros_like(q0))
    mujoco.mj_forward(model, data)

    hold = args.seconds / len(args.poses) if args.seconds else args.hold
    n_control = int(round(hold * len(args.poses) / args.dt))
    hand.set_pose(args.poses[0], immediate=True)

    times, q_hist, target_hist, ctrl_hist = [], [], [], []
    for k in range(n_control):
        t = k * args.dt
        pose = args.poses[min(int(t // hold), len(args.poses) - 1)]
        if pose != hand.cfg.pose:
            hand.cfg.pose = pose
            hand.set_pose(pose)

        hand.step(args.dt)
        q = body.q(data)
        dq = body.dq(data)
        body_tau = kp * (q0 - q) - kd * dq
        ctrl = hand.assemble_ctrl(body_tau, data=data)
        for _ in range(args.substeps):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

        q_hand, _ = hand.read_state(data)
        times.append(data.time)
        q_hist.append(q_hand)
        target_hist.append(hand.targets())
        ctrl_hist.append(ctrl)

        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f"simulation diverged at t={data.time:.2f}s")

    q_hand = np.asarray(q_hist)
    targets = np.asarray(target_hist)
    err = np.abs(q_hand - targets)
    print("\nfinal hand pose error (rad):")
    for i, name in enumerate(HAND_JOINT_NAMES):
        print(f"  {name:32s} target={targets[-1, i]:+.3f}  q={q_hand[-1, i]:+.3f}  err={err[-1, i]:.3f}")
    print(f"\nmax err over run: {err.max():.3f} rad, mean: {err.mean():.3f} rad")
    print(f"body drift from default pose: {np.abs(body.q(data) - q0).max():.3f} rad")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save.with_suffix(".npz"),
            time=np.asarray(times),
            hand_qpos=q_hand,
            hand_targets=targets,
            ctrl=np.asarray(ctrl_hist),
            joint_names=np.asarray(HAND_JOINT_NAMES),
        )
        print(f"saved trajectory -> {args.save.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
