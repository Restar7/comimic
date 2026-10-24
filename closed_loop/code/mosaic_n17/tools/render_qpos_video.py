#!/usr/bin/env python3
"""Re-render a closed-loop qpos trajectory (``--log-qpos``) into a MuJoCo video.

The loop records ``qpos.npy`` (one row per 50 Hz control tick).  Rendering it
afterwards means a *successful* run never has to be repeated to get a video --
important because the closed loop is numerically non-deterministic (two runs at
the same seed can diverge).

Usage::

    python -m mosaic_n17.tools.render_qpos_video \
        --qpos runs/.../task1_s0/qpos.npy --scene <scene.xml> \
        --out runs/.../task1_s0_replay.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .closed_loop_mujoco import DEFAULT_SCENES, _FfmpegVideoWriter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qpos", type=Path, required=True)
    ap.add_argument("--scene", type=Path, default=None)
    ap.add_argument("--scene-name", default="scene_43dof_two_table.xml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--world-camera", default="side_view")
    ap.add_argument("--ego-camera", default="head_camera")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument(
        "--stride", type=int, default=2, help="render every N control ticks (2 -> 25 fps)"
    )
    args = ap.parse_args()

    scene = args.scene if args.scene is not None else Path(DEFAULT_SCENES) / args.scene_name
    qpos = np.load(args.qpos)
    if qpos.ndim != 2:
        raise SystemExit(f"qpos.npy must be [T, nq], got {qpos.shape}")

    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    if qpos.shape[1] != m.nq:
        raise SystemExit(f"qpos width {qpos.shape[1]} != model nq {m.nq}")

    half_w = args.width // 2
    renderer = mujoco.Renderer(m, args.height, half_w)

    def camera(name: str):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam.fixedcamid < 0:
            raise SystemExit(f"camera {name!r} not in {scene}")
        return cam

    world_cam, ego_cam = camera(args.world_camera), camera(args.ego_camera)
    writer = _FfmpegVideoWriter(args.out, args.width, args.height, 50.0 / args.stride)
    try:
        for tick in range(0, qpos.shape[0], args.stride):
            d.qpos[:] = qpos[tick]
            mujoco.mj_forward(m, d)
            frames = []
            for cam in (world_cam, ego_cam):
                renderer.update_scene(d, cam)
                frames.append(renderer.render())
            writer.write(np.concatenate(frames, axis=1))
    finally:
        writer.release()
    print(f"wrote {args.out} ({qpos.shape[0]} ticks, stride {args.stride})", flush=True)


if __name__ == "__main__":
    main()
