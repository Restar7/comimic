#!/usr/bin/env python3
"""Verify that every deployment scene loads and (optionally) renders.

    python verify_scenes.py
    python verify_scenes.py --render out/     # writes one png per scene

Requires only `mujoco` (pip install mujoco).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
SCENES = [
    "scene_43dof.xml",
    "scene_pickplace.xml",
    "scene_43dof_two_table.xml",
    "scene_43dof_handover.xml",
    "scene_43dof_bend_and_pick.xml",
    "scene_43dof_two_table_carton_aligned.xml",
    "scene_twotables.xml",
    "scene_twotables_carton_aligned.xml",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=Path, default=None, help="directory for preview pngs")
    ap.add_argument("--camera", default="auto", help="\"auto\" or a scene camera name")
    args = ap.parse_args()

    if args.render:
        import cv2

        args.render.mkdir(parents=True, exist_ok=True)

    ok = 0
    for name in SCENES:
        path = HERE / "g1" / name
        if not path.exists():
            print(f"{name:40s} MISSING")
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"{name:40s} FAIL  {str(exc)[:70]}")
            continue
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cams = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
        ]
        print(f"{name:40s} nq={model.nq:3d} ngeom={model.ngeom:4d} cams={cams}")
        ok += 1

        if args.render:
            renderer = mujoco.Renderer(model, 360, 480)
            cam = mujoco.MjvCamera()
            if args.camera == "auto":
                # frame the whole scene, not the robot's head
                mujoco.mjv_defaultCamera(cam)
                cam.lookat[:] = [2.0, 0.0, 0.8]
                cam.distance = 4.0
                cam.azimuth = 130.0
                cam.elevation = -20.0
            else:
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = max(cid, 0)
            renderer.update_scene(data, cam)
            img = renderer.render()
            cv2.imwrite(str(args.render / f"{name.replace('.xml', '')}.png"), img[..., ::-1])
            renderer.close()

    print(f"\n{ok}/{len(SCENES)} scenes loaded")


if __name__ == "__main__":
    main()
