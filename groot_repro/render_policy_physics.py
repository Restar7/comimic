"""Render the GR00T policy's own output as a video (physics closed loop).

Loop:
    MuJoCo head_camera render  ->  GR00T policy -> motion tokens (40 x 64)
    token[t] -> SONIC decoder -> joint targets -> robot pose -> next render

The SONIC decoder output is tracked by a PD loop inside MuJoCo, so the robot
actually balances and the base moves under physics.  The observation history
fed to the decoder must use `q - default_angles`; missing that offset makes the
decoder command nonsense and the robot falls within a second.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq

import g1_sim as G
from sonic_loop import SonicController, CONTROL_DT, HISTORY

SCENE = ("/home/wusichen/SONIC_DATA/deployment/gear_sonic/data/robot_model/"
         "model_data/g1/scene_43dof_two_table.xml")
SUBSTEPS = 10  # 50 Hz control / 500 Hz physics
DATASET = "/home/wusichen/SONIC_DATA/2026-07-14/datasets/2026-07-14-20-30-01"
PROMPT = (
    "Pick up the carton from the table in front of you, turn around, walk "
    "to the table behind you, and place the carton into the box."
)


def load_initial_pose():
    return G.DEFAULT_ANGLES.copy(), np.zeros(7), np.zeros(7)


def build_policy_obs(q_mj, hand_l, hand_r, rgb, prompt):
    state = {
        "left_leg": q_mj[0:6],
        "right_leg": q_mj[6:12],
        "waist": q_mj[12:15],
        "left_arm": q_mj[15:22],
        "right_arm": q_mj[22:29],
        "left_hand": hand_l,
        "right_hand": hand_r,
        "projected_gravity": np.array([0.0, 0.0, -1.0]),
    }
    return {
        "video": {"ego_view": np.asarray(rgb, np.uint8)[None, None]},
        "state": {k: np.asarray(v, np.float32)[None, None] for k, v in state.items()},
        "language": {"annotation.human.task_description": [[prompt]]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/wusichen/groot_repro/ckpt_sirius_50k")
    ap.add_argument("--chunks", type=int, default=12, help="policy calls (0.8 s each)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", default="/home/wusichen/groot_repro/out/policy_rollout.mp4")
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--world-camera", default="side_view")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-gib", default="3.0GiB")
    args = ap.parse_args()

    q0, hl0, hr0 = load_initial_pose()

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    smap = G.SceneMap(m)
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, 0.781]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    smap.set_q(d, q0)
    smap.set_hands(d, np.concatenate([hl0, hr0]))
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")

    world_cam = mujoco.MjvCamera()
    wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.world_camera)
    if wid >= 0:
        world_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        world_cam.fixedcamid = wid
    else:
        mujoco.mjv_defaultCamera(world_cam)
        world_cam.lookat[:] = [0.35, 0.0, 0.55]
        world_cam.distance = 2.4
        world_cam.azimuth = 118.0
        world_cam.elevation = -18.0

    def render(view="head"):
        renderer.update_scene(d, cam if view == "head" else world_cam)
        return renderer.render()

    # policy
    import meta_patch

    meta_patch.install()
    if args.device.startswith("cuda"):
        import gpu_patch

        gpu_patch.install(max_gpu_gib=args.gpu_gib)
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(embodiment_tag="unitree_g1_sonic", model_path=args.ckpt,
                         device=args.device, strict=True)

    ctrl = SonicController("sonic_models/model_decoder.onnx", init_q=q0)
    ctrl.hist.clear()
    for _ in range(HISTORY):
        ctrl.hist.append({
            "base_ang_vel": np.zeros(3),
            "q": (q0 - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
            "dq": np.zeros(29),
            "last_action": np.zeros(29),
            "gravity": np.array([0.0, 0.0, -1.0]),
        })

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 480))
    log = []
    t_start = time.time()
    for chunk in range(args.chunks):
        rgb = render()
        obs = build_policy_obs(smap.q(d), smap.hands(d)[:7], smap.hands(d)[7:], rgb, args.prompt)
        t0 = time.time()
        action, _ = policy.get_action(obs)
        infer_s = time.time() - t0
        tokens = np.asarray(action["motion_token"], np.float32).reshape(-1, 64)[:40]
        hands_l = np.asarray(action["left_hand_joints"], np.float32).reshape(-1, 7)[:40]
        hands_r = np.asarray(action["right_hand_joints"], np.float32).reshape(-1, 7)[:40]

        for t in range(40):
            q_now = smap.q(d).copy()
            dq_now = smap.dq(d).copy()
            q_target = ctrl.step(tokens[t], q_now, dq_now, d.qpos[3:7].copy(), d.qvel[3:6].copy())
            for _ in range(SUBSTEPS):
                smap.set_ctrl(d, np.clip(
                    G.KP * (q_target - smap.q(d)) - G.KD * smap.dq(d), -88, 88))
                hq = smap.hands(d)
                hdq = smap.hand_dq(d)
                smap.set_hand_ctrl(d, np.concatenate([
                    np.clip(2.0 * (hands_l[t] - hq[:7]) - 0.05 * hdq[:7], -2.4, 2.4),
                    np.clip(2.0 * (hands_r[t] - hq[7:]) - 0.05 * hdq[7:], -2.4, 2.4),
                ]))
                mujoco.mj_step(m, d)
            if t % 2 == 0:  # 50 Hz control -> 25 fps video
                world = render("world")
                ego = render("head")
                frame = np.concatenate([world, ego], axis=1)
                cv2.putText(frame, "policy output (world view)", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "ego_view fed to the policy", (650, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame[..., ::-1])
        log.append({"chunk": chunk, "infer_seconds": round(infer_s, 2)})
        print(f"chunk {chunk+1}/{args.chunks}: policy {infer_s:.1f}s, "
              f"token range [{tokens.min():.2f},{tokens.max():.2f}]", flush=True)

    writer.release()
    Path(args.out).with_suffix(".json").write_text(json.dumps(log, indent=2))
    print(f"wrote {args.out} ({args.chunks*0.8:.1f}s of motion, "
          f"{time.time()-t_start:.0f}s wall)")


if __name__ == "__main__":
    main()
