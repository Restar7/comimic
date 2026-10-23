"""Run one policy episode headless and log MuJoCo ground truth for success checks.

Same closed loop as render_policy_physics.py, but instead of a video it writes a
`state.jsonl` in the format the deployment's check_*_success.py scripts expect:
one record per 50 Hz control tick with `elapsed`, `tracked_bodies`, base pose.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

import g1_sim as G
from sonic_loop import CONTROL_DT, HISTORY, SonicController

SCENE_DIR = (
    "/home/wusichen/SONIC_DATA/deployment/gear_sonic/data/robot_model/model_data/g1"
)

TASKS = {
    "apple": dict(
        scene=f"{SCENE_DIR}/scene_pickplace.xml",
        prompt=(
            "Pick up the apple from the table in front of you, turn around, walk "
            "to the table behind you, and place the apple into the box."
        ),
        track=["apple", "box"],
        target="apple",
        goal="box",
        world_camera="side_view",
    ),
    "carton": dict(
        scene=f"{SCENE_DIR}/scene_43dof_two_table.xml",
        prompt=(
            "Pick up the carton from the table in front of you, turn around, walk "
            "to the table behind you, and place the carton into the box."
        ),
        track=["sugar_box", "box"],
        target="sugar_box",
        goal="box",
        world_camera="side_view",
    ),
    "redcup": dict(
        scene=f"{SCENE_DIR}/scene_43dof.xml",
        prompt=(
            "Pick up the red cup, place it into the upper drawer of the cabinet, "
            "and close the drawer."
        ),
        track=["mug", "drawer"],
        target="mug",
        goal="drawer",
        world_camera="side_view",
    ),
    "handover": dict(
        scene=f"{SCENE_DIR}/scene_43dof_handover.xml",
        prompt=(
            "pick up the carton with the right hand, hand it over to the left hand, "
            "and place it into the box"
        ),
        track=["sugar_box", "tgt_box"],
        target="sugar_box",
        goal="tgt_box",
        world_camera="front_view",
    ),
}


def build_obs(q_mj, hand_l, hand_r, rgb, prompt):
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
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--chunks", type=int, default=75, help="75 chunks = 60 s")
    ap.add_argument("--ckpt", default="/home/wusichen/groot_repro/ckpt_sirius_50k")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-gib", default="2.4GiB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/home/wusichen/groot_repro/out/runs")
    args = ap.parse_args()

    cfg = TASKS[args.task]
    out_dir = Path(args.out_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.jsonl"

    m = mujoco.MjModel.from_xml_path(cfg["scene"])
    d = mujoco.MjData(m)
    smap = G.SceneMap(m)
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, 0.781]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    smap.set_q(d, G.DEFAULT_ANGLES)
    smap.set_hands(d, np.zeros(14))
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")

    body_ids = {}
    for name in cfg["track"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise SystemExit(f"body {name} not in {cfg['scene']}")
        body_ids[name] = bid

    import meta_patch

    meta_patch.install()
    if args.device.startswith("cuda"):
        import gpu_patch

        gpu_patch.install(max_gpu_gib=args.gpu_gib)
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag="unitree_g1_sonic",
        model_path=args.ckpt,
        device=args.device,
        strict=True,
    )

    ctrl = SonicController("sonic_models/model_decoder.onnx", init_q=G.DEFAULT_ANGLES)
    ctrl.hist.clear()
    for _ in range(HISTORY):
        ctrl.hist.append(
            {
                "base_ang_vel": np.zeros(3),
                "q": np.zeros(29),
                "dq": np.zeros(29),
                "last_action": np.zeros(29),
                "gravity": np.array([0.0, 0.0, -1.0]),
            }
        )

    sub = int(round(CONTROL_DT / m.opt.timestep))
    rows = []
    t0 = time.time()
    tick_global = 0
    for chunk in range(args.chunks):
        import torch

        renderer.update_scene(d, cam)
        rgb = renderer.render()
        obs = build_obs(smap.q(d), smap.hands(d)[:7], smap.hands(d)[7:], rgb, cfg["prompt"])
        torch.manual_seed(args.seed * 100000 + chunk)
        np.random.seed(args.seed * 100000 + chunk)
        t1 = time.time()
        action, _ = policy.get_action(obs)
        infer_s = time.time() - t1
        tokens = np.asarray(action["motion_token"], np.float32).reshape(-1, 64)[:40]
        hands_l = np.asarray(action["left_hand_joints"], np.float32).reshape(-1, 7)[:40]
        hands_r = np.asarray(action["right_hand_joints"], np.float32).reshape(-1, 7)[:40]

        for t in range(40):
            q_target = ctrl.step(
                tokens[t],
                d.qpos[7:36].copy(),
                d.qvel[6:35].copy(),
                d.qpos[3:7].copy(),
                d.qvel[3:6].copy(),
            )
            for _ in range(sub):
                smap.set_ctrl(d, np.clip(
                    G.KP * (q_target - smap.q(d)) - G.KD * smap.dq(d), -88, 88))
                hq = smap.hands(d)
                hdq = smap.hand_dq(d)
                smap.set_hand_ctrl(d, np.concatenate([
                    np.clip(2.0 * (hands_l[t] - hq[:7]) - 0.05 * hdq[:7], -2.4, 2.4),
                    np.clip(2.0 * (hands_r[t] - hq[7:]) - 0.05 * hdq[7:], -2.4, 2.4),
                ]))
                mujoco.mj_step(m, d)

            tilt = float(
                np.degrees(
                    np.arccos(np.clip(G.quat_to_rotmat(d.qpos[3:7])[2, 2], -1, 1))
                )
            )
            rows.append(
                {
                    "elapsed": round(tick_global * CONTROL_DT, 4),
                    "base_position": d.qpos[0:3].tolist(),
                    "tilt_deg": tilt,
                    "tracked_bodies": {
                        name: {"position": d.xpos[bid].tolist()}
                        for name, bid in body_ids.items()
                    },
                }
            )
            tick_global += 1
        print(
            f"[{args.task}] chunk {chunk+1}/{args.chunks} infer {infer_s:.1f}s "
            f"tilt {rows[-1]['tilt_deg']:.0f}",
            flush=True,
        )

    state_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    meta = {
        "task": args.task,
        "scene": cfg["scene"],
        "prompt": cfg["prompt"],
        "seconds": round(tick_global * CONTROL_DT, 1),
        "wall_seconds": round(time.time() - t0, 1),
        "final_tilt_deg": rows[-1]["tilt_deg"],
        "final_base": rows[-1]["base_position"],
        "fell": rows[-1]["tilt_deg"] > 60 or rows[-1]["base_position"][2] < 0.35,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {state_path}")


if __name__ == "__main__":
    main()
