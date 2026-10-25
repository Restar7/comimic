#!/usr/bin/env python3
"""Deployment closed loop with the **SuperMimic WBC** as executor (the "supermimic decoder" route).

    ego camera -> GR00T VLA (joint-finetuned ckpt) -> 40x64 token + hands
              -> SuperMimic WBC policy (token-conditioned, from the joint ckpt)
              -> 29 joint targets -> 500 Hz PD -> MuJoCo

This is the pairing the ablation plan calls ``B（ft）``: the VLA that was back-propagated
through SuperMimic, executed BY that same SuperMimic WBC — not by the SONIC release
decoder that ``mosaic_n17/tools/closed_loop_mujoco.py`` still uses.

Usage:
    python3 scripts/closed_loop_vla_supermimic.py \
        --vla-ckpt runs/ckpt_vla_ft4000 --joint-ckpt runs/joint_diff_ft318/joint_004000.pt \
        --task task1 --steps 600 --out-video runs/closed_loop_sm/task1_ft4000.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "groot_repro"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_SCENES = Path(
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/SONIC_DATA/deployment/"
    "gear_sonic/data/robot_model/model_data/g1"
)
DEFAULT_RELEASE = Path(
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/"
    "gear_sonic_deploy/policy/release"
)
DEFAULT_REPRO = ROOT / "groot_repro"

TASKS = {
    "task1": {
        "scene": "scene_43dof_two_table.xml",
        "prompt": ("Navigate to the table ahead, grasp the milk carton, then turn around and "
                   "proceed to the other table to place the carton into the basket."),
    },
    "task2": {
        "scene": "scene_43dof.xml",
        "prompt": "Place the red cup into the drawer and close the drawer completely.",
    },
    "task3": {
        "scene": "scene_43dof_handover.xml",
        "prompt": "Retrieve the milk carton from the right plate and place it onto the left plate.",
    },
    "task4": {
        "scene": "scene_43dof_bend_and_pick.xml",
        "prompt": ("Squat down to grasp the aluminum can on the low chair, then stand up and "
                   "place it on the table ahead."),
    },
}


class VideoWriter:
    @staticmethod
    def _ffmpeg() -> str:
        import shutil
        import subprocess  # noqa: F401

        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        shim = ROOT / "closed_loop/bin/ffmpeg"
        if shim.exists():
            return str(shim)
        found = sorted((ROOT / "third_party/gr00t_deps/imageio_ffmpeg/binaries").glob("ffmpeg-*"))
        if found:
            return str(found[-1])
        raise SystemExit("no ffmpeg binary found")

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        import subprocess

        path.parent.mkdir(parents=True, exist_ok=True)
        self.shape = (height, width, 3)
        self.frames = 0
        self.proc = subprocess.Popen(
            [self._ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        arr = np.ascontiguousarray(frame, dtype=np.uint8)
        if arr.shape != self.shape:
            raise ValueError(f"frame {arr.shape} != {self.shape}")
        self.proc.stdin.write(arr.tobytes())
        self.frames += 1

    def release(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")


def observation(data, smap, renderer, cam, G, prompt):
    renderer.update_scene(data, cam)
    ego = renderer.render()
    q = smap.q(data)
    obs = {
        "video": {"ego_view": np.asarray(ego, np.uint8)[None, None]},
        "state": {
            "left_leg": q[0:6][None, None].astype(np.float32),
            "right_leg": q[6:12][None, None].astype(np.float32),
            "waist": q[12:15][None, None].astype(np.float32),
            "left_arm": q[15:22][None, None].astype(np.float32),
            "right_arm": q[22:29][None, None].astype(np.float32),
            "left_hand": smap.hands(data)[:7][None, None].astype(np.float32),
            "right_hand": smap.hands(data)[7:][None, None].astype(np.float32),
            "projected_gravity": G.projected_gravity(np.asarray(data.qpos[3:7], np.float64))[
                None, None].astype(np.float32),
        },
        "language": {"annotation.human.task_description": [[prompt]]},
    }
    return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vla-ckpt", type=Path, required=True, help="GR00T checkpoint dir (offline view)")
    ap.add_argument("--joint-ckpt", type=Path, required=True, help="joint trainer state (SuperMimic WBC)")
    ap.add_argument("--task", choices=sorted(TASKS), default="task1")
    ap.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES)
    ap.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    ap.add_argument("--repro-dir", type=Path, default=DEFAULT_REPRO)
    ap.add_argument("--steps", type=int, default=600, help="control ticks @50 Hz")
    ap.add_argument("--execute-steps", type=int, default=20, help="ticks executed per 40-tick plan")
    ap.add_argument("--out-video", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-gib", default="40GiB")
    ap.add_argument("--world-camera", default="side_view")
    ap.add_argument("--ego-camera", default="head_camera")
    ap.add_argument("--video-every", type=int, default=2)
    ap.add_argument("--hand-mode", choices=("vla", "open"), default="vla")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    cfg = TASKS[args.task]
    scene_path = args.scenes_dir / cfg["scene"]

    sys.path.insert(0, str(args.repro_dir))
    sys.path.insert(0, str(ROOT / "closed_loop/code"))
    sys.path.insert(0, str(ROOT / "Isaac-GR00T"))
    import meta_patch  # noqa: PLC0415

    meta_patch.install()
    if args.device.startswith("cuda"):
        import gpu_patch  # noqa: PLC0415

        gpu_patch.install(max_gpu_gib=args.gpu_gib)
    import mujoco  # noqa: PLC0415
    import g1_sim as G  # noqa: PLC0415
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: PLC0415

    from diff_vla_wbc.supermimic_backend import build_g1_assets  # noqa: PLC0415
    from diff_vla_wbc.train import JointTrainer  # noqa: PLC0415

    device = torch.device(args.device)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    smap = G.SceneMap(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.781]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    smap.set_q(data, G.DEFAULT_ANGLES.copy())
    smap.set_hands(data, np.zeros(14))
    mujoco.mj_forward(model, data)
    # Same spawn fix as the token-exact loop: the deploy MJCF puts the feet ~3.5 cm into
    # the floor; without this the first solver step fires a huge impulse.
    for _ in range(4):
        if data.ncon == 0:
            break
        deepest = min(float(c.dist) for c in data.contact[: data.ncon])
        if deepest >= -1e-4:
            break
        data.qpos[2] += -deepest + 1e-3
        mujoco.mj_forward(model, data)
    print(f"[sim] scene={cfg['scene']} nq={model.nq} spawn z={data.qpos[2]:.3f} "
          f"contacts={data.ncon}", flush=True)

    assets = build_g1_assets(device=device)
    bundle = JointTrainer.load_bundle_from_checkpoint(args.joint_ckpt, assets)
    if int(getattr(bundle.policy, "token_dim", 0)) != 64:
        raise SystemExit("SuperMimic WBC is not token-conditioned")
    print(f"[sm] WBC {args.joint_ckpt} (token-conditioned)", flush=True)

    policy = Gr00tPolicy(embodiment_tag="unitree_g1_sonic", model_path=str(args.vla_ckpt),
                         device=args.device, strict=True)
    print(f"[vla] {args.vla_ckpt}", flush=True)

    renderer = mujoco.Renderer(model, 480, 640)
    world_cam = mujoco.MjvCamera()
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.world_camera)
    world_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    world_cam.fixedcamid = wid if wid >= 0 else 0
    ego_cam = mujoco.MjvCamera()
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.ego_camera)
    if eid < 0:
        raise SystemExit(f"scene has no {args.ego_camera!r} camera")
    ego_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    ego_cam.fixedcamid = eid

    def render(cam):
        renderer.update_scene(data, cam)
        return renderer.render()

    writer = VideoWriter(args.out_video, 1280, 480, 50.0 / args.video_every)
    sub = int(round(0.02 / model.opt.timestep))
    xi = torch.zeros(1, 33, device=device)
    history = None
    chunk = None
    consumed = 0
    plans = 0
    rows = []
    with torch.no_grad():
        for tick in range(args.steps):
            if chunk is None or consumed >= args.execute_steps:
                obs = observation(data, smap, renderer, ego_cam, G, cfg["prompt"])
                action, _info = policy.get_action(obs)
                tokens = np.asarray(action["motion_token"], np.float32).reshape(-1, 64)[:40]
                hands = np.concatenate((
                    np.asarray(action["left_hand_joints"], np.float32).reshape(-1, 7)[:40],
                    np.asarray(action["right_hand_joints"], np.float32).reshape(-1, 7)[:40],
                ), axis=1)
                chunk = (tokens, hands)
                consumed = 0
                plans += 1
                print(f"[plan {plans}] tick={tick} token=[{tokens[0].min():.2f},{tokens[0].max():.2f}]",
                      flush=True)
            tokens, hands = chunk
            idx = min(consumed, len(tokens) - 1)
            token = torch.as_tensor(tokens[idx], dtype=torch.float32, device=device).reshape(1, 64)

            state = np.concatenate([data.qpos[0:3], data.qpos[3:7], smap.q(data),
                                    data.qvel[0:3], data.qvel[3:6], smap.dq(data)])
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            act29 = bundle.act(bundle.flatten_history(history), token) if history is not None \
                else torch.zeros(1, 29, device=device)
            q_target = (assets.default_joint_pos + assets.action_scale * act29[0]).cpu().numpy()
            hand_t = np.zeros(14) if args.hand_mode == "open" else hands[idx]
            hand_t = np.clip(hand_t, -1.5, 1.5)

            for _ in range(sub):
                smap.set_ctrl(data, np.clip(G.KP * (q_target - smap.q(data))
                                            - G.KD * smap.dq(data), -88, 88))
                hq = smap.hands(data)
                smap.set_hand_ctrl(data, np.concatenate((
                    np.clip(2.0 * (hand_t[:7] - hq[:7]) - 0.05 * smap.hand_dq(data)[:7], -2.4, 2.4),
                    np.clip(2.0 * (hand_t[7:] - hq[7:]) - 0.05 * smap.hand_dq(data)[7:], -2.4, 2.4),
                )))
                mujoco.mj_step(model, data)

            state = np.concatenate([data.qpos[0:3], data.qpos[3:7], smap.q(data),
                                    data.qvel[0:3], data.qvel[3:6], smap.dq(data)])
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            clean = bundle.observation(state_t, act29, xi)
            history = clean.unsqueeze(0).repeat(10, 1, 1) if history is None \
                else torch.cat((history[1:], clean.unsqueeze(0)), dim=0)

            tilt = float(np.degrees(np.arccos(np.clip(
                G.quat_to_rotmat(data.qpos[3:7])[2, 2], -1, 1))))
            rows.append({"tick": tick, "z": float(data.qpos[2]), "tilt_deg": tilt})
            if tick % 100 == 0:
                print(f"[{tick:4d}] z={data.qpos[2]:.3f} tilt={tilt:5.1f}° plans={plans}",
                      flush=True)
            if tick % args.video_every == 0:
                writer.write(np.concatenate((render(world_cam), render(ego_cam)), axis=1))
            consumed += 1
    writer.release()
    renderer.close()

    summary = {
        "task": args.task, "vla": str(args.vla_ckpt), "wbc": str(args.joint_ckpt),
        "scene": cfg["scene"], "ticks": args.steps, "seconds": round(args.steps * 0.02, 2),
        "plans": plans,
        "min_z": float(min(r["z"] for r in rows)),
        "max_tilt_deg": float(max(r["tilt_deg"] for r in rows)),
        "final_z": rows[-1]["z"], "final_tilt_deg": rows[-1]["tilt_deg"],
        "fell": bool(max(r["tilt_deg"] for r in rows) > 60.0 or min(r["z"] for r in rows) < 0.45),
        "video": str(args.out_video),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({**summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
