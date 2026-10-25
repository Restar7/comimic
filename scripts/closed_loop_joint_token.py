#!/usr/bin/env python3
"""Closed-loop MuJoCo inference of a **token-conditioned** joint/SuperMimic ckpt, with video.

Difference from ``scripts/eval_joint_token_mujoco.py`` (which uses ``diff_vla_wbc.mujoco_env``):

* the sim is the *deployment* G1 scene the closed-loop videos use
  (``SONIC_DATA/.../g1/scene_43dof*.xml`` + ``groot_repro/g1_sim.py`` name-based joint map),
  with the real motor limits and a **500 Hz PD** recomputed inside every physics substep
  (the ``mujoco_env`` version holds one torque for the whole 20 ms control period, which is
  numerically unstable at these gains);
* every control tick is rendered (world view | head camera) into an mp4.

Conditioning is the **recorded** 64-d SONIC token of the ft318 npz (``g1_motion_token_64``),
one token per tick: this measures the WBC on a perfect token (oracle-token closed loop).
Hands are driven by the recorded ``g1_hand_joints_14`` unless ``--hand-mode zero``.

Usage:
    python3 scripts/closed_loop_joint_token.py \
        --joint-ckpt runs/joint_sm24k_hand/joint_002000.pt \
        --clip data/npz/g1_tasks/task_1_...__ep000000.npz --steps 600 \
        --out-video runs/closed_loop_joint/task1_hand2000.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Must be set before mujoco is imported (``diff_vla_wbc`` pulls it in): the renderer
# picks its GL backend at import time.
os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "groot_repro"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diff_vla_wbc.supermimic_backend import (  # noqa: E402
    SuperMimicBundle,
    build_g1_assets,
    load_motion_library,
)
from diff_vla_wbc.train import JointTrainer  # noqa: E402

DEFAULT_SCENES = Path(
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/SONIC_DATA/deployment/"
    "gear_sonic/data/robot_model/model_data/g1"
)


class VideoWriter:
    """H.264 writer through ffmpeg (system binary, repo shim, or imageio-ffmpeg)."""

    @staticmethod
    def _ffmpeg() -> str:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        shim = ROOT / "closed_loop/bin/ffmpeg"
        if shim.exists():
            return str(shim)
        found = sorted((ROOT / "third_party/gr00t_deps/imageio_ffmpeg/binaries").glob("ffmpeg-*"))
        if found:
            return str(found[-1])
        raise SystemExit("no ffmpeg binary found (install ffmpeg or the imageio-ffmpeg wheel)")

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        exe = self._ffmpeg()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.proc = subprocess.Popen(
            [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self.shape = (height, width, 3)

    def write(self, frame: np.ndarray) -> None:
        array = np.ascontiguousarray(frame, dtype=np.uint8)
        if array.shape != self.shape:
            raise ValueError(f"frame {array.shape} != {self.shape}")
        self.proc.stdin.write(array.tobytes())
        self.frames += 1

    def release(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-ckpt", type=Path, default=None, help="joint trainer state")
    ap.add_argument("--supermimic-ckpt", type=Path, default=None,
                    help="plain SuperMimic ckpt (baseline before any joint finetune)")
    ap.add_argument("--clip", type=Path, required=True, help="ft318 npz clip")
    ap.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES)
    ap.add_argument("--scene", default="scene_43dof.xml")
    ap.add_argument("--steps", type=int, default=600, help="control ticks @50 Hz (600 = 12 s)")
    ap.add_argument("--out-video", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--world-camera", default="side_view")
    ap.add_argument("--ego-camera", default="head_camera")
    ap.add_argument("--video-every", type=int, default=2, help="frames per N ticks (2 -> 25 fps)")
    ap.add_argument("--hand-mode", choices=("recorded", "zero"), default="recorded")
    ap.add_argument("--spawn-lift", choices=("auto", "none"), default="auto",
                    help="auto = lift the spawn so the feet just touch the floor")
    ap.add_argument("--torque-limit", type=float, default=88.0, help="Nm, same as the deploy loop")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco  # noqa: PLC0415

    import g1_sim as G  # noqa: PLC0415

    device = torch.device(args.device)
    assets = build_g1_assets(device=device)
    if args.joint_ckpt:
        bundle = JointTrainer.load_bundle_from_checkpoint(args.joint_ckpt, assets)
    elif args.supermimic_ckpt:
        bundle = SuperMimicBundle.from_checkpoint(args.supermimic_ckpt, assets)
    else:
        raise SystemExit("pass --joint-ckpt or --supermimic-ckpt")
    if int(getattr(bundle.policy, "token_dim", 0)) != 64:
        raise SystemExit("this script is for 64-d token-conditioned checkpoints")

    library = load_motion_library([args.clip], assets, device=device)
    clip = library.clips[0]
    frames = int(clip.end - clip.start)
    step_count = min(args.steps, frames - 1)
    npz = np.load(args.clip)
    tokens = npz["g1_motion_token_64"]
    hands = npz["g1_hand_joints_14"] if args.hand_mode == "recorded" else np.zeros((len(tokens), 14))

    scene_path = args.scenes_dir / args.scene
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    smap = G.SceneMap(model)
    mujoco.mj_resetData(model, data)
    qpos, qvel = library.ref_qpos_qvel(torch.tensor([int(clip.start)], device=library.device))
    data.qpos[0:3] = qpos[0][:3].cpu().numpy()
    data.qpos[3:7] = qpos[0][3:7].cpu().numpy()
    smap.set_q(data, qpos[0][7:36].cpu().numpy())
    smap.set_hands(data, np.zeros(14))
    mujoco.mj_forward(model, data)
    # The ft318 reference z was estimated from foot contact through FK, so spawning exactly on
    # it pushes the foot collision boxes ~3.5 cm into the floor; the first solver step then fires
    # a huge normal impulse and the rollout is decided by contact noise.  Lift until the deepest
    # contact is just touching.
    if args.spawn_lift == "auto":
        for _ in range(4):
            if data.ncon == 0:
                break
            deepest = min(float(c.dist) for c in data.contact[: data.ncon])
            if deepest >= -1e-4:
                break
            data.qpos[2] += -deepest + 1e-3
            mujoco.mj_forward(model, data)
        print(f"[sim] spawn lifted to z={data.qpos[2]:.4f} (deepest contact "
              f"{min((float(c.dist) for c in data.contact[: data.ncon]), default=0.0):+.4f} m)")
    print(f"[sim] scene={args.scene} nq={model.nq} nu={model.nu} dt={model.opt.timestep} "
          f"spawn z={data.qpos[2]:.3f} contacts={data.ncon}")

    renderer = mujoco.Renderer(model, 480, 640)
    world_cam = mujoco.MjvCamera()
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.world_camera)
    world_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    world_cam.fixedcamid = wid if wid >= 0 else 0
    ego_cam = mujoco.MjvCamera()
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.ego_camera)
    ego_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    if eid < 0:
        raise SystemExit(f"scene has no {args.ego_camera!r} camera")
    ego_cam.fixedcamid = eid

    def render(cam) -> np.ndarray:
        renderer.update_scene(data, cam)
        return renderer.render()

    writer = VideoWriter(args.out_video, 1280, 480, 50.0 / args.video_every)
    hist = bundle.reference_start(library, torch.tensor([int(clip.start)], device=library.device))
    history = hist["observations"].to(device)
    xi = torch.zeros(1, 33, device=device)
    sub = int(round(0.02 / model.opt.timestep))
    dt_hand = 0.05

    rows = []
    with torch.no_grad():
        for t in range(step_count):
            tok = torch.as_tensor(tokens[min(t, len(tokens) - 1)], dtype=torch.float32,
                                  device=device).reshape(1, 64)
            action = bundle.act(bundle.flatten_history(history), tok)
            q_target = (assets.default_joint_pos + assets.action_scale * action[0]).cpu().numpy()
            hand_t = np.clip(hands[min(t, len(hands) - 1)], -1.5, 1.5)
            for _ in range(sub):
                smap.set_ctrl(data, np.clip(G.KP * (q_target - smap.q(data))
                                            - G.KD * smap.dq(data), -args.torque_limit, args.torque_limit))
                hq = smap.hands(data)
                smap.set_hand_ctrl(data, np.concatenate((
                    np.clip(2.0 * (hand_t[:7] - hq[:7]) - 0.05 * smap.hand_dq(data)[:7], -2.4, 2.4),
                    np.clip(2.0 * (hand_t[7:] - hq[7:]) - 0.05 * smap.hand_dq(data)[7:], -2.4, 2.4),
                )))
                mujoco.mj_step(model, data)
            state = np.concatenate([data.qpos[0:3], data.qpos[3:7], smap.q(data),
                                    data.qvel[0:3], data.qvel[3:6], smap.dq(data)])
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            clean = bundle.observation(state_t, action, xi)
            history = torch.cat((history[1:], clean.unsqueeze(0)), dim=0)

            ref = library.ref(torch.tensor([min(t, frames - 1)], device=library.device))
            joint_err = float(np.abs(smap.q(data) - ref.joint_pos[0].cpu().numpy()).mean())
            tilt = float(np.degrees(np.arccos(np.clip(G.quat_to_rotmat(data.qpos[3:7])[2, 2], -1, 1))))
            rot = G.quat_to_rotmat(data.qpos[3:7])
            yaw = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
            xy = float(np.linalg.norm(data.qpos[0:2]))
            rows.append({"tick": t, "t": round(t * 0.02, 3), "z": float(data.qpos[2]),
                         "tilt_deg": tilt, "yaw_deg": yaw, "xy": xy, "joint_err": joint_err})
            if t % args.video_every == 0:
                writer.write(np.concatenate((render(world_cam), render(ego_cam)), axis=1))
            if t % 100 == 0:
                print(f"[{t:4d}] z={data.qpos[2]:.3f} tilt={tilt:5.1f}° joint_err={joint_err:.3f}",
                      flush=True)
    writer.release()
    renderer.close()

    summary = {
        "clip": Path(args.clip).stem,
        "ckpt": str(args.joint_ckpt or args.supermimic_ckpt),
        "scene": args.scene,
        "ticks": step_count,
        "seconds": round(step_count * 0.02, 2),
        "mean_joint_err": float(np.mean([r["joint_err"] for r in rows])),
        "final_joint_err": rows[-1]["joint_err"],
        "min_z": float(np.min([r["z"] for r in rows])),
        "max_tilt_deg": float(np.max([r["tilt_deg"] for r in rows])),
        "final_tilt_deg": rows[-1]["tilt_deg"],
        "fell": bool(max(r["tilt_deg"] for r in rows) > 60.0 or min(r["z"] for r in rows) < 0.45),
        "max_abs_yaw_deg": float(max(abs(r["yaw_deg"]) for r in rows)),
        "final_yaw_deg": rows[-1]["yaw_deg"],
        "max_xy_drift": float(max(r["xy"] for r in rows)),
        "video": str(args.out_video),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=False))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({**summary, "rows": rows}, indent=2))
        print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
