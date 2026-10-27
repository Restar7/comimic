#!/usr/bin/env python3
"""MuJoCo closed loop: GR00T-N1.7 VLA -> 64-d SONIC token -> SuperMimic C6h WBC -> G1.

Chain (the deployment chain, with the two checkpoints the user asked for)::

    head_camera RGB + 64-d robot state + instruction
        -> VLA (``runs/.../groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000``)
        -> 40 x (64 token + 14 finger)
        -> SuperMimic WBC ``runs/S1_handover/WBC_A20000_wmvel_cmd.pt`` (C6h, iter 20000)
        -> 29 joint position targets -> 500 Hz PD -> MuJoCo

The VLA is spoken to over the **same HTTP protocol** HumanoidArena uses
(``scripts/ha_vla_latent64_server.py``), so the loop exercises the deployment interface
rather than a private shortcut.  The WBC is loaded through the Isaac/trackerlab bundle
(``ISAAC_JOINT``) so the 93-d observation, the HA-aligned default pose and the
``q_target = default_q + a`` action convention are exactly the WBC's training convention.

Video: left = third-person tracking camera, right = the exact head camera frame fed to the
VLA.  One file per task + a JSON with survival / tilt / latency.

Usage::

    # 1) start the VLA server (minimal PYTHONPATH: gr00t_deps' skimage is a cp312 partial copy)
    PYTHONPATH=$PWD:$PWD/Isaac-GR00T:$PWD/third_party/gr00t_deps/_noflash \\
      python3 scripts/ha_vla_latent64_server.py \\
        --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \\
        --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights \\
        --port 18090 --device cuda:0 --bf16 --horizon 40 --fm-steps 4 --decode codec

    # 2) closed loop
    SUPERMIMIC_HA_ALIGN=1 python3 scripts/closed_loop_vla_c6h.py \\
        --task ha_double_desk --steps 600 \\
        --out-video runs/closed_loop_vla_c6h/ha_double_desk.mp4
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# Renderer backend has to be chosen before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "groot_repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_SCENES = Path(
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/SONIC_DATA/deployment/"
    "gear_sonic/data/robot_model/model_data/g1"
)
DEFAULT_VLA = ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000"
DEFAULT_WBC = ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt"
DEFAULT_PACKED = ROOT / "runs/isaac_motion/ha_cmd/packed.pt"

CONTROL_DT = 0.02  # 50 Hz, the WBC / VLA deployment period

TASKS = {
    "ha_double_desk": {
        "scene": "scene_ha_double_desk.xml",
        "prompt": "Pick up the object from the double desk and place it into the basket.",
        "spawn": [0.0, 0.0, 0.800],
        "spawn_lift": 0.028,
    },
    "task1": {
        "scene": "scene_43dof_two_table.xml",
        "prompt": ("Navigate to the table ahead, grasp the milk carton, then turn around and "
                   "proceed to the other table to place the carton into the basket."),
        "spawn": [0.0, 0.0, 0.781],
        "spawn_lift": 0.0,
    },
    "task2": {
        "scene": "scene_43dof.xml",
        "prompt": "Place the red cup into the drawer and close the drawer completely.",
        "spawn": [0.0, 0.0, 0.781],
        "spawn_lift": 0.0,
    },
    "task3": {
        "scene": "scene_43dof_handover.xml",
        "prompt": "Retrieve the milk carton from the right plate and place it onto the left plate.",
        "spawn": [0.0, 0.0, 0.781],
        "spawn_lift": 0.0,
    },
    "task4": {
        "scene": "scene_43dof_bend_and_pick.xml",
        "prompt": ("Squat down to grasp the aluminum can on the low chair, then stand up and "
                   "place it on the table ahead."),
        "spawn": [0.0, 0.0, 0.781],
        "spawn_lift": 0.0,
    },
}


# --------------------------------------------------------------------------- video
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
        raise SystemExit("no ffmpeg binary found")

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
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


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 8
    for text in lines:
        draw.rectangle([6, y - 2, 6 + 7 * len(text), y + 13], fill=(0, 0, 0))
        draw.text((9, y), text, fill=(255, 235, 59))
        y += 17
    return np.asarray(img)


# --------------------------------------------------------------------------- helpers
def rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    """Same row-layout 6D used by ``probe_vla_offline.py`` (the tested HA request)."""
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    m = R.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return m[:, :2].reshape(6).astype(np.float32)


def state64(q: np.ndarray, dq: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """HA ``vla_state`` = root rot6d(6) + dof_pos(29) + dof_vel(29)."""
    return np.concatenate([rot6d(quat), np.asarray(q, np.float32),
                           np.asarray(dq, np.float32)]).astype(np.float32)


def post_infer(url: str, rgb: np.ndarray, state: np.ndarray, prompt: str,
               timeout: float = 180.0) -> tuple[np.ndarray, np.ndarray, float]:
    body = json.dumps({
        "observation": {
            "images": {"front": {"shape": list(rgb.shape), "dtype": "uint8",
                                 "data_b64": base64.b64encode(np.ascontiguousarray(rgb).tobytes()).decode()}},
            "state": state.tolist(),
        },
        "robot_type": "unitree_g1_refpose_v3_1",
        "return_chunk": True,
        "task": prompt,
    }).encode()
    req = urllib.request.Request(f"{url}/infer", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    dt = time.time() - t0
    chunk = np.asarray(payload["latent64_chunk"], dtype=np.float32)
    hands = np.asarray(payload.get("hands14_chunk", np.zeros((len(chunk), 14), np.float32)),
                       dtype=np.float32)
    return chunk, hands, dt


def gravity_body(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return r.T @ np.array([0.0, 0.0, -1.0])


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="ha_double_desk")
    ap.add_argument("--vla-ckpt", type=Path, default=DEFAULT_VLA, help="(provenance only)")
    ap.add_argument("--wbc-ckpt", type=Path, default=DEFAULT_WBC)
    ap.add_argument("--packed-motion", type=Path, default=DEFAULT_PACKED)
    ap.add_argument("--server-url", default="http://127.0.0.1:18090")
    ap.add_argument("--scene", type=Path, default=None, help="override the task's MJCF")
    ap.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--steps", type=int, default=600, help="control ticks @ 50 Hz (600 = 12 s)")
    ap.add_argument("--execute-steps", type=int, default=40,
                    help="ticks executed per 40-tick VLA chunk (deployment consumes all 40)")
    ap.add_argument("--settle-ticks", type=int, default=25,
                    help="hold the default pose before the VLA takes over")
    ap.add_argument("--out-video", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ego-camera", default="head_camera")
    ap.add_argument("--video-every", type=int, default=2, help="2 -> 25 fps")
    ap.add_argument("--hand-mode", choices=("vla", "open"), default="vla")
    ap.add_argument("--action-scale", choices=("spec", "one"), default="spec",
                    help="spec = trackerlab C6h convention; one = our joint-finetune convention")
    ap.add_argument("--no-pillow", action="store_true", help="skip the burned-in labels")
    ap.add_argument("--world-mode", choices=("track", "fixed"), default="track")
    ap.add_argument("--world-lookat", type=float, nargs=3, default=[0.75, 0.6, 0.85])
    ap.add_argument("--world-distance", type=float, default=4.2)
    ap.add_argument("--world-azimuth", type=float, default=135.0)
    ap.add_argument("--world-elevation", type=float, default=-15.0)
    args = ap.parse_args()

    cfg = TASKS[args.task]
    scene_path = args.scene or (args.scenes_dir / cfg["scene"])
    prompt = args.prompt or cfg["prompt"]

    import mujoco
    import torch

    import g1_sim as G
    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle

    # ---- WBC (C6h) -----------------------------------------------------------
    bundle = load_isaac_bundle(
        str(args.wbc_ckpt), packed_motion=str(args.packed_motion), device=args.device,
    )
    # Gains / effort / default pose come from the checkpoint's own ``robot_spec`` (HA align),
    # not from the live robot registry: that is the exact controller the WBC was trained against.
    ck = torch.load(str(args.wbc_ckpt), map_location="cpu", weights_only=False)
    rs = ck.get("robot_spec")
    if rs is None:
        raise SystemExit("WBC checkpoint has no robot_spec; cannot align the MuJoCo controller")
    q0 = np.asarray(rs["default_q"], dtype=np.float64)
    kp = np.asarray(rs["kp"], dtype=np.float64)
    kd = np.asarray(rs["kd"], dtype=np.float64)
    effort = np.asarray(rs["effort_limit"], dtype=np.float64)
    # trackerlab env: q_target = default_q + xi + a * action_scale (env.py:364).  The upstream
    # C6h run therefore speaks *scaled* actions; ``one`` is only for our joint finetune.
    spec_scale = np.asarray(rs["action_scale"], dtype=np.float64)
    action_scale = spec_scale if args.action_scale == "spec" else np.ones(29)
    if not np.allclose(q0, bundle.default_joint_pos.detach().cpu().numpy(), atol=1e-6):
        raise SystemExit("robot_spec.default_q != bundle default pose (mount a matching USD/spec)")
    print(f"[wbc] {args.wbc_ckpt.name} iter={bundle.source_iteration} obs_mode={bundle.obs_mode} "
          f"hist={bundle.hist} action_scale={args.action_scale} "
          f"kp[hip_pitch/knee/ankle/elbow]={kp[0]:.0f}/{kp[3]:.0f}/{kp[4]:.0f}/{kp[18]:.0f}", flush=True)

    # ---- MuJoCo --------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    smap = G.SceneMap(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.asarray(cfg["spawn"], np.float64)
    data.qpos[0:3][2] += float(cfg["spawn_lift"])
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    smap.set_q(data, q0)
    smap.set_hands(data, np.zeros(14))
    mujoco.mj_forward(model, data)
    print(f"[sim] scene={scene_path.name} nq={model.nq} nu={model.nu} dt={model.opt.timestep} "
          f"spawn={np.round(data.qpos[0:3], 3)} contacts={data.ncon}", flush=True)

    # joint limits (MuJoCo order = STANDARD_29 order, looked up by name)
    q_lo = np.empty(29)
    q_hi = np.empty(29)
    for i, qadr in enumerate(smap.body_q):
        jid = int(np.where(model.jnt_qposadr == qadr)[0][0])
        lo, hi = model.jnt_range[jid]
        q_lo[i] = lo if model.jnt_limited[jid] else -3.0
        q_hi[i] = hi if model.jnt_limited[jid] else 3.0

    renderer = mujoco.Renderer(model, 480, 640)
    ego_cam = mujoco.MjvCamera()
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.ego_camera)
    if eid < 0:
        raise SystemExit(f"scene has no {args.ego_camera!r} camera")
    ego_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    ego_cam.fixedcamid = eid
    world_cam = mujoco.MjvCamera()
    world_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    world_cam.distance = args.world_distance
    world_cam.azimuth = args.world_azimuth
    world_cam.elevation = args.world_elevation

    def look_world() -> None:
        if args.world_mode == "fixed":
            world_cam.lookat[:] = args.world_lookat
            return
        p = data.qpos[0:3]
        world_cam.lookat[:] = [float(p[0]), float(p[1]), float(p[2]) + 0.55]

    def render(cam) -> np.ndarray:
        renderer.update_scene(data, cam)
        return renderer.render()

    # ---- VLA server ----------------------------------------------------------
    health = f"{args.server_url}/"
    try:
        with urllib.request.urlopen(health, timeout=5) as r:
            info = json.loads(r.read())
        print(f"[vla] {args.server_url} -> {info}", flush=True)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"VLA server not reachable at {args.server_url}: {exc}")

    writer = VideoWriter(args.out_video, 1280, 480, 50.0 / args.video_every)
    xi = torch.zeros(1, 33, device=args.device)
    dummy_ref = FrameRef(torch.zeros(1, dtype=torch.long, device=args.device))
    sub = max(1, int(round(CONTROL_DT / model.opt.timestep)))
    print(f"[loop] {sub} physics substeps per control tick", flush=True)

    def flat_state() -> torch.Tensor:
        quat = np.asarray(data.qpos[3:7], np.float64)
        q = np.asarray(smap.q(data), np.float32)
        qd = np.asarray(smap.dq(data), np.float32)
        v = np.asarray(data.qvel[0:3], np.float32)
        w_body = np.asarray(data.qvel[3:6], np.float64)
        from g1_sim import quat_to_rotmat

        w_world = (quat_to_rotmat(quat) @ w_body).astype(np.float32)
        s = np.concatenate([data.qpos[0:3], quat, q, v, w_world, qd]).astype(np.float32)
        return torch.as_tensor(s, device=args.device).unsqueeze(0)

    def current_obs(prev_action: np.ndarray) -> torch.Tensor:
        a = torch.as_tensor(prev_action, dtype=torch.float32, device=args.device).unsqueeze(0)
        with torch.no_grad():
            return bundle.observation(flat_state(), a, xi, dummy_ref)

    def step_physics(q_target: np.ndarray, hand_target: np.ndarray) -> None:
        for _ in range(sub):
            tau = kp * (q_target - smap.q(data)) - kd * smap.dq(data)
            smap.set_ctrl(data, np.clip(tau, -effort, effort))
            hq = smap.hands(data)
            hdq = smap.hand_dq(data)
            smap.set_hand_ctrl(data, np.concatenate((
                np.clip(2.0 * (hand_target[:7] - hq[:7]) - 0.05 * hdq[:7], -2.4, 2.4),
                np.clip(2.0 * (hand_target[7:] - hq[7:]) - 0.05 * hdq[7:], -2.4, 2.4),
            )))
            mujoco.mj_step(model, data)

    def tilt_deg() -> float:
        from g1_sim import quat_to_rotmat

        up = float(quat_to_rotmat(np.asarray(data.qpos[3:7], np.float64))[2, 2])
        return float(np.degrees(np.arccos(np.clip(up, -1.0, 1.0))))

    # ---- settle on the default pose -----------------------------------------
    for _ in range(max(0, args.settle_ticks)):
        step_physics(q0, np.zeros(14))
    print(f"[settle] z={data.qpos[2]:.3f} tilt={tilt_deg():.1f}°", flush=True)

    hist = current_obs(np.zeros(29)).unsqueeze(0).repeat(10, 1, 1)
    rows: list[dict] = []
    latencies: list[float] = []
    plans = 0
    consumed = 10**9
    tokens = np.zeros((1, 64), np.float32)
    hands = np.zeros((1, 14), np.float32)
    last_act = np.zeros(29, np.float32)
    t_start = time.time()

    with torch.no_grad():
        for tick in range(args.steps):
            if consumed >= min(args.execute_steps, len(tokens)):
                look_world()
                ego = render(ego_cam)
                s64 = state64(np.asarray(smap.q(data), np.float32),
                              np.asarray(smap.dq(data), np.float32),
                              np.asarray(data.qpos[3:7], np.float64))
                tokens, hands, dt = post_infer(args.server_url, ego, s64, prompt)
                latencies.append(dt)
                plans += 1
                consumed = 0
                print(f"[plan {plans:3d}] tick={tick:4d} infer={dt * 1e3:.0f}ms "
                      f"tok_absmean={np.abs(tokens[0]).mean():.4f} "
                      f"hand_max={np.abs(hands[0]).max():.2f}", flush=True)

            idx = min(consumed, len(tokens) - 1)
            token = torch.as_tensor(tokens[idx], dtype=torch.float32,
                                    device=args.device).reshape(1, 64)
            act = bundle.act(bundle.flatten_history(hist), token)
            act_np = act[0].detach().cpu().numpy().astype(np.float64)
            q_target = np.clip(q0 + action_scale * act_np, q_lo, q_hi)
            hand_target = np.zeros(14) if args.hand_mode == "open" else np.clip(hands[idx], -1.5, 1.5)

            step_physics(q_target, hand_target)

            obs = current_obs(act_np.astype(np.float32))
            hist = torch.cat((hist[1:], obs.unsqueeze(0)), dim=0)
            last_act = act_np.astype(np.float32)
            consumed += 1

            p = data.qpos[0:3]
            rows.append({
                "tick": tick, "plans": plans,
                "z": float(p[2]), "x": float(p[0]), "y": float(p[1]),
                "tilt_deg": tilt_deg(),
                "token_absmean": float(np.abs(tokens[idx]).mean()),
            })
            if tick % 100 == 0:
                print(f"[{tick:4d}] z={p[2]:.3f} xy=({p[0]:+.2f},{p[1]:+.2f}) "
                      f"tilt={rows[-1]['tilt_deg']:5.1f}° plans={plans}", flush=True)
            if tick % args.video_every == 0:
                look_world()
                left = render(world_cam)
                right = render(ego_cam)
                frame = np.concatenate((left, right), axis=1)
                if not args.no_pillow:
                    secs = tick * CONTROL_DT
                    frame = overlay(frame, [
                        f"VLA ckpt-8000 + SuperMimic C6h(20k)  |  {args.task}  t={secs:5.2f}s",
                        f"world view (left) / head_camera fed to VLA (right)",
                        f"z={p[2]:.2f}m tilt={rows[-1]['tilt_deg']:4.1f}deg plan={plans}",
                    ])
                writer.write(frame)
    writer.release()
    renderer.close()

    zs = [r["z"] for r in rows]
    tilts = [r["tilt_deg"] for r in rows]
    xy = np.array([[r["x"], r["y"]] for r in rows])
    summary = {
        "task": args.task,
        "scene": str(scene_path),
        "prompt": prompt,
        "vla_ckpt": str(args.vla_ckpt),
        "wbc_ckpt": str(args.wbc_ckpt),
        "wbc_iteration": int(bundle.source_iteration),
        "obs_mode": bundle.obs_mode,
        "ticks": args.steps,
        "seconds": round(args.steps * CONTROL_DT, 2),
        "plans": plans,
        "infer_ms_mean": float(np.mean(latencies) * 1e3) if latencies else None,
        "infer_ms_max": float(np.max(latencies) * 1e3) if latencies else None,
        "wall_sec": round(time.time() - t_start, 2),
        "min_z": float(min(zs)),
        "final_z": float(zs[-1]),
        "max_tilt_deg": float(max(tilts)),
        "final_tilt_deg": float(tilts[-1]),
        "travel_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "fell": bool(max(tilts) > 60.0 or min(zs) < 0.45),
        "video": str(args.out_video),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({**summary, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
