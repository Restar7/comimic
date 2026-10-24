#!/usr/bin/env python3
"""Closed-loop MuJoCo inference for the four SONIC tasks, with live view.

Loop: head_camera -> GR00T policy -> 40x64 motion tokens + hands
      -> frozen SONIC decoder -> 29 joint targets -> 500 Hz PD -> mj_step

Modes
-----
``--chunks N`` (default 40): fixed-length rollout of ``N`` deployment blocks
(one block = ``SONIC78_V1.horizon`` = 40 control ticks = 0.8 s); optional
``--video out.mp4``.
``--until-success``: keep calling the policy until the task predicate succeeds
or ``--max-seconds`` is reached; state rows are written every control tick.

Replanning follows the deployment contract: the controller executes
``--execute-steps`` (default 20 = 2.5 Hz) of every 40-tick plan and then replans
while the old plan still has an unexecuted tail.  ``--rtc`` reuses that tail as
the new plan's frozen prefix (zero-shot RTC, no retraining).  The MuJoCo loop is
synchronous, so ``--rtc-frozen-ticks`` is a continuity parameter here, not a
latency budget; on a real 50 Hz controller it must be ``ceil(latency/20ms)+1``.

The live view renders the world view and the ego view into a MuJoCo passive
viewer window; headless runs use EGL (``MUJOCO_GL=egl``) and write mp4 instead.

Success predicates are the deployment-style ones from the reference repo
(``check_success.py``) for task_1/task_2; task_3/task_4 use explicit movement /
lift proxies documented in the runbook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mosaic_n17.contracts import SONIC78_V1, TOKEN_ABS_BOUND, TickSchedule  # noqa: E402
from mosaic_n17.policy.rtc import RtcRequest, RtcWindow, align_prefix_tail, encode_committed_action  # noqa: E402


HORIZON = SONIC78_V1.horizon


DEFAULT_REPRO = "/tmp/issac_groot/groot_repro"
DEFAULT_RELEASE = (
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/"
    "gear_sonic_deploy/policy/release"
)
DEFAULT_SCENES = (
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/SONIC_DATA/deployment/"
    "gear_sonic/data/robot_model/model_data/g1"
)


class _FfmpegVideoWriter:
    """H.264 writer through ffmpeg; cv2's mp4v encoder costs ~58 ms/frame here."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to write --video")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.shape = (height, width, 3)
        self.frames = 0
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        array = np.asarray(frame, dtype=np.uint8)
        if array.shape != self.shape:
            raise ValueError(f"video frame must be {self.shape}, got {array.shape}")
        self._process.stdin.write(np.ascontiguousarray(array).tobytes())
        self.frames += 1

    def release(self) -> None:
        self._process.stdin.close()
        stderr = self._process.stderr.read().decode("utf-8", "replace")
        if self._process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.strip()}")


TASKS = {
    "task1": {
        "scene": "scene_43dof_two_table.xml",
        "prompt": (
            "Navigate to the table ahead, grasp the milk carton, then turn around and "
            "proceed to the other table to place the carton into the basket."
        ),
        "track": ["sugar_box", "box"],
        "success": "carton",
        "world_camera": "side_view",
    },
    "task2": {
        "scene": "scene_43dof.xml",
        "prompt": "Place the red cup into the drawer and close the drawer completely.",
        "track": ["mug", "drawer"],
        "success": "redcup",
        "world_camera": "side_view",
    },
    "task3": {
        "scene": "scene_43dof_handover.xml",
        "prompt": "Retrieve the milk carton from the right plate and place it onto the left plate.",
        "track": ["sugar_box", "tgt_box"],
        "targets": ["tgt_box"],
        "success": "place:sugar_box:tgt_box",
        "world_camera": "side_view",
    },
    "task4": {
        "scene": "scene_43dof_bend_and_pick.xml",
        "prompt": (
            "Squat down to grasp the aluminum can on the low chair, then stand up and "
            "place it on the table ahead."
        ),
        "track": ["tomato_soup_can", "table_A"],
        "targets": ["table_A"],
        "success": "place:tomato_soup_can:table_A",
        "world_camera": "side_view",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--ckpt", required=True, help="offline-loadable GR00T checkpoint view")
    ap.add_argument("--release-dir", type=Path, default=Path(DEFAULT_RELEASE))
    ap.add_argument("--scenes-dir", type=Path, default=Path(DEFAULT_SCENES))
    ap.add_argument("--repro-dir", type=Path, default=Path(DEFAULT_REPRO))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chunks", type=int, default=40, help="fixed-length mode chunk count")
    ap.add_argument("--until-success", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument(
        "--viewer",
        action="store_true",
        help="live MuJoCo window on $DISPLAY (requires MUJOCO_GL=glfw, e.g. noVNC :1)",
    )
    ap.add_argument("--view-every", type=int, default=5, help="render every N control ticks")
    ap.add_argument("--video", type=Path, help="optional mp4 (world|ego)")
    ap.add_argument(
        "--video-every",
        type=int,
        default=2,
        help="write a video frame every N control ticks (2 -> 25 fps, 4 -> 12.5 fps)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-gib", default="40GiB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--execute-steps",
        type=int,
        default=SONIC78_V1.execute_steps,
        help="ticks executed per plan (must divide 40; 20 = 2.5 Hz deployment cadence, "
        "40 reproduces the pre-RTC 40-tick runs)",
    )
    ap.add_argument(
        "--rtc",
        action="store_true",
        help="reuse the unexecuted tail of the committed plan as the next plan's prefix",
    )
    ap.add_argument(
        "--rtc-frozen-ticks",
        type=int,
        default=2,
        help="F: leading ticks pinned to the committed tail (sim continuity parameter)",
    )
    ap.add_argument("--rtc-ramp-rate", type=float, default=6.0)
    ap.add_argument(
        "--rtc-overlap-ticks",
        type=int,
        default=6,
        help="latency budget: ticks pinned to the committed plan (F..O ramp); the "
        "remaining executed ticks are freshly denoised",
    )
    ap.add_argument(
        "--log-action",
        action="store_true",
        help="also write the executed token and q_target per tick (for RTC on/off continuity metrics)",
    )
    ap.add_argument(
        "--log-qpos",
        action="store_true",
        help="save the per-tick qpos trajectory (qpos.npy) so any run can be re-rendered offline",
    )
    ap.add_argument(
        "--log-ego-hash",
        action="store_true",
        help="print a short sha256 of the ego image each plan (checks whether rendering changes the observation)",
    )
    ap.add_argument(
        "--profile-stages",
        action="store_true",
        help="print render/policy timings and one repeated policy call per plan",
    )
    ap.add_argument("--k", type=int, default=1, help="candidates per plan (K>=1)")
    ap.add_argument(
        "--k-select",
        choices=("first", "gate"),
        default="first",
        help="'first' = fixed primary candidate (no scorer); 'gate' needs a calibrated AC scorer",
    )
    args = ap.parse_args()

    cfg = TASKS[args.task]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene = args.scenes_dir / cfg["scene"]
    if not scene.is_file():
        raise SystemExit(f"scene not found: {scene}")
    sys.path.insert(0, str(args.repro_dir))
    import meta_patch  # noqa: PLC0415

    meta_patch.install()
    if args.device.startswith("cuda"):
        import gpu_patch  # noqa: PLC0415

        gpu_patch.install(max_gpu_gib=args.gpu_gib)
    import check_success  # noqa: PLC0415
    import g1_sim as G  # noqa: PLC0415
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    import sonic_loop  # noqa: PLC0415
    from sonic_loop import CONTROL_DT, HISTORY, SonicController  # noqa: PLC0415

    class _QuietDecoder(sonic_loop.Decoder):
        """ORT CPU pool with 1 thread and no spin-wait.

        The default ORT pool spins on all cores between decoder calls; the policy's
        image preprocessing then drops from milliseconds to seconds, which shows up
        as a 3.4 s "inference" per plan instead of the ~0.1 s the model needs.
        """

        def __init__(self, onnx_path):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            self.sess = ort.InferenceSession(
                onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
            self.input = self.sess.get_inputs()[0].name

    sonic_loop.Decoder = _QuietDecoder
    import torch  # noqa: PLC0415

    # RTC on/off comparisons are only meaningful at the same seed: the action head
    # draws its own Gaussian initialisation on every plan.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    smap = G.SceneMap(m)
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, 0.781]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    smap.set_q(d, G.DEFAULT_ANGLES.copy())
    smap.set_hands(d, np.zeros(14))
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    world_cam = mujoco.MjvCamera()
    wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cfg["world_camera"])
    if wid >= 0:
        world_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        world_cam.fixedcamid = wid
    else:
        mujoco.mjv_defaultCamera(world_cam)
        world_cam.lookat[:] = [0.35, 0.0, 0.55]
        world_cam.distance = 2.4
        world_cam.azimuth = 118.0
        world_cam.elevation = -18.0

    def render(camera):
        renderer.update_scene(d, camera)
        return renderer.render()

    body_ids = {}
    for name in cfg["track"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise SystemExit(f"body {name!r} not in {scene}")
        body_ids[name] = bid

    if args.rtc:
        from mosaic_n17.policy.rtc_policy import RtcGr00tPolicy  # noqa: PLC0415

        policy = RtcGr00tPolicy(
            embodiment_tag="unitree_g1_sonic",
            model_path=args.ckpt,
            device=args.device,
            strict=True,
        )
    else:
        policy = Gr00tPolicy(
            embodiment_tag="unitree_g1_sonic",
            model_path=args.ckpt,
            device=args.device,
            strict=True,
        )
    base_policy = policy
    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    if args.k > 1:
        from mosaic_n17.policy.policy_wrapper import MosaicPolicyWrapper  # noqa: PLC0415

        if args.k_select == "gate":
            raise SystemExit(
                "--k-select gate needs a calibrated AC scorer; it is not wired into this loop"
            )
        policy = MosaicPolicyWrapper(
            policy, K=args.k, selection="first", shadow=False, primary_index=0
        )

    target_frames = {name: _target_frame(m, d, body_ids[name]) for name in cfg.get("targets", [])}
    ctrl = SonicController(str(args.release_dir / "model_decoder.onnx"), init_q=G.DEFAULT_ANGLES)
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
    writer = None
    if args.video:
        if args.video_every < 1:
            raise SystemExit("--video-every must be >= 1")
        writer = _FfmpegVideoWriter(args.video, 1280, 480, 50.0 / args.video_every)
    viewer = None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
        pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis
            viewer.cam.distance = 2.6
            viewer.cam.azimuth = 120.0
            viewer.cam.elevation = -18.0
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = wid if wid >= 0 else 0

    if not 0 < args.execute_steps <= HORIZON or HORIZON % args.execute_steps:
        raise SystemExit(
            f"--execute-steps must be a divisor of {HORIZON} (2.5/5/10 Hz control cadence)"
        )
    schedule = TickSchedule(execute_steps=args.execute_steps)
    replan_hz = schedule.replan_hz
    if args.rtc and args.execute_steps >= HORIZON:
        raise SystemExit(
            "--rtc needs --execute-steps < 40: a fully consumed plan leaves no tail to reuse"
        )
    if args.rtc:
        window = RtcWindow(
            overlap=args.rtc_overlap_ticks,
            frozen=args.rtc_frozen_ticks,
            ramp_rate=args.rtc_ramp_rate,
        )
        if args.rtc_overlap_ticks >= args.execute_steps:
            raise SystemExit(
                "--rtc-overlap-ticks must be < --execute-steps: otherwise every executed tick "
                "is a frozen/ramped replay of the old plan and no fresh row is executed"
            )

    rows: list[dict] = []
    qpos_frames: list[np.ndarray] = []
    blocks = args.chunks if not args.until_success else 10**9
    max_plans = blocks * (HORIZON // args.execute_steps)
    max_ticks = int(np.ceil(args.max_seconds / CONTROL_DT)) + 1
    tick = 0
    plans = 0
    consumed = 0
    start_tick = 0
    chunk: tuple | None = None
    committed_norm: np.ndarray | None = None
    plan_info = {"index": 0, "overlap": 0, "frozen": 0}
    success: dict = {"success": False}
    prefix_checks = 0
    skipped = 0
    overlap_hist: dict[str, int] = {}
    max_prefix_abs_diff = 0.0
    max_candidate_spread = 0.0
    started = time.time()

    while tick < max_ticks:
        if chunk is None or consumed >= args.execute_steps:
            # Stop only when a new plan would be needed: the last allowed plan
            # must still execute all of its --execute-steps ticks.
            if plans >= max_plans:
                break
            render_t0 = time.time()
            obs, ego_hash = _observation(d, smap, renderer, cam, G, cfg, hash_ego=args.log_ego_hash)
            render_s = time.time() - render_t0
            anchor = tick
            has_plan = chunk is not None
            j = consumed if has_plan else 0
            overlap = min(args.rtc_overlap_ticks, HORIZON - j) if args.rtc else 0
            frozen = args.rtc_frozen_ticks if args.rtc else 0
            request = _rtc_request(args.rtc, committed_norm, j, overlap, frozen, args)
            t0 = time.time()
            if args.rtc or args.k > 1:
                action, info = policy.get_action(obs, options=_policy_options(anchor, request))
            else:
                action, info = policy.get_action(obs)
            infer_s = time.time() - t0
            if args.profile_stages:
                stages = _profile_stages(base_policy, obs)
                print(
                    f"[profile] plan {plans + 1} render={render_s:.3f}s policy={infer_s:.3f}s "
                    + " ".join(f"{name}={value:.3f}s" for name, value in stages.items()),
                    flush=True,
                )
            selected = info.get("selected_index")
            selected_index = 0 if selected is None else int(selected[0])
            spread = float(info.get("candidate_spread", 0.0))
            max_candidate_spread = max(max_candidate_spread, spread)
            chunk = _decode_chunk(action)
            had_norm = committed_norm is not None
            if request is not None:
                committed_norm, diff = _commit_prefix(
                    request, committed_norm, info, j, frozen, anchor
                )
                prefix_checks += 1
                max_prefix_abs_diff = max(max_prefix_abs_diff, diff)
                overlap_hist[str(overlap)] = overlap_hist.get(str(overlap), 0) + 1
                prefix_label = "yes"
            elif not args.rtc:
                prefix_label = "off"
            elif had_norm:
                skipped += 1
                prefix_label = "skip"
            elif "action_normalized" in info:
                # K candidates already return normalized rows for the selected
                # candidate; reusing them avoids a decode/normalize round trip.
                committed_norm = np.asarray(info["action_normalized"], np.float32)
                prefix_label = "select"
            else:
                # First plan under --rtc (K=1): no sampler output, so the committed
                # chunk is encoded through the official normalizer.
                committed_norm = _encode_commit(base_policy, chunk, window)
                prefix_label = "encode"
            start_tick, consumed = anchor, 0
            plans += 1
            plan_info = {
                "index": plans - 1,
                "overlap": overlap,
                "frozen": frozen,
                "candidate": selected_index,
                "candidate_spread": spread,
            }
            ego_note = f"ego={ego_hash[:12]} " if args.log_ego_hash else ""
            print(
                f"[{args.task}] plan {plans} anchor={anchor} j={j} O={overlap} F={frozen} "
                f"K={args.k} sel={selected_index} spread={spread:.3f} infer={infer_s:.2f}s "
                f"{ego_note}token=[{chunk[0].min():.2f},{chunk[0].max():.2f}] "
                f"prefix={prefix_label}",
                flush=True,
            )
        index = tick - start_tick
        if not 0 <= index < HORIZON:
            raise RuntimeError(
                f"tick {tick}: plan index {index} outside [0,{HORIZON}); the committed plan is stale"
            )
        tokens, hands_l, hands_r = chunk
        token, hand_l, hand_r = tokens[index], hands_l[index], hands_r[index]
        # smap resolves the 29 body joints by name: the MJCF interleaves the
        # hands (legs+waist, left arm, left hand, right arm, right hand), so a
        # contiguous body slice would feed the left hand in place of the right arm.
        q_target = ctrl.step(
            token,
            smap.q(d).copy(),
            smap.dq(d).copy(),
            d.qpos[3:7].copy(),
            d.qvel[3:6].copy(),
        )
        for _ in range(sub):
            smap.set_ctrl(
                d,
                np.clip(G.KP * (q_target - smap.q(d)) - G.KD * smap.dq(d), -88, 88),
            )
            hq = smap.hands(d)
            hdq = smap.hand_dq(d)
            smap.set_hand_ctrl(
                d,
                np.concatenate(
                    (
                        np.clip(2.0 * (hand_l - hq[:7]) - 0.05 * hdq[:7], -2.4, 2.4),
                        np.clip(2.0 * (hand_r - hq[7:]) - 0.05 * hdq[7:], -2.4, 2.4),
                    )
                ),
            )
            mujoco.mj_step(m, d)
        tilt = float(np.degrees(np.arccos(np.clip(G.quat_to_rotmat(d.qpos[3:7])[2, 2], -1, 1))))
        if args.log_qpos:
            qpos_frames.append(np.asarray(d.qpos, dtype=np.float32).copy())
        row = {
            "elapsed": round(tick * CONTROL_DT, 4),
            "base_position": d.qpos[0:3].tolist(),
            "tilt_deg": tilt,
            "tracked_bodies": {
                name: {"position": d.xpos[bid].tolist()} for name, bid in body_ids.items()
            },
            "plan": plan_info["index"],
            "index": index,
            "overlap": plan_info["overlap"],
            "frozen": plan_info["frozen"],
            "candidate": plan_info["candidate"],
            "candidate_spread": plan_info["candidate_spread"],
        }
        if args.log_action:
            row["token"] = np.round(token, 5).tolist()
            row["q_target"] = np.round(q_target, 5).tolist()
        rows.append(row)
        if viewer is not None and tick % args.view_every == 0:
            viewer.sync()
        if writer is not None and tick % args.video_every == 0:
            frame = np.concatenate((render(world_cam), render(cam)), axis=1)
            writer.write(frame)
        tick += 1
        consumed += 1
        if consumed < args.execute_steps and tick < max_ticks:
            continue
        success = _check_success(cfg["success"], rows, check_success, target_frames)
        print(
            f"[{args.task}] plan {plans} done t={tick * CONTROL_DT:.1f}s "
            f"tilt={rows[-1]['tilt_deg']:.0f} success={success}",
            flush=True,
        )
        if success.get("success") or tick * CONTROL_DT >= args.max_seconds:
            break

    if writer is not None:
        writer.release()
    if viewer is not None:
        viewer.close()
    if args.log_qpos and qpos_frames:
        np.save(args.out_dir / "qpos.npy", np.stack(qpos_frames))
    state_path = args.out_dir / "state.jsonl"
    state_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    meta = {
        "task": args.task,
        "scene": cfg["scene"],
        "prompt": cfg["prompt"],
        "chunks": plans,
        "plans": plans,
        "execute_steps": args.execute_steps,
        "replan_hz": replan_hz,
        "k": args.k,
        "k_select": args.k_select if args.k > 1 else None,
        "seconds": round(tick * CONTROL_DT, 2),
        "wall_seconds": round(time.time() - started, 1),
        "final_tilt_deg": rows[-1]["tilt_deg"],
        "fell": rows[-1]["tilt_deg"] > 60 or rows[-1]["base_position"][2] < 0.35,
        "success": success,
        "rtc": {
            "enabled": bool(args.rtc),
            "frozen_ticks": args.rtc_frozen_ticks if args.rtc else 0,
            "ramp_rate": args.rtc_ramp_rate if args.rtc else 0.0,
            "overlap_ticks": args.rtc_overlap_ticks,
            "prefix_checks": prefix_checks,
            "skipped": skipped,
            "overlap_hist": overlap_hist,
            "max_prefix_abs_diff": max_prefix_abs_diff,
            "max_candidate_spread": max_candidate_spread,
        },
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def _observation(d, smap, renderer, cam, G, cfg, *, hash_ego: bool = False):
    """Render the ego view and assemble the policy observation."""

    renderer.update_scene(d, cam)
    ego = renderer.render()
    digest = hashlib.sha256(np.ascontiguousarray(ego).tobytes()).hexdigest() if hash_ego else ""
    obs = {
        "video": {"ego_view": np.asarray(ego, np.uint8)[None, None]},
        "state": {
            "left_leg": smap.q(d)[0:6][None, None].astype(np.float32),
            "right_leg": smap.q(d)[6:12][None, None].astype(np.float32),
            "waist": smap.q(d)[12:15][None, None].astype(np.float32),
            "left_arm": smap.q(d)[15:22][None, None].astype(np.float32),
            "right_arm": smap.q(d)[22:29][None, None].astype(np.float32),
            "left_hand": smap.hands(d)[:7][None, None].astype(np.float32),
            "right_hand": smap.hands(d)[7:][None, None].astype(np.float32),
            "projected_gravity": G.projected_gravity(np.asarray(d.qpos[3:7], np.float64))[
                None, None
            ].astype(np.float32),
        },
        "language": {"annotation.human.task_description": [[cfg["prompt"]]]},
    }
    return obs, digest


def _decode_chunk(action: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one policy action dict into ``(tokens[H,64], left[H,7], right[H,7])``."""

    tokens = np.asarray(action["motion_token"], np.float32).reshape(-1, 64)[:HORIZON]
    hands_l = np.asarray(action["left_hand_joints"], np.float32).reshape(-1, 7)[:HORIZON]
    hands_r = np.asarray(action["right_hand_joints"], np.float32).reshape(-1, 7)[:HORIZON]
    if (
        tokens.shape != (HORIZON, 64)
        or hands_l.shape != (HORIZON, 7)
        or hands_r.shape != (HORIZON, 7)
    ):
        raise RuntimeError(f"policy returned a malformed chunk: {tokens.shape}")
    # The official VLA rejects |motion_token| > 1.25; a clipped token would silently
    # change the command sent to the decoder, so fail closed instead.
    if float(np.abs(tokens).max()) > TOKEN_ABS_BOUND:
        raise RuntimeError(
            f"motion token out of contract: |token|max={float(np.abs(tokens).max()):.3f} "
            f"> {TOKEN_ABS_BOUND}"
        )
    return tokens, hands_l, hands_r


def _rtc_request(enabled, committed_norm, j, overlap, frozen, args):
    """Build the RTC request with the old plan's next rows re-aligned into the tail.

    The official sampler pins the new chunk's first ``overlap`` rows to the last
    ``overlap`` rows of the tensor it receives, so the old plan's rows
    ``[j, j+overlap)`` must be copied there; passing the whole chunk with
    ``overlap = H - j`` would make every executed tick a replay of the old plan's
    never-executed future.
    """

    if not enabled or committed_norm is None:
        return None
    if frozen >= overlap or overlap - frozen < 2:
        return None
    window = RtcWindow(overlap=overlap, frozen=frozen, ramp_rate=args.rtc_ramp_rate)
    prefix = align_prefix_tail(committed_norm, start=j, overlap=overlap, horizon=HORIZON)
    return RtcRequest(window=window, previous_action=prefix)


def _encode_commit(policy, chunk, window):
    """Normalize a decoded chunk for the next RTC window (first plan only)."""

    import torch  # imported lazily: only the RTC path needs a dtype round trip

    raw = np.concatenate((chunk[0], chunk[1], chunk[2]), axis=-1)
    encoded = encode_committed_action(
        policy.processor, raw, window=window, embodiment_tag="UNITREE_G1_SONIC"
    )
    # The policy casts the prefix to the model dtype (bf16) before the sampler sees
    # it, so store the same quantized values or the frozen-prefix check will fail.
    tensor = torch.from_numpy(np.asarray(encoded, np.float32))
    return tensor.to(dtype=policy.model.dtype).float().numpy()


def _policy_options(anchor: int, request) -> dict:
    options: dict = {"anchor_tick": anchor}
    if request is not None:
        options["rtc"] = request.as_mapping()
    return options


def _commit_prefix(request, committed_norm, info, j, frozen, anchor):
    """Verify the frozen prefix and return the new committed normalized chunk."""

    if "action_normalized" not in info:
        raise RuntimeError("RTC policy did not return action_normalized; cannot commit a prefix")
    new_norm = np.asarray(info["action_normalized"], np.float32)
    expected = committed_norm[:, j : j + frozen, :]
    head = new_norm[:, :frozen, :]
    if not np.array_equal(head, expected):
        raise RuntimeError(
            f"anchor {anchor}: RTC prefix mismatch between the committed rows "
            f"[{j}:{j + frozen}] and the new chunk; refusing to switch plans"
        )
    diff = float(np.abs(head - expected).max()) if frozen else 0.0
    return new_norm, diff


def _profile_stages(policy, observation: dict) -> dict[str, float]:
    """Time the policy stages inside the loop process (diagnostics only)."""

    from gr00t.policy.gr00t_policy import MessageType, _rec_to_dtype  # noqa: PLC0415
    import torch  # noqa: PLC0415

    def sync() -> None:
        torch.cuda.synchronize()

    t0 = time.time()
    unbatched = policy._unbatch_observation(observation)
    processed, states = [], []
    for ob in unbatched:
        step = policy._to_vla_step_data(ob)
        states.append(step.states)
        processed.append(
            policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        )
    collated = _rec_to_dtype(policy.collate_fn(processed), dtype=torch.bfloat16)
    sync()
    t1 = time.time()
    with torch.inference_mode():
        backbone_inputs, action_inputs = policy.model.prepare_input(collated["inputs"])
        backbone_out = policy.model.backbone(backbone_inputs)
    sync()
    t2 = time.time()
    with torch.inference_mode():
        head_out = policy.model.action_head.get_action(backbone_out, action_inputs, None)
    sync()
    t3 = time.time()
    with torch.inference_mode():
        normalized = head_out["action_pred"].float()
        batched = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in policy.modality_configs["state"].modality_keys
        }
        policy.processor.decode_action(normalized.cpu().numpy(), policy.embodiment_tag, batched)
    sync()
    t4 = time.time()
    return {
        "preproc": t1 - t0,
        "backbone": t2 - t1,
        "action_head": t3 - t2,
        "decode": t4 - t3,
    }


def _target_frame(m, d, body_id: int) -> dict:
    """Axis-aligned top surface of a static target body (table top, tray, box floor)."""

    geoms = [
        gid
        for gid in range(m.ngeom)
        if int(m.geom_bodyid[gid]) == body_id
        and int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    ]
    if not geoms:
        raise RuntimeError(f"target body {body_id} has no box geom")
    gid = max(geoms, key=lambda g: float(m.geom_size[g][0] * m.geom_size[g][1]))
    center = np.asarray(d.geom_xpos[gid], float)
    half = np.asarray(m.geom_size[gid], float)
    return {
        "xy": center[:2].tolist(),
        "half_xy": half[:2].tolist(),
        "top_z": float(center[2] + half[2]),
    }


def _place_success(
    rows: list[dict],
    name: str,
    target: dict,
    *,
    xy_frac: float = 0.85,
    z_lo: float = -0.02,
    z_hi: float = 0.10,
    stable_ticks: int = 25,
    stable_tol: float = 0.01,
) -> dict:
    """``True`` when the object comes to rest on the target's top surface.

    The object must be inside the footprint, its centre within ``[z_lo, z_hi]`` of the
    surface, and its height must not move more than ``stable_tol`` over the last
    ``stable_ticks`` ticks -- a can held above the table is *not* placed.
    """

    window = rows[-stable_ticks:]
    positions = np.asarray([row["tracked_bodies"][name]["position"] for row in window], dtype=float)
    x, y, z = (float(value) for value in positions[-1])
    dx = abs(x - target["xy"][0])
    dy = abs(y - target["xy"][1])
    z_rel = z - target["top_z"]
    inside = dx <= target["half_xy"][0] * xy_frac and dy <= target["half_xy"][1] * xy_frac
    drift = float(positions[:, 2].max() - positions[:, 2].min())
    stable = drift <= stable_tol
    ok = bool(inside and z_lo <= z_rel <= z_hi and stable)
    return {
        "success": ok,
        "stage": f"place z_rel={z_rel:.3f} dx={dx:.3f} dy={dy:.3f} drift={drift:.3f}",
        "z_rel": z_rel,
        "dx": dx,
        "dy": dy,
        "stable": stable,
    }


def _check_success(kind: str, rows: list[dict], check_success, target_frames=None) -> dict:
    """Evaluate the task predicate on the rows collected so far."""

    target_frames = {} if target_frames is None else target_frames
    if len(rows) < 25:
        return {"success": False, "stage": "warmup"}
    if kind == "carton":
        return check_success.check_carton(rows)
    if kind == "redcup":
        return check_success.check_redcup(rows)
    parts = kind.split(":")
    if parts[0] == "place":
        name, target_name = parts[1], parts[2]
        if target_name not in target_frames:
            raise KeyError(f"no target frame for {target_name!r}")
        return _place_success(rows, name, target_frames[target_name])
    if parts[0] == "move":
        name, threshold = parts[1], float(parts[2])
        start = np.asarray(rows[0]["tracked_bodies"][name]["position"])
        last = np.asarray(rows[-1]["tracked_bodies"][name]["position"])
        moved = float(np.linalg.norm(last - start))
        return {"success": moved >= threshold, "stage": f"moved={moved:.3f}m"}
    if parts[0] == "lift":
        name, threshold = parts[1], float(parts[2])
        zs = np.asarray([row["tracked_bodies"][name]["position"][2] for row in rows])
        lift = float(zs.max() - zs[0])
        return {"success": lift >= threshold, "stage": f"lift={lift:.3f}m"}
    raise ValueError(f"unknown success kind {kind!r}")


if __name__ == "__main__":
    main()
