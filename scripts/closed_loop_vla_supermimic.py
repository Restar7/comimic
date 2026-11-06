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
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# 闭环**必须**能渲 ego 视角（VLA 的输入；--no-video 只是不写 mp4），但 GL 后端在机器之间
# 差异很大，踩过两次（见 docs §5.2 第 4/5/7 条）：
#   1) run_j1_rounds.sh 为省显存全局 `MUJOCO_GL=disable` ⇒ `mujoco.Renderer` 拿不到 GLContext；
#   2) PyOpenGL 在 linux 上"看到 $DISPLAY 就选 GLX"，而 GLX 平台没有 EGL 符号
#      ⇒ `'NoneType' object has no attribute 'eglQueryString'`；
#   3) H200 上把它们都设对了仍然炸：PyOpenGL 的 EGL 平台在那里起不来，静默回落到 GLX。
# 所以不再猜——下面 `_import_mujoco()` 按顺序**真试**一遍（egl → osmesa），哪个能 import 用哪个。
GL_BACKENDS: tuple[str, ...] = ("egl", "osmesa")


def _import_mujoco():
    """Import ``mujoco`` with the first GL backend that actually works on this box."""
    forced = os.environ.get("SM_GL_BACKEND", "").strip()
    backends = (forced,) if forced else GL_BACKENDS
    errors: list[str] = []
    for backend in backends:
        os.environ["MUJOCO_GL"] = backend
        if backend == "egl":
            os.environ["PYOPENGL_PLATFORM"] = "egl"   # 否则 PyOpenGL 按 $DISPLAY 选 GLX
        else:
            os.environ.pop("PYOPENGL_PLATFORM", None)
        for name in [m for m in list(sys.modules) if m.split(".")[0] in ("mujoco", "OpenGL")]:
            del sys.modules[name]                     # 上一次失败可能留了半截模块
        try:
            import mujoco  # noqa: PLC0415

            print(f"[gl] backend={backend}", flush=True)
            return mujoco
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{backend}: {type(exc).__name__}: {exc}")
            print(f"[gl][warn] {backend} 不可用 → 试下一个（{type(exc).__name__}）", flush=True)
    raise SystemExit("[fail] 没有可用的 GL 后端（本机渲不出 ego 视角）：\n  " + "\n  ".join(errors))

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "groot_repro"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

def _pin_hf_cache() -> None:
    """Make the local Cosmos-Reason2-2B cache actually be used by ``--arm vla``.

    Why this exists: ``third_party/gr00t_deps/sitecustomize.py`` runs at interpreter start-up
    and imports transformers → huggingface_hub *before* any of our code. ``huggingface_hub``
    freezes ``HF_HUB_CACHE`` / ``HF_HUB_OFFLINE`` into module constants at import time, so the
    ``os.environ.setdefault(...)`` calls in ``main()`` are dead code: the cache stayed at
    ``~/.cache/huggingface/hub`` and a HEAD call went to huggingface.co, where the gated
    ``nvidia/Cosmos-Reason2-2B`` answers 401 (2026-10-29 03:45: all 4 B-line episodes died there).
    Patching the already-loaded constants is the only in-process fix; the shell runbooks also
    export HF_HOME/HF_HUB_OFFLINE before launching python.
    """
    home = os.environ.get(
        "HF_HOME", "/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface"
    )
    cache = str(Path(home).expanduser() / "hub")
    offline = os.environ.get("HF_HUB_OFFLINE", "1") == "1"
    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.HF_HUB_CACHE = cache
        hub_constants.HF_HUB_OFFLINE = offline
    except Exception:  # huggingface_hub 还没被拉起来的干净环境里不需要补
        pass
    try:  # transformers 另外把 cache 路径 / offline 标志复制成自己的模块常量
        import transformers.utils.hub as tf_hub

        for name in ("TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
            if hasattr(tf_hub, name):
                setattr(tf_hub, name, cache)
        tf_hub._is_offline_mode = offline
    except Exception:
        pass
    if not Path(cache).is_dir():
        print(f"[warn] HF cache 不存在: {cache}（--arm vla 会尝试联网，可能 401）", flush=True)


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

_VLM_REPO_ID = "nvidia/Cosmos-Reason2-2B"


def _offline_vla_view(ckpt: Path) -> Path:
    """Return a checkpoint view whose ``model_name`` points at the **local** VLM.

    ``Gr00tPolicy`` reads ``model_name`` from ``config.json`` (backbone) and
    ``processor_config.json`` (tokenizer/processor); both are pinned to the gated repo id
    ``nvidia/Cosmos-Reason2-2B``, so even with the 4.88 GB weights already in the HF cache the
    loader still calls the Hub and gets 401 (2026-10-29 03:50, second B-line failure: the
    ``model_info`` HEAD inside the tokenizer path). The view is nothing but symlinks plus those
    two rewritten fields, so ``AutoModel.from_pretrained(view)`` never touches the network.
    """
    try:
        from diff_vla_wbc.gr00t_vlm import resolve_vlm_source

        source, origin = resolve_vlm_source()
    except Exception as exc:  # noqa: BLE001 - 拿不到本地权重就退回原 ckpt（可能联网 401）
        print(f"[warn] 解析本地 VLM 权重失败（{exc}）→ 直接用原 ckpt", flush=True)
        return ckpt
    if origin == "repo-id":
        print("[warn] 本地没有 VLM 权重（只解析到 repo id）→ 直接用原 ckpt", flush=True)
        return ckpt

    src = str(Path(source).resolve())
    view = ROOT / "runs/offline_views" / f"{ckpt.parent.name}__{ckpt.name}"
    stamp = view / ".vlm_source"
    if stamp.is_file() and stamp.read_text().strip() == src and (view / "config.json").is_file():
        return view

    tmp = view.with_name(view.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    for item in ckpt.iterdir():
        if item.name in ("config.json", "processor_config.json"):
            continue
        (tmp / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    for name in ("config.json", "processor_config.json"):
        path = ckpt / name
        if not path.is_file():
            continue
        if name == "config.json":
            payload = json.loads(path.read_text())
            payload["model_name"] = src
            text = json.dumps(payload, indent=2)
        else:  # 保留统计量/字段结构，只换掉里面的 gated repo id
            text = json.dumps(json.loads(path.read_text()), indent=2).replace(_VLM_REPO_ID, src)
        (tmp / name).write_text(text)
    (tmp / ".vlm_source").write_text(src + "\n")
    shutil.rmtree(view, ignore_errors=True)
    tmp.rename(view)
    print(f"[vla-view] model_name → {src}（离线，{view}）", flush=True)
    return view


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
    ap.add_argument("--out-video", type=Path, default=None, help="不给且 --no-video 时不写视频")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-gib", default="40GiB")
    ap.add_argument("--world-camera", default="side_view")
    ap.add_argument("--ego-camera", default="head_camera")
    ap.add_argument("--video-every", type=int, default=2)
    ap.add_argument("--hand-mode", choices=("vla", "open"), default="vla")
    # ---- B 线采集（给 WM 保鲜 / critic / b_start 用；见 docs/S3_END_TO_END_PLAN_20261028.md §1）----
    ap.add_argument("--dump", type=Path, default=None,
                    help="把每条 episode 的 {states, actions, tokens, q_target_applied} 存成 .pt")
    # ---- 回放模式（B 线采集，frames = 真库行号；b_start/critic/reward 都靠它）----
    ap.add_argument("--replay-lib", type=Path, default=None,
                    help="参考库（compact .pt）。给了就进回放模式：从 --start-frame 起步、每 tick 推 1 帧")
    ap.add_argument("--start-frame", type=int, default=0, help="起始库行号（回放模式）")
    ap.add_argument("--joint-src", type=Path, default=None,
                    help="--arm joint 用：录制的 episode parquet（读 action.wbc 做 PD 回放）")
    ap.add_argument("--arm", choices=("vla", "gt", "joint"), default="vla",
                    help="gt = 录制 token（不加载 VLA，快，用来验链路）；vla = 真实 VLA 推理（采漂移）")
    ap.add_argument("--tokens", type=Path, default=None, help="录制 token 包（--arm gt 必需）")
    ap.add_argument("--no-video", action="store_true", default=False, help="只采集不录像（快很多）")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    _pin_hf_cache()

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
    mujoco = _import_mujoco()
    import g1_sim as G  # noqa: PLC0415
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: PLC0415

    from diff_vla_wbc.supermimic_backend import build_g1_assets  # noqa: PLC0415
    from diff_vla_wbc.train import JointTrainer  # noqa: PLC0415

    device = torch.device(args.device)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    smap = G.SceneMap(model)
    mujoco.mj_resetData(model, data)
    replay = args.replay_lib is not None
    lib = None
    tok_pack = None
    if replay:
        # 回放模式：root + 29 关节 + qd **一起**从库里读（与 4090 交接的 SM_INIT_FROM_LIB=1 同义）
        # 用本仓的加载器：它会把 mjlab 的 G1 XML 指向我们自己的资产（系统 python 里 mjlab 是 shim，
        # 直接构造 MotionLibrary 会 `mjlab.__file__` 崩）
        from diff_vla_wbc.supermimic_backend import build_g1_assets, load_motion_library
        _assets0 = build_g1_assets(device="cpu")
        lib = load_motion_library([str(args.replay_lib)], _assets0, compact=True, device="cpu")
        f0 = int(args.start_frame)
        qpos, qvel = lib.ref_qpos_qvel(torch.tensor([f0]))
        # 场景 nq 包含手指与物体；只写机器人那一段（qpos 36 = 3+4+29，qvel 35 = 3+3+29）
        data.qpos[0:36] = qpos[0].numpy()
        data.qvel[0:35] = qvel[0].numpy()
        mujoco.mj_forward(model, data)
        if args.arm == "gt":
            if args.tokens is None:
                raise SystemExit("--arm gt 需要 --tokens（录制 token 包）")
            tok_pack = torch.load(str(args.tokens), map_location="cpu", weights_only=False)["tokens"].float()
        print(f"[replay] lib={args.replay_lib} start_frame={f0} arm={args.arm} "
              f"frames={lib.num_frames}", flush=True)
    else:
        data.qpos[0:3] = [0.0, 0.0, 0.781]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        smap.set_q(data, G.DEFAULT_ANGLES.copy())
        smap.set_hands(data, np.zeros(14))
        mujoco.mj_forward(model, data)
    # Same spawn fix as the token-exact loop: the deploy MJCF puts the feet ~3.5 cm into
    # the floor; without this the first solver step fires a huge impulse.
    # 2026-10-30 修 bug：原来只要**任何**接触穿透就把整机往上抬 `-deepest`，而
    # `deepest` 可能是"手/手臂插在桌子里"（实测 -0.87 m！）⇒ 机器人被抬到桌子高度
    # 1.649 m 悬空，然后自由落体摔成一坨（f62251 那集的 tilt 94-115° 就是这个）。
    # 修复：单次抬升上限 3 cm、累计上限 5 cm；超限说明不是"脚陷进地板"，只警告不改高度。
    _lift_total = 0.0
    for _ in range(4):
        if data.ncon == 0:
            break
        deepest = min(float(c.dist) for c in data.contact[: data.ncon])
        if deepest >= -1e-4:
            break
        _d = float(min(-deepest + 1e-3, 0.03))
        _d = float(min(_d, max(0.05 - _lift_total, 0.0)))
        if _d <= 0.0:
            print(f"[sim][warn] 初始穿透 {deepest * 100:.1f} cm 超出抬升上限"
                  f"（累计已抬 {_lift_total * 100:.1f} cm），不再抬升以免悬空", flush=True)
            break
        data.qpos[2] += _d
        _lift_total += _d
        mujoco.mj_forward(model, data)
    print(f"[sim] scene={cfg['scene']} nq={model.nq} spawn z={data.qpos[2]:.3f} "
          f"contacts={data.ncon}", flush=True)

    assets = build_g1_assets(device=device)
    bundle = JointTrainer.load_bundle_from_checkpoint(args.joint_ckpt, assets)
    JQ = JH = None
    if args.arm == "joint":
        if args.joint_src is None:
            raise SystemExit("--arm joint 需要 --joint-src <episode.parquet>")
        import pyarrow.parquet as _pq  # noqa: PLC0415
        _aw = np.array(_pq.ParquetFile(str(args.joint_src)).read(columns=["action.wbc"])
                       .column("action.wbc").to_pylist(), np.float64)
        # 43 布局: [腿12, 腰3, 左臂7, 左手7, 右臂7, 右手7] ⇒ 身体 29 = [0:22]+[29:36]（= MuJoCo 序）
        JQ = _aw[:, list(range(22)) + list(range(29, 36))]
        JH = np.concatenate([_aw[:, 22:29], _aw[:, 36:43]], axis=1)
        print(f"[replay] {args.joint_src.name}: {len(JQ)} 帧（PD 回放录制关节，不用 VLA/WBC）",
              flush=True)
    if int(getattr(bundle.policy, "token_dim", 0)) != 64:
        raise SystemExit("SuperMimic WBC is not token-conditioned")
    print(f"[sm] WBC {args.joint_ckpt} (token-conditioned)", flush=True)

    policy = None
    if not (replay and args.arm in ("gt", "joint")):
        policy = Gr00tPolicy(embodiment_tag="unitree_g1_sonic",
                             model_path=str(_offline_vla_view(args.vla_ckpt)),
                             device=args.device, strict=True)
        print(f"[vla] {args.vla_ckpt}", flush=True)
    else:
        print("[vla] skipped（--arm gt：用录制 token 驱动）", flush=True)

    # gt + 不录像时完全不需要渲染（也就不需要 EGL/display）
    renderer = (None if (args.arm in ("gt", "joint") and (args.no_video or args.out_video is None))
                else mujoco.Renderer(model, 480, 640))
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

    writer = (None if (args.no_video or args.out_video is None)
              else VideoWriter(args.out_video, 1280, 480, 50.0 / args.video_every))
    sub = int(round(0.02 / model.opt.timestep))
    xi = torch.zeros(1, 33, device=device)
    history = None
    chunk = (np.zeros((40, 64), np.float32), np.zeros((40, 14), np.float32)) \
        if args.arm == "joint" else None
    consumed = 0
    plans = 0
    rows = []
    # B 线 dump：按控制 tick 记 (s_t, a_t, token_t, s_{t+1})。动作 = **真正下发**的归一化动作
    # （mjlab: q_target = default_q + action_scale * a，所以记录 a 本身即可）。
    rec_states, rec_next, rec_actions, rec_tokens, rec_qt = [], [], [], [], []
    rec_frames, rec_rewards = [], []
    cur_frame = int(args.start_frame) if replay else -1
    _a_prev = torch.zeros(1, 29, device=device)
    with torch.no_grad():
        for tick in range(args.steps):
            if replay and args.arm == "gt":
                # 回放：每 tick 参考前进 1 帧，token 用数据集录制值（与训练同空间）
                cur_frame = min(cur_frame + 1, int(lib.num_frames) - 2)
                rec_frames.append(cur_frame)
                tk = tok_pack[cur_frame].reshape(1, 64).to(device)
                tokens = np.repeat(tk.cpu().numpy(), 40, axis=0)
                hands = np.zeros((40, 14), np.float32)
                chunk = (tokens, hands)
                plan_tick = tick
            elif args.arm != "joint" and (chunk is None or consumed >= args.execute_steps):
                obs = observation(data, smap, renderer, ego_cam, G, cfg["prompt"])
                action, _info = policy.get_action(obs)
                if replay:
                    cur_frame = min(cur_frame + 1, int(lib.num_frames) - 2)
                    rec_frames.append(cur_frame)
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
            idx = 0 if (replay and args.arm == "gt") else min(consumed, len(tokens) - 1)
            token = torch.as_tensor(tokens[idx], dtype=torch.float32, device=device).reshape(1, 64)

            state = np.concatenate([data.qpos[0:3], data.qpos[3:7], smap.q(data),
                                    data.qvel[0:3], data.qvel[3:6], smap.dq(data)])
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            if args.arm == "joint":
                _fr = min(int(cur_frame), len(JQ) - 1)
                cur_frame += 1
                q_target = JQ[_fr].astype(np.float32)
                act29 = torch.zeros(1, 29, device=device)
            else:
                act29 = bundle.act(bundle.flatten_history(history), token) if history is not None \
                    else torch.zeros(1, 29, device=device)
                q_target = (assets.default_joint_pos + assets.action_scale * act29[0]).cpu().numpy()
            if args.dump is not None:
                rec_states.append(state.copy())
                rec_actions.append(act29[0].detach().cpu().numpy().astype(np.float32))
                rec_tokens.append(token[0].detach().cpu().numpy().astype(np.float32))
                rec_qt.append(q_target.astype(np.float32))
            if args.arm == "joint":
                hand_t = JH[min(int(cur_frame) - 1, len(JH) - 1)]
            else:
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
            if args.dump is not None:
                rec_next.append(state.copy())
            clean = bundle.observation(state_t, act29, xi)
            history = clean.unsqueeze(0).repeat(10, 1, 1) if history is None \
                else torch.cat((history[1:], clean.unsqueeze(0)), dim=0)

            if replay:
                _at = torch.as_tensor(act29[0].detach().cpu().numpy(), dtype=torch.float32).reshape(1, 29)
                _rf = bundle.reference_frame(lib, torch.tensor([cur_frame]))
                # 库在 CPU、state 在 GPU：ReferenceFrame 的 6 个张量都要搬到 device
                _ref = _rf.__class__(
                    _rf.body_pos_w.to(device), _rf.body_quat_w.to(device),
                    _rf.body_lin_vel_w.to(device), _rf.body_ang_vel_w.to(device),
                    _rf.root_pos_w.to(device), _rf.root_quat_w.to(device))
                _terms = bundle.reward(state_t, _at.to(device), _a_prev.to(device), _ref)
                rec_rewards.append(float(_terms["total"]))
                _a_prev = _at
            tilt = float(np.degrees(np.arccos(np.clip(
                G.quat_to_rotmat(data.qpos[3:7])[2, 2], -1, 1))))
            rows.append({"tick": tick, "z": float(data.qpos[2]), "tilt_deg": tilt})
            if tick % 100 == 0:
                print(f"[{tick:4d}] z={data.qpos[2]:.3f} tilt={tilt:5.1f}° plans={plans}",
                      flush=True)
            if writer is not None and tick % args.video_every == 0:
                writer.write(np.concatenate((render(world_cam), render(ego_cam)), axis=1))
            if not (replay and args.arm in ("gt", "joint")):
                consumed += 1
    if writer is not None:
        writer.release()
    if renderer is not None:
        renderer.close()

    summary = {
        "task": args.task, "vla": str(args.vla_ckpt), "wbc": str(args.joint_ckpt),
        "scene": cfg["scene"], "ticks": args.steps, "seconds": round(args.steps * 0.02, 2),
        "plans": plans,
        "min_z": float(min(r["z"] for r in rows)),
        "max_tilt_deg": float(max(r["tilt_deg"] for r in rows)),
        "final_z": rows[-1]["z"], "final_tilt_deg": rows[-1]["tilt_deg"],
        "fell": bool(max(r["tilt_deg"] for r in rows) > 60.0
                       or (min(r["z"] for r in rows) < 0.30
                           and max(r["tilt_deg"] for r in rows) > 30.0)),
        # 2026-10-31 修：原来 `min_z < 0.45` 会把 **深蹲**判成摔倒 ——
        # task4 就是"蹲下捡地上铝罐"，骨盆正常就到 0.44（实测该集 max_tilt 仅 14.7°），
        # 结果 80 集健康门 fell=1/80 直接把整条流程中止。
        # 摔 = 倾角大 或 （够低 且 已经在倾倒）。
        "video": str(args.out_video) if args.out_video else None,
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({**summary, "rows": rows}, indent=2))

    if args.dump is not None:
        import torch as _torch

        n = len(rec_actions)
        dump = {
            "states": _torch.as_tensor(np.stack(rec_states[:n])).float(),          # (T,71) s_t
            "next_states": _torch.as_tensor(np.stack(rec_next[:n])).float(),       # (T,71) s_{t+1}
            "actions": _torch.as_tensor(np.stack(rec_actions)).float(),            # (T,29) 归一化动作 a
            "tokens": _torch.as_tensor(np.stack(rec_tokens)).float(),              # (T,64) 环内 token
            "q_target_applied": _torch.as_tensor(np.stack(rec_qt)).float(),        # (T,29) 弧度
            # 回放模式：frames = **真库行号**（BStartSource 认这个）；自由闭环为 -1
            "frames": (_torch.as_tensor(rec_frames[:n], dtype=_torch.long) if rec_frames
                       else _torch.full((n,), -1, dtype=_torch.long)),
            "rewards": (_torch.as_tensor(rec_rewards[:n], dtype=_torch.float32) if rec_rewards
                        else _torch.zeros(n)),
            "meta": {
                **summary, "dt": 0.02, "joint_order": "standard",
                "state_layout": "[p(3), quat(4), q(29), v_w(3), w_b(3), qd(29)]",
                "action_semantics": "a_exec (q_target = default_q + action_scale * a)",
                "has_ref": bool(rec_rewards),
                "frames_source": "library_row" if replay else "none",
                "cfg": {"SM_TOKEN_SOURCE": ("online" if (replay and args.arm == "gt")
                                            else ("http" if replay else "free")),
                        "SM_INIT_FROM_LIB": "1" if replay else "0",
                        "SM_START_FRAME": int(args.start_frame),
                        "SM_HAND": "0"},
                "arm": str(args.arm), "replay_lib": str(args.replay_lib) if replay else None,
                "source": "closed_loop_vla_supermimic",
            },
        }
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        _torch.save(dump, str(args.dump))
        print(f"[dump] {n} ticks -> {args.dump}", flush=True)


if __name__ == "__main__":
    main()
