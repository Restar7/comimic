#!/usr/bin/env python3
"""Reproduce a HumanoidArena episode's **initial state** in Isaac, and verify the scene match.

Two things must line up for "闭环推理在正确的场景里、和训练完全一致":

1. **场景/初始位姿** —— the sim must spawn the robot and every object where the recording had
   them.  The raw npz carries exactly that:
     ``robot_root_position[0]`` / ``robot_root_orientation[0]``
     ``episode_init_env_obj_<name>_{position,orientation,linear_velocity,angular_velocity}``
   plus the task yaml (``tasks/common_env_config/<task>_sonic.yaml``) gives the reset ranges and
   the object ``prim_paths``.

2. **首帧画面** —— if the first rendered frame does not match the recording's first frame
   pixel-wise, the scene is *not* the training scene, no matter what the JSON says.

This script does both halves offline (no Isaac, no GPU):

    # 1) emit the initial-state override for one episode
    python3 scripts/ha_scene_bootstrap.py init \
        --npz data/ha_raw/HOI_double_desk/tw/<ep>.npz --task HOI_double_desk

    # 2) dump the recorded first frame (needs the mp4 next to the npz)
    python3 scripts/ha_scene_bootstrap.py first-frame \
        --npz data/ha_raw/HOI_double_desk/tw/<ep>.npz --out runs/ha_bootstrap/<ep>_first_frame.png

    # 3) compare a simulated first frame against the recorded one
    python3 scripts/ha_scene_bootstrap.py compare \
        --reference runs/ha_bootstrap/<ep>_first_frame.png --sim <sim_first_frame.png>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HA_ROOT = Path("/inspire/hdd/global_user/liumengfan-253108110079/qwk-workspace/HumanoidArena")
YAML_DIR = HA_ROOT / "isaaclab_twist2_g1" / "tasks" / "common_env_config"
FFMPEG = ROOT / "closed_loop" / "bin" / "ffmpeg"

TASK_TO_YAML = {
    "HOI_double_desk": "doubledesk_sonic.yaml",
    "HOI_grap_cup": "selective_reaching_sonic.yaml",
    "HOI_pp_box": "pickplace_box_sonic.yaml",
    "HSI_boxing": "boxing_bag_sonic.yaml",
    "HSI_open_door": "opendoor_sonic.yaml",
    "HSI_sit_sofa": "livingroom_sitsofa_sonic.yaml",
    "HSI_vision_navi": "small_warehouse_vision_navigation_sonic.yaml",
    "HOI_football_v2": "football_single_sonic.yaml",
}


def _yaml_blocks(path: Path) -> dict:
    """Tiny reader for the ``deterministic_object_resets`` block (no PyYAML dependency)."""
    text = path.read_text()
    out: dict[str, dict] = {}
    cur = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if "record_name:" in line:
            cur = line.split("record_name:")[1].strip()
            out[cur] = {}
        elif cur and "prim_paths:" in line:
            tail = line.split("prim_paths:", 1)[1].strip()
            out[cur]["prim_paths_raw"] = tail
        elif cur and "x:" in line and "pose_range" not in line:
            out[cur].setdefault("pose_range", {})["x"] = line.split(":", 1)[1].strip()
        elif cur and "y:" in line and "pose_range" not in line:
            out[cur].setdefault("pose_range", {})["y"] = line.split(":", 1)[1].strip()
        elif cur and "z:" in line and "pose_range" not in line:
            out[cur].setdefault("pose_range", {})["z"] = line.split(":", 1)[1].strip()
        elif cur and "yaw:" in line:
            out[cur].setdefault("pose_range", {})["yaw"] = line.split(":", 1)[1].strip()
    return out


def cmd_init(args) -> int:
    z = np.load(args.npz, allow_pickle=False)
    task = args.task or args.npz.parents[1].name
    payload: dict = {
        "source_npz": str(args.npz),
        "task": task,
        "task_env_name": str(z["task"]) if "task" in z.files else None,
        "num_frames": int(z["num_frames"]),
        "schema_version": str(z["schema_version"]) if "schema_version" in z.files else None,
        "robot": {},
        "objects": {},
        "yaml_reset_ranges": None,
    }
    if "robot_root_position" in z.files:
        payload["robot"]["position_xyz"] = np.asarray(z["robot_root_position"][0], float).tolist()
        payload["robot"]["orientation_wxyz"] = np.asarray(z["robot_root_orientation"][0], float).tolist()
    # ``env_obj_<name>_position`` -> <name>; names may contain underscores, so slice by prefix.
    names = sorted({
        k[len("env_obj_"):-len("_position")] for k in z.files
        if k.startswith("env_obj_") and k.endswith("_position")
    })
    for name in names:
        entry: dict = {}
        if f"env_obj_{name}_position" in z.files:
            entry["measured_start_position"] = np.asarray(z[f"env_obj_{name}_position"][0], float).tolist()
        if f"env_obj_{name}_orientation" in z.files:
            entry["measured_start_orientation_wxyz"] = np.asarray(
                z[f"env_obj_{name}_orientation"][0], float).tolist()
        for key, suffix in (("position", "position_xyz"), ("orientation", "orientation_wxyz"),
                            ("linear_velocity", "linear_velocity"), ("angular_velocity", "angular_velocity")):
            # 真实键名带 episode_ 前缀（见 npz key dump）
            full = f"episode_init_env_obj_{name}_{key}"
            if full in z.files:
                entry[f"recorded_{suffix}"] = np.asarray(z[full], float).reshape(-1).tolist()
        payload["objects"][name] = entry

    yaml_path = YAML_DIR / TASK_TO_YAML.get(task, "")
    if yaml_path.is_file():
        payload["yaml_reset_ranges"] = {"file": str(yaml_path), "blocks": _yaml_blocks(yaml_path)}

    out = Path(args.out) if args.out else (
        ROOT / "runs" / "ha_bootstrap" / f"{task}__{args.npz.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"[written] {out}")
    print(f"  robot pos: {payload['robot'].get('position_xyz')}")
    print(f"  objects  : {sorted(payload['objects'])}")
    if payload["yaml_reset_ranges"]:
        print(f"  yaml     : {yaml_path.name} ({len(payload['yaml_reset_ranges']['blocks'])} reset blocks)")
    else:
        print("  yaml     : (none found — check TASK_TO_YAML)")
    return 0


def _recorded_video(npz: Path) -> Path | None:
    video = npz.parent / "videos" / f"{npz.stem}_front_rgb.mp4"
    return video if video.is_file() else None


def cmd_first_frame(args) -> int:
    video = _recorded_video(args.npz)
    if video is None:
        print(f"[fail] no recorded video next to {args.npz}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = FFMPEG if FFMPEG.is_file() else Path("ffmpeg")
    cmd = [str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-vf", "select=eq(n\\,0)", "-frames:v", "1", str(out)]
    if subprocess.call(cmd) != 0 or not out.is_file():
        print(f"[fail] ffmpeg could not extract frame 0 from {video}", file=sys.stderr)
        return 1
    print(f"[written] {out}  (frame 0 of {video.name})")
    return 0


def _png_size(path: Path) -> tuple[int, int]:
    """Read width/height straight from the PNG IHDR chunk (no PIL)."""
    with open(path, "rb") as fh:
        head = fh.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    width = int.from_bytes(head[16:20], "big")
    height = int.from_bytes(head[20:24], "big")
    return height, width


def _load_rgb_png(path: Path) -> np.ndarray:
    """Read a PNG without PIL: use ffmpeg to convert to raw rgb24 on stdout."""
    ffmpeg = FFMPEG if FFMPEG.is_file() else Path("ffmpeg")
    out = subprocess.run(
        [str(ffmpeg), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    h, w = _png_size(path)
    if h * w * 3 != len(out):
        raise ValueError(f"{path}: decoded {len(out)} bytes but header says {h}x{w}")
    return np.frombuffer(out, dtype=np.uint8).reshape(h, w, 3)


def cmd_compare(args) -> int:
    ref = _load_rgb_png(Path(args.reference))
    sim = _load_rgb_png(Path(args.sim))
    if ref.shape != sim.shape:
        print(f"[fail] shape mismatch: reference {ref.shape} vs sim {sim.shape}", file=sys.stderr)
        return 1
    a = ref.astype(np.float32) / 255.0
    b = sim.astype(np.float32) / 255.0
    mae = float(np.abs(a - b).mean())
    mse = float(((a - b) ** 2).mean())
    psnr = float("inf") if mse == 0 else float(-10.0 * np.log10(mse))
    # 结构上的粗略判据：亮度直方图相关 + 边缘图相关（不需要 skimage）
    ga = a.mean(-1); gb = b.mean(-1)
    hist_r = float(np.corrcoef(np.histogram(ga, 32, (0, 1))[0], np.histogram(gb, 32, (0, 1))[0])[0, 1])
    ex = np.abs(np.diff(ga, axis=0)); ey = np.abs(np.diff(gb, axis=0))
    edge_r = float(np.corrcoef(ex.ravel(), ey.ravel())[0, 1])
    verdict = "MATCH" if (mae < args.max_mae and psnr > args.min_psnr) else "MISMATCH"
    print(json.dumps({
        "reference": str(args.reference), "sim": str(args.sim), "shape": list(ref.shape),
        "mae": round(mae, 5), "psnr_db": round(psnr, 3),
        "hist_corr": round(hist_r, 4), "edge_corr": round(edge_r, 4),
        "thresholds": {"max_mae": args.max_mae, "min_psnr": args.min_psnr},
        "verdict": verdict,
    }, indent=1))
    return 0 if verdict == "MATCH" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("init", help="emit the Isaac initial-state override for one episode")
    p1.add_argument("--npz", type=Path, required=True)
    p1.add_argument("--task", default=None)
    p1.add_argument("--out", default=None)
    p1.set_defaults(func=cmd_init)

    p2 = sub.add_parser("first-frame", help="dump the recorded first frame as PNG")
    p2.add_argument("--npz", type=Path, required=True)
    p2.add_argument("--out", type=Path, required=True)
    p2.set_defaults(func=cmd_first_frame)

    p3 = sub.add_parser("compare", help="compare a simulated first frame to the recorded one")
    p3.add_argument("--reference", type=Path, required=True)
    p3.add_argument("--sim", type=Path, required=True)
    p3.add_argument("--max-mae", type=float, default=0.05)
    p3.add_argument("--min-psnr", type=float, default=20.0)
    p3.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
