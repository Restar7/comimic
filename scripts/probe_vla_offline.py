#!/usr/bin/env python3
"""Offline VLA probe: recorded (image, state) -> predicted token/hands vs recorded labels.

Answers "is the official-finetune VLA actually predicting the recorded action?" without
running Isaac.  Feeds one frame of a recorded HA session to the latent64 VLA server and
compares the returned 64-d token + 14-d hands with the dataset labels.

Usage::

    python3 scripts/probe_vla_offline.py --task HOI_double_desk__tw --episode 0 \
        --frames 200 600 1000 --port 18101
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
INSTR = {
    "HOI_double_desk": "Pick up the object from the double desk and place it into the basket.",
    "HOI_grap_cup": "Grasp the cup on the table and lift it up.",
    "HOI_pp_box": "Pick up the box and place it at the target location.",
    "HSI_boxing": "Throw punches at the target following the reference motion.",
    "HSI_open_door": "Walk to the door, open it, and step through.",
    "HSI_sit_sofa": "Walk to the sofa and sit down.",
    "HSI_vision_navi": "Navigate to the goal location shown in the instruction.",
}
FPS = 50


def _video_frame(path: Path, idx: int) -> np.ndarray:
    """Decode one frame as HxWx3 uint8 via ffmpeg (no cv2/av dependency)."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})",
         "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(480, 640, 3)


def _rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    m = R.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return m[:, :2].reshape(6).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--port", type=int, default=18101)
    args = ap.parse_args()

    sess = ROOT / "data" / "ha_sessions" / args.task
    pq_path = sess / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    video = sess / "videos" / "chunk-000" / "observation.images.ego_view" / \
        f"episode_{args.episode:06d}.mp4"
    t = pq.read_table(pq_path)
    state = np.array([np.array(x.as_py(), dtype=np.float32) for x in t.column("observation.state")])
    root = np.array([np.array(x.as_py(), dtype=np.float32) for x in t.column("observation.root_orientation")])
    tok = np.array([np.array(x.as_py(), dtype=np.float32) for x in t.column("action.motion_token")])
    wbc = np.array([np.array(x.as_py(), dtype=np.float32) for x in t.column("action.wbc")])
    hands_lab = wbc[:, -14:]
    task_key = "_".join(args.task.split("__")[0].split("_")[:3])
    instruction = INSTR.get(task_key, "")
    print(f"[probe] task={args.task} ep={args.episode} T={len(state)} instruction={instruction!r}")

    mean_tok = tok.mean(0)
    base_mse = float(np.mean((tok - mean_tok) ** 2))
    print(f"[probe] label token var (MSE vs mean) = {base_mse:.4f}; "
          f"label token absmean={np.abs(tok).mean():.4f}")

    import urllib.request
    for f in args.frames:
        joints = state[f, :29]
        vel = (state[f, :29] - state[max(f - 1, 0), :29]) * FPS
        s64 = np.concatenate([_rot6d(root[f]), joints, vel]).astype(np.float32)
        img = _video_frame(video, f)
        body = json.dumps({
            "observation": {
                "images": {"front": {"shape": list(img.shape), "dtype": "uint8",
                                     "data_b64": base64.b64encode(img.tobytes()).decode()}},
                "state": s64.tolist(),
            },
            "robot_type": "unitree_g1_refpose_v3_1",
            "return_chunk": True,
            "task": instruction,
        }).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/infer", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        chunk = np.asarray(resp["latent64_chunk"], dtype=np.float32)
        hands = np.asarray(resp.get("hands14_chunk", np.zeros((len(chunk), 14), np.float32)),
                           dtype=np.float32)
        pred = chunk[0]
        mse = float(np.mean((pred - tok[f]) ** 2))
        corr = float(np.corrcoef(pred, tok[f])[0, 1])
        # chunk vs the recorded 40-step label window (if within the clip)
        n = min(len(chunk), len(tok) - f)
        chunk_mse = float(np.mean((chunk[:n] - tok[f:f + n]) ** 2))
        h_max = float(np.abs(hands[0]).max())
        h_lab = float(np.abs(hands_lab[f]).max())
        print(f"[probe] f={f:5d} token_mse={mse:.4f} corr={corr:+.3f} "
              f"chunk_mse={chunk_mse:.4f} | base_mse={base_mse:.4f} | "
              f"hands_pred_max={h_max:.3f} hands_label_max={h_lab:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
