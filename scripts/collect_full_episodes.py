#!/usr/bin/env python3
"""Collect full-length B episodes for the 5-round plan.

Unlike `s3_collect_parallel.sh` (fixed MAX_STEPS, random start, margin filter),
this runner picks clips per task, starts at frame 0 (or a uniform start), and
sets each episode's MAX_STEPS to its own remaining clip length.

Server must be started separately (base or current head):
  PORT=18180 GPU=2 CKPT=.../checkpoint-8000 bash scripts/start_vla_server.sh
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK_PREFIX = {
    "DoubleDesk": "HOI_double_desk",
    "PPBox": "HOI_pp_box",
    "Boxing": "HSI_boxing",
    "OpenDoor": "HSI_open_door",
    "SitSofa": "HSI_sit_sofa",
    "VisionNavi": "HSI_vision_navi",
    "GrapCup": "HOI_grap_cup",
}


def pick_clips(task: str, n: int, seed: int) -> list[dict]:
    packed = torch.load(str(ROOT / "runs/isaac_motion/ha_cmd/packed.pt"), map_location="cpu", weights_only=False)
    names, starts, lengths = packed["names"], packed["starts"], packed["lengths"]
    pre = TASK_PREFIX[task]
    idx = [i for i, name in enumerate(names) if name.startswith(pre)]
    if not idx:
        raise SystemExit(f"no clips for {task} ({pre})")
    idx.sort(key=lambda i: int(lengths[i]))
    if n >= len(idx):
        picked = idx
    else:
        # stratified by length quantile; unique positions
        pos = []
        for k in range(n):
            p = round(k * (len(idx) - 1) / max(1, n - 1))
            while p in pos:
                p = (p + 1) % len(idx)
            pos.append(p)
        picked = [idx[p] for p in pos]
    return [{
        "task": task,
        "clip": names[i],
        "row": int(starts[i]),
        "length": int(lengths[i]),
        "start": 0,
        "seed": seed,
    } for i in picked]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--tasks", default="DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi")
    ap.add_argument("--episodes", type=int, default=30, help="episodes per task")
    ap.add_argument("--out-root", default="", help="default runs/B_<round>")
    ap.add_argument("--server-url", default="http://127.0.0.1:18180")
    ap.add_argument("--sm-ckpt", default=str(ROOT / "runs/S1_handover/WBC_A24000_wmvel_v6.pt"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = args.tasks.split()
    specs = []
    for t in tasks:
        specs += pick_clips(t, args.episodes, args.seed)
    out_root = Path(args.out_root) if args.out_root else ROOT / "runs" / f"B_{args.round}"
    log_root = ROOT / "logs" / f"full_{args.round}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    print(f"[full] {len(specs)} episodes, tasks={tasks}, jobs={args.jobs}, gpus={args.num_gpus}", flush=True)
    for s in specs:
        print(f"[full] {s['task']:10s} row={s['row']:7d} length={s['length']:5d} clip={s['clip']}", flush=True)
    if args.dry_run:
        return 0

    lock = threading.Lock()
    done = 0
    failed = 0

    def run(i: int, s: dict):
        gpu = args.gpu_offset + (i % args.num_gpus)
        name = f"vla_{s['task']}_f{s['row']}_r{s['seed']}"
        out = out_root / name
        log = log_root / f"{name}.log"
        dump = out / "dump"
        if dump.exists() and any(dump.glob("*.pt")):
            return i, s, gpu, 0, 0.0, str(out), str(log), "skip"
        env = os.environ.copy()
        env.update({
            "ARM": "vla",
            "TASK": s["task"],
            "SEED": "0",
            "RESET_SEED": str(s["seed"]),
            "START_FRAME": str(s["row"]),
            "MAX_STEPS": str(s["length"]),
            "GPU": str(gpu),
            "SERVER_URL": args.server_url,
            "SM_CKPT": args.sm_ckpt,
            "OUT": str(out),
        })
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.run(["bash", str(ROOT / "scripts/s3_collect_arm.sh")],
                                env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        return i, s, gpu, rc, time.time() - t0, str(out), str(log), "run"

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(run, i, s) for i, s in enumerate(specs)]
        for fu in as_completed(futs):
            i, s, gpu, rc, dt, out, log, mode = fu.result()
            with lock:
                done += 1
                if rc != 0:
                    failed += 1
            print(f"[full] {done}/{len(specs)} {mode} rc={rc} {s['task']} row={s['row']} "
                  f"L={s['length']} gpu={gpu} {dt:.0f}s {out}", flush=True)

    print(f"[full] DONE total={len(specs)} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
