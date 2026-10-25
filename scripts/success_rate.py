#!/usr/bin/env python3
"""Multi-seed closed-loop success rate for one or more checkpoints.

This is the measurement harness for ablation #1 ("does our backprop finetune
actually help?"): run a checkpoint through the MuJoCo closed loop on every task
and seed, collect the deployment-style success predicates from each run's
``meta.json``, and report per-task and overall success rates with Wilson 95%
confidence intervals.

A single episode is binary, so 5 seeds x 4 tasks = 20 episodes gives a 5%
granularity — two arms must differ by >= 2 episodes before the intervals stop
overlapping. Prefer >= 30 episodes per arm for anything you want to claim.

Why subprocess: ``closed_loop_mujoco.py`` boots its own CUDA context, MuJoCo EGL
renderer and policy; keeping each episode in a fresh process means one OOM or
crash cannot poison the rest of the sweep, and seeds are truly independent.

Examples
--------
    # plan only (no GPU work)
    python3 scripts/success_rate.py \\
        --arm base=runs/ckpt_vla20k \\
        --arm ft4000=runs/ckpt_vla_ft4000 \\
        --seeds 0 1 2 3 4 --dry-run

    # run the sweep (4 tasks x 5 seeds x 2 arms = 40 episodes)
    python3 scripts/success_rate.py \\
        --arm base=runs/ckpt_vla20k \\
        --arm ft4000=runs/ckpt_vla_ft4000 \\
        --seeds 0 1 2 3 4 --chunks 40 --video

Resumable: an episode whose ``meta.json`` already exists is skipped, so a
crashed sweep can be restarted with the same command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOSED_LOOP = ROOT / "closed_loop"
# NB: the hand-streaming / deployment-cadence loop lives under mosaic_n17/tools,
# NOT the older closed_loop/code/closed_loop_mujoco.py (that one has no --hand-mode).
RUNNER = CLOSED_LOOP / "code" / "mosaic_n17" / "tools" / "closed_loop_mujoco.py"
G1_TASKS = ROOT / "data" / "npz" / "g1_tasks"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_manifest(path: Path) -> dict:
    """Cheap but unambiguous checkpoint identity (no 5 GB hashing)."""
    files = []
    for name in ("config.json", "model.safetensors.index.json", "statistics.json",
                 "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
                 "model-00003-of-00003.safetensors"):
        p = path / name
        if not p.exists():
            continue
        entry = {"file": name, "bytes": p.stat().st_size,
                 "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                 "symlink": p.is_symlink()}
        if p.stat().st_size < 50_000_000:  # config/index are tiny; hash them
            entry["sha256"] = sha256_file(p)
        else:
            entry["target"] = str(p.resolve())
        files.append(entry)
    return {"path": str(path.resolve()), "files": files}


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"],
                                      text=True)
        return bool(out.strip())
    except Exception:
        return True


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal at n<30)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


_DEPS_OK: bool | None = None


def build_env(device: str) -> dict[str, str]:
    """Same environment ``closed_loop/run_tasks.sh`` sets up (ffmpeg shim, EGL, deps).

    ``third_party/gr00t_deps`` contains a **partial** ``onnxruntime`` tree built
    for another Python; putting it on ``PYTHONPATH`` shadows a perfectly good
    site-packages install and every episode dies with
    ``No module named onnxruntime.capi.onnxruntime_pybind11_state``.
    Include it only when the interpreter cannot import the deps on its own.
    """
    global _DEPS_OK
    env = dict(os.environ)
    deps = ROOT / "third_party" / "gr00t_deps"
    noflash = deps / "_noflash"
    base = [CLOSED_LOOP / "code", ROOT / "Isaac-GR00T",
            CLOSED_LOOP / "assets" / "repro", noflash]
    env["PATH"] = f"{CLOSED_LOOP / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = ":".join(str(p) for p in base)
    env["MUJOCO_GL"] = "egl"
    env.setdefault("HF_HOME", "/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface")
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "8")
    env["CUDA_VISIBLE_DEVICES"] = device
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if _DEPS_OK is None:
        probe = ("import onnxruntime, tyro, transformers; "
                 "from gr00t.policy.gr00t_policy import Gr00tPolicy")
        rc = subprocess.call([sys.executable, "-c", probe], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _DEPS_OK = rc == 0
        if not _DEPS_OK:
            print(f"[env] interpreter {sys.executable} needs {deps} on PYTHONPATH",
                  flush=True)
    if not _DEPS_OK:
        env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{deps}"
    return env


def calib_npz(task: str) -> Path | None:
    """Recorded npz for the hand trend calibration (same rule as run_tasks.sh).

    The recordings are named ``task_1_...`` while the loop calls the task
    ``task1`` (and ``task_1``); accept every spelling instead of silently
    dropping the calibration (which used to make ``--hand-mode trend`` fall
    through to raw VLA hand values).
    """
    match = re.search(r"(\d+)", task)
    patterns = []
    if match:
        n = match.group(1)
        patterns = [f"task_{n}_*.npz", f"task{n}_*.npz"]
    patterns.append(f"{task}_*.npz")
    for pattern in patterns:
        hits = sorted(G1_TASKS.glob(pattern))
        if hits:
            return hits[0]
    return None


def episode(
    *, arm: str, ckpt: Path, task: str, seed: int, chunks: int, execute_steps: int,
    device: str, out_root: Path, video: bool, hand_mode: str, extra: list[str],
) -> dict:
    out_dir = out_root / arm / f"{task}_s{seed}"
    meta_path = out_dir / "meta.json"
    calib = calib_npz(task)
    if meta_path.exists():
        # Resumable: a finished episode is reused, but only if it really is the
        # same arm/checkpoint (a stale directory must not silently poison the
        # comparison).
        meta = json.loads(meta_path.read_text())
        prev = meta.get("ckpt")
        if prev and Path(prev).resolve() != ckpt.resolve():
            raise SystemExit(
                f"{meta_path} was produced with ckpt={prev}, but this run asks for "
                f"{ckpt}. Delete the stale episode or use a fresh --out."
            )
        meta.setdefault("arm", arm)
        meta.setdefault("task", task)
        meta.setdefault("seed", seed)
        meta.setdefault("ckpt", str(ckpt))
        meta["hand_mode"] = hand_mode
        meta["calib_npz"] = str(calib) if calib else None
        meta["chunks_arg"] = chunks
        meta["execute_steps_arg"] = execute_steps
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUNNER),
        "--task", task, "--ckpt", str(ckpt), "--chunks", str(chunks),
        "--execute-steps", str(execute_steps), "--device", "cuda:0",
        "--seed", str(seed), "--hand-mode", hand_mode, "--log-hands",
        "--scenes-dir", str(CLOSED_LOOP / "assets" / "scenes"),
        "--release-dir", str(CLOSED_LOOP / "assets" / "release"),
        "--repro-dir", str(CLOSED_LOOP / "assets" / "repro"),
        "--out-dir", str(out_dir),
    ]
    if calib is not None:
        cmd += ["--hand-calib-npz", str(calib)]
    if video:
        cmd += ["--video", str(out_root / arm / "videos" / f"{task}_s{seed}.mp4")]
    cmd += extra

    t0 = time.time()
    log = out_dir / "run.log"
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=build_env(device), cwd=str(ROOT))
    wall = round(time.time() - t0, 1)
    if rc != 0 or not meta_path.exists():
        return {
            "task": task, "seed": seed, "arm": arm, "failed_to_run": True, "rc": rc,
            "wall_seconds": wall, "log": str(log), "ckpt": str(ckpt),
            "hand_mode": hand_mode, "calib_npz": str(calib) if calib else None,
            "chunks_arg": chunks, "execute_steps_arg": execute_steps,
        }
    meta = json.loads(meta_path.read_text())
    meta.update({
        "arm": arm, "task": task, "seed": seed, "ckpt": str(ckpt),
        "wall_seconds": wall, "proc_wall_seconds": wall,
        "hand_mode": hand_mode, "calib_npz": str(calib) if calib else None,
        "calib_sha256": sha256_file(calib) if calib else None,
        "chunks_arg": chunks, "execute_steps_arg": execute_steps,
        "runner_sha256": sha256_file(RUNNER),
        "log": str(log),
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return round(s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2, 4)


def mcnemar_exact(b01: int, b10: int) -> float:
    """Two-sided exact McNemar test on paired binary outcomes."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, 2.0 * tail), 4)


def _numeric_details(rows: list[dict]) -> dict:
    """Aggregate the continuous quantities the predicates already log."""
    keys: set[str] = set()
    for r in rows:
        keys.update(k for k, v in (r.get("success") or {}).items()
                    if k != "success" and _finite(v))
    out: dict = {}
    for key in sorted(keys):
        vals = [float((r.get("success") or {})[key]) for r in rows
                if _finite((r.get("success") or {}).get(key))]
        if vals:
            out[key] = {"n": len(vals), "mean": _mean(vals), "median": _median(vals),
                        "min": round(min(vals), 4), "max": round(max(vals), 4)}
    return out


def _arm_task_stats(sel: list[dict]) -> dict:
    ok = [r for r in sel if not r.get("failed_to_run")]
    failed = len(sel) - len(ok)
    k = sum(1 for r in ok if (r.get("success") or {}).get("success"))
    n = len(ok)
    lo, hi = wilson(k, n)
    fell = sum(1 for r in ok if r.get("fell"))
    walls = [float(r["wall_seconds"]) for r in ok if _finite(r.get("wall_seconds"))]
    stages: dict[str, int] = {}
    for r in ok:
        stage = (r.get("success") or {}).get("stage")
        if stage:
            key = str(stage)[:120]
            stages[key] = stages.get(key, 0) + 1
    cont: dict[str, dict] = {}
    for key in ("dx", "dy", "z_rel", "drift"):
        vals = [float((r.get("success") or {})[key]) for r in ok
                if _finite((r.get("success") or {}).get(key))]
        if vals:
            cont[key] = {"n": len(vals), "mean": _mean(vals), "median": _median(vals),
                         "min": round(min(vals), 4), "max": round(max(vals), 4)}
    success_times = [float(r["success"]["success_time_s"]) for r in ok
                     if _finite((r.get("success") or {}).get("success_time_s"))]
    return {
        "success": k, "episodes": n, "rate": round(k / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "failed_to_run": failed,
        "fell": fell, "not_fell": n - fell,
        "not_fell_rate": round((n - fell) / n, 4) if n else 0.0,
        "wall_seconds_mean": _mean(walls),
        "wall_seconds_median": _median(walls),
        "wall_seconds_total": round(sum(walls), 1),
        "stages": stages,
        "continuous": cont,
        "success_time_s_mean": _mean(success_times),
        "success_time_s_median": _median(success_times),
        "predicate_details": _numeric_details(ok),
    }


def _paired_stats(rows: list[dict], arms: list[str]) -> dict:
    """Pair episodes by (task, seed): same seed => same initial noise, so the
    comparison is paired, not two independent binomials."""
    by: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        if "arm" not in r or "task" not in r or r.get("failed_to_run"):
            continue
        by[(r["arm"], r["task"], r["seed"])] = r
    tasks = sorted({r["task"] for r in rows if "task" in r})
    out: dict = {}
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            per_task = {}
            tot = {"pairs": 0, "a_only": 0, "b_only": 0, "both": 0, "neither": 0}
            for task in tasks:
                seeds = sorted({s for (arm, t, s) in by if arm == a and t == task}
                               & {s for (arm, t, s) in by if arm == b and t == task})
                a_only = b_only = both = neither = 0
                for seed in seeds:
                    ra, rb = by[(a, task, seed)], by[(b, task, seed)]
                    sa = bool((ra.get("success") or {}).get("success"))
                    sb = bool((rb.get("success") or {}).get("success"))
                    if sa and sb:
                        both += 1
                    elif sa:
                        a_only += 1
                    elif sb:
                        b_only += 1
                    else:
                        neither += 1
                entry = {
                    "pairs": len(seeds), "a_only": a_only, "b_only": b_only,
                    "both": both, "neither": neither,
                    "net_b_minus_a": b_only - a_only,
                    "mcnemar_p": mcnemar_exact(a_only, b_only),
                }
                per_task[task] = entry
                for key in tot:
                    tot[key] += entry[key]
            tot["net_b_minus_a"] = tot["b_only"] - tot["a_only"]
            tot["mcnemar_p"] = mcnemar_exact(tot["a_only"], tot["b_only"])
            out[f"{a}_vs_{b}"] = {"per_task": per_task, "overall": tot}
    return out


def summarize(rows: list[dict], arms: list[str], ckpts: dict[str, str] | None = None) -> dict:
    rows = [r for r in rows if "arm" in r and "task" in r]
    tasks = sorted({r["task"] for r in rows})
    out: dict = {"per_arm": {}, "tasks": tasks, "arms": list(arms),
                 "ckpts": ckpts or {}, "generated_at": datetime.now(timezone.utc).isoformat()}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        per_task = {}
        for task in tasks:
            per_task[task] = _arm_task_stats([r for r in arm_rows if r["task"] == task])
        k = sum(v["success"] for v in per_task.values())
        # Failed-to-run episodes are reported, never counted as failures.
        n = sum(v["episodes"] for v in per_task.values())
        failed = sum(v["failed_to_run"] for v in per_task.values())
        lo, hi = wilson(k, n)
        out["per_arm"][arm] = {
            "per_task": per_task, "success": k, "episodes": n, "failed_to_run": failed,
            "rate": round(k / n, 4) if n else 0.0, "ci95": [round(lo, 4), round(hi, 4)],
            "fell": sum(v["fell"] for v in per_task.values()),
            "not_fell": sum(v["not_fell"] for v in per_task.values()),
            "not_fell_rate": round(sum(v["not_fell"] for v in per_task.values()) / n, 4) if n else 0.0,
            "wall_seconds_total": round(sum(v["wall_seconds_total"] for v in per_task.values()), 1),
        }
    out["paired"] = _paired_stats(rows, arms)
    out["total_episodes"] = len(rows)
    out["failed_to_run"] = sum(1 for r in rows if r.get("failed_to_run"))
    # Protocol consistency: both arms must be measured with the same harness knobs.
    protocol = {}
    for r in rows:
        key = (r.get("chunks_arg"), r.get("execute_steps_arg"), r.get("hand_mode"),
               Path(r["calib_npz"]).name if r.get("calib_npz") else None)
        protocol.setdefault(r["arm"], set()).add(key)
    out["protocol_by_arm"] = {arm: sorted(str(v) for v in vals)
                              for arm, vals in protocol.items()}
    out["protocol_consistent"] = len({v for vals in protocol.values() for v in vals}) <= 1
    return out


def markdown(summary: dict, arms: list[str]) -> str:
    tasks = summary.get("tasks") or sorted({t for a in arms for t in summary["per_arm"][a]["per_task"]})
    per = summary["per_arm"]
    lines = ["# A1 — 有无联合微调（闭环成功率）", ""]
    lines += [f"_generated: {summary.get('generated_at', '?')} · "
              f"episodes: {summary.get('total_episodes')} · "
              f"failed_to_run: {summary.get('failed_to_run')} · "
              f"protocol_consistent: {summary.get('protocol_consistent')}_", ""]
    if summary.get("ckpts"):
        lines += ["| arm | checkpoint |", "|---|---|"]
        for arm in arms:
            lines.append(f"| `{arm}` | `{summary['ckpts'].get(arm, '?')}` |")
        lines.append("")

    lines += ["## 1. 成功率（主指标）", "",
              "| task | " + " | ".join(arms) + " |",
              "|---|" + "|".join(["---"] * len(arms)) + "|"]
    for task in tasks:
        cells = []
        for arm in arms:
            v = per[arm]["per_task"][task]
            cells.append(f"{v['success']}/{v['episodes']} ({v['rate'] * 100:.0f}%) "
                         f"[{v['ci95'][0] * 100:.0f}, {v['ci95'][1] * 100:.0f}]")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    cells = []
    for arm in arms:
        v = per[arm]
        cells.append(f"**{v['success']}/{v['episodes']} ({v['rate'] * 100:.0f}%)** "
                     f"[{v['ci95'][0] * 100:.0f}, {v['ci95'][1] * 100:.0f}]")
    lines.append("| **all** | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 2. 未摔倒率", "",
              "| task | " + " | ".join(arms) + " |",
              "|---|" + "|".join(["---"] * len(arms)) + "|"]
    for task in tasks:
        cells = [f"{per[a]['per_task'][task]['not_fell']}/{per[a]['per_task'][task]['episodes']} "
                 f"({per[a]['per_task'][task]['not_fell_rate'] * 100:.0f}%)" for a in arms]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines.append("")

    cont_tasks = [t for t in tasks
                  if all("dx" in per[a]["per_task"][t]["continuous"] for a in arms)]
    if cont_tasks:
        lines += ["## 3. task3/task4 连续量（place 谓词；成功要求 dx/dy 在目标footprint内、",
                  "z_rel∈[-0.02, 0.10]、末 25 tick 高度漂移≤0.01 m）", ""]
        for task in cont_tasks:
            lines += [f"### {task}", "",
                      "| arm | dx mean | dy mean | z_rel mean | drift mean | episodes |",
                      "|---|---|---|---|---|---|"]
            for arm in arms:
                c = per[arm]["per_task"][task]["continuous"]
                lines.append(
                    f"| {arm} | {c['dx']['mean']:.4f} | {c['dy']['mean']:.4f} | "
                    f"{c['z_rel']['mean']:.4f} | "
                    f"{c.get('drift', {}).get('mean', float('nan')):.4f} | "
                    f"{per[arm]['per_task'][task]['episodes']} |")
            lines.append("")

    paired = summary.get("paired") or {}
    if paired:
        lines += ["## 4. 配对检验（同 task/seed 配对；McNemar 精确检验）", ""]
        for name, block in paired.items():
            a, b = name.split("_vs_")
            lines += [f"`{a}` vs `{b}`：b_only = {b} 独赢，a_only = {a} 独赢；"
                      "p 为双侧精确 McNemar。", "",
                      "| task | pairs | a_only | b_only | both | neither | net(b-a) | p |",
                      "|---|---|---|---|---|---|---|---|"]
            for task, v in block["per_task"].items():
                lines.append(f"| {task} | {v['pairs']} | {v['a_only']} | {v['b_only']} | "
                             f"{v['both']} | {v['neither']} | {v['net_b_minus_a']} | {v['mcnemar_p']} |")
            v = block["overall"]
            lines.append(f"| **all** | {v['pairs']} | {v['a_only']} | {v['b_only']} | "
                         f"{v['both']} | {v['neither']} | {v['net_b_minus_a']} | {v['mcnemar_p']} |")
            lines.append("")

    lines += ["## 5. 运行完整性", "",
              "| arm | episodes | failed_to_run | fell | wall_total_s | wall_mean_s |",
              "|---|---|---|---|---|---|"]
    for arm in arms:
        walls = [per[arm]['per_task'][t]['wall_seconds_mean'] for t in tasks]
        walls = [w for w in walls if w is not None]
        lines.append(f"| {arm} | {per[arm]['episodes']} | {per[arm]['failed_to_run']} | "
                     f"{per[arm]['fell']} | {per[arm]['wall_seconds_total']} | "
                     f"{_mean(walls)} |")
    lines += ["", "## 6. 判读与边界", "",
              "- 二进制 episode 的粒度是 1/n；n=40/臂时，单任务 n=10，**两臂差 ≥2 个 episode 才勉强算信号**，"
              "结论以配对 McNemar 的 p 值为准。",
              "- 执行器固定为仓库内 SONIC decoder（`model_decoder.onnx`，两臂同一文件），"
              "因此本实验只消融 VLA 权重，不混入 WBC 差异。",
              "- 这是 **SONIC 4 任务 MuJoCo 代理口径**，不是 HumanoidArena 7 任务 Isaac 口径；"
              "绝对值不可与 arena 榜单比，但两臂内部可比。",
              "- 每 episode 是独立进程，seed 同时控制动作头的初始噪声；同 seed 两臂是配对样本。",
              ""]
    if summary.get("failed_to_run"):
        lines += [f"> ⚠️ 有 {summary['failed_to_run']} 个 episode 未跑成（failed_to_run），"
                  "已被排除在成功率分母之外；请用同命令续跑补齐后再引用本表。", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=CKPT",
                    help="repeatable; NAME is the label used in the report, CKPT a GR00T "
                         "checkpoint dir (see scripts/export_vla_ckpt.py)")
    ap.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3", "task4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--chunks", type=int, default=40,
                    help="one chunk = 2 plans x 20 ticks = 0.8 s; 40 chunks = 32 s")
    ap.add_argument("--execute-steps", type=int, default=20, help="2.5 Hz deployment cadence")
    ap.add_argument("--hand-mode", default="trend",
                    choices=["vla", "trend", "fixed-open", "fixed-grasp"])
    ap.add_argument("--device", default="0", help="value for CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "success_rate")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="recompute summary.json/SUMMARY.md from an existing rows.json")
    ap.add_argument("--retries", type=int, default=1,
                    help="retry an episode that crashed (fresh process, same seed)")
    ap.add_argument("--allow-failures", action="store_true",
                    help="exit 0 even if some episodes failed to run (default: exit 2)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra flags for the runner")
    args = ap.parse_args()

    # Hard policy (2026-10-24): this project permanently uses the HA + IsaacLab
    # chain. The MuJoCo closed loop below is kept only for historical
    # reproduction and must never be used for official ablation numbers.
    if os.environ.get("COMIMIC_ALLOW_MUJOCO_ROUTE", "0") != "1":
        raise SystemExit(
            "[data-policy] MuJoCo closed loop is permanently disabled.\n"
            "  Official closed-loop evaluation: bash scripts/run_a1_ha_isaac.sh\n"
            "  (HA data + IsaacLab/HumanoidArena; see docs/DATA_POLICY_HA_ONLY.md)\n"
            "  Historical reproduction only: COMIMIC_ALLOW_MUJOCO_ROUTE=1 ..."
        )

    arms: list[tuple[str, Path]] = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=CKPT, got {spec!r}")
        name, path = spec.split("=", 1)
        p = Path(path)
        if not (p / "model.safetensors.index.json").exists() and not (p / "config.json").exists():
            raise SystemExit(f"{p} does not look like a GR00T checkpoint dir")
        arms.append((name.strip(), p))
    names = [a for a, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate arm names: {names}")
    ckpts = {name: str(p.resolve()) for name, p in arms}

    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "rows.json"
    if args.report_only:
        if not rows_path.exists():
            raise SystemExit(f"--report-only needs {rows_path}")
        rows = json.loads(rows_path.read_text())
        summary = summarize(rows, names, ckpts)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        (args.out / "SUMMARY.md").write_text(markdown(summary, names))
        print(markdown(summary, names))
        return

    plan = [(a, p, t, s) for a, p in arms for t in args.tasks for s in args.seeds]
    est = len(plan) * 70 / 60.0
    print(f"[plan] {len(plan)} episodes ({len(names)} arms x {len(args.tasks)} tasks x "
          f"{len(args.seeds)} seeds), ~{est:.0f} min at ~70 s/episode")

    manifest = {
        "kind": "A1 joint-finetune ablation (SONIC 4-task MuJoCo closed loop)",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "runner": {"path": str(RUNNER.resolve()), "sha256": sha256_file(RUNNER)},
        "decoder_onnx": {
            "path": str((CLOSED_LOOP / "assets" / "release" / "model_decoder.onnx").resolve()),
            "sha256": sha256_file(CLOSED_LOOP / "assets" / "release" / "model_decoder.onnx"),
        },
        "scenes": {
            task: {
                "scene": scene,
                "sha256": sha256_file(CLOSED_LOOP / "assets" / "scenes" / scene)
                if (CLOSED_LOOP / "assets" / "scenes" / scene).exists() else None,
            }
            for task, scene in {
                "task1": "scene_43dof_two_table.xml",
                "task2": "scene_43dof.xml",
                "task3": "scene_43dof_handover.xml",
                "task4": "scene_43dof_bend_and_pick.xml",
            }.items()
        },
        "calibration": {
            task: ({"path": str(calib_npz(task)), "sha256": sha256_file(calib_npz(task))}
                   if calib_npz(task) else None)
            for task in args.tasks
        },
        "protocol": {
            "tasks": args.tasks, "seeds": args.seeds, "chunks": args.chunks,
            "execute_steps": args.execute_steps, "hand_mode": args.hand_mode,
            "device": args.device, "extra": args.extra,
            "ckpts": ckpts,
            "checkpoints": {name: ckpt_manifest(p) for name, p in arms},
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        for arm, ckpt, task, seed in plan:
            print(f"  {arm:>12} {task} seed={seed} ckpt={ckpt} calib={calib_npz(task)}")
        print(f"[dry-run] manifest -> {args.out / 'manifest.json'}")
        return

    rows: list[dict] = []
    for i, (arm, ckpt, task, seed) in enumerate(plan, 1):
        meta = None
        for attempt in range(args.retries + 1):
            meta = episode(
                arm=arm, ckpt=ckpt, task=task, seed=seed, chunks=args.chunks,
                execute_steps=args.execute_steps, device=args.device, out_root=args.out,
                video=args.video, hand_mode=args.hand_mode, extra=list(args.extra),
            )
            if not meta.get("failed_to_run"):
                break
            print(f"[retry {attempt + 1}/{args.retries}] {arm} {task} seed={seed} "
                  f"rc={meta.get('rc')} log={meta.get('log')}", flush=True)
        assert meta is not None
        rows.append(meta)
        ok = (meta.get("success") or {}).get("success")
        print(f"[{i}/{len(plan)}] {arm} {task} seed={seed} success={ok} "
              f"fell={meta.get('fell')} wall={meta.get('wall_seconds')}s", flush=True)
        rows_path.write_text(json.dumps(rows, indent=2, default=str))
        # write the report after every episode so a crash never loses the curve
        summary = summarize(rows, names, ckpts)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        (args.out / "SUMMARY.md").write_text(markdown(summary, names))

    summary = summarize(rows, names, ckpts)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["results"] = {
        "total_episodes": summary["total_episodes"],
        "failed_to_run": summary["failed_to_run"],
        "per_arm": {arm: {"success": summary["per_arm"][arm]["success"],
                          "episodes": summary["per_arm"][arm]["episodes"],
                          "rate": summary["per_arm"][arm]["rate"]} for arm in names},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "SUMMARY.md").write_text(markdown(summary, names))
    print(markdown(summary, names))
    if summary["failed_to_run"] and not args.allow_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
