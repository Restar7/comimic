#!/usr/bin/env python3
"""P0b G0 验收：检查 ``ha_cmd_exec/packed.pt`` 是否自洽（FK / 根高 / 索引结构 / 与 ha_cmd 对比）。

G0 门（docs/PLAN_REFSTART_WM_PERTURB_20261102.md §4）::

    FK(执行 q, 执行根) vs body_pos_w 脚位置      max < 3 cm
    根高度（pelvis z）与执行根                    < 2 cm
    帧数 / clip 数                               699 clips，帧数与原始 npz 一致
    与 ha_cmd 对比                                q 差 ~0.39 rad、根相同

``body_pos_w`` 来自 ``data/npz/ha_tokens/<TASK>__<src>__epNNNNNN.npz``（该文件就是用同一个
``G1Kinematics`` 对 (root, joint_pos) 做的 FK，见 scripts/prepare_ha_tokens_npz.py）——
本检查因此验证的是**打包后的库行**（fp16）能否复现录制时的身体位姿，覆盖"关节顺序 / 根
frame / 帧对齐 / fp16 截断"这几类最致命的错位。

用法::

    python3 scripts/check_exec_lib.py --packed runs/isaac_motion/ha_cmd_exec/packed.pt \\
        --report runs/isaac_motion/ha_cmd_exec/report.json
    python3 scripts/check_exec_lib.py --packed ... --report ... --device cuda:0 --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GATE_FOOT_M = 0.03
GATE_ROOT_Z_M = 0.02
NUM_JOINTS = 29
REF_DIM = 13 + 2 * NUM_JOINTS
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
PELVIS = "pelvis"


def _fk_kinematics(device: str):
    from diff_vla_wbc.compat import ensure_supermimic_importable, load_g1_model

    ensure_supermimic_importable()
    from supermimic.kinematics.g1_fk import G1Kinematics

    return G1Kinematics(load_g1_model(), device=device)


def _body_npz_map(token_npz_dir: Path) -> dict[str, Path]:
    """clip 名 -> 含 body_pos_w 的 npz（按 npz 里的 ``source`` 字段权威映射）。"""
    out: dict[str, Path] = {}
    if not token_npz_dir.is_dir():
        return out
    for f in sorted(token_npz_dir.glob("*.npz")):
        try:
            src = str(np.load(f, allow_pickle=True)["source"])
        except Exception:  # noqa: BLE001
            continue
        sp = Path(src)
        try:
            name = f"{sp.parent.parent.name}__{sp.parent.name}__ep{int(sp.stem.rsplit('_', 1)[-1]):06d}"
        except ValueError:
            continue
        out[name] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd_exec/report.json")
    ap.add_argument("--ref-lib", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--token-npz-dir", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fk-batch", type=int, default=2048)
    a = ap.parse_args()

    t_start = time.time()
    d = torch.load(a.packed, map_location="cpu", weights_only=False)
    ref, names = d["ref"], list(d["names"])
    starts = [int(x) for x in d["starts"]]
    lengths = [int(x) for x in d["lengths"]]
    n_frames = int(ref.shape[0])

    print(f"[check] {a.packed}: ref={tuple(ref.shape)} {ref.dtype} clips={len(names)} "
          f"frames={n_frames} ({n_frames / 50.0 / 3600.0:.2f} h)")

    # ------------------------------------------------------------------ 1. 结构
    problems: list[str] = []
    if ref.dim() != 2 or ref.shape[1] != REF_DIM:
        problems.append(f"ref shape {tuple(ref.shape)} != (N,{REF_DIM})")
    if starts and starts[0] != 0:
        problems.append("starts[0] != 0")
    if any(b <= a0 for a0, b in zip(starts, starts[1:])):
        problems.append("starts not strictly increasing")
    if starts[-1] + lengths[-1] != n_frames:
        problems.append(f"index covers {starts[-1] + lengths[-1]} rows, ref has {n_frames}")
    if not bool(torch.isfinite(ref.float()).all()):
        problems.append("ref contains NaN/inf")
    quat = ref[:, 3:7].float()
    qnorm_dev = float((quat.norm(dim=-1) - 1).abs().max())
    if qnorm_dev > 5e-3:
        problems.append(f"quaternion norm deviates by {qnorm_dev:.2e}")

    ref_cmp = None
    if a.ref_lib and Path(a.ref_lib).is_file():
        r = torch.load(a.ref_lib, map_location="cpu", weights_only=False)
        same_names = list(r["names"]) == names
        same_lengths = [int(x) for x in r["lengths"]] == lengths
        same_starts = [int(x) for x in r["starts"]] == starts
        dq = (ref.float()[:, 13:42] - r["ref"].float()[:, 13:42]).abs()
        droot = (ref.float()[:, 0:7] - r["ref"].float()[:, 0:7]).abs()
        ref_cmp = {
            "ref_lib": str(a.ref_lib),
            "index_structure_identical": bool(same_names and same_lengths and same_starts),
            "names_identical": same_names,
            "lengths_identical": same_lengths,
            "starts_identical": same_starts,
            "q_abs_mean_vs_ha_cmd": float(dq.mean()),
            "q_abs_median_vs_ha_cmd": float(dq.median()),
            "q_abs_max_vs_ha_cmd": float(dq.max()),
            "root_abs_max_vs_ha_cmd": float(droot.max()),
        }
        if not (same_names and same_lengths and same_starts):
            problems.append("index structure differs from ha_cmd (HARD CONSTRAINT violated)")
        print(f"[check] vs ha_cmd: mean|Δq|={ref_cmp['q_abs_mean_vs_ha_cmd']:.4f} rad "
              f"(median {ref_cmp['q_abs_median_vs_ha_cmd']:.4f}), "
              f"root max|Δ|={ref_cmp['root_abs_max_vs_ha_cmd']:.2e}, "
              f"structure_identical={ref_cmp['index_structure_identical']}")

    # ------------------------------------------------------------------ 2. FK
    npz_map = _body_npz_map(a.token_npz_dir)
    print(f"[check] body_pos_w npz index: {len(npz_map)} clips (from {a.token_npz_dir})")
    kin = _fk_kinematics(a.device)
    body_names = list(kin.body_names)
    foot_idx = [body_names.index(b) for b in FEET if b in body_names]
    pel_idx = body_names.index(PELVIS) if PELVIS in body_names else 0
    missing_feet = [b for b in FEET if b not in body_names]
    if missing_feet:
        print(f"[check] WARN: FK model has no {missing_feet}; foot gate skipped")
    if not npz_map:
        print("[check] WARN: no body_pos_w npz found; FK gate skipped")
    if a.limit:
        names_sel = names[: a.limit]
    else:
        names_sel = names
    idx_of = {n: i for i, n in enumerate(names)}

    rows: list[dict] = []
    dev = torch.device(a.device)
    t_fk = time.time()
    for k, name in enumerate(names_sel):
        i = idx_of[name]
        s, T = starts[i], lengths[i]
        row = {"name": name, "frames": T, "clip_index": i}
        npz_path = npz_map.get(name)
        if npz_path is None:
            row["fk"] = None
            rows.append(row)
            continue
        with np.load(npz_path, allow_pickle=True) as z:
            bpos = np.asarray(z["body_pos_w"], dtype=np.float32)
            if bpos.shape[0] != T:
                problems.append(f"{name}: body_pos_w has {bpos.shape[0]} frames, library {T}")
                T = min(T, bpos.shape[0])
        lib = ref[s: s + T].float()
        qpos = torch.cat([lib[:, 0:7], lib[:, 13:42]], dim=1).to(dev)
        qvel = torch.zeros((T, 35), dtype=torch.float32, device=dev)
        tgt = torch.as_tensor(bpos[:T], dtype=torch.float32, device=dev)
        pos_chunks = []
        with torch.no_grad():
            for b in range(0, T, a.fk_batch):
                pos_chunks.append(kin(qpos[b: b + a.fk_batch], qvel[b: b + a.fk_batch])["pos"])
        fk_pos = torch.cat(pos_chunks, dim=0).cpu()
        foot_err = (float((fk_pos[:, foot_idx] - tgt[:, foot_idx]).norm(dim=-1).max())
                    if foot_idx else float("nan"))
        pel_err = (float((fk_pos[:, pel_idx, 2] - tgt[:, pel_idx, 2]).abs().max())
                   if pel_idx < tgt.shape[1] else float("nan"))
        # 库里的 root（p）与 FK 出来的 pelvis 必须是同一个点：body_pos_w 的 pelvis 行就是录制时的
        # robot_root_position（prepare_ha_tokens_npz.py 用同一个 FK 生成），所以这里用 pelvis 行当 root 参照。
        pel_vs_root = float((fk_pos[:, pel_idx] - tgt[:, pel_idx]).abs().max())
        row["fk"] = {
            "foot_max_err_m": foot_err,
            "pelvis_z_max_err_m": pel_err,
            "pelvis_vs_root_max_err_m": pel_vs_root,
        }
        rows.append(row)
        if (k + 1) % 100 == 0 or k + 1 == len(names_sel):
            done = sum(r["frames"] for r in rows)
            print(f"[check] FK {k + 1}/{len(names_sel)} clips, {done} frames, "
                  f"{time.time() - t_fk:.1f}s", flush=True)

    fk_rows = [r for r in rows if r.get("fk")]
    foot_vals = [r["fk"]["foot_max_err_m"] for r in fk_rows]
    pel_vals = [r["fk"]["pelvis_z_max_err_m"] for r in fk_rows]
    root_vals = [r["fk"]["pelvis_vs_root_max_err_m"] for r in fk_rows]
    sum_fk = {
        "clips_checked": len(fk_rows),
        "foot_max_err_m": max(foot_vals) if foot_vals else None,
        "foot_p99_err_m": float(np.percentile(foot_vals, 99)) if foot_vals else None,
        "foot_mean_err_m": float(np.mean(foot_vals)) if foot_vals else None,
        "pelvis_z_max_err_m": max(pel_vals) if pel_vals else None,
        "pelvis_vs_root_max_err_m": max(root_vals) if root_vals else None,
        "worst_clips": sorted(fk_rows, key=lambda r: -r["fk"]["foot_max_err_m"])[:5],
    }

    # ------------------------------------------------------------------ 3. 帧数
    frame_ok = True
    if npz_map:
        bad = 0
        for i, name in enumerate(names):
            p = npz_map.get(name)
            if p is None:
                continue
            with np.load(p, allow_pickle=True) as z:
                n = int(z["num_frames"]) if "num_frames" in z.files else z["joint_pos"].shape[0]
            if n != lengths[i]:
                bad += 1
                if bad <= 5:
                    problems.append(f"{name}: library {lengths[i]} frames, npz {n}")
        frame_ok = bad == 0

    # ------------------------------------------------------------------ 4. 判门
    gates = {
        "index_structure_identical": None if ref_cmp is None else ref_cmp["index_structure_identical"],
        "clip_count_699": len(names) == 699,
        "frames_match_npz": frame_ok,
        "foot_fk_max_lt_3cm": None if not foot_vals else (max(foot_vals) < GATE_FOOT_M),
        "pelvis_z_max_lt_2cm": None if not pel_vals else (max(pel_vals) < GATE_ROOT_Z_M),
        "quat_norm_ok": qnorm_dev <= 5e-3,
        "no_nan_inf": bool(torch.isfinite(ref.float()).all()),
    }
    hard_fail = [k for k, v in gates.items() if v is False]
    verdict = "PASS" if not hard_fail else "FAIL"

    report = {
        "packed": str(a.packed),
        "ref_lib": str(a.ref_lib) if a.ref_lib else None,
        "dtype": str(ref.dtype),
        "num_clips": len(names),
        "num_frames": n_frames,
        "hours_50hz": round(n_frames / 50.0 / 3600.0, 3),
        "quat_norm_max_deviation": qnorm_dev,
        "compare_to_ha_cmd": ref_cmp,
        "fk": sum_fk,
        "gates": gates,
        "hard_failures": hard_fail,
        "problems": problems,
        "verdict": verdict,
        "gate_thresholds": {"foot_max_err_m": GATE_FOOT_M, "pelvis_z_max_err_m": GATE_ROOT_Z_M,
                            "expected_clips": 699},
        "fk_model": "G1Kinematics(MJCF/STANDARD_29) vs data/npz/ha_tokens body_pos_w",
        "clips": rows,
        "elapsed_s": round(time.time() - t_start, 1),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[check] report -> {a.report}")
    print(f"[check] FK foot max = {sum_fk['foot_max_err_m']} m, pelvis z max = "
          f"{sum_fk['pelvis_z_max_err_m']} m, pelvis-vs-root = "
          f"{sum_fk['pelvis_vs_root_max_err_m']} m")
    print(f"[check] gates: {json.dumps(gates, ensure_ascii=False)}")
    print(f"[check] VERDICT = {verdict}" + (f" (fail: {hard_fail})" if hard_fail else ""))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
