#!/usr/bin/env python3
"""从原始 HA npz 的**物体真值**算任务奖励 / 分级成功（2026-11-02）。

用途：
  1. G2：验证 `ha_cmd_exec` + 新 WBC 的**参考本身**是不是成功（参考成功率 > 0）；
  2. A' 里存一份任务奖励（只用于评测，不进 WM 训练）；
  3. P5 的分级成功（reach / grasp / lift / place）。

数据字段（`data/ha_raw/<TASK>/<src>/<clip>.npz`）：
  env_obj_hammer_position (T,3) / env_obj_basket_position (T,3)
  episode_init_env_obj_hammer_position / _basket_position (3,)
  body_pos_w (T,30,3) + body_names（手腕用于 reach/grasp 代理）
  robot_qpos_before_decimation / robot_root_position / robot_root_orientation

判据（可调）：
  reach  = 任一腕–锤子距离 < --reach
  lift   = 锤子 z > 初始 z + --lift
  grasp  = reach 且 lift
  place  = ||锤子xy − 篮子xy|| < --r-xy 且 |锤子z − 篮子z| < --dz
  success= place（任一步） ⇒ 这一集成功

用法：
  python3 scripts/reward_from_objects.py --npz-dir data/ha_raw \
      --packed runs/isaac_motion/ha_cmd_exec/packed.pt \
      --out runs/wm_real/exp_object_metrics.json --scan
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _wrist_indices(names: list[str]) -> list[int]:
    out = []
    for i, n in enumerate(names):
        if "wrist" in n and ("left" in n or "right" in n):
            out.append(i)
    return out


def _metrics(z, *, r_xy: float, dz: float, lift: float, reach: float, with_steps: bool = False):
    # 非 DoubleDesk 任务没有 hammer/basket 字段 → 跳过（返回 None）
    if "env_obj_hammer_position" not in z.files or "env_obj_basket_position" not in z.files:
        return None
    ham = np.asarray(z["env_obj_hammer_position"], dtype=np.float64)      # (T,3)
    bas = np.asarray(z["env_obj_basket_position"], dtype=np.float64)      # (T,3)
    _h0 = (z["episode_init_env_obj_hammer_position"]
           if "episode_init_env_obj_hammer_position" in z.files else ham[0])
    ham0 = np.asarray(_h0, dtype=np.float64)
    d_xy = np.linalg.norm(ham[:, :2] - bas[:, :2], axis=1)
    d_z = ham[:, 2] - bas[:, 2]
    place = (d_xy < r_xy) & (np.abs(d_z) < dz)
    lift = ham[:, 2] > (ham0[2] + lift)

    wrist_d = np.full(len(ham), np.inf)
    if "body_pos_w" in z and "body_names" in z:
        names = [str(x) for x in z["body_names"]]
        idx = _wrist_indices(names)
        if idx:
            bp = np.asarray(z["body_pos_w"], dtype=np.float64)            # (T,B,3)
            wrist_d = np.linalg.norm(bp[:, idx, :] - ham[:, None, :], axis=-1).min(axis=1)
    reach_mask = wrist_d < reach
    grasp = reach_mask & lift

    out = {
        "frames": int(len(ham)),
        "success": bool(place.any()),
        "place_steps": int(place.sum()),
        "first_success_frame": int(np.argmax(place)) if place.any() else None,
        "lift_steps": int(lift.sum()),
        "grasp_steps": int(grasp.sum()),
        "reach_steps": int(reach_mask.sum()),
        "min_hammer_basket_xy": float(d_xy.min()),
        "final_hammer_basket_xy": float(d_xy[-1]),
        "max_hammer_z": float(ham[:, 2].max()),
        "init_hammer_z": float(ham0[2]),
        "wrist_min_dist": float(wrist_d.min()) if np.isfinite(wrist_d).any() else None,
    }
    if with_steps:
        out.update({
            # 每步奖励（place=1，lift 给 0.3 的 shaping，可改）
            "_reward": (place.astype(np.float32) + 0.3 * lift.astype(np.float32)).tolist(),
            "_place": place.astype(np.int8).tolist(),
            "_lift": lift.astype(np.int8).tolist(),
            "_reach": reach_mask.astype(np.int8).tolist(),
        })
    return out


def _clip_ts(name: str) -> str | None:
    m = re.search(r"(\d{13,})", name)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, default=ROOT / "data/ha_raw")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/exp_object_metrics.json")
    ap.add_argument("--r-xy", type=float, default=0.15)
    ap.add_argument("--dz", type=float, default=0.25)
    ap.add_argument("--lift", type=float, default=0.03)
    ap.add_argument("--reach", type=float, default=0.20)
    ap.add_argument("--scan", action="store_true", help="扫描整个 npz-dir 并汇总")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--with-steps", action="store_true", help="输出每步 place/lift/reach（单 clip 用；699 clip 会很大）")
    ap.add_argument("--weights-out", type=Path, default=ROOT / "runs/wm_real/exp_start_weights.pt", help="每帧起点权重输出")
    ap.add_argument("--phase-boost", type=float, default=4.0, help="关键段权重 = 1 + boost*phase")
    a = ap.parse_args()

    files = sorted(a.npz_dir.rglob("*.npz")) if a.scan else sorted(a.npz_dir.rglob("*.npz"))
    if a.max_clips > 0:
        files = files[: a.max_clips]
    if not files:
        raise SystemExit(f"no npz under {a.npz_dir}")

    # clip 名（timestamp）→ packed 里的全局行
    starts = None
    if a.packed.exists():
        import torch  # noqa: PLC0415

        p = torch.load(str(a.packed), map_location="cpu", weights_only=False)
        starts = {str(n).split("ep")[-1]: int(s) for n, s in zip(p["names"], p["starts"])}
        _lengths = {str(n).split("ep")[-1]: int(l) for n, l in zip(p["names"], p["lengths"])}
        _names_by_ts = {str(n).split("ep")[-1]: str(n) for n in p["names"]}
        _n_frames = int(p["ref"].shape[0])

    rows = []
    n_success = 0
    n_skipped = 0
    _weights_local: dict[str, np.ndarray] = {}
    for f in files:
        with np.load(f) as z:
            m = _metrics(z, r_xy=a.r_xy, dz=a.dz, lift=a.lift, reach=a.reach, with_steps=True)
            if m is None:
                n_skipped += 1
                continue
            ts = _clip_ts(f.name)
            # 2026-11-02：关键段权重（reach/lift/place 任一为 1）
            _ph = np.maximum.reduce([np.asarray(m["_reach"]), np.asarray(m["_lift"]),
                                     np.asarray(m["_place"])]).astype(np.float32)
            _w = 1.0 + float(a.phase_boost) * _ph
            if len(_w) >= 5:
                _w = np.convolve(_w, np.ones(5, dtype=np.float32) / 5.0, mode="same")
            if ts:
                _weights_local[ts] = _w.astype(np.float32)
            for _k in ("_reward", "_place", "_lift", "_reach"):
                m.pop(_k, None)
            m["clip"] = f.stem
            m["npz"] = str(f)
            m["global_start"] = starts.get(ts) if starts else None
            rows.append(m)
            n_success += int(m["success"])

    # 2026-11-02：每帧起点权重 = 关键段权重 × 任务均衡，写 .pt
    W = np.ones(int(_n_frames), dtype=np.float32)
    _assigned = np.zeros(int(_n_frames), dtype=bool)
    if starts:
        for ts, w in _weights_local.items():
            st = starts.get(ts)
            L = _lengths.get(ts)
            if st is None or not L:
                continue
            W[st:st + L] = w[:L]
            _assigned[st:st + L] = True
        # 任务均衡：7 个任务各自总权重 = N/7
        _task_frames: dict[str, int] = {}
        for n in _names_by_ts.values():
            _t = n.split("__")[0]
            _task_frames[_t] = _task_frames.get(_t, 0) + _lengths[str(n).split("ep")[-1]]
        _nt = max(1, len(_task_frames))
        for n in _names_by_ts.values():
            _t = n.split("__")[0]
            st = starts[str(n).split("ep")[-1]]
            L = _lengths[str(n).split("ep")[-1]]
            if _task_frames.get(_t, 0) > 0:
                W[st:st + L] *= float(_n_frames) / (_nt * _task_frames[_t])
        W = np.maximum(W, 1e-6)
        W = W / float(W.mean())
        a.weights_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": torch.from_numpy(W),
                    "meta": {"sampler": "phase_task", "phase_boost": float(a.phase_boost),
                             "assigned_frames": int(_assigned.sum()), "n_frames": int(_n_frames),
                             "n_tasks": int(len(_task_frames)), "criteria": {"r_xy": a.r_xy, "dz": a.dz,
                             "lift": a.lift, "reach": a.reach}}},
                   str(a.weights_out))
        print(f"[obj] start_weights -> {a.weights_out} "
              f"assigned={int(_assigned.sum())}/{int(_n_frames)} mean={float(W.mean()):.3f} "
              f"max={float(W.max()):.3f}")

    place_steps = np.array([r["place_steps"] for r in rows], dtype=np.float64)
    summer = {
        "n_clips": len(rows),
        "n_success": n_success,
        "success_rate": round(n_success / max(len(rows), 1), 4),
        "n_skipped_no_object": n_skipped,
        "mean_place_steps": float(place_steps.mean()) if len(place_steps) else 0.0,
        "criteria": {"r_xy": a.r_xy, "dz": a.dz, "lift": a.lift, "reach": a.reach},
        "packed": str(a.packed),
        "npz_dir": str(a.npz_dir),
        "rows": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summer, ensure_ascii=False, indent=1))
    print(f"[obj] clips={len(rows)} success={n_success} ({summer['success_rate']:.1%}) "
          f"mean_place_steps={summer['mean_place_steps']:.1f} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
