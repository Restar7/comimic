#!/usr/bin/env python3
"""P0b：构建 **执行参考库**（executed reference library）``runs/isaac_motion/ha_cmd_exec``。

为什么需要（docs/PLAN_REFSTART_WM_PERTURB_20261102.md §1）：

* 旧库 ``ha_cmd`` 是**混血**：q/qd 来自**命令流** ``encoder_motion_joint_{pos,vel}_hist``，
  root 却来自**执行轨迹** ``robot_root_position/orientation``；两者差 mean|Δq|≈0.39 rad、
  lag ≈ +8 帧（0.16 s）。拿命令 q 当跟踪目标 = 让机器人追一条物理上追不上的轨迹，
  WBC 只能折中 → 参考回放 0.32 m（xy）、1327 步摔。
* ``data/ha_raw/<TASK>/<src>/<clip>.npz`` 里**同时**有执行 q / 执行 qd / 执行根 / 根速度，
  所以新库把整个 canonical 71 维全部换成"执行"量，token（输入条件）仍由命令流编码，
  训练目标变成 **(执行 q, 执行根)**。

字段映射（canonical 布局 ``p(3) quat_wxyz(4) v(3) w(3) q(29) qd(29)`` = 71）::

    p     <- robot_root_position             (world)
    quat  <- robot_root_orientation          (wxyz, world)
    q     <- robot_qpos_before_decimation    (STANDARD_29 / MJCF 序；已用 FK 逐位验证)
    v_w   <- robot_root_lin_vel_world        (world)
    w_b   <- robot_root_ang_vel_world        (world)
    qd    <- robot_qvel_before_decimation    (STANDARD_29)

**硬约束（否则图像 / 指令 / token sidecar 全错位）**：clip 名、每条 clip 的帧数、
``starts/lengths`` 排布必须和 ``ha_cmd`` 库**逐项一致**；只换 ref 里的数值，不动索引结构。
脚本默认开 ``--match-lib``（自动找 ``runs/isaac_motion/ha_cmd/packed.pt``）做逐项断言。

``--x20-repeats N`` 只额外产出 ``packed_x20.pt``（每条 clip 重复 N 次、名字加
``__HA_CMDr0..r{N-1}``，与 ``bones_seed_plus_ha_cmd_x20.pt`` 的 HA 段同构），
供训练库构建使用；``packed.pt`` 永远是索引结构不变的 699 条。

用法::

    python3 scripts/build_ha_exec_lib.py --limit 3 --out /tmp/ha_cmd_exec_smoke   # 冒烟
    python3 scripts/build_ha_exec_lib.py --npz-dir data/ha_raw \\
        --out runs/isaac_motion/ha_cmd_exec --x20-repeats 20                      # 全量 699
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

REF_LAYOUT = "canonical"          # [p(3), quat_wxyz(4), v_w(3), w_b(3), q(29), qd(29)]
NUM_JOINTS = 29
REF_DIM = 13 + 2 * NUM_JOINTS     # 71


def clip_files(npz_dir: Path) -> list[Path]:
    return sorted(npz_dir.glob("*/*/*.npz"))


def clip_name(p: Path) -> str:
    """``<TASK>__<src>__ep<timestamp>`` —— 与 build_ha_command_lib.py / ha_cmd 库同名。"""
    ep = p.stem.rsplit("_", 1)[-1]
    return f"{p.parent.parent.name}__{p.parent.name}__ep{int(ep):06d}"


def read_executed(path: Path) -> tuple[np.ndarray, int]:
    """原始 npz -> (ref (T,71) float32, T)。只读需要的成员（npz 里还有 1 GB 级 depth）。"""
    z = np.load(path, allow_pickle=True)
    need = (
        "robot_root_position",
        "robot_root_orientation",
        "robot_root_lin_vel_world",
        "robot_root_ang_vel_world",
        "robot_qpos_before_decimation",
        "robot_qvel_before_decimation",
    )
    missing = [k for k in need if k not in z.files]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    p = np.asarray(z["robot_root_position"], dtype=np.float32)
    quat = np.asarray(z["robot_root_orientation"], dtype=np.float32)      # wxyz
    v = np.asarray(z["robot_root_lin_vel_world"], dtype=np.float32)
    w = np.asarray(z["robot_root_ang_vel_world"], dtype=np.float32)
    q = np.asarray(z["robot_qpos_before_decimation"], dtype=np.float32)   # STANDARD_29
    qd = np.asarray(z["robot_qvel_before_decimation"], dtype=np.float32)
    T = p.shape[0]
    for name, arr in (("quat", quat), ("v_w", v), ("w_b", w), ("q", q), ("qd", qd)):
        if arr.shape[0] != T:
            raise ValueError(f"{path}: {name} has {arr.shape[0]} frames, root has {T}")
    if q.shape[1] != NUM_JOINTS or qd.shape[1] != NUM_JOINTS:
        raise ValueError(f"{path}: q/qd width {q.shape[1]}/{qd.shape[1]} != {NUM_JOINTS}")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError(f"{path}: non-finite values in root/joint arrays")
    ref = np.concatenate([p, quat, v, w, q, qd], axis=1)
    if ref.shape[1] != REF_DIM:
        raise ValueError(f"{path}: ref dim {ref.shape[1]} != {REF_DIM}")
    return ref, T


def _structure(names: list[str], lengths: list[int]) -> dict:
    starts = (np.cumsum([0] + lengths[:-1])).tolist()
    return {"names": names, "starts": [int(s) for s in starts],
            "lengths": [int(x) for x in lengths]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, default=ROOT / "data/ha_raw")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd_exec")
    ap.add_argument("--x20-repeats", type=int, default=1,
                    help=">1 时额外产出 packed_x20.pt（训练库用；packed.pt 结构不变）")
    ap.add_argument("--match-lib", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                    help="索引结构参照库；传空字符串关闭断言")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    a = ap.parse_args()

    files = clip_files(a.npz_dir)
    if a.limit:
        files = files[: a.limit]
    if not files:
        raise SystemExit(f"no npz under {a.npz_dir}")

    names: list[str] = []
    refs: list[torch.Tensor] = []
    lengths: list[int] = []
    q_stats: list[dict] = []
    t0 = time.time()
    dtype = getattr(torch, a.dtype)
    for i, p in enumerate(files):
        name = clip_name(p)
        ref, T = read_executed(p)
        names.append(name)
        lengths.append(int(T))
        refs.append(torch.from_numpy(ref).to(dtype))
        q_stats.append({"name": name, "frames": int(T),
                        "q_abs_mean": float(np.abs(ref[:, 13:42]).mean()),
                        "root_z_mean": float(ref[:, 2].mean())})
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            print(f"[ha-exec] {i + 1}/{len(files)} clips, {sum(lengths)} frames, "
                  f"{time.time() - t0:.1f}s", flush=True)

    ref = torch.cat(refs, dim=0)
    struct = _structure(names, lengths)

    # ---- 硬约束：与 ha_cmd 逐项一致 -------------------------------------------
    match_info = None
    if a.match_lib and Path(a.match_lib).is_file():
        d = torch.load(a.match_lib, map_location="cpu", weights_only=False)
        ref_names = list(d["names"])
        ref_starts = [int(x) for x in d["starts"]]
        ref_lengths = [int(x) for x in d["lengths"]]
        if a.limit:
            print(f"[ha-exec] --limit={a.limit}：跳过与 {a.match_lib.name} 的逐项断言")
        else:
            if names != ref_names:
                bad = next((k for k, (x, y) in enumerate(zip(names, ref_names)) if x != y), None)
                raise SystemExit(f"[ha-exec] FATAL: clip names differ from ha_cmd at {bad}: "
                                 f"{names[bad] if bad is not None else 'len'} vs "
                                 f"{ref_names[bad] if bad is not None else len(ref_names)}")
            if lengths != ref_lengths:
                bad = next(k for k, (x, y) in enumerate(zip(lengths, ref_lengths)) if x != y)
                raise SystemExit(f"[ha-exec] FATAL: lengths differ at clip {bad} "
                                 f"({names[bad]}): {lengths[bad]} vs {ref_lengths[bad]}")
            if struct["starts"] != ref_starts:
                raise SystemExit("[ha-exec] FATAL: starts differ from ha_cmd")
            # 与旧库对比：q 差（预期 ~0.39 rad）、root 是否相同（预期相同）
            old = d["ref"].float()
            dq = (ref.float()[:, 13:42] - old[:, 13:42]).abs()
            dqd = (ref.float()[:, 42:71] - old[:, 42:71]).abs()
            droot = (ref.float()[:, 0:7] - old[:, 0:7]).abs()
            match_info = {
                "ref_lib": str(a.match_lib),
                "index_structure_identical": True,
                "q_abs_mean_vs_ha_cmd": float(dq.mean()),
                "q_abs_max_vs_ha_cmd": float(dq.max()),
                "qd_abs_mean_vs_ha_cmd": float(dqd.mean()),
                "root_abs_max_vs_ha_cmd": float(droot.max()),
            }
            print(f"[ha-exec] vs ha_cmd: mean|Δq|={match_info['q_abs_mean_vs_ha_cmd']:.4f} rad, "
                  f"max={match_info['q_abs_max_vs_ha_cmd']:.3f}, "
                  f"root max|Δ|={match_info['root_abs_max_vs_ha_cmd']:.2e} (预期 0)")

    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / "packed.pt"
    packed = {"ref": ref, "names": names, "starts": struct["starts"], "lengths": struct["lengths"]}
    torch.save(packed, out)
    print(f"[ha-exec] wrote {out} clips={len(names)} frames={int(ref.shape[0])} "
          f"({out.stat().st_size / 1e6:.1f} MB)")

    # ---- 可选：x20 展开版（训练库构建用）--------------------------------------
    x20_info = None
    if a.x20_repeats and a.x20_repeats > 1:
        r = int(a.x20_repeats)
        x_names, x_refs, x_lengths = [], [], []
        for k in range(r):
            for j, name in enumerate(names):
                x_names.append(f"{name}__HA_CMDr{k}")
                x_refs.append(ref[struct["starts"][j]: struct["starts"][j] + lengths[j]])
                x_lengths.append(lengths[j])
        x_struct = _structure(x_names, x_lengths)
        x_out = a.out / "packed_x20.pt"
        torch.save({"ref": torch.cat(x_refs), "names": x_names,
                    "starts": x_struct["starts"], "lengths": x_struct["lengths"]}, x_out)
        x20_info = {"repeats": r, "path": str(x_out), "num_clips": len(x_names),
                    "num_frames": int(sum(x_lengths))}
        print(f"[ha-exec] wrote {x_out} clips={len(x_names)} frames={sum(x_lengths)}")

    meta = {
        "ref_layout": REF_LAYOUT,
        "src_fps": 50.0,
        "control_dt": 0.02,
        "dtype": a.dtype,
        "num_clips": len(names),
        "num_frames": int(ref.shape[0]),
        "source": "data/ha_raw robot_qpos/qvel_before_decimation + robot_root_* (EXECUTED trajectory)",
        "note": ("q/qd/root/vel all from the measured execution; token stays the command-stream "
                 "encoder label. index structure mirrors runs/isaac_motion/ha_cmd/packed.pt"),
        "npz_dir": str(a.npz_dir),
        "match_ha_cmd": match_info,
        "x20": x20_info,
        "q_abs_mean_over_lib": float(np.mean([s["q_abs_mean"] for s in q_stats])),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in ("num_clips", "num_frames", "dtype")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
