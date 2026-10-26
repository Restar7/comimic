#!/usr/bin/env python3
"""S0.4 后续 / **S2 触发判定**：标签 token（= `encoder_latent`）能不能在环复现？

背景：`docs/S0_AUDIT_REPORT_20261026.md` §4 实测「标签 `g1_motion_token_64` == 录制
`encoder_latent`」，而端口侧车（`data/port_tokens`，`SonicTokenBuilder` 装配）与标签
MSE=0.0854 / corr=0.057。复核附录 A 因此把 S2（重标端口 + VLA 微调）降级为"仅当 builder
修不出来才执行"。**本脚本把"修不修得出来"变成逐位数字。**

三个实验（每个 clip 都跑）：

A. 用录制里现成的 1762 输入切片跑冻结 encoder：
   ``command = obs[4:584]``、``anchor = obs[601:661]`` → 是否逐位复现 `encoder_latent`？
   （验证 encoder 权重/偏移/布局）

B. 拆 command 的 580 维：**字段主序** ``[pos×10, vel×10]``；确认 10 份是不是**当前参考帧**
   （`human_joint_pos[t]` / 当前参考速度），以及 anchor 60 维是不是 10 份同一个相对朝向。

C. 只用"训练环里拿得到"的量重建：
   ``command = concat([cur_pos]×10, [cur_vel]×10)``（当前帧重复，字段主序，关节顺序不转置）
   ``anchor  = concat([anchor6[t]]×10)``  → 是否逐位复现 `encoder_latent`？

判定：
* A、C 都过 ⇒ 标签空间可在环复现 ⇒ **S2（重标端口）永久退役**；
* A 过、C 不过 ⇒ 标签依赖录制时才有的量 ⇒ S2 保留为备选；
* A 不过 ⇒ encoder 切片/权重有问题，先修 probe。

用法::

    python3 scripts/probe_token_reproduction.py --clips 1
    python3 scripts/probe_token_reproduction.py --clips 2 \
        --out runs/S2_gates/g0b_token_reproduction.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.sonic_encoder import (  # noqa: E402
    ANCHOR_MF_OFFSET,
    COMMAND_FRAME_DIM,
    COMMAND_FRAMES,
    COMMAND_OFFSET,
    G1MotionEncoder,
    mjcf_to_isaaclab_perm,
)

#: 每个任务取前 N 条
TASKS = (
    "HOI_double_desk__tw",
    "HOI_pp_box__yb",
    "HSI_vision_navi__yb",
    "HSI_boxing__zz",
)


def _clips(task: str, n: int) -> list[Path]:
    task_name, op = task.split("__")
    return sorted((ROOT / "data" / "ha_raw" / task_name / op).glob("*.npz"))[:n]


def _maxabs(a, b) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def _exact_frac(a, b, tol: float = 1e-9) -> float:
    return float((np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) < tol).mean())


def _port_sidecar_name(path: Path) -> str:
    """raw npz -> 端口侧车名（侧车按 sorted(raw) 的索引命名）。"""
    names = {
        "PickPlace-DoubleDesk": "HOI_double_desk__tw",
        "ArtVIP-Tabletop-GrapCup": "HOI_grap_cup__yb",
        "ArtVIP-Livingroom-GrapCup": "HOI_grap_cup__yb",
        "PickPlace-Box": "HOI_pp_box__yb",
        "Boxing": "HSI_boxing__zz",
        "OpenDoor": "HSI_open_door__zz",
        "SitSofa": "HSI_sit_sofa__zz",
        "VisionNavigation": "HSI_vision_navi__yb",
    }
    session = next((v for k, v in names.items() if k in path.name), None)
    if session is None:
        return path.stem
    task, op = session.split("__")
    raws = sorted((ROOT / "data" / "ha_raw" / task / op).glob("*.npz"))
    return f"{session}__ep{raws.index(path):06d}"


def probe_clip(enc: G1MotionEncoder, path: Path, want_port: bool = True) -> dict:
    with np.load(path) as z:
        T = int(z["num_frames"])
        obs = np.asarray(z["encoder_input"], dtype=np.float32)[:T]
        latent = np.asarray(z["encoder_latent"], dtype=np.float32)[:T]
        anchor_win = np.asarray(z["encoder_anchor_window"], dtype=np.float32)[:T]
        ref_pos = np.asarray(z["human_joint_pos"], dtype=np.float32)[:T]
        ref_vel = np.asarray(z["encoder_motion_joint_vel_hist"], dtype=np.float32)[:T, -1]

    out: dict = {"clip": path.stem, "frames": T}

    # ---- A: 录制输入 -> encoder -> latent
    cmd_rec = obs[:, COMMAND_OFFSET : COMMAND_OFFSET + COMMAND_FRAMES * COMMAND_FRAME_DIM]
    anc_rec = obs[:, ANCHOR_MF_OFFSET : ANCHOR_MF_OFFSET + COMMAND_FRAMES * 6].reshape(T, COMMAND_FRAMES, 6)
    with torch.no_grad():
        tok_A = enc(torch.from_numpy(cmd_rec.reshape(T, COMMAND_FRAMES, COMMAND_FRAME_DIM)),
                    torch.from_numpy(anc_rec)).numpy()
    out["A_recorded_input"] = {"max_abs": _maxabs(tok_A, latent), "exact_frac": _exact_frac(tok_A, latent)}

    # ---- B: 580 维的字段主序拆解（10 份是不是同一个当前帧）
    pos_block = cmd_rec[:, : COMMAND_FRAMES * 29].reshape(T, COMMAND_FRAMES, 29)
    vel_block = cmd_rec[:, COMMAND_FRAMES * 29 :].reshape(T, COMMAND_FRAMES, 29)
    dup_pos = max(_maxabs(pos_block[:, j], ref_pos) for j in range(COMMAND_FRAMES))
    dup_vel = max(_maxabs(vel_block[:, j], ref_vel) for j in range(COMMAND_FRAMES))
    dup_anchor = max(_maxabs(anc_rec[:, j], anc_rec[:, 0]) for j in range(COMMAND_FRAMES))
    out["B_layout"] = {
        "frame_major_would_be_wrong": True,
        "dup_pos_max_abs": dup_pos,
        "dup_vel_max_abs": dup_vel,
        "dup_anchor_max_abs": dup_anchor,
        "pos_block_is_current_ref_frame": dup_pos < 1e-6,
        "vel_block_is_current_ref_vel": dup_vel < 1e-6,
        "anchor_is_10_copies": dup_anchor < 1e-6,
    }

    # ---- C: 只用"当前参考帧 + anchor"重建（在环可得）
    cmd_cur = np.concatenate(
        [np.repeat(ref_pos[:, None, :], COMMAND_FRAMES, axis=1).reshape(T, -1),
         np.repeat(ref_vel[:, None, :], COMMAND_FRAMES, axis=1).reshape(T, -1)],
        axis=-1,
    ).astype(np.float32)
    with torch.no_grad():
        tok_C = enc(torch.from_numpy(cmd_cur.reshape(T, COMMAND_FRAMES, COMMAND_FRAME_DIM)),
                    torch.from_numpy(anc_rec)).numpy()
    out["C_current_frame_rebuild"] = {"max_abs": _maxabs(tok_C, latent), "exact_frac": _exact_frac(tok_C, latent)}

    # ---- D: 现端口侧车的口径（未来 10 帧升序 + PERM_MJCF_TO_ISAACLAB + 恒等 anchor）
    perm = mjcf_to_isaaclab_perm()
    fut = np.arange(T)[:, None] + np.arange(COMMAND_FRAMES)[None, :]
    fut = np.clip(fut, 0, T - 1)
    # 用录制里的参考运动当库（离线等价物）
    pos_fut = ref_pos[fut][:, :, perm].reshape(T, -1)
    vel_fut = ref_vel[fut][:, :, perm].reshape(T, -1)
    cmd_fut = np.concatenate([pos_fut, vel_fut], axis=-1).reshape(T, COMMAND_FRAMES, COMMAND_FRAME_DIM)
    id_anchor = np.zeros((T, COMMAND_FRAMES, 6), np.float32)
    id_anchor[..., 0] = 1.0
    id_anchor[..., 3] = 1.0
    with torch.no_grad():
        tok_D = enc(torch.from_numpy(cmd_fut.astype(np.float32)), torch.from_numpy(id_anchor)).numpy()
    out["D_future_perm_identity_anchor"] = {"max_abs": _maxabs(tok_D, latent), "exact_frac": _exact_frac(tok_D, latent)}
    if want_port:
        sidecar = ROOT / "data" / "port_tokens" / "ha_tokens" / f"{_port_sidecar_name(path)}.npy"
        if sidecar.is_file():
            port = np.load(sidecar)[:T]
            out["D_port_sidecar"] = {
                "path": str(sidecar.relative_to(ROOT)),
                "mse_vs_label": float(((port - latent) ** 2).mean()),
                "corr_vs_label": float(np.corrcoef(port.ravel(), latent.ravel())[0, 1]),
                "max_abs_vs_D_recipe": _maxabs(port, tok_D),
                "exact_frac_vs_D_recipe": _exact_frac(port, tok_D),
            }
    return out


def probe_token_npz(enc: G1MotionEncoder, path: Path) -> dict:
    """用 `data/npz/ha_tokens` 的 clip 跑 parity（S1 修好 builder 后的验收口径）。

    注意口径（本轮实测）：encoder 的 ``command`` 用的是**参考轨迹**（raw 里的
    ``human_joint_pos`` / ``encoder_motion_joint_vel_hist[-1]``），**不是** npz 的
    ``joint_pos``（那是机器人实测执行轨迹）；拿执行轨迹去装 command 只能到 exact≈0.35–0.46。
    这里两种都算，把差异钉在数字上。
    """
    with np.load(path) as z:
        pos_exec = np.asarray(z["joint_pos"], dtype=np.float32)
        vel_exec = np.asarray(z["joint_vel"], dtype=np.float32)
        label = np.asarray(z["g1_motion_token_64"], dtype=np.float32)
        src = Path(str(z["source"]))
    raw = ROOT / "data" / "ha_raw"
    cand = list(raw.glob(f"*/*/{src.name}.npz"))
    if not cand:
        cand = list(raw.glob(f"*/*/*{src.stem}*.npz"))
    if not cand:
        return {"clip": path.stem, "error": f"raw npz for {src.name} not found"}
    with np.load(cand[0]) as rz:
        anchor6 = np.asarray(rz["consumed_anchor_rot6d"], dtype=np.float32)
        ref_pos = np.asarray(rz["human_joint_pos"], dtype=np.float32)
        ref_vel = np.asarray(rz["encoder_motion_joint_vel_hist"], dtype=np.float32)[:, -1]
        T0 = int(rz["num_frames"])
    T = min(len(pos_exec), len(vel_exec), len(label), len(anchor6), T0)

    def _tok(pos, vel):
        cmd = np.concatenate(
            [np.repeat(pos[:T, None, :], COMMAND_FRAMES, axis=1).reshape(T, -1),
             np.repeat(vel[:T, None, :], COMMAND_FRAMES, axis=1).reshape(T, -1)],
            axis=-1,
        ).reshape(T, COMMAND_FRAMES, COMMAND_FRAME_DIM)
        anc = np.repeat(anchor6[:T, None, :], COMMAND_FRAMES, axis=1)
        with torch.no_grad():
            return enc(torch.from_numpy(np.ascontiguousarray(cmd)),
                       torch.from_numpy(np.ascontiguousarray(anc))).numpy()

    tok_ref = _tok(ref_pos, ref_vel)
    tok_exec = _tok(pos_exec, vel_exec)
    return {
        "clip": path.stem,
        "frames": T,
        "ref_source": "human_joint_pos + encoder_motion_joint_vel_hist[-1]",
        "max_abs": _maxabs(tok_ref, label[:T]),
        "exact_frac": _exact_frac(tok_ref, label[:T]),
        "exec_source_exact_frac": _exact_frac(tok_exec, label[:T]),
        "exec_source_max_abs": _maxabs(tok_exec, label[:T]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=1, help="每个任务取前 N 条 clip")
    ap.add_argument("--from-token-npz", type=int, default=0,
                    help="额外用 data/npz/ha_tokens 的 N 条 clip 跑 parity（S1 修好 builder 后的验收口径）")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/S2_gates/g0b_token_reproduction.json")
    args = ap.parse_args()

    enc = G1MotionEncoder.from_onnx(args.onnx)
    print(f"[probe] encoder params={enc.info()['params']} fsq={enc.info()['fsq']}", flush=True)

    rows = []
    for task in TASKS:
        for path in _clips(task, args.clips):
            row = probe_clip(enc, path)
            rows.append(row)
            a, c, d = (row["A_recorded_input"], row["C_current_frame_rebuild"],
                       row["D_future_perm_identity_anchor"])
            port = row.get("D_port_sidecar", {})
            print(
                f"[{task:22s}] T={row['frames']:4d} "
                f"A(max={a['max_abs']:.1e} exact={a['exact_frac']:.4f}) "
                f"C(max={c['max_abs']:.1e} exact={c['exact_frac']:.4f}) "
                f"D(max={d['max_abs']:.1e} exact={d['exact_frac']:.3f}) "
                + (f"sidecar≈D(exact={port['exact_frac_vs_D_recipe']:.3f}) corr={port['corr_vs_label']:.3f}"
                   if port else "sidecar=n/a"),
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"encoder": enc.info(), "clips": rows}

    # ---- 额外：用 data/npz/ha_tokens（参考运动 + 录制 anchor）跑 parity = 修好 builder 后的验收口径
    if args.from_token_npz:
        pool = sorted((ROOT / "data" / "npz" / "ha_tokens").glob("*.npz"))
        step = max(1, len(pool) // args.from_token_npz)
        picks = pool[::step][: args.from_token_npz]
        parity = [probe_token_npz(enc, p) for p in picks]
        payload["token_npz_parity"] = parity
        for r in parity:
            if "error" in r:
                print(f"[parity] {r['clip']}: {r['error']}", flush=True)
            else:
                print(f"[parity] {r['clip'][:44]:44s} T={r['frames']:4d} "
                      f"exact={r['exact_frac']:.4f} max|Δ|={r['max_abs']:.2e}", flush=True)

    args.out.write_text(json.dumps(payload, indent=1))

    A_ok = all(r["A_recorded_input"]["exact_frac"] > 0.999 for r in rows)
    C_ok = all(r["C_current_frame_rebuild"]["exact_frac"] > 0.999 for r in rows)
    verdict = (
        "标签空间可复现（S2 永久退役）" if A_ok and C_ok
        else "标签空间部分依赖录制量（S2 保留为备选）" if A_ok
        else "encoder 切片/权重有问题（先修 probe，别谈 S2）"
    )
    print(f"[probe] -> {args.out}")
    print(f"[判定] A={'pass' if A_ok else 'fail'} C={'pass' if C_ok else 'fail'} => {verdict}", flush=True)
    return 0 if A_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
