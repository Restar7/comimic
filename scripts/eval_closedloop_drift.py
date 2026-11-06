#!/usr/bin/env python3
"""闭环 dump 上的**真实引擎漂移**（2026-11-05）。

为什么不能只看净位移
--------------------
`eval_closedloop_progress.py` 报的 `net`（净位移）只是"崩没崩"的探测器：
round-3 崩掉时 0.111 m、base 1.0 m 能分开，但分化不出"谁跟得更准"。
官方 `success` 在我们的时间预算里**永远够不着**（DoubleDesk 示范中位数 939 帧 ≈37 s，
闭环只跑 1000 步）。所以真正的判据只能是：

> **在真实引擎（Isaac）里，机器人相对"理想指令流"漂了多少。**

这正是训练奖励用的那一套量（`huber_err`：位置 / 朝向 / 关节 / 线速度 / 角速度），
只不过训练时是在**冻结 WM 的想象里**算的，这里在**真引擎**里算。两个数一对比，
就知道"想象里的改善"有没有落到真机上。

怎么算（全部是**免对齐**的口径）
--------------------------------
dump 的 `steps.jsonl` 每步都有 `root_pos / root_quat_wxyz / q / qd /
root_lin_vel_b / root_ang_vel_b / frame`，`frame` 指向运动库 `packed.pt` 的 `ref` 行。

* 位置漂移：`‖(p_t − p_0) − (r_t − r_0)‖` —— 减掉起点，世界坐标系的平移/朝向被消掉，
  剩下的就是"走了同样远、但走歪了多少"。**累积漂移**再额外报一条（对时间的和）。
* 朝向漂移：`angle(q_0⁻¹q_t, r_0⁻¹r_t)` —— 同样是相对量，单位度。
* 关节漂移：`‖(q_t − q_0) − (rq_t − rq_0)‖`，**两种关节序都算**取小者，并报两者比值
  （比值 ≫1 说明序选错了，脚本会自己提示）。
* 速度误差：`‖v_t − v_ref_t‖`、`‖ω_t − ω_ref_t‖` —— 速度本来就是自由量，不需要对齐。

参照系（判读用）
----------------
* 理想 = 0（参考轨迹自己比自己）。
* 采集语料 A' 的逐行漂移 p50 = **0.32 cm / p90 0.87 cm**（`runs/a_prime_sonic_leash`），
  这是"一个跟得不错的数据生成器"的水平，可以当"好"的锚点。

用法::

    python3 scripts/eval_closedloop_drift.py \
        --runs base:=runs/closedloop_r3/base_s0 f100:=runs/closedloop_r3/f100_s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def quat_mul(a, b):
    """Hamilton product, wxyz。"""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conj(q):
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_angle(q):
    """旋转角（弧度）。q 已归一化。"""
    w = np.clip(np.abs(q[..., 0]), -1.0, 1.0)
    return 2.0 * np.arccos(w)


def load_steps(run: Path) -> dict[str, np.ndarray] | None:
    p = run / "dump" / "steps.jsonl"
    if not p.exists():
        return None
    keys = ["frame", "q", "qd", "root_pos", "root_quat_wxyz",
            "root_lin_vel_b", "root_ang_vel_b", "applied", "latent"]
    buf = {k: [] for k in keys}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        for k in keys:
            if k in d:
                buf[k].append(d[k])
    return {k: np.asarray(v, dtype=np.float64) for k, v in buf.items() if v}


def analyse(run: Path, ref: np.ndarray, corr_settle: int = 20,
            until: int | None = None) -> dict | None:
    s = load_steps(run)
    if s is None or len(s["frame"]) < 50:
        return None
    # ★ 2026-11-05 夜间：`until` 只统计**前 until 步**。为什么要这个：
    #   1000 步里后半段是"站桩"（`docs/STALL_MECHANISM_AND_SA4_VERDICT_20261105.md` §3），
    #   联合微调被训练的是 20 步（0.4 s）想象窗口，它的收益只可能先出现在**接近段**；
    #   把整段中位数当唯一判据，等于拿"站桩时长"去稀释动作质量。
    if until is not None and 0 < until < len(s["frame"]):
        s = {k: v[:until] for k, v in s.items()}
    F = s["frame"].astype(np.int64)
    F = np.clip(F, 0, ref.shape[0] - 1)
    R = ref[F]                                   # (T,71) 参考帧

    # ---- 位置 / 朝向：减掉起点，消掉世界系偏移 ------------------------------
    dp_act = s["root_pos"] - s["root_pos"][0]
    dp_ref = R[:, 0:3] - R[0, 0:3]
    e_pos = np.linalg.norm(dp_act - dp_ref, axis=-1)          # (T,) 米

    qa = s["root_quat_wxyz"] / np.linalg.norm(s["root_quat_wxyz"], axis=-1, keepdims=True)
    qr = R[:, 3:7] / np.maximum(np.linalg.norm(R[:, 3:7], axis=-1, keepdims=True), 1e-9)
    rel_a = quat_mul(quat_conj(qa[0:1]), qa)
    rel_r = quat_mul(quat_conj(qr[0:1]), qr)
    e_ori = np.degrees(quat_angle(quat_mul(quat_conj(rel_r), rel_a)))  # (T,) 度

    # ---- 关节：相对量 + 两种序取小 ------------------------------------------
    dq_act = s["q"] - s["q"][0]
    dq_ref = R[:, 7:36] - R[0, 7:36]
    e_q_a = np.linalg.norm(dq_act - dq_ref, axis=-1)
    e_q_b = np.linalg.norm(dq_act - dq_ref[:, ::-1], axis=-1)
    swap = float(np.mean(e_q_b)) < float(np.mean(e_q_a))
    e_q = e_q_b if swap else e_q_a
    ratio = float(np.mean(e_q_b) / max(np.mean(e_q_a), 1e-9))

    # ---- 速度（自由量，免对齐）---------------------------------------------
    e_v = np.linalg.norm(s["root_lin_vel_b"] - R[:, 36:39], axis=-1)      # m/s
    e_w = np.linalg.norm(s["root_ang_vel_b"] - R[:, 39:42], axis=-1)      # rad/s
    e_qd = np.linalg.norm(s["qd"] - R[:, 42:71], axis=-1)                 # rad/s

    def stat(x, sl=None):
        y = x if sl is None else x[sl]
        return (float(np.median(y)), float(np.quantile(y, 0.9)), float(np.max(y)))

    k = corr_settle
    out = {
        "n": int(len(F)),
        "pos_p50": stat(e_pos)[0], "pos_p90": stat(e_pos)[1], "pos_max": stat(e_pos)[2],
        "pos_tail_p50": stat(e_pos, slice(-200, None))[0],
        "ori_p50": stat(e_ori)[0], "ori_p90": stat(e_ori)[1],
        "q_p50": stat(e_q)[0], "q_p90": stat(e_q)[1],
        "v_p50": stat(e_v)[0], "w_p50": stat(e_w)[0], "qd_p50": stat(e_qd)[0],
        "q_swap": swap, "q_ratio": ratio,
        "q_settle_p50": stat(e_q, slice(k, None))[0] if len(F) > k else float("nan"),
        "interp": float(np.mean(e_pos[k:2 * k]) if len(F) > 2 * k else np.nan),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="TAG=DIR")
    ap.add_argument("--packed", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--settle", type=int, default=20, help="跳过前 N 步（换 token 的瞬态）")
    ap.add_argument("--until", type=int, default=0,
                    help="只统计前 N 步（0=全部）。用来把'接近段'与'站桩段'分开看")
    a = ap.parse_args()

    import torch
    pk = torch.load(a.packed, map_location="cpu", weights_only=False)
    ref = torch.as_tensor(pk["ref"]).float().numpy()
    print(f"[drift] 参考库 {ref.shape[0]} 帧；A' 采集语料逐行漂移 p50 0.32 cm / p90 0.87 cm 为'好'的锚点")
    print(f"[drift] 口径：全部**减掉起点**（消掉世界系平移/朝向），故与 reset 位姿无关\n")

    print("[drift] ⚠ 可信度：root 位置/朝向两列**可信**（两套布局在 0:3 / 3:7 上一致，且已减掉起点）；"
          "\n        关节/速度四列**暂不可信**：dump 的 q 与运动库 canonical 的 q 之间存在"
          "关节序 + 零位差（实测直接比 2.45 rad、按 B 置换后仍 0.42 rad），\n"
          "        要先把这两件事定死（见 docs/PIPELINE_CURRENT_20261105.md §11）才读得出结论。")
    hdr = (f"{'arm':<10} {'pos p50':>8} {'pos p90':>8} {'pos 尾部':>8} "
           f"{'ori p50':>8} {'q?':>7} {'q 尾部?':>9} "
           f"{'|v|?':>7} {'|w|?':>7} {'|qd|?':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for spec in a.runs:
        tag, _, d = spec.partition("=")
        run = Path(d)
        run = run if run.is_absolute() else (ROOT / run)
        r = analyse(run, ref, corr_settle=a.settle, until=(a.until or None))
        if r is None:
            print(f"{tag:<10} （没有 dump/steps.jsonl，跳过）")
            continue
        rows.append((tag, r))
        print(f"{tag:<10} {r['pos_p50']*100:8.2f} {r['pos_p90']*100:8.2f} "
              f"{r['pos_tail_p50']*100:8.2f} {r['ori_p50']:8.2f} {r['q_p50']:8.3f} "
              f"{r['q_settle_p50']:11.3f} {r['v_p50']:9.3f} {r['w_p50']:9.3f} {r['qd_p50']:10.3f}")
        if r["q_swap"]:
            print(f"{'':<10} ⚠ 关节序取了**反序**（比值 {r['q_ratio']:.2f}）——说明 dump 的 q 与 ref 不同序")
    if len(rows) >= 2:
        print("\n[drift] 相对第一行的变化（负数 = 更贴近理想指令）:")
        b = rows[0][1]
        for tag, r in rows[1:]:
            d = lambda k: (r[k] - b[k]) / max(abs(b[k]), 1e-9) * 100
            print(f"  {tag:<10} 位置 {d('pos_p50'):+6.1f}%   朝向 {d('ori_p50'):+6.1f}%   "
                  f"关节 {d('q_p50'):+6.1f}%   线速度 {d('v_p50'):+6.1f}%   "
                  f"角速度 {d('w_p50'):+6.1f}%")
    print("\n[drift] 判读：位置/朝向/关节是**相对起点**的漂移（越小越跟手）；"
          "\n        线速度/角速度的误差才是解释'为什么位置会飘'的因果量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
