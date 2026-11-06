#!/usr/bin/env python3
"""用官方 kinematic planner 把一集 LeRobot 记录**重建出带 root 位置的参考轨迹**。

背景（2026-10-30）
------------------
官方奖励 `tracking_anchor_pos`（0.5 / σ0.3）监督的是**绝对世界根位置**，它的真值来自训练用的
motion 参考（`.pkl`，`MotionLib` 加载，含整段 `global_translation`）。那份 `.pkl` 是**采集时**
由 kinematic planner 实时产出的，不是 LeRobot 日志。

我们的 LeRobot 日志里**没有任何根位置**（`observation.state` 只有 43 关节、`root_orientation`
只有朝向、`eef_state` 是基座相对），所以我们导出的 `ft318_compact.pt` 的 root x/y 恒为 0 ——
于是同一个奖励项退化成"待在原地"。

本脚本把缺的那一半补回来：按官方 C++ 部署的同一套规则（10 Hz planner 线程、replan 触发条件、
30 Hz→50 Hz 重采样、8 帧交叉淡入）把数据集录下来的 `planner_movement/facing/speed` 喂给
`planner_sonic.onnx`，得到**逐帧带 root 位置**的参考轨迹。

**自带对拍**：`observation.root_orientation` 是录下来的。如果重建出的根朝向和它吻合，
说明重建可信（这是唯一能验证位置链路的量，因为位置本身没录）。

用法：
    PY=third_party/SuperMimic-Reproduced/.venv/bin/python
    $PY scripts/replay_planner_ref.py --parquet <episode_000000.parquet> --out /tmp/p0.npz
    $PY scripts/replay_planner_ref.py --dataset <session_dir> --episodes 0 1 2 --out-dir /tmp/ref
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANNER = (
    ROOT.parent / "GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
)
FPS = 50.0            # LeRobot 记录频率
# 数据集的 43 关节布局 → 身体 29（STANDARD_29 / MuJoCo MJCF 顺序）的索引
BODY29_IDX = list(range(22)) + list(range(29, 36))
PLANNER_HZ = 10.0     # 官方 planner 线程
OUT_HZ = 30.0         # planner 模型输出频率
LOOKAHEAD = 2         # 官方 motion_look_ahead_steps（50 Hz 帧）
CROSSFADE = 8         # 官方 8 帧交叉淡入（50 Hz）
REPLAN_PERIOD_WALK = 1.0   # 官方"walk 类"周期 replan
MIN_TOKENS = 6


# ------------------------------------------------------------------ 基础几何
def quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def quat_slerp(a, b, t):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    th = math.acos(max(-1.0, min(1.0, d)))
    out = (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)
    return out / np.linalg.norm(out)


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(th):
    return np.array([math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ 数据读取
def load_episode(parquet: Path) -> dict:
    import pyarrow.parquet as pq

    cols = ["observation.state", "observation.root_orientation", "teleop.planner_movement",
            "teleop.planner_facing", "teleop.planner_speed", "teleop.planner_mode"]
    have = set(pq.ParquetFile(str(parquet)).schema_arrow.names)
    cols = [c for c in cols if c in have]
    t = pq.ParquetFile(str(parquet)).read(columns=cols)
    d = {c: np.array(t.column(c).to_pylist()) for c in cols}
    state = d["observation.state"]
    # ★ 数据集 43 关节的顺序是【腿12 + 腰3 + 左臂7 + 左手7 + 右臂7 + 右手7】，
    #   身体 29 个必须按 BODY29_IDX 取，不能写 state[:, :29]（那会把左手当成右臂）。
    q29 = state[:, BODY29_IDX]
    out = {
        "q29": q29.astype(np.float64),
        "root_quat": d["observation.root_orientation"].astype(np.float64),
        "move": d.get("teleop.planner_movement", np.zeros((len(state), 3))).astype(np.float64),
        "face": d.get("teleop.planner_facing", np.zeros((len(state), 3))).astype(np.float64),
        "speed": d.get("teleop.planner_speed", np.full(len(state), -1.0)).astype(np.float64).reshape(-1),
        "pmode": d.get("teleop.planner_mode", np.zeros(len(state), np.int64)).astype(np.int64).reshape(-1),
        "n": len(state),
    }
    return out


# ------------------------------------------------------------------ planner
class Planner:
    def __init__(self, path: Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        # 多进程并行时**必须**限制 intra-op 线程：ORT 默认每会话开满核，
        # 6 个 worker × 128 线程 = 768 线程互相抢，实测比限流慢 5 倍（load 167/128）。
        nthr = int(os.environ.get("PLC_ORT_THREADS", "0") or 0)
        if nthr > 0:
            so.intra_op_num_threads = nthr
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.calls = 0
        self.secs = 0.0
        self.names = [i.name for i in self.sess.get_inputs()]

    def __call__(self, ctx, *, mode, target_vel, move, face, seed=0, height=-1.0):
        assert ctx.shape == (4, 36), ctx.shape
        inp = {
            "context_mujoco_qpos": ctx[None].astype(np.float32),
            "target_vel": np.array([target_vel], np.float32),
            "mode": np.array([mode], np.int64),
            "movement_direction": np.array([move], np.float32),
            "facing_direction": np.array([face], np.float32),
            "random_seed": np.array([seed], np.int64),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": np.ones((1, 11), np.int64),
            "height": np.array([height], np.float32),
        }
        inp = {k: inp[k] for k in self.names}
        t = time.time()
        res = self.sess.run(None, inp)
        self.secs += time.time() - t
        self.calls += 1
        qpos, nf = res[0][0], int(np.asarray(res[1]).reshape(-1)[0])
        return qpos[:nf].astype(np.float64)


def resample_30_to_50(q30: np.ndarray) -> np.ndarray:
    """30 Hz → 50 Hz：位置线性插值、四元数 slerp（与官方 resampling 一致）。"""
    n30 = len(q30)
    dur = (n30 - 1) / OUT_HZ
    n50 = int(dur * FPS) + 1
    out = np.zeros((n50, 36), np.float64)
    for k in range(n50):
        t = k / FPS
        u = min(t * OUT_HZ, n30 - 1)
        i0 = int(math.floor(u))
        i1 = min(i0 + 1, n30 - 1)
        w = u - i0
        out[k, 0:3] = (1 - w) * q30[i0, 0:3] + w * q30[i1, 0:3]
        out[k, 3:7] = quat_slerp(q30[i0, 3:7], q30[i1, 3:7], w)
        out[k, 7:36] = (1 - w) * q30[i0, 7:36] + w * q30[i1, 7:36]
    return out


def infer_mode(move, speed, pmode) -> int:
    """把数据集里的指令映射到 planner 的 clip 索引。

    数据集 `teleop.planner_mode` 只有 {0,1}（是"static/non-static"语义，不是 clip 索引），
    所以 mode 只能按指令推断：不动→idle(0)，在动→walk(2)。
    """
    moving = float(np.linalg.norm(move[:2])) > 1e-5
    if not moving and speed <= 0.0:
        return 0
    return 2 if speed <= 0.0 or speed >= 0.25 else 1


def _last_filled(filled: np.ndarray, f: int) -> int:
    """从 f 往前找最近一个已规划帧（官方做法：把上一份规划播完再切）。"""
    j = min(f, len(filled) - 1)
    while j >= 0 and not filled[j]:
        j -= 1
    return j


def replay(ep: dict, planner: Planner, *, seed: int = 0, verbose: bool = False) -> dict:
    n = ep["n"]
    q29, rq = ep["q29"], ep["root_quat"]
    move, face, speed = ep["move"], ep["face"], ep["speed"]

    traj = np.zeros((n, 36), np.float64)      # [root_pos(3), root_quat(4), dof(29)]
    traj[:, 2] = 0.75
    traj[:, 3:7] = rq[0]                      # 初始朝向对齐录制值（同一世界系，可直接对拍）
    traj[:, 7:36] = q29
    filled = np.zeros(n, bool)                # 由 planner 写过的帧

    step = int(round(FPS / PLANNER_HZ))       # 5 帧 @50Hz
    cur = 0
    last_mode = last_face = last_speed = last_move = None
    last_replan_t = -1e9
    filled_until = -1                          # 已规划到的最远帧（不够就强制重规划）
    bad = 0
    max_calls = 20 * n // step + 50

    while cur < n - 1 and planner.calls < max_calls + 10000:
        m = infer_mode(move[cur], speed[cur], ep["pmode"][cur])
        fvec = face[cur] if np.linalg.norm(face[cur][:2]) > 1e-5 else move[cur]
        if np.linalg.norm(fvec[:2]) < 1e-5:
            fvec = np.array([math.cos(yaw_of(rq[cur])), math.sin(yaw_of(rq[cur])), 0.0])

        mode_ch = (m != last_mode) or (last_face is not None and
                                       np.linalg.norm(fvec[:2] - last_face[:2]) > 1e-3)
        dyn_ch = (last_speed is not None and abs(speed[cur] - last_speed) > 1e-3) or \
                 (last_move is not None and np.linalg.norm(move[cur][:2] - last_move[:2]) > 1e-3)
        timer = (cur / FPS - last_replan_t) >= REPLAN_PERIOD_WALK
        moving = float(np.linalg.norm(move[cur][:2])) > 1e-5
        starved = (cur + LOOKAHEAD + step) > filled_until   # 上一份规划快用完了
        # 2026-10-30 修 bug：原来 `dyn_ch and m != 0` 会在**命令停下来那一刻**（m 变成 0=idle）
        # 拒绝重规划 ⇒ 机器人继续把上一份"走路"规划播完（最长 ~2 s）⇒ 参考系统性**多走**。
        # 官方规则是"速度/方向变化就重规划"（不管新 mode 是不是 idle），这里照做。
        need = (last_mode is None) or mode_ch or dyn_ch or (timer and moving) or starved
        if not need:
            cur += step
            continue

        # 4 帧 context @30Hz，起点 = cur + lookahead（官方做法：从**规划轨迹**采样；
        # 未规划到的帧用"最近一个已规划帧"顶住 = 官方把上一份动画播完的行为）
        tgt = speed[cur] if speed[cur] > 0 else -1.0
        ctx = np.zeros((4, 36), np.float64)
        for j in range(4):
            f = min(n - 1, cur + LOOKAHEAD + int(round(j * FPS / OUT_HZ)))
            g = f if filled[f] else _last_filled(filled, f)
            if g >= 0:
                ctx[j, 0:3], ctx[j, 3:7] = traj[g, 0:3], traj[g, 3:7]
            else:
                ctx[j, 0:3], ctx[j, 3:7] = (0.0, 0.0, 0.75), rq[0]
            ctx[j, 7:36] = q29[f]          # dof 用**真实关节**（策略跟踪误差 2.5°，可忽略）
        q30 = planner(ctx, mode=m, target_vel=tgt, move=move[cur], face=fvec, seed=seed)
        seq = resample_30_to_50(q30)

        # 安全闸：非有限 / 从 context 起跳 >1 m / 高度出界 ⇒ 丢弃这次规划
        jump = float(np.linalg.norm(seq[0, 0:3] - ctx[0, 0:3])) if len(seq) else 9.9
        zmid = float(np.median(seq[:, 2])) if len(seq) else 0.0
        if (not np.isfinite(seq).all()) or jump > 1.0 or not (0.35 < zmid < 1.15):
            bad += 1
            cur += 1
            if bad > 200:
                print(f"[warn] 连续丢弃 {bad} 次规划，提前结束 @frame {cur}")
                break
            continue

        s = cur + LOOKAHEAD
        k = min(len(seq), n - s)
        if k > 0:
            for i in range(k):
                w = 1.0 if i >= CROSSFADE or not filled[s + i] else (i + 1) / CROSSFADE
                traj[s + i, 0:3] = (1 - w) * traj[s + i, 0:3] + w * seq[i, 0:3]
                if filled[s + i]:
                    traj[s + i, 3:7] = quat_slerp(traj[s + i, 3:7], seq[i, 3:7], w)
                else:
                    traj[s + i, 3:7] = seq[i, 3:7]
                traj[s + i, 7:36] = q29[s + i]     # dof 保留真实值（我们只要 root 那一半）
                filled[s + i] = True
            filled_until = max(filled_until, s + k - 1)
            bad = 0
        last_mode, last_face, last_speed, last_move = m, fvec.copy(), speed[cur], move[cur].copy()
        last_replan_t = cur / FPS
        cur += step
    return {"traj": traj, "filled": filled, "bad": bad}


def report(ep: dict, res: dict, name: str = "") -> dict:
    traj, filled = res["traj"], res["filled"]
    rq = ep["root_quat"]
    y_pred = yaw_of(traj[:, 3:7])
    y_rec = yaw_of(rq)
    ok = filled
    dy = wrap(y_pred[ok] - y_rec[ok])
    p = traj[:, 0:3]
    m = filled.mean()
    if ok.sum() > 10:
        corr = float(np.corrcoef(y_pred[ok], y_rec[ok])[0, 1])
    else:
        corr = float("nan")
    print(f"\n[{name}] frames={len(traj)} planner 覆盖 {100 * m:.0f}% "
          f"calls={res.get('calls', '?')}")
    print(f"  重建水平轨迹: 净位移 {np.linalg.norm(p[-1, :2] - p[0, :2]) * 100:.1f} cm"
          f" | 路径长 {np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum() * 100:.1f} cm"
          f" | xy 包围盒 {np.ptp(p[:, 0]) * 100:.0f}×{np.ptp(p[:, 1]) * 100:.0f} cm")
    print(f"  根高度 z: [{p[:, 2].min():.3f}, {p[:, 2].max():.3f}]")
    print(f"  ★ 对拍 yaw（重建 vs 录制）: MAE={np.degrees(np.abs(dy)).mean():.1f}° "
          f"中位={np.degrees(np.median(np.abs(dy))):.1f}° "
          f"p90={np.degrees(np.quantile(np.abs(dy), 0.9)):.1f}° corr={corr:.3f}")
    return {"yaw_mae": float(np.degrees(np.abs(dy)).mean()), "yaw_corr": corr,
            "travel_cm": float(np.linalg.norm(p[-1, :2] - p[0, :2]) * 100),
            "path_cm": float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum() * 100),
            "fill": float(m)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    ap.add_argument("--parquet", type=Path, default=None)
    ap.add_argument("--dataset", type=Path, default=None, help="session 目录（含 data/chunk-000）")
    ap.add_argument("--episodes", nargs="*", type=int, default=[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    planner = Planner(a.planner)
    targets: list[Path] = []
    if a.parquet is not None:
        targets = [a.parquet]
    elif a.dataset is not None:
        cand = sorted(glob.glob(str(a.dataset / "data" / "chunk-*" / "episode_*.parquet")))
        if not cand:
            raise SystemExit(f"[fail] {a.dataset} 里没有 episode_*.parquet")
        targets = [Path(cand[i]) for i in a.episodes if i < len(cand)]
    else:
        raise SystemExit("需要 --parquet 或 --dataset")

    summaries = []
    for p in targets:
        ep = load_episode(p)
        res = replay(ep, planner, seed=a.seed)
        res["calls"] = planner.calls
        s = report(ep, res, name=p.stem)
        summaries.append(s)
        if a.out is not None:
            np.savez_compressed(a.out, traj=res["traj"].astype(np.float32),
                                filled=res["filled"], q29=ep["q29"].astype(np.float32),
                                root_quat=ep["root_quat"].astype(np.float32))
            print(f"  [out] {a.out}")
        if a.out_dir is not None:
            a.out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.out_dir / f"{p.stem}.npz",
                                traj=res["traj"].astype(np.float32), filled=res["filled"],
                                q29=ep["q29"].astype(np.float32),
                                root_quat=ep["root_quat"].astype(np.float32))
    print(f"\n[planner] {planner.calls} 次调用, {planner.secs:.1f}s total, "
          f"{1000 * planner.secs / max(1, planner.calls):.0f} ms/call")
    if len(summaries) > 1:
        import statistics as st
        print(f"[汇总] yaw MAE 中位 {st.median(s['yaw_mae'] for s in summaries):.1f}° | "
              f"净位移中位 {st.median(s['travel_cm'] for s in summaries):.0f} cm | "
              f"路径中位 {st.median(s['path_cm'] for s in summaries):.0f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
