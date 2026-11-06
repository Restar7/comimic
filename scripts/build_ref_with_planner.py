#!/usr/bin/env python3
"""给 318 库**补齐根轨迹**（用官方 kinematic planner 重建），产出 `ft318_compact_v2.pt`。

为什么需要（2026-10-30 追到的根因）
------------------------------------
官方奖励 `tracking_anchor_pos`（0.5 / std 0.3）监督的是绝对世界根位置，真值来自训练用的
motion 参考（`.pkl` / `MotionLib`，含整段 `global_translation`）。那份参考是**采集时**由
kinematic planner 实时产出的；而 LeRobot 日志只记了 43 关节 + 根朝向，**没有任何根位置**。

我们导出的 `ft318_compact.pt` 因此 root x/y ≡ 0（473,876 帧实测），同一个奖励项退化成
"待在原地" —— 这是"走路"这件事唯一的缺口。

本脚本：
  1. 用**内容匹配**（长度 + 根四元数序列）把每个 library clip 对到源 parquet
     （库里的 `epNNNNNN` 与 parquet 的 `episode_XXXXXX` **编号不是同一套**，必须按内容对）
  2. 对每集跑 `replay_planner_ref.replay()`：喂录下来的 planner 指令，逐 0.1 s 重规划，
     得到带 root 位置的轨迹
  3. 默认把每集**归一化**到"第一帧 root xy=0、yaw=0"（与评测 spawn 约定一致）
  4. 写 `ft318_compact_v2.pt`：与 v1 **完全同格式**（ref/names/starts/lengths），
     只是 root x/y 由 0 换成真值；关节/朝向/速度通道原样保留

自带验收：逐集 `yaw MAE`（重建 vs 录制的 `observation.root_orientation`）+ 覆盖率 + 净位移。

用法：
    PY=third_party/SuperMimic-Reproduced/.venv/bin/python
    $PY scripts/build_ref_with_planner.py --limit 8 --workers 2          # 试跑
    $PY scripts/build_ref_with_planner.py --out data/library/ft318_compact_v2.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_planner_ref as R  # noqa: E402

SONIC_DATA = ROOT.parent / "SONIC_DATA"
CLIP_NAMES = ["observation.state", "observation.root_orientation"]


# ------------------------------------------------------------------ 源索引
def index_sources(sonic: Path, sessions: list[str]) -> dict[str, list[dict]]:
    """{session: [{"path","rows","quat"(n,4) }]}，一次读完只留需要的两列。"""
    import pyarrow.parquet as pq

    out: dict[str, list[dict]] = {}
    for s in sessions:
        pats = [str(sonic / "**" / "datasets" / s / "data" / "chunk-*" / "episode_*.parquet")]
        files: list[str] = []
        for pat in pats:
            files += glob.glob(pat, recursive=True, include_hidden=True)
        files = sorted(set(files))
        rows = []
        for f in files:
            try:
                t = pq.ParquetFile(f).read(columns=CLIP_NAMES)
            except Exception:
                continue
            st = np.array(t.column("observation.state").to_pylist(), np.float64)
            q = np.array(t.column("observation.root_orientation").to_pylist(), np.float64)
            rows.append({"path": f, "rows": len(st), "quat": q})
        out[s] = rows
        print(f"[index] {s}: {len(rows)} parquet", flush=True)
    return out


def match_clip(idx: list[dict], length: int, quat: np.ndarray) -> tuple[int, float]:
    """按 (长度相等, 根四元数序列最小误差) 找源文件。返回 (下标, 误差)；找不到 (-1, inf)。"""
    best_i, best_e = -1, float("inf")
    for i, c in enumerate(idx):
        if c["rows"] != length:
            continue
        m = min(len(quat), c["rows"])
        e = float(np.mean(1.0 - np.abs((c["quat"][:m] * quat[:m]).sum(1))))
        if e < best_e:
            best_i, best_e = i, e
    return best_i, best_e


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ------------------------------------------------------------------ 单集任务（跑在子进程）
_PLANNER_CACHE: dict[str, "R.Planner"] = {}


def _planner(path: str) -> "R.Planner":
    """每个 worker 进程只加载一次 ONNX（774 MB，重复加载会白花 3 s/clip）。"""
    if path not in _PLANNER_CACHE:
        _PLANNER_CACHE[path] = R.Planner(Path(path))
    return _PLANNER_CACHE[path]


def _job(task: dict) -> dict:
    ep = R.load_episode(Path(task["parquet"]))
    planner = _planner(task["planner"])
    res = R.replay(ep, planner, seed=0)
    traj, filled = res["traj"], res["filled"]
    n = min(task["length"], len(traj))
    pos = traj[:n, 0:3].copy()
    quat = traj[:n, 3:7].copy()

    # 未覆盖的帧向前顶住（官方也是把上一份规划播完）
    for k in range(n):
        if not filled[k]:
            g = k - 1
            while g >= 0 and not filled[g]:
                g -= 1
            if g >= 0:
                pos[k], quat[k] = pos[g], quat[g]

    # ---- 参考 x/y 来源（2026-10-30 判别实验定案）----
    # 三个候选（同一窗口、12 s、8 个高信号 clip 实测）：
    #   指令积分 694-934cm / planner 原始 112-317cm / 指令速度 150-345cm / **物理重放 133-207cm**
    #   中位: 重放/planner = 0.65、重放/指令积分 = 0.23
    # ⇒ 指令积分不是距离（是愿望，错）；v6 的"指令速度"是人造混合体（可能超可达，错）。
    #   官方训练参考 = planner 的输出（`.pkl` 就是它）⇒ **默认就用 planner 原始输出，不缩放**。
    #   `--pace` 时才用"方向 planner + 速度指令"那版（保留作对照，不推荐）。
    _mv = ep["move"][:n, :2]
    _moving = np.linalg.norm(_mv, axis=1) > 1e-5
    _v = np.where(ep["speed"][:n] > 0, ep["speed"][:n], 0.573) * _moving
    imp_path = float(_v.sum() / R.FPS) * 100.0
    if task.get("pace", False):
        def _win_sum(a, w):
            m = len(a)
            c = np.concatenate(([0.0], np.cumsum(np.asarray(a, np.float64))))
            i0 = np.maximum(0, np.arange(m) - w // 2)
            i1 = np.minimum(m, np.arange(m) + w // 2 + 1)
            return c[i1] - c[i0]
        _dir = np.zeros((n, 2))
        _dir[1:] = np.diff(pos[:n, :2], axis=0)
        _sm = np.stack([_win_sum(_dir[:, 0], int(1.0 * R.FPS)),
                        _win_sum(_dir[:, 1], int(1.0 * R.FPS))], -1)
        _mag = np.linalg.norm(_sm, axis=1)
        _valid = _mag > 1e-9
        _unit = np.zeros_like(_sm)
        if _valid.any():
            _unit[_valid] = _sm[_valid] / _mag[_valid, None]
            _idx = np.where(_valid, np.arange(n), -1)
            np.maximum.accumulate(_idx, out=_idx)
            _idx[_idx < 0] = np.where(_valid)[0][0]
            _unit = _unit[_idx]
        else:
            _unit[:] = np.array([1.0, 0.0])
        _xy = np.cumsum(_unit * (_v / R.FPS)[:, None], axis=0)
        _xy -= _xy[0]
        pos[:n, 0:2] = _xy
    rec_path = float(np.linalg.norm(np.diff(pos[:n, :2], axis=0), axis=1).sum()) * 100.0
    scale = 1.0

    # 归一化：用**录制朝向**的第一帧把整集刚体旋转到 yaw=0（保持相对几何不变），
    # 因为 reference 只当"运动参考"用，不要求和遥操作场景的世界系一致。
    dyaw = float(-yaw_of(ep["root_quat"][0])) if task.get("normalize", True) else 0.0
    if dyaw != 0.0:
        c, s_ = math.cos(dyaw), math.sin(dyaw)
        Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        pos = pos @ Rz.T
    pos[:, 0] -= pos[0, 0]
    pos[:, 1] -= pos[0, 1]

    err = np.abs(R.wrap(yaw_of(traj[:n, 3:7]) - yaw_of(ep["root_quat"][:n])))
    return {
        "name": task["name"], "parquet": task["parquet"], "n": int(n),
        "pos": pos.astype(np.float32), "dyaw": dyaw,
        "fill": float(filled[:n].mean()),
        "yaw_mae_deg": float(np.degrees(err.mean())),
        "pace_scale": float(scale), "imp_path_cm": imp_path, "rec_path_cm": rec_path,
        "imp_path_cm": imp_path, "rec_path_cm": rec_path,
        "travel_cm": float(np.linalg.norm(pos[-1, :2] - pos[0, :2]) * 100),
        "path_cm": float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum() * 100),
        "z_min": float(pos[:, 2].min()), "z_max": float(pos[:, 2].max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, default=ROOT / "data/library/ft318_compact.pt")
    ap.add_argument("--sonic-data", type=Path, default=SONIC_DATA)
    ap.add_argument("--planner", type=Path, default=R.DEFAULT_PLANNER)
    ap.add_argument("--out", type=Path, default=ROOT / "data/library/ft318_compact_v2.pt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--pace", action="store_true",
                    help="改用\"方向 planner + 速度指令\"的对照版本（默认=planner 原始输出，官方口径）")
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()

    import torch

    lib = torch.load(str(a.library), map_location="cpu", weights_only=False)
    ref, starts, lengths, names = lib["ref"], lib["starts"], lib["lengths"], lib["names"]
    names = [str(x) for x in names]
    n_clip = len(names)
    if a.limit:
        n_clip = min(n_clip, a.limit)
    print(f"[lib] {a.library.name}: {len(names)} clip, {ref.shape[0]} 帧; 本次处理 {n_clip}")

    sessions = sorted({n.split("__")[1] for n in names[:n_clip]})
    t0 = time.time()
    idx = index_sources(a.sonic_data, sessions)
    print(f"[index] {time.time() - t0:.0f}s")

    tasks, unmatched = [], []
    for i, nm in enumerate(names[:n_clip]):
        s = nm.split("__")[1]
        L = int(lengths[i])
        s0 = int(starts[i])
        q = ref[s0:s0 + L, 3:7].float().numpy().astype(np.float64)
        j, e = match_clip(idx[s], L, q)
        if j < 0 or e > 5e-3:
            unmatched.append((nm, L, e))
            continue
        tasks.append({"name": nm, "parquet": idx[s][j]["path"], "length": L,
                      "planner": str(a.planner), "normalize": not a.no_normalize})
    print(f"[match] 匹配 {len(tasks)}/{n_clip}，未匹配 {len(unmatched)}")
    for u in unmatched[:6]:
        print(f"   [miss] {u[0][-32:]} len={u[1]} err={u[2]:.2e}")
    if not tasks:
        raise SystemExit("[fail] 没有任何 clip 匹配上")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(_job, tasks), 1):
            results.append(r)
            if k % 20 == 0 or k == len(tasks):
                print(f"  [{k}/{len(tasks)}] {time.time() - t0:.0f}s  "
                      f"yaw_mae(中位)={np.median([x['yaw_mae_deg'] for x in results]):.1f}°",
                      flush=True)

    # 组装 v2
    ref2 = ref.clone()
    for i, nm in enumerate(names[:n_clip]):
        s0, L = int(starts[i]), int(lengths[i])
        r = next((x for x in results if x["name"] == nm), None)
        if r is None:
            continue
        n = min(L, r["n"])
        # ★ 只填 (x, y)；库里的 z 是**由关节 FK 推出的骨盆高度**，编码"蹲/弯腰"，
        #   planner 的 z 是它自己步态的高度，写进去会把 task4 的弯腰段抹掉。
        ref2[s0:s0 + n, 0:2] = torch.from_numpy(r["pos"][:n, 0:2])
        # 朝向保留**录制的**（真值），只做同一个刚体旋转，保持与 xy 同系
        if r["dyaw"] != 0.0:
            qz = torch.tensor([math.cos(r["dyaw"] / 2), 0.0, 0.0,
                               math.sin(r["dyaw"] / 2)])
            q = ref2[s0:s0 + n, 3:7].double()
            w1, x1, y1, z1 = q.unbind(-1)
            w2, x2, y2, z2 = qz[0], qz[1], qz[2], qz[3]
            ref2[s0:s0 + n, 3:7] = torch.stack([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1).float()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": ref2, "names": lib["names"], "starts": starts, "lengths": lengths}, a.out)
    print(f"\n[out] {a.out}  ({a.out.stat().st_size / 1e6:.0f} MB)")

    mae = np.array([x["yaw_mae_deg"] for x in results])
    trav = np.array([x["travel_cm"] for x in results])
    path = np.array([x["path_cm"] for x in results])
    fill = np.array([x["fill"] for x in results])
    print(f"[验收] n={len(results)}  yaw MAE: p50={np.median(mae):.1f}° p90={np.quantile(mae, .9):.1f}° "
          f"max={mae.max():.1f}°   corr 未计入")
    print(f"       覆盖率 p50={np.median(fill) * 100:.0f}% min={fill.min() * 100:.0f}%")
    print(f"       净位移 cm: p10={np.quantile(trav, .1):.0f} p50={np.median(trav):.0f} "
          f"p90={np.quantile(trav, .9):.0f} max={trav.max():.0f}")
    print(f"       路径长 cm: p50={np.median(path):.0f} p90={np.quantile(path, .9):.0f} "
          f"max={path.max():.0f}")
    still0 = int(((ref2[:, 0].abs() < 1e-6) & (ref2[:, 1].abs() < 1e-6)).sum())
    print(f"       v2 里 x=y=0 的帧: {still0}/{ref.shape[0]}（v1 是全部）")
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(
            {"n": len(results), "unmatched": [u[0] for u in unmatched],
             "yaw_mae_p50": float(np.median(mae)), "yaw_mae_p90": float(np.quantile(mae, .9)),
             "travel_p50_cm": float(np.median(trav)), "path_p50_cm": float(np.median(path)),
             "fill_p50": float(np.median(fill)),
             "per_clip": [{k2: v for k2, v in x.items() if k2 not in ("pos", "quat")}
                          for x in results]}, indent=1, ensure_ascii=False))
        print(f"[report] {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
