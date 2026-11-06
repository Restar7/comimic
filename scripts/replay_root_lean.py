#!/usr/bin/env python3
"""把录制的**关节目标**灌回 MuJoCo，让物理自己走出根轨迹（= 参考的 x/y）。

为什么（2026-10-30 判别实验）：同一批窗口里
    指令积分 694-934cm / planner 原始 112-317cm / 物理重放 133-207cm
中位 重放/planner = 0.65。而参考里存的**关节**是录制真值 ⇒ 参考必须和这些关节自洽，
否则"关节全部跟住"也只能拿到 0.65（奖励天花板），策略永远被判"走不够"。
重放 = 同一批关节在本物理里的必然结果 ⇒ 自洽、可达。

用法（每进程处理一个分片）：
    python3 scripts/replay_root_lean.py --list runs/replay_root/list.json --shard 0 --nshard 32 \
        --out runs/replay_root/root_v9.npz
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENES = Path("/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/SONIC_DATA/deployment/"
              "gear_sonic/data/robot_model/model_data/g1")
DT_CTRL, SUB = 0.02, 40
BODY = list(range(22)) + list(range(29, 36))     # 43 → 身体 29（= STANDARD_29 / MuJoCo 声明序）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--scene", default="scene_43dof.xml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "groot_repro"))
    import mujoco
    import g1_sim as G

    model = mujoco.MjModel.from_xml_path(str(SCENES / a.scene))
    data = mujoco.MjData(model)
    smap = G.SceneMap(model)
    q0 = G.DEFAULT_ANGLES.copy()
    items = json.load(open(a.list))
    mine = [x for i, x in enumerate(items) if i % a.nshard == a.shard]
    print(f"[shard {a.shard}/{a.nshard}] {len(mine)} clips", flush=True)

    out = {}
    import pyarrow.parquet as pq
    for it in mine:
        ci, parq = it["i"], it["parquet"]
        t = pq.ParquetFile(parq).read(columns=["action.wbc", "observation.root_orientation"])
        A = np.array(t.column("action.wbc").to_pylist(), np.float64)
        RQ = np.array(t.column("observation.root_orientation").to_pylist(), np.float64)
        T = len(A) if a.max_frames <= 0 else min(len(A), a.max_frames)
        data.qpos[:] = 0.0; data.qvel[:] = 0.0
        data.qpos[0:3] = (0.0, 0.0, 0.75); data.qpos[3:7] = RQ[0]
        data.qpos[smap.body_q] = G.DEFAULT_ANGLES
        mujoco.mj_forward(model, data)
        P = np.zeros((T, 3), np.float32); Q = np.zeros((T, 4), np.float32)
        for k in range(T):
            P[k] = data.qpos[0:3]; Q[k] = data.qpos[3:7]
            qt = A[k, BODY]
            for _ in range(SUB):
                tau = G.KP * (qt - smap.q(data)) - G.KD * smap.dq(data)
                smap.set_ctrl(data, np.clip(tau, -88.0, 88.0))
                mujoco.mj_step(model, data)
        out[ci] = {"pos": P, "quat": Q}
        print(f"  clip{ci} {T} 帧 净位移={np.linalg.norm(P[-1,:2]-P[0,:2])*100:.0f}cm", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **{f"p{k}": v["pos"] for k, v in out.items()},
                        **{f"q{k}": v["quat"] for k, v in out.items()})
    print(f"[out] {a.out} ({len(out)} clips)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
