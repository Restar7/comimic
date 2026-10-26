#!/usr/bin/env python3
"""从 HA 录制数据造**真实转移** (s_t, a_t, s_{t+1})，给 WM 保鲜用（不需要仿真器）。

为什么：冻结的 WM 是在旧策略分布上训的；token 一漂，想象就进外推区。
SuperMimic 每个 iter 用 replay buffer 的真仿真转移重训 WM；我们没有仿真器，但 HA 录制里就有真转移：

    s_t     = [pelvis_pos, pelvis_quat(wxyz), joint_pos,                        ← 执行轨迹（npz）
               root_lin_vel_w, root_ang_vel_w, joint_vel]                       ← 执行轨迹（npz）
    a_t     = ( q_cmd(t+1) − default_q ) / action_scale                          ← 命令流（ha_cmd 库）
    s_{t+1} = s_{t+1}

`a_t` 是"完美跟踪者会输出的动作"（WBC 的动作语义就是 q_target=default_q+a*scale），
所以 (s_t, a_t, s_{t+1}) 就是"部署时执行器给这个动作，机器人实际怎么动"的真实样本（含跟踪滞后）。

输出：runs/wm_real/ha_transitions.pt
    states (N,71) / actions (N,29) / next_states (N,71) / clip_ids (N,) / frames (N,)

    python3 scripts/build_real_transitions.py
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"),
)
sys.path.insert(0, str(ROOT))
# trackerTask（robot spec）在 trackerlab-supermimic/source 下
for _p in (ROOT / "trackerlab-supermimic/source/trackerTask",
           ROOT / "trackerlab-supermimic/source/trackerLab"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/ha_transitions.pt")
    a = ap.parse_args()

    from trackerTask.supermimic.robots import get_robot

    spec = get_robot("g1")
    dq = torch.tensor(spec.default_q, dtype=torch.float32)
    sc = torch.tensor(spec.action_scale, dtype=torch.float32)
    print(f"[real] spec HA_ALIGN={os.environ.get('SUPERMIMIC_HA_ALIGN')} "
          f"default_q[3]={dq[3]:.3f} scale mean={sc.mean():.4f}", flush=True)

    packed = torch.load(str(a.packed), map_location="cpu", weights_only=False)
    cmd_q = packed["ref"][:, 13:13 + 29].float()          # 命令流关节位置
    names = list(packed["names"])
    lengths = [int(x) for x in packed["lengths"]]
    starts = [int(x) for x in packed["starts"]]
    cmd_q = (cmd_q - dq) / sc                              # 全库动作（按帧）

    files = sorted(a.npz_dir.glob("*.npz"))
    if len(files) != len(names):
        raise SystemExit(f"npz 数 {len(files)} != 库 clip 数 {len(names)}")

    states, actions, nexts, clip_ids, frames_idx = [], [], [], [], []
    t0 = time.time()
    for ci, p in enumerate(files):
        with np.load(p) as z:
            pos = np.asarray(z["body_pos_w"][:, 0], np.float32)      # (T,3)
            quat = np.asarray(z["body_quat_w"][:, 0], np.float32)    # (T,4) wxyz
            lin = np.asarray(z["body_lin_vel_w"][:, 0], np.float32)
            ang = np.asarray(z["body_ang_vel_w"][:, 0], np.float32)
            q = np.asarray(z["joint_pos"], np.float32)
            qd = np.asarray(z["joint_vel"], np.float32)
        T = min(len(pos), len(q), lengths[ci])
        # !! 录制的 root 线/角速度全是 0（实测 std=0），必须用差分补 —— 否则 WM 会把速度
        #    通道拟合成常数，归一化时 1/std^2 还会炸到 1e12。
        dt = 1.0 / float(z["fps"]) if "fps" in z else 0.02
        lin = np.gradient(pos[:T].astype(np.float64), dt, axis=0).astype(np.float32)
        q = q.astype(np.float32)
        _qa, _qb = quat[:-1].astype(np.float64), quat[1:].astype(np.float64)
        def _qmul(x, y):
            x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            y0, y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
            return np.stack([x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
                             x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
                             x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
                             x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0], -1)
        _qinv = np.concatenate([_qb[:, :1], -_qb[:, 1:]], 1)
        _dq = _qmul(_qa, _qinv)
        _dq = _dq / (np.linalg.norm(_dq, axis=-1, keepdims=True) + 1e-9)
        _dq = np.where(_dq[:, :1] < 0, -_dq, _dq)
        ang = (2.0 * _dq[:, 1:] / dt).astype(np.float32)
        ang = np.concatenate([ang, ang[-1:]], 0)
        s = np.concatenate([pos[:T], quat[:T], q[:T], lin[:T], ang[:T], qd[:T]], 1)  # (T,71)
        st = starts[ci]
        act = cmd_q[st + 1 : st + T + 1]                          # a_t 用 t+1 的命令（q_target 语义）
        if act.shape[0] < T:
            act = torch.cat([act, act[-1:].expand(T - act.shape[0], -1)], 0)
        n = T - 1
        states.append(torch.from_numpy(s[:n]))
        nexts.append(torch.from_numpy(s[1:n + 1]))
        actions.append(act[:n])
        clip_ids.append(torch.full((n,), ci, dtype=torch.int32))
        frames_idx.append(torch.arange(n, dtype=torch.int64) + st)
        if (ci + 1) % 100 == 0:
            print(f"  [{ci+1}/{len(files)}] {time.time()-t0:.0f}s", flush=True)

    out = {
        "states": torch.cat(states), "actions": torch.cat(actions),
        "next_states": torch.cat(nexts), "clip_ids": torch.cat(clip_ids),
        "frames": torch.cat(frames_idx),
        "default_q": dq, "action_scale": sc,
        "meta": {"source": str(a.npz_dir), "cmd": str(a.packed),
                 "note": "s=executed(q from npz, root vel from npz); a=(cmd_q(t+1)-dq)/scale"},
    }
    _st = out["states"]
    out["scales"] = {
        "v": _st[:, 36:39].std(0).clamp_min(1e-2),
        "w": _st[:, 39:42].std(0).clamp_min(1e-2),
        "qd": _st[:, 42:71].std(0).clamp_min(1e-2),
        "p": _st[:, 0:3].std(0).clamp_min(1e-2),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    n = out["states"].shape[0]
    print(f"[real] {n} transitions -> {a.out}  ({a.out.stat().st_size/1e6:.0f} MB)", flush=True)
    # 健康检查：一步位移 / 动作幅值
    print(f"[real] |dp| mean={float((out['next_states'][:, :3]-out['states'][:, :3]).norm(dim=-1).mean()):.5f} m "
          f"| |a| mean={float(out['actions'].abs().mean()):.4f} "
          f"| |dq| mean={float((out['next_states'][:, 7:36]-out['states'][:, 7:36]).abs().mean()):.5f} rad",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
