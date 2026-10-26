#!/usr/bin/env python3
"""MuJoCo 闭环实验台：用官方 SONIC decoder + 官方 PD 常数跑闭环，逐帧转储。

为什么不用 Isaac：本节点（新 pod）Isaac 的 RTX/Vulkan 枚举不到 GPU，起不来；而这条
MuJoCo 链（`groot_repro/g1_sim.py` + `sonic_loop.py`）的常数是从
`policy_parameters.hpp` 逐项抄的（ACTION_SCALE / KP / KD / DEFAULT_ANGLES / 关节映射），
和被评测的 Isaac provider 用的是同一套执行器语义，且能 100% 逐帧打点。

四个实验臂（`--mode`）：
  gt_tokens        录制 token + 闭环 decoder（= 同一条链，token 用真值）
  zeros_tokens     zero token 对照（decoder 收到的“静默”输入）
  recorded_target 直接把录制里的 29 维 joint target 灌进去（跳过 decoder，测 plant+PD）
  bias0.09         GT token + 常数偏移（复现“token 没做 codec 反归一化”的偏差）

用法（要 isaac venv 的 python，它有 mujoco + onnxruntime）：
  PY=third_party/isaacsim_env/bin/python3
  $PY scripts/dbg_mujoco_sonic_loop.py --mode gt_tokens --steps 300 \
      --npz data/ha_raw/HOI_double_desk/tw/xxx.npz --out runs/dbg_mujoco/gt.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROOT = ROOT / "groot_repro"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GROOT))

import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402

import g1_sim as G  # noqa: E402
from sonic_loop import SonicController, CONTROL_DT  # noqa: E402

SCENE = Path("/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/"
             "GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
DECODER = Path("/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/"
               "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")


def _tilt_deg(quat_wxyz: np.ndarray) -> float:
    r = G.quat_to_rotmat(np.asarray(quat_wxyz, np.float64))
    return float(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gt_tokens",
                    choices=("gt_tokens", "zeros_tokens", "recorded_target", "bias"))
    ap.add_argument("--bias", type=float, default=0.088, help="mode=bias 时的 token 常数偏移")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--decoder", type=Path, default=DECODER)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    tokens = np.asarray(z["encoder_latent"], np.float64)               # (T,64) SONIC 原始空间
    tgt_sonic = np.asarray(z["decoder_target_action"], np.float64)     # (T,29) SONIC(=HA_V31) 序
    q_ha = np.asarray(z["robot_qpos_before_decimation"], np.float64)   # (T,29) HA_V31 序
    rec_z = np.asarray(z["robot_root_position"], np.float64)[:, 2]

    import joint_contract as JC
    q_mj0 = np.stack([JC.to_standard(q, src="ha_v31") for q in q_ha])  # MuJoCo/STANDARD 序

    m = mujoco.MjModel.from_xml_path(str(args.scene))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    # 起始 root 用录制高度；贴在原点，避免场景里和其它物体初始重叠
    d.qpos[0:3] = [0.0, 0.0, 0.80]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:36] = q_mj0[0]
    d.qpos[36:50] = 0.0
    mujoco.mj_forward(m, d)

    ctrl = SonicController(str(args.decoder), init_q=q_mj0[0])
    sub = int(round(CONTROL_DT / m.opt.timestep))
    n = int(min(args.steps, len(tokens)))

    rec = {k: [] for k in ("t", "token", "obs994", "action", "q_target", "q", "dq",
                           "root_pos", "root_quat", "tilt", "torque", "sat")}
    print(f"[mode] {args.mode}  steps={n}  sub={sub}  dt={m.opt.timestep}")
    for tick in range(n):
        q = d.qpos[7:36].copy()
        dq = d.qvel[6:35].copy()
        quat = d.qpos[3:7].copy()
        ang = d.qvel[3:6].copy()

        token = np.zeros(64) if args.mode == "zeros_tokens" else tokens[tick].copy()
        if args.mode == "bias":
            token = token + args.bias

        if args.mode == "recorded_target":
            q_target = np.asarray(tgt_sonic[tick], np.float64)[G.ISAACLAB_TO_MUJOCO]
            obs994 = np.zeros(994)
            action = (q_target - G.DEFAULT_ANGLES) / G.ACTION_SCALE
        else:
            action = ctrl.decoder(ctrl.build_obs(token))
            q_target = G.DEFAULT_ANGLES + action[G.ISAACLAB_TO_MUJOCO] * G.ACTION_SCALE
            obs994 = ctrl.build_obs(token)
            ctrl.hist.append({
                "base_ang_vel": np.asarray(ang, float),
                "q": (q - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
                "dq": dq[G.MUJOCO_TO_ISAACLAB],
                "last_action": action,
                "gravity": G.projected_gravity(quat),
            })

        tau = G.KP * (q_target - q) - G.KD * dq
        eff = np.array([G.EFFORT[m_] for m_ in G.MOTOR], np.float64)
        tau_c = np.clip(tau, -eff, eff)
        for _ in range(sub):
            d.ctrl[:29] = tau_c
            d.ctrl[29:] = 0.0
            mujoco.mj_step(m, d)

        rec["t"].append(tick * CONTROL_DT)
        rec["token"].append(token)
        rec["obs994"].append(obs994)
        rec["action"].append(action)
        rec["q_target"].append(q_target)
        rec["q"].append(q.copy())
        rec["dq"].append(dq)
        rec["root_pos"].append(d.qpos[0:3].copy())
        rec["root_quat"].append(d.qpos[3:7].copy())
        rec["tilt"].append(_tilt_deg(d.qpos[3:7]))
        rec["torque"].append(tau)
        rec["sat"].append(float(np.mean(np.abs(tau) >= eff * 0.999)))

        if tick % 50 == 0 or tick == n - 1:
            print(f"  t={tick*CONTROL_DT:5.2f}s z={d.qpos[2]:.3f} "
                  f"tilt={rec['tilt'][-1]:7.1f}deg |tau|max={np.abs(tau).max():7.1f} "
                  f"rec_z={rec_z[tick]:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, mode=args.mode, n=n, rec_z=rec_z[:n],
                        **{k: np.asarray(v) for k, v in rec.items()})
    zz = np.asarray(rec["root_pos"])
    tilt = np.asarray(rec["tilt"])
    fell = zz[:, 2] < 0.45
    print(f"[result] {args.mode}: z start={zz[0,2]:.3f} end={zz[-1,2]:.3f} min={zz[:,2].min():.3f} "
          f"| 录制 z mean={rec_z[:n].mean():.3f} | tilt max={tilt.max():.1f}deg "
          f"| 首次 z<0.45 = {int(np.argmax(fell)) if fell.any() else -1} | saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
