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
    ap.add_argument("--init-history", default="recorded",
                    choices=("recorded", "provider", "zeros", "legacy"))
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
    # 起始 root：**自动贴地**。录制里的 root z（Isaac 的 HA 资产）和这个 MuJoCo 模型差
    # 3~5 cm；直接用录制的 0.80 会让机器人先自由落体一次，落地冲击足以毁掉闭环。
    SM = G.SceneMap(m)                    # 按名字取索引：模型里手部关节/执行器插在中间，
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]    # 用 qpos[7:36] 会把左手指当成右臂
    SM.set_q(d, q_mj0[0])
    SM.set_hands(d, np.zeros(len(G.HAND_JOINT_NAMES)))
    # 实测：该模型在录制首帧姿态下，root z ≈ 0.752 才刚接触地面（0.74 已穿地 1.7 cm，
    # 0.76 还悬空）。这里用实测值，避免“先自由落体一次”。
    z0 = 0.752
    d.qpos[0:3] = [0.0, 0.0, z0]
    mujoco.mj_forward(m, d)
    print(f"[init] root z = {z0:.4f} (录制 z={z['robot_root_position'][0,2]:.3f}, ncon={d.ncon})")

    ctrl = SonicController(str(args.decoder), init_q=q_mj0[0])

    # ---- 历史缓冲初始化：这是“上层喂给下层”的第一个接口 -------------------------
    # recorded : 用录制第 0 帧真实的 10 帧历史（ang_vel / q-delta / dq / last_action / gravity）
    # provider : 复刻 Isaac provider `_setup_buffers()` 的初值
    #            q-delta=0, dq=0, ang_vel=0, last_action=DEFAULT_ANGLES(错), gravity=0(错)
    # zeros    : last_action 与 gravity 全零（更激进的反例）
    if args.init_history in ("recorded", "provider", "zeros"):
        if args.init_history == "recorded":
            h_av = np.asarray(z["decoder_ang_vel_hist"][0], np.float64)[::-1]
            h_qp = np.asarray(z["encoder_robot_joint_pos_hist"][0], np.float64)[::-1]
            h_qv = np.asarray(z["encoder_robot_joint_vel_hist"][0], np.float64)[::-1]
            h_la = np.asarray(z["decoder_last_action_hist"][0], np.float64)[::-1]
            h_gv = np.asarray(z["decoder_gravity_dir_hist"][0], np.float64)[::-1]
            assert h_av.shape == (10, 3) and h_qp.shape == (10, 29), (h_av.shape, h_qp.shape)
        elif args.init_history == "provider":
            h_av = np.zeros((10, 3))
            h_qp = np.zeros((10, 29))
            h_qv = np.zeros((10, 29))
            h_la = np.tile(G.DEFAULT_ANGLES, (10, 1))     # ← Isaac provider 的初值
            h_gv = np.zeros((10, 3))                      # ← Isaac provider 的初值
        else:
            h_av = np.zeros((10, 3))
            h_qp = np.zeros((10, 29))
            h_qv = np.zeros((10, 29))
            h_la = np.zeros((10, 29))
            h_gv = np.tile(np.array([0.0, 0.0, -1.0]), (10, 1))
        ctrl.hist.clear()
        for i in range(10):
            ctrl.hist.append({"base_ang_vel": h_av[i], "q": h_qp[i], "dq": h_qv[i],
                              "last_action": h_la[i], "gravity": h_gv[i]})
        print(f"[init] history seeded: {args.init_history}")
    else:
        print("[init] SonicController default (absolute-angle history)")

    sub = int(round(CONTROL_DT / m.opt.timestep))
    n = int(min(args.steps, len(tokens)))

    rec = {k: [] for k in ("t", "token", "obs994", "action", "q_target", "q", "dq",
                           "root_pos", "root_quat", "tilt", "torque", "sat")}
    print(f"[mode] {args.mode}  steps={n}  sub={sub}  dt={m.opt.timestep}")
    for tick in range(n):
        q = SM.q(d).copy()
        dq = SM.dq(d).copy()
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
            SM.set_ctrl(d, tau_c)
            SM.set_hand_ctrl(d, np.zeros(len(G.HAND_JOINT_NAMES)))
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
