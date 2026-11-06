#!/usr/bin/env python3
"""最小 G1 采集环（A 线）：冻结 WBC 在 Isaac 真物理里走，存 (s, a, s', frame)。

**本脚本刻意不 import 我们训练栈的任何模块**（diff_vla_wbc / trackerLab MDP 会连锁依赖
MuJoCo 常量，在 Isaac venv 里炸）。这里只需要：
  * trackerLab.algorithms.supermimic.networks.Policy（纯 torch，已在 Isaac venv 验证可用）
  * ckpt 里的 pi 权重 + robot_spec（joint_names / default_q / action_scale）
奖励**不在这里算**：它只依赖 (s, a, prev_a, frame)，全部落盘后由
scripts/compute_collect_reward.py 在系统 python 里用**训练同一个 bundle.reward** 补算。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"),
)
for _p in (ROOT / "trackerlab-supermimic/source/trackerTask",
           ROOT / "trackerlab-supermimic/source/trackerLab"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--token-npy", type=Path, required=True)
parser.add_argument("--clip", type=str, required=True)
parser.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--exec-steps", type=int, default=20)
parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/onpolicy.pt")
parser.add_argument("--warmup", type=int, default=20, help="记录前的 warm-up 步数（让观测历史自洽）")
parser.add_argument("--log-every", type=int, default=25)
parser.add_argument("--action-noise-std", type=float, default=0.05,
                    help="[旧口径] 动作空间里的**绝对** σ。只在 --action-noise-rad=0 时生效。"
                         "问题：|a| 是重尾（最大 18.6），同一绝对 σ 对不同关节的相对强度差很多。")
parser.add_argument("--action-noise-rad", type=float, default=0.0,
                    help="★[推荐] 以**关节角弧度**给噪声：δ_j = N(0,(rad/scale_j)²)。"
                         "这样扰动的「角度」对所有关节一致。0.014 ≈ 旧口径的中位等效（0.05×0.275）。"
                         ">0 时忽略 --action-noise-std。")
parser.add_argument("--noise-mode", choices=("gauss", "uniform"), default="gauss")
parser.add_argument("--perturb-hold", type=int, default=20)
parser.add_argument("--perturb-hold-jitter", type=int, default=5,
                    help="hold 抖动（步）。原来 hold==exec_steps==20 ⇒ 噪声永远落在 chunk 边界，"
                         "WM 会学到\"扰动只出现在 t≡0 (mod 20)\"的伪相关。5 ⇒ hold∈{15..25}。")
parser.add_argument("--action-clip", type=float, default=0.0,
                    help="动作截断（0=不截断，与训练口径一致；该 WBC 训练时 action_limit=None）")
parser.add_argument("--start-perturb-xy", type=float, default=0.10)
parser.add_argument("--start-perturb-yaw", type=float, default=0.10)
parser.add_argument("--start-perturb-dq", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--packed", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
args_cli = parser.parse_args()

app = AppLauncher(args_cli).app
print("[collect] app ready", flush=True)

import os  # noqa: E402
import torch  # noqa: E402
torch.manual_seed(int(args_cli.seed))
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from trackerLab.algorithms.supermimic.networks import Policy  # noqa: E402

DEV = torch.device(getattr(args_cli, "device", None) or "cuda:0")
USD = os.environ["SUPERMIMIC_ROBOT_USD"]
HIST = 10


@configclass
class SceneCfg(InteractiveSceneCfg):
    # InteractiveScene 只认 AssetBaseCfg 风格的条目，裸 GroundPlaneCfg 会报 Unknown asset config
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    # robot 在 main() 里按 ckpt 的 robot_spec 动态构造（PD 增益必须显式给，见下）
    robot: ArticulationCfg = None


def main() -> int:
    ck = torch.load(str(args_cli.wbc_ckpt), map_location="cpu", weights_only=False)
    rs = ck["robot_spec"]
    joint_names = list(rs["joint_names"])
    default_q = torch.tensor(rs["default_q"], dtype=torch.float32, device=DEV)
    scale = torch.tensor(rs["action_scale"], dtype=torch.float32, device=DEV)
    nj = len(joint_names)
    pol = Policy(obs_dim=93, goal_dim=0, hist=HIST, token_dim=64, num_joints=nj,
                 action_limit=None)
    miss, unexp = pol.load_state_dict(ck["pi"], strict=False)
    if miss or unexp:
        raise SystemExit(f"[collect] policy 权重不匹配 missing={miss[:4]} unexpected={unexp[:4]}")
    pol = pol.to(DEV).eval()
    print(f"[collect] WBC loaded: joints={nj}", flush=True)

    toks = np.load(str(args_cli.token_npy)).astype(np.float32)
    if toks.ndim != 2 or toks.shape[1] != 64:
        raise SystemExit(f"[collect] token 流形状异常 {toks.shape}")
    print(f"[collect] tokens={toks.shape} clip={args_cli.clip}", flush=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 200.0, device=str(DEV)))
    # InteractiveSceneCfg 的 num_envs/env_spacing 是 MISSING 必填字段，必须在实例化时给
    # ★ 执行器 PD 增益必须显式设：USD 自带的 drive 增益远低于训练时的 kp/kd，
    #   不设的话关节几乎不动（实测根位移只有录制的 0.15×）。数值来自 ckpt 的 robot_spec。
    from isaaclab.actuators import ImplicitActuatorCfg  # noqa: PLC0415

    kp = torch.as_tensor(rs["kp"], dtype=torch.float32)
    kd = torch.as_tensor(rs["kd"], dtype=torch.float32)
    ef = torch.as_tensor(rs.get("effort_limit", [0.0] * nj), dtype=torch.float32)
    acts = {
        f"j{i}": ImplicitActuatorCfg(
            joint_names_expr=[joint_names[i]],
            stiffness=float(kp[i]), damping=float(kd[i]),
            effort_limit=float(ef[i]) if float(ef[i]) > 0 else None,
        )
        for i in range(nj)
    }
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.80)),
        actuators=acts,
    )
    cfg = SceneCfg(num_envs=int(args_cli.num_envs), env_spacing=2.5)
    cfg.robot = robot_cfg
    print(f"[collect] 执行器增益: kp mean={float(kp.mean()):.1f} kd mean={float(kd.mean()):.2f} "
          f"effort mean={float(ef.mean()):.1f}", flush=True)
    scene = InteractiveScene(cfg)
    robot: Articulation = scene["robot"]
    sim.reset()
    scene.update(sim.get_physics_dt())
    n = int(robot.num_instances)

    isaac_names = list(robot.joint_names)
    missing = [j for j in joint_names if j not in isaac_names]
    if missing:
        print(f"[collect] USD 缺关节 {missing[:5]}；USD joints[:8]={isaac_names[:8]}", flush=True)
        return 3
    idx = [isaac_names.index(j) for j in joint_names]
    idx_t = torch.tensor(idx, dtype=torch.long, device=DEV)
    print(f"[collect] scene ready num_envs={n} 关节映射 OK（前 5: "
          f"{[(joint_names[i], idx[i]) for i in range(5)]}）", flush=True)

    # ★ 根位姿必须先对齐**参考帧**：奖励里的 e_root_pos/e_root_ori 是绝对位姿差，
    #   如果仿真从 (0,0,0.8) 起步而参考在别处，奖励会被一个巨大常数偏置吞掉（实测 e_root_pos≈20）。
    ref = None
    try:
        # 2026-11-02：新 A' 必须用执行参考库（ha_cmd_exec），不是旧的 ha_cmd
        packed = torch.load(str(args_cli.packed), map_location="cpu",
                         weights_only=False)
        ref = torch.as_tensor(packed["ref"]).float()
        starts = torch.as_tensor(packed["starts"]).long()
        names = list(packed["names"])
        if args_cli.clip in names:
            ci = names.index(args_cli.clip)
        else:
            # npz 的 stem（ep000000）与库里的原始名（ep<timestamp>）不一致：按**同序号**回退
            stems = [q.stem for q in sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))]
            if args_cli.clip not in stems:
                raise KeyError(f"{args_cli.clip} 既不在库里也不在 npz 里")
            ci = stems.index(args_cli.clip)
            print(f"[collect] clip 名按同序号回退: {args_cli.clip} -> {names[ci]}", flush=True)
        row = ref[int(starts[ci]) + int(args_cli.start_frame)]
        root_p = row[:3]
        # canonical 的 quat 是 wxyz（与 flat 布局一致），Isaac 要 wxyz
        root_q = row[3:7]
        pose = torch.zeros(n, 7, device=DEV)
        pose[:, :3] = root_p.to(DEV)
        pose[:, 3:] = root_q.to(DEV)
        robot.write_root_pose_to_sim(pose)
        print(f"[collect] 根位姿对齐参考帧: p={root_p.tolist()} q={root_q.tolist()}", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[collect] ⚠️ 根位姿对齐失败（奖励的 root 项会偏）: {_e}", flush=True)

    # 2026-11-03：frames 必须写**全局 packed 行**（starts[ci]+local），
    # 否则 compute_collect_reward 对 starts>0 的 clip 会查错参考（局部帧从 0 开始）。
    _frame_offset = 0
    try:
        _frame_offset = int(starts[ci])
    except Exception:  # noqa: BLE001
        _frame_offset = 0

    # ★ 关节初值必须取**参考帧的关节角**（不是 default_q）：片段第 0 帧往往离默认姿态很远，
    #   否则机器人一出生就在"追赶"，全程滞后（实测根位移只有录制的 1/3）。
    q_init_ref = None
    if ref is not None:
        row0 = ref[int(starts[ci]) + int(args_cli.start_frame)]
        q_init_ref = row0[13:13 + nj].to(DEV)
    q0 = torch.zeros(n, robot.num_joints, device=DEV)
    q0[:, idx_t] = (q_init_ref if q_init_ref is not None else default_q).unsqueeze(0)
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    scene.update(sim.get_physics_dt())
    print(f"[collect] 关节初值 = {'参考帧' if q_init_ref is not None else 'default_q'}", flush=True)

    # ---- L2 起点扰动（2026-11-02）----
    if max(args_cli.start_perturb_xy, args_cli.start_perturb_yaw, args_cli.start_perturb_dq) > 0:
        _xy = float(args_cli.start_perturb_xy); _yaw = float(args_cli.start_perturb_yaw); _dq = float(args_cli.start_perturb_dq)
        _rp = robot.data.root_pos_w.clone()
        _rp[:, :2] += torch.randn(n, 2, device=DEV) * _xy
        _rq = robot.data.root_quat_w.clone()
        if _yaw > 0:
            _a = torch.randn(n, device=DEV) * _yaw
            _c, _s = torch.cos(_a / 2), torch.sin(_a / 2)
            _w, _x, _y, _z = _rq.unbind(-1)
            _rq = torch.stack([_c * _w - _s * _z, _c * _x - _s * _y, _c * _y + _s * _x, _c * _z + _s * _w], dim=-1)
        robot.write_root_pose_to_sim(torch.cat([_rp, _rq], dim=-1))
        if _dq > 0:
            _q2 = robot.data.joint_pos.clone()
            _q2[:, idx_t] += torch.randn(n, len(idx_t), device=DEV) * _dq
            robot.write_joint_state_to_sim(_q2, torch.zeros_like(_q2))
        scene.update(sim.get_physics_dt())
        print(f"[collect] L2 start perturb: xy={_xy} yaw={_yaw} dq={_dq}", flush=True)

    # ★ warm-up：让 WBC 的 10 帧观测历史与 token 自洽后再开始记录（否则前 ~10 步是垃圾输入）
    warm = int(args_cli.warmup)
    if warm > 0:
        last_a = torch.zeros(n, nj, device=DEV)
        hist = torch.zeros(n, HIST, 93, device=DEV)
        for wstep in range(warm):
            row = toks[min(int(args_cli.start_frame) + wstep, toks.shape[0] - 1)]
            tok_w = torch.as_tensor(row, device=DEV, dtype=torch.float32).unsqueeze(0).expand(n, -1)
            q_w = robot.data.joint_pos[:, idx_t]
            dq_w = robot.data.joint_vel[:, idx_t]
            obs_w = torch.cat((q_w - default_q, dq_w, robot.data.root_ang_vel_b,
                               robot.data.projected_gravity_b, last_a), dim=-1)
            hist = torch.cat((hist[:, 1:], obs_w.unsqueeze(1)), dim=1)
            with torch.no_grad():
                a_w = pol(hist, tok_w)
            robot.set_joint_position_target(
                default_q.unsqueeze(0) + a_w * scale.unsqueeze(0),
                joint_ids=[int(i) for i in idx])
            for _ in range(4):
                sim.step()
            scene.update(sim.get_physics_dt())
            last_a = a_w.detach()
        print(f"[collect] warm-up {warm} 步完成", flush=True)

    last_a = torch.zeros(n, nj, device=DEV)
    hist = torch.zeros(n, HIST, 93, device=DEV)
    _delta = None
    _next_hold = 0                     # 下一次重采噪声的**绝对步号**（不再用 step % hold）
    _printed_mode = False
    S, A, S2, F = [], [], [], []
    t0 = time.time()
    for step in range(int(args_cli.steps)):
        if step % int(args_cli.exec_steps) == 0:
            lo = int(args_cli.start_frame) + step
            row = toks[min(lo, toks.shape[0] - 1)]
            tok = torch.as_tensor(row, device=DEV, dtype=torch.float32).unsqueeze(0).expand(n, -1)
        q = robot.data.joint_pos[:, idx_t]
        dq = robot.data.joint_vel[:, idx_t]
        obs = torch.cat((q - default_q, dq, robot.data.root_ang_vel_b,
                         robot.data.projected_gravity_b, last_a), dim=-1)
        hist = torch.cat((hist[:, 1:], obs.unsqueeze(1)), dim=1)
        state = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w, q,
                           robot.data.root_lin_vel_w, robot.data.root_ang_vel_w, dq), dim=-1)
        with torch.no_grad():
            a = pol(hist, tok)
        # ---- L1 动作噪声（2026-11-04 改形状：逐关节尺度 + hold 抖动）----
        _sig_rad = float(args_cli.action_noise_rad)
        _sig_abs = float(args_cli.action_noise_std)
        if _sig_rad > 0.0 or _sig_abs > 0.0:
            if _delta is None or step >= _next_hold:
                # ★ 逐关节尺度：以**关节角弧度**给噪声，再除以 action_scale 换到动作空间。
                #   为什么：action_scale 实测 0.167~0.750（4.5×），旧的绝对 σ=0.05 在最大关节上
                #   相当于 2.15°、在最小关节上只有 0.48° —— 同一个参数对不同关节意思不同。
                if _sig_rad > 0.0:
                    _sig = torch.full((nj,), _sig_rad, device=DEV) / scale.to(DEV).clamp_min(1e-6)
                    _mode = f"rad={_sig_rad:g}（逐关节，等效动作 σ∈[{( _sig_rad/scale.max()):.3f},"
                    _mode += f"{( _sig_rad/scale.min()):.3f}]）"
                else:
                    _sig = torch.full((nj,), _sig_abs, device=DEV)
                    _mode = f"abs={_sig_abs:g}（旧口径，绝对动作 σ）"
                if args_cli.noise_mode == "gauss":
                    _delta = torch.randn_like(a) * _sig.unsqueeze(0)
                else:
                    _delta = (torch.rand_like(a) * 2.0 - 1.0) * _sig.unsqueeze(0)
                # ② hold 抖动：不再与 exec_steps 同相
                _base_h = max(1, int(args_cli.perturb_hold))
                _jit = max(0, int(args_cli.perturb_hold_jitter))
                _h = _base_h + (int(torch.randint(-_jit, _jit + 1, (1,)).item()) if _jit > 0 else 0)
                _next_hold = step + max(1, _h)
                if not _printed_mode:
                    print(f"[collect] L1 noise mode = {_mode}  hold={_base_h}±{_jit}", flush=True)
                    _printed_mode = True
            a = a + _delta
            # ★ 2026-11-03 修：原来这里是 `.clamp(-1, 1)`，是**致命 bug**。这个 WBC 训练时
            #   `action_limit=None`（动作无界），实测 |a| 最大 18.6、34% 的分量超过 1.0；
            #   一旦 clamp 到 ±1，动作被砍掉大半 → 机器人 40 步内必倒。
            #   证据：把 σ 从 0.05 降到 1e-6，倒伏比例完全不变（69% vs 70%），说明跟噪声无关。
            #   只有显式给 --action-clip 才截断。
            if float(args_cli.action_clip) > 0:
                a = a.clamp(-float(args_cli.action_clip), float(args_cli.action_clip))
        robot.set_joint_position_target(
            default_q.unsqueeze(0) + a * scale.unsqueeze(0),
            joint_ids=[int(i) for i in idx])
        for _ in range(4):
            sim.step()
        scene.update(sim.get_physics_dt())
        nxt = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w,
                         robot.data.joint_pos[:, idx_t], robot.data.root_lin_vel_w,
                         robot.data.root_ang_vel_w, robot.data.joint_vel[:, idx_t]), dim=-1)
        S.append(state.cpu()); A.append(a.cpu()); S2.append(nxt.cpu())
        F.append(torch.full((n,), _frame_offset + int(args_cli.start_frame) + step, dtype=torch.long))
        last_a = a.detach()
        if step % int(args_cli.log_every) == 0:
            up = 1 - 2 * (robot.data.root_quat_w[:, 1] ** 2 + robot.data.root_quat_w[:, 2] ** 2)
            print(f"[collect] {step}/{args_cli.steps} "
                  f"z={float(robot.data.root_pos_w[:, 2].mean()):.3f} "
                  f"|a|={float(a.abs().mean()):.3f} up={float(up.mean()):.3f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    out = {
        "states": torch.cat(S), "actions": torch.cat(A), "next_states": torch.cat(S2),
        "frames": torch.cat(F), "clip": args_cli.clip, "num_envs": n,
        "steps": int(args_cli.steps), "token_npy": str(args_cli.token_npy),
        "meta": {"layout": "[p(3),quat_wxyz(4),q(29),v(3),w(3),qd(29)]",
                 "joint_names": joint_names, "exec_steps": int(args_cli.exec_steps),
                 "wbc_ckpt": str(args_cli.wbc_ckpt),
                 "note": "奖励未计算，见 scripts/compute_collect_reward.py"},
    }
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(args_cli.out))
    print(f"[collect] saved {args_cli.out}  n={out['states'].shape[0]} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    # 2026-11-03：Isaac 的 app.close() 在本容器偶发挂起（数据已存但进程不退）。
    # 数据已落盘，直接 _exit；跳过 close 的清理。
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
