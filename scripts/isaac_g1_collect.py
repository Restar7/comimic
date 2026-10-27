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
parser.add_argument("--log-every", type=int, default=25)
args_cli = parser.parse_args()

app = AppLauncher(args_cli).app
print("[collect] app ready", flush=True)

import torch  # noqa: E402
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
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.80)),
        actuators={},
    )


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
    cfg = SceneCfg(num_envs=int(args_cli.num_envs), env_spacing=2.5)
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

    q0 = torch.zeros(n, robot.num_joints, device=DEV)
    q0[:, idx_t] = default_q.unsqueeze(0)
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    scene.update(sim.get_physics_dt())

    last_a = torch.zeros(n, nj, device=DEV)
    hist = torch.zeros(n, HIST, 93, device=DEV)
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
        F.append(torch.full((n,), int(args_cli.start_frame) + step, dtype=torch.long))
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
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
