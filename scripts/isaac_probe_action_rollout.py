#!/usr/bin/env python3
"""在**真物理**里执行 job 里的两串动作（±梯度方向），录下每步状态。

配套 `scripts/probe_wm_gradient_truth.py`（plan/score）。跑法（必须用 Isaac venv）：
    source scripts/isaac_env.sh
    $ISAAC_VENV/bin/python -u scripts/isaac_probe_action_rollout.py \
        --job /tmp/job_h2.pt --out /tmp/roll_h2.pt --num-envs 96 --device cuda:0 --headless

约定（与 isaac_bw_pert_collect.py 完全一致，已验证可跑）：
  * 起点状态 = job 里的 `states_ha`（**ha_v31 序**），按 `idx_ha` 写进 isaac 关节
  * 动作 = job 里的 `acts_*`（**standard 序**），`q_target = default_q + a*scale`，
    用 `set_joint_position_target(..., joint_ids=idx_std)` 下发
  * 每个控制步 = 4 个物理子步（dt=1/200 ⇒ 20 ms，与训练口径一致）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "trackerlab-supermimic/source/trackerLab",
          ROOT / "trackerlab-supermimic/source/trackerTask"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_USD = ("third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/"
               "g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=0, help="0=用 job 里的 N")
    ap.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_v6.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--headless", action="store_true", default=True)
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    args_cli, _ = ap.parse_known_args()
    app = AppLauncher(args_cli).app  # noqa: F841

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.actuators import ImplicitActuatorCfg  # noqa: PLC0415
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: PLC0415
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: PLC0415
    from isaaclab.sim import SimulationContext  # noqa: PLC0415
    from isaaclab.utils import configclass  # noqa: PLC0415

    global DEV
    dev = str(args_cli.device)
    if ":" not in dev and torch.cuda.is_available():
        dev += ":0"
    DEV = torch.device(dev)
    if DEV.type == "cuda":
        torch.cuda.set_device(DEV)
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    job = torch.load(str(args_cli.job), map_location="cpu", weights_only=False)
    S0 = job["states_ha"].float()
    AP = job["acts_plus"].float()
    AM = job["acts_minus"].float()
    H = int(job["horizon"])
    N = int(S0.shape[0]) if args_cli.num_envs <= 0 else int(args_cli.num_envs)
    S0, AP, AM = S0[:N], AP[:N], AM[:N]
    log(f"[probe-roll] job={args_cli.job.name} N={N} H={H} DEV={DEV}")

    ck = torch.load(str(args_cli.wbc_ckpt), map_location="cpu", weights_only=False)
    rs = ck["robot_spec"]
    joint_names = list(rs["joint_names"])                    # STANDARD_29
    default_q = torch.tensor(rs["default_q"], dtype=torch.float32, device=DEV)
    scale = torch.tensor(rs["action_scale"], dtype=torch.float32, device=DEV)
    nj = len(joint_names)
    # HA_V31 关节名（与 dump 的 71 维存储序一致）：**job 里带的就是 dump 记录的 HA 序**
    ha_names = list(job.get("joint_names") or joint_names)
    if len(ha_names) != nj:
        raise SystemExit(f"[probe-roll] job joint_names {len(ha_names)} != spec {nj}")
    usd_path = (rs.get("usd") or os.environ.get("SUPERMIMIC_ROBOT_USD")
                or str(ROOT / DEFAULT_USD))

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = None

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 200.0, device=str(DEV)))
    kp = torch.as_tensor(rs["kp"], dtype=torch.float32)
    kd = torch.as_tensor(rs["kd"], dtype=torch.float32)
    ef = torch.as_tensor(rs.get("effort_limit", [0.0] * nj), dtype=torch.float32)
    acts_cfg = {f"j{i}": ImplicitActuatorCfg(joint_names_expr=[joint_names[i]],
                                            stiffness=float(kp[i]), damping=float(kd[i]),
                                            effort_limit=float(ef[i]) if float(ef[i]) > 0 else None)
                for i in range(nj)}
    cfg = SceneCfg(num_envs=N, env_spacing=2.5)
    cfg.robot = ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
                               spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
                               init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.80)),
                               actuators=acts_cfg)
    scene = InteractiveScene(cfg)
    robot: Articulation = scene["robot"]
    sim.reset()
    scene.update(sim.get_physics_dt())

    isaac_names = list(robot.joint_names)
    idx_std = torch.tensor([isaac_names.index(j) for j in joint_names], dtype=torch.long, device=DEV)
    idx_ha = torch.tensor([isaac_names.index(j) for j in ha_names], dtype=torch.long, device=DEV)

    def write_state(row):
        pose = torch.zeros(N, 7, device=DEV)
        pose[:, :3] = row[:, :3].to(DEV)
        pose[:, 3:] = row[:, 3:7].to(DEV)
        vel = torch.zeros(N, 6, device=DEV)
        vel[:, :3] = row[:, 36:39].to(DEV)
        vel[:, 3:] = row[:, 39:42].to(DEV)
        q0 = torch.zeros(N, robot.num_joints, device=DEV)
        qd0 = torch.zeros_like(q0)
        q0[:, idx_ha] = row[:, 7:7 + nj].to(DEV)
        qd0[:, idx_ha] = row[:, 42:42 + nj].to(DEV)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(vel)
        robot.write_joint_state_to_sim(q0, qd0)
        scene.update(sim.get_physics_dt())

    def read_state():
        return torch.cat((robot.data.root_pos_w, robot.data.root_quat_w,
                          robot.data.joint_pos[:, idx_ha], robot.data.root_lin_vel_w,
                          robot.data.root_ang_vel_w, robot.data.joint_vel[:, idx_ha]), -1).cpu()

    out = {}
    for tag, A in (("plus", AP), ("minus", AM)):
        write_state(S0)
        recs = []
        for t in range(H):
            q_tgt = default_q.unsqueeze(0) + A[:, t].to(DEV) * scale.unsqueeze(0)
            robot.set_joint_position_target(q_tgt, joint_ids=[int(i) for i in idx_std])
            robot.write_data_to_sim()
            for _ in range(4):
                sim.step()
            scene.update(sim.get_physics_dt())
            recs.append(read_state())
        out[f"states_{tag}"] = torch.stack(recs, 1)          # (N,H,71) ha 序
        log(f"[probe-roll] {tag}: 末状态 root_pos 均值={out[f'states_{tag}'][:, -1, :3].mean(0).tolist()}")
    out["meta"] = {"job": str(args_cli.job), "H": H, "N": N, "sign": ["plus", "minus"]}
    torch.save(out, args_cli.out)
    log(f"[probe-roll] saved {args_cli.out}")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
