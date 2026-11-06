#!/usr/bin/env python3
"""A2 验收：带手 43-dof 资产真的能进 mjlab，且**观测 93 / 动作 29 / nu 43**。

跑法：
    python3 scripts/check_hand_asset.py                 # 43dof 分支
    python3 scripts/check_hand_asset.py --baseline      # 旧的 29-dof，做对照

判据（三条全绿才算 A2 过）：
    policy obs 宽 = 93      动作维 = 29      sim 的 nu = 43（手 14 个 actuator 在模型里）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party/SuperMimic-Reproduced"
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="用旧 29-dof 资产做对照")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if not a.baseline:
        os.environ["SM_G1_ASSET"] = "43dof"
    else:
        os.environ.pop("SM_G1_ASSET", None)

    from supermimic.envs.tracking_env_cfg import (
        G1_43DOF_ASSET,
        SuperMimicEnvCfgArgs,
        make_env,
    )

    npz = sorted(str(p) for p in (ROOT / "data/npz/g1_tasks").rglob("*.npz"))[:4]
    assert npz, "找不到 data/npz/g1_tasks 下的 npz"
    args = SuperMimicEnvCfgArgs(
        npz_files=npz, num_envs=a.num_envs, record_transitions=False,
        obs_noise=False, reset_noise=False, domain_randomization=False,
        compact_library=True,
    )
    env = make_env(args, device=a.device)

    act_dim = int(env.action_space.shape[-1])
    nu = int(env.sim.mj_model.nu) if hasattr(env, "sim") else int(env.mj_model.nu)
    # 直接看真实观测张量（mjlab 的 observation_space 在拼接前是 None）
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    hist = int(getattr(args, "history_length", 10))
    policy_dim = int(obs["policy"].shape[-1]) // hist      # 单帧宽（history 默认 10）
    goal_dim = int(obs["goal"].shape[-1])

    print(f"[check] branch={'43dof' if G1_43DOF_ASSET == '43dof' else 'baseline-29dof'}")
    print(f"[check] env={type(env).__name__} num_envs={a.num_envs}")
    print(f"[check] obs.policy(单帧)={policy_dim} × hist{hist}  obs.goal={goal_dim}  act={act_dim}  nu={nu}")
    model = env.sim.mj_model if hasattr(env, "sim") else env.mj_model
    jn = [model.actuator(i).name for i in range(model.nu)]
    hand = [n for n in jn if "hand" in n]
    print(f"[check] actuators={len(jn)} 其中手部={len(hand)} e.g. {hand[:3]}")
    print(f"[check] model: nbody={model.nbody} nq={model.nq} nv={model.nv} nu={model.nu} "
          f"njnt={model.njnt}  ← 带场景时应比纯机器人(nbody=45,nq=50)更大")
    try:
        bodies = [model.body(i).name for i in range(model.nbody)]
        objs = [b for b in bodies if any(k in b for k in ("table", "can", "box", "tray", "cabinet", "drawer", "pitcher", "mug", "banana", "brick", "sugar", "bleach", "mustard", "pudding"))]
        print(f"[check] 场景物体 body({len(objs)}): {objs[:8]}")
    except Exception as exc:
        print(f"[check] (body 名读取失败: {type(exc).__name__})")

    ok = True
    if policy_dim != 93:
        print(f"[FAIL] policy obs 宽 {policy_dim}，期望 93"); ok = False
    if act_dim != 29:
        print(f"[FAIL] 动作维 {act_dim}，期望 29（说明手被执行器正则捞进去了）"); ok = False
    want_nu = 29 if a.baseline else 43
    if nu != want_nu:
        print(f"[FAIL] nu={nu}，期望 {want_nu}"); ok = False
    print("[PASS] A2 验收通过" if ok else "[FAIL] 见上面几条")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
