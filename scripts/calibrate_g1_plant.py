#!/usr/bin/env python3
"""G1 gate: coarse plant calibration of the MuJoCo G1 against ft318 recorded targets.

`docs/MUJOCO_SONIC_JOINTFT_PLAN_20261104.md` §2.2 的结论是：318 的 token/目标都能复现，
但在 MuJoCo 里**基座会前倾摔倒**，因为录制姿态的质心-支撑边界只剩 2–3 cm，而当前 MJCF 的
质量分布/接触/踝驱动压不住。这个脚本就是 §4 的粗扫工具：**同一个回放指标，扫几个物理旋钮**。

指标（越大越好）：

* ``survive``：从第 0 帧起跑 target 回放，能撑多少步（上限 ``--steps``）；
* ``tilt``：结束时的前倾角；
* ``drift``：基座 xy 漂移（米）。

扫描维度：

* ``--com-shifts``：``torso_link`` 质心 x 偏移（负 = 往后）——最敏感、最便宜；
* ``--frictions``：地面/脚底摩擦系数（改 ``geom_friction``）；
* ``--gains``：``ha`` / ``deploy`` 两套执行器表。

用法::

    third_party/SuperMimic-Reproduced/.venv/bin/python scripts/calibrate_g1_plant.py \
        --com-shifts 0,-0.02,-0.04,-0.06,-0.08,-0.10 --steps 120 \
        --out runs/plant_calib.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "groot_repro"))


def _replay_module():
    spec = importlib.util.spec_from_file_location(
        "replay_ft318_mujoco", ROOT / "scripts" / "replay_ft318_mujoco.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(rp, ep, *, com_shift: float, friction: float | None, gains: str, steps: int,
             ankle_scale: float = 1.0, kp_scale: float = 1.0, init: str = "recorded") -> dict:
    plant = rp.Plant(rp.DEFAULT_SCENE, gains=gains)
    if kp_scale != 1.0:
        plant.kp = plant.kp * kp_scale
        plant.kd = plant.kd * kp_scale
    if ankle_scale != 1.0:
        plant.kp[4] *= ankle_scale
        plant.kp[5] *= ankle_scale
        plant.kp[10] *= ankle_scale
        plant.kp[11] *= ankle_scale
        plant.kd[4] *= ankle_scale
        plant.kd[5] *= ankle_scale
        plant.kd[10] *= ankle_scale
        plant.kd[11] *= ankle_scale
    m = plant.model
    if com_shift:
        import mujoco

        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        m.body_ipos[bid][0] += com_shift
    if friction is not None:
        m.geom_friction[:, 0] = friction
    init_state = rp.frame_init_state(plant, ep, 0) if init == "recorded" else {}
    plant.reset(ep["body_state"][0], ep["hands_state"][0],
                quat=init_state.get("quat", (1, 0, 0, 0)), dq=init_state.get("dq"),
                base_lin_vel=init_state.get("base_lin_vel"),
                base_ang_vel=init_state.get("base_ang_vel"))

    up, roots, mae, tilt_err = [], [], [], []
    for t in range(steps):
        plant.apply_target(ep["body_wbc"][t], ep["hands_wbc"][t])
        up.append(plant.up())
        roots.append(plant.root()[:2].copy())
        mae.append(float(np.abs(plant.q() - ep["body_state"][t + 1]).mean()))
        tilt_err.append(abs(plant.tilt_deg() - float(np.degrees(np.arccos(
            np.clip(ep["root_quat"][t + 1, 0] ** 2 - np.sum(ep["root_quat"][t + 1, 1:] ** 2), -1, 1))))))
        if plant.up() < 0.5:
            break
    roots = np.stack(roots)
    return {
        "com_shift": com_shift,
        "friction": friction,
        "gains": gains,
        "survive": len(up),
        "fallen": bool(plant.up() < 0.5),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(up[-1], -1, 1)))),
        "drift_m": float(np.linalg.norm(roots[-1] - roots[0])),
        "joint_mae": float(np.mean(mae)),
        "tilt_mae_deg": float(np.mean(tilt_err)),
        "ankle_scale": ankle_scale,
        "kp_scale": kp_scale,
        "init": init,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-index", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--com-shifts", default="0,-0.02,-0.04,-0.06,-0.08,-0.10")
    ap.add_argument("--frictions", default="")
    ap.add_argument("--gains", default="ha,deploy")
    ap.add_argument("--ankle-scales", default="1.0")
    ap.add_argument("--kp-scales", default="1.0")
    ap.add_argument("--init", default="recorded", choices=("rest", "recorded"))
    ap.add_argument("--out", type=Path, default=ROOT / "runs/plant_calib.json")
    args = ap.parse_args()

    rp = _replay_module()
    ep = rp.load_episode(args.set_index, args.episode)
    com_shifts = [float(x) for x in args.com_shifts.split(",") if x.strip()]
    frictions = [float(x) for x in args.frictions.split(",") if x.strip()] or [None]
    rows = []
    ankle_scales = [float(x) for x in args.ankle_scales.split(",") if x.strip()]
    kp_scales = [float(x) for x in args.kp_scales.split(",") if x.strip()]
    for gains in [g.strip() for g in args.gains.split(",") if g.strip()]:
        for fr in frictions:
            for kps in kp_scales:
                for ans in ankle_scales:
                    for shift in com_shifts:
                        row = run_case(rp, ep, com_shift=shift, friction=fr, gains=gains,
                                       steps=args.steps, ankle_scale=ans, kp_scale=kps,
                                       init=args.init)
                        rows.append(row)
                        print(
                            f"gains={gains:6s} fr={fr} kp={kps:4.2f} ank={ans:4.2f} com={shift:+.2f}"
                            f" -> survive {row['survive']:3d} tilt {row['tilt_deg']:5.1f}"
                            f" tiltMAE {row['tilt_mae_deg']:5.1f} drift {row['drift_m']:.3f}"
                            f" mae {row['joint_mae']:.3f} fallen={row['fallen']}"
                        )

    best = sorted(rows, key=lambda r: (r["tilt_mae_deg"], -r["survive"]))[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"episode": f"{ep['task'][0]}/{args.episode}",
                                    "steps": args.steps, "best": best, "rows": rows},
                                   indent=1, ensure_ascii=False))
    print(f"\n[best] {json.dumps(best, ensure_ascii=False)}")
    print(f"[out]  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
