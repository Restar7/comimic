#!/usr/bin/env python3
"""Per-task (ft318 四任务) imaginary-rollout eval of a joint/SuperMimic checkpoint.

This is the same metric the trainer logs as ``state_term`` / ``rollout_reward``
(frozen world model + analytic tracking reward, ``imagine_with_tokens``), but

* conditioned on the **recorded** 64-d token (``g1_motion_token_64``), i.e. the
  oracle-token upper bound — no GR00T forward pass needed;
* sliced **per task** instead of the two mixed eval batches in the training run.

Usage:
    python3 scripts/eval_joint_tasks_ft318.py \
        --joint-ckpt runs/joint_sm24k_hand/joint_001000.pt --tag hand@1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.compat import quat_error_magnitude  # noqa: E402
from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource  # noqa: E402
from diff_vla_wbc.rollout import LossWeights, compute_joint_loss, imagine_with_tokens  # noqa: E402
from diff_vla_wbc.supermimic_backend import (  # noqa: E402
    SuperMimicBundle,
    build_g1_assets,
    load_motion_library,
)
from diff_vla_wbc.train import JointTrainer  # noqa: E402

TASKS = {
    "task_1": "meeting_room_milk_basket",
    "task_2": "office_red_cup_drawer",
    "task_3": "kitchen_plate_milk_transfer",
    "task_4": "diningroom_can_low_chair",
}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-dir", type=Path, default=ROOT / "data/npz/g1_tasks")
    ap.add_argument("--joint-ckpt", type=Path, default=None)
    ap.add_argument("--supermimic-ckpt", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2, help="batches per task")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/eval_joint_tasks")
    a = ap.parse_args()

    device = torch.device(a.device)
    assets = build_g1_assets(device=device)
    if a.joint_ckpt:
        bundle = JointTrainer.load_bundle_from_checkpoint(a.joint_ckpt, assets)
    elif a.supermimic_ckpt:
        bundle = SuperMimicBundle.from_checkpoint(a.supermimic_ckpt, assets)
    else:
        raise SystemExit("pass --joint-ckpt or --supermimic-ckpt")
    weights = LossWeights(**torch.load(a.joint_ckpt, map_location="cpu", weights_only=False)["config"]["weights"]) \
        if a.joint_ckpt else LossWeights()
    print(f"[info] tag={a.tag} ckpt={a.joint_ckpt or a.supermimic_ckpt}")

    rows = []
    for task, name in TASKS.items():
        clips = sorted(a.motion_dir.glob(f"{task}_{name}__*.npz"))
        if not clips:
            raise SystemExit(f"no clips for {task} under {a.motion_dir}")
        library = load_motion_library(clips, assets, device=device)
        provider = MotionBatchProvider(
            bundle, library, batch_size=a.batch, horizon=a.horizon,
            token_source=None, token_dim=64, token_horizon=40, seed=12345,
        )
        recorded = RecordedTokenSource(a.motion_dir, library, token_horizon=40, token_dim=64,
                                       device=device)
        st, rw, pos_e, ori_e, q_e, vel_e, surv = [], [], [], [], [], [], []
        for i in range(a.samples):
            b = provider.sample()
            tok = recorded({"frames": b.frames})
            ro = imagine_with_tokens(
                bundle, tokens=tok, state=b.start_state, obs_hist=b.obs_hist,
                history=b.history, xi=b.xi, references=b.references,
                prev_action=b.prev_action, horizon=a.horizon, obs_noise=0.0,
            )
            _, terms = compute_joint_loss(ro, teacher_states=b.teacher_states,
                                          gamma=0.99, weights=weights)
            st.append(float(terms.get("state", torch.zeros(()))))
            rw.append(float(ro.rewards.mean()))
            n = min(ro.states.shape[0] - 1, b.teacher_states.shape[1])
            pred, tgt = ro.states[1:n + 1], b.teacher_states[:, :n].permute(1, 0, 2)
            pos_e.append(float(F.mse_loss(pred[..., 0:3], tgt[..., 0:3])))
            ori_e.append(float(quat_error_magnitude(pred[..., 3:7], tgt[..., 3:7]).square().mean()))
            q_e.append(float(F.mse_loss(pred[..., 7:36], tgt[..., 7:36])))
            vel_e.append(float(F.mse_loss(pred[..., 36:71], tgt[..., 36:71])))
            surv.append(float(ro.survival[-1].mean()))
        row = {
            "task": task, "name": name, "clips": len(clips),
            "state_term": sum(st) / len(st), "rollout_reward": sum(rw) / len(rw),
            "state_pos": sum(pos_e) / len(pos_e), "state_ori": sum(ori_e) / len(ori_e),
            "state_q": sum(q_e) / len(q_e), "state_vel": sum(vel_e) / len(vel_e),
            "survival": sum(surv) / len(surv),
        }
        rows.append(row)
        print(f"[{task}] state_term={row['state_term']:.4f} reward={row['rollout_reward']:+.4f} "
              f"pos={row['state_pos']:.2e} ori={row['state_ori']:.4f} q={row['state_q']:.4f} "
              f"vel={row['state_vel']:.4f} survival={row['survival']:.3f}")

    def avg(k: str) -> float:
        return float(sum(r[k] for r in rows) / len(rows))

    summary = {"tag": a.tag, "ckpt": str(a.joint_ckpt or a.supermimic_ckpt),
               "horizon": a.horizon, "batch": a.batch, "samples": a.samples,
               "mean_state_term": avg("state_term"), "mean_rollout_reward": avg("rollout_reward"),
               "mean_state_pos": avg("state_pos"), "mean_state_ori": avg("state_ori"),
               "mean_state_q": avg("state_q"), "mean_state_vel": avg("state_vel"),
               "tasks": rows}
    print("[summary] " + json.dumps({k: v for k, v in summary.items() if k != "tasks"}))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.tag.replace('/', '_')}.json").write_text(json.dumps(summary, indent=2))
    print(f"[saved] {a.out / (a.tag.replace('/', '_') + '.json')}")


if __name__ == "__main__":
    main()
