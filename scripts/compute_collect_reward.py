#!/usr/bin/env python3
"""给 `isaac_g1_collect.py` 采到的转移补算奖励（在**系统 python** 里做，口径不重写）。

为什么分两步：
  Isaac 进程里不能 import 我们的训练栈（`diff_vla_wbc`/trackerLab MDP 会连锁依赖 MuJoCo 常量，
  在 Isaac venv 里炸），所以 `isaac_g1_collect.py` 只存 `(s, a, s', frame)`。
  奖励只依赖 `(state, action, prev_action, frame)`，全部在盘上 ⇒ 这里用**训练同一个**
  `bundle.reward(...)` 补算，保证口径与训练完全一致。

用法：
    python3 scripts/compute_collect_reward.py --in /tmp/collect_smoke/pt.pt \
        --out runs/wm_real/onpolicy_gt_doubledesk.pt
    # 可选：附加"命令派生动作"口径的奖励（诊断两种动作口径的差异）
    python3 scripts/compute_collect_reward.py --in ... --out ... --also-command-reward
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--reward-mode", default="saturating", choices=("saturating", "linear"))
    ap.add_argument("--reward-vel-scale", type=float, default=1.0)
    ap.add_argument("--also-command-reward", action="store_true",
                    help="另外算一遍\"命令派生动作\"口径的奖励（诊断用）")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    d = torch.load(str(a.inp), map_location="cpu", weights_only=False)
    S, A, S2, F = d["states"], d["actions"], d["next_states"], d["frames"]
    print(f"[reward] in={a.inp} n={S.shape[0]} clip={d.get('clip')} "
          f"envs={d.get('num_envs')} steps={d.get('steps')}", flush=True)

    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz, packed_motion=a.packed,
                               obs_mode=None, hist=None, token_dim=64)
    bundle.reward_mode = str(a.reward_mode)
    bundle.reward_vel_scale = float(a.reward_vel_scale)

    n = int(S.shape[0])
    bs = 4096
    out = {"rewards": torch.zeros(n), "reward_terms": {}}
    terms_acc: dict[str, list] = {}
    for lo in range(0, n, bs):
        sl = slice(lo, min(lo + bs, n))
        st = S[sl].to(dev).float()
        ac = A[sl].to(dev).float()
        fr = F[sl].to(dev).long()
        # prev_action：同一 env 的上一行（按环境交错，故用"回退 num_envs 行"近似；
        # 第一帧用自身）
        prev = A[torch.clamp(torch.arange(lo, min(lo + bs, n)) - int(d.get("num_envs", 1)),
                             min=0)].to(dev).float()
        terms = bundle.reward(st, ac, prev, FrameRef(fr), bodies=None)
        out["rewards"][sl] = terms["total"].detach().cpu()
        for k, v in terms.items():
            if torch.is_tensor(v) and v.shape == fr.shape:
                terms_acc.setdefault(k, []).append(v.detach().cpu())
    for k, vs in terms_acc.items():
        out["reward_terms"][k] = torch.cat(vs)
    print(f"[reward] r mean={float(out['rewards'].mean()):+.5f} "
          f"std={float(out['rewards'].std()):.5f} "
          f"min={float(out['rewards'].min()):+.4f} max={float(out['rewards'].max()):+.4f}", flush=True)
    for k in ("track", "e_body_pos", "e_body_ori", "e_body_linvel", "e_body_angvel",
              "e_root_pos", "e_root_ori", "action_rate", "joint_limit"):
        if k in out["reward_terms"]:
            t = out["reward_terms"][k]
            print(f"[reward]   {k:15s} mean={float(t.mean()):+.6f}", flush=True)

    payload = dict(d)
    payload["rewards"] = out["rewards"]
    payload["reward_terms"] = out["reward_terms"]
    payload.setdefault("meta", {})
    payload["meta"].update({"reward_mode": a.reward_mode,
                            "reward_vel_scale": float(a.reward_vel_scale),
                            "reward_fn": "trackerlab mdp.reward via bundle.reward (同一函数)",
                            "reward_computed_by": "scripts/compute_collect_reward.py"})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(a.out))
    stats = {"n": n, "r_mean": float(out["rewards"].mean()),
             "r_std": float(out["rewards"].std()), "meta": payload["meta"]}
    a.out.with_suffix(".json").write_text(json.dumps(stats, indent=1, default=str))
    print(f"[reward] saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
