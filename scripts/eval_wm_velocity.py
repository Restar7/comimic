#!/usr/bin/env python3
"""Measure the WM velocity channel and the policy rollout for a SuperMimic checkpoint.

This is the *acceptance* counterpart of ``scripts/finetune_wm_velocity_policy.py``: it does
no training, only the three numbers the handovers use to decide whether a retrain worked:

* ``qd_corr`` / ``qd_ratio`` — the WM's one-step joint-velocity prediction against the
  reference window, driven by the **policy's own** actions (``a = pi(obs_ref, token_ref)``,
  i.e. the same input the closed loop feeds the WM);
* ``pos_mse`` / ``joint_mse`` — position / joint-position one-step error (should not regress);
* ``state_term`` / ``reward`` — the end-to-end imagined rollout with the token chunk
  (``imagine_with_tokens``), which is the quantity the joint training optimises.

The structure (``wm_integration``) is taken from the checkpoint: a ckpt with the key trains
as explicit integration, a legacy ckpt stays residual.

Usage:
    python3 scripts/eval_wm_velocity.py --ckpt trackerlab-supermimic/logs/bones_seed_wmvel_1025/last.pt
    python3 scripts/eval_wm_velocity.py --ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/last.pt --tag old
    python3 scripts/eval_wm_velocity.py --ckpt ... --tokens label     # HA 标签 token（对照用）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "trackerlab-supermimic/source/trackerLab"):
    sys.path.insert(0, str(p))

from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource  # noqa: E402
from diff_vla_wbc.isaac_joint import flat_state_tuple, load_isaac_bundle  # noqa: E402
from diff_vla_wbc.rollout import (  # noqa: E402
    LossWeights,
    _flat_history,
    compute_joint_loss,
    imagine_with_tokens,
)

PORT_TOKENS = ROOT / "data/port_tokens/ha_tokens"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_tokens/packed.pt")
    ap.add_argument("--tokens", choices=("port", "label"), default="port",
                    help="port = data/port_tokens/ha_tokens (the WBC's conditioning space, "
                         "what the joint training uses); label = the npz g1_motion_token_64")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8, help="how many batches to average")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--tag", default="", help="label printed with the JSON line")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device)
    bundle = load_isaac_bundle(
        a.ckpt, device=dev, npz=sorted(a.npz_dir.glob("*.npz")), packed_motion=a.packed,
        packed_dir=None, obs_mode=None, hist=10, token_dim=64,
    )
    lib = bundle.library
    token_dir = PORT_TOKENS if a.tokens == "port" else a.npz_dir
    rec = RecordedTokenSource(token_dir, lib, token_horizon=40, token_dim=64, device=dev)
    prov = MotionBatchProvider(bundle, lib, batch_size=a.batch, horizon=40, token_source=None,
                               token_dim=64, token_horizon=40, seed=12345, hand_dir=None)

    def batch():
        b = prov.sample()
        tok = rec({"frames": b.frames})
        obs = _flat_history(bundle, b.history)
        with torch.no_grad():
            act = bundle.act(obs, tok[:, 0], ref=b.references[0])
        return b, tok, act

    # ---- WM one-step, policy actions ------------------------------------------------
    acc: dict[str, list[float]] = {}
    xi0 = torch.zeros(a.batch, 33, device=dev)
    for _ in range(a.samples):
        b, _, act = batch()
        S = flat_state_tuple(b.start_state)
        S1 = flat_state_tuple(b.teacher_states[:, 0])
        with torch.no_grad():
            out = bundle.wm.step(S, act, xi0)
            pred_qd, ref_qd = out[5], S1[5]
            # 真正的逐维 Pearson 相关：cov / (σx·σy)，再对维度取平均（不是全局均值之比，
            # 那个在偏移大时会 >1）。
            def _corr(x, y):
                xc, yc = x - x.mean(0), y - y.mean(0)
                return float(((xc * yc).mean(0) / (xc.std(0) * yc.std(0) + 1e-9)).mean())

            def _geo(a, b):  # 旋转矩阵之间的测地角（rad）
                tr = (a.transpose(-1, -2) @ b).diagonal(dim1=-2, dim2=-1).sum(-1)
                return torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))

            vals = {
                "qd_corr": _corr(pred_qd, ref_qd),
                "qd_ratio": float(pred_qd.abs().mean() / (ref_qd.abs().mean() + 1e-9)),
                "joint_mse": float((out[4] - S1[4]).square().mean()),
                "pos_mse": float((out[0] - S1[0]).square().mean()),
                "pos_err_m": float((out[0] - S1[0]).norm(dim=-1).mean()),
                "rot_err_deg": float(_geo(out[1], S1[1]).mean() * 180.0 / 3.141592653589793),
                "linvel_corr": _corr(out[2], S1[2]),
                # 根角速度：这是旧（混帧）公式真正伤到的通道，修好后必须 > 0.9
                "root_w_corr": _corr(out[3], S1[3]),
                "root_w_ratio": float(out[3].abs().mean() / (S1[3].abs().mean() + 1e-9)),
            }
            for k, v in vals.items():
                acc.setdefault(k, []).append(v)

    # ---- end-to-end imagined rollout ------------------------------------------------
    W0 = LossWeights()
    W0.state_vel = 0.0        # new standard: velocity does not enter the state regulariser
    roll: dict[str, list[float]] = {}
    for _ in range(a.samples):
        b, tok, _ = batch()
        with torch.no_grad():
            ro = imagine_with_tokens(bundle, tokens=tok, state=b.start_state, obs_hist=b.obs_hist,
                                     history=b.history, xi=b.xi, references=b.references,
                                     prev_action=b.prev_action, horizon=a.horizon, obs_noise=0.0)
            _, terms = compute_joint_loss(ro, teacher_states=b.teacher_states, gamma=0.99, weights=W0)
            vals = {"state_term": float(terms["state"]), "reward": float(ro.rewards.mean()),
                    "survival": float(ro.survival.mean())}
            for k, v in vals.items():
                roll.setdefault(k, []).append(v)

    out = {
        "tag": a.tag or a.ckpt.name,
        "ckpt": str(a.ckpt),
        "wm_integration": getattr(bundle.wm, "explicit_integration", None) and "explicit_dt%g" % bundle.wm.dt or "residual",
        "tokens": a.tokens,
        "wm": {k: round(sum(v) / len(v), 5) for k, v in acc.items()},
        "rollout": {k: round(sum(v) / len(v), 5) for k, v in roll.items()},
        "samples": a.samples,
        "batch": a.batch,
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
