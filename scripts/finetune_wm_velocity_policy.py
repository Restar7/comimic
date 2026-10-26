#!/usr/bin/env python3
"""Calibrate the WM velocity channel with the *policy's own* actions as the WM input.

The WM's action input lives in the space the WBC emits (it was trained on PhysX rollouts of
that policy).  Fine-tuning it with hand-made `a = Δq` breaks that match (policy rollouts blow
up), so here the input is a = π(obs_ref, token_ref) -- identical to what the rollout feeds.
Only the velocity dims are emphasised; positions are kept as a regulariser.

Acceptance: policy rollout (port token) state_term must drop well below the 2.87 baseline and
the WM's qd correlation with the reference must be > 0.9 (was 0.155).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "trackerlab-supermimic/source/trackerLab"):
    sys.path.insert(0, str(p))

# LOCKED 2026-10-26 (S1.1)：与 evaluate.py / eval_wm_velocity.py 同一把口径锁。
if os.environ.get("SM_ALLOW_SCALE_MISMATCH") != "1":
    os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
    os.environ.setdefault(
        "SUPERMIMIC_ROBOT_USD",
        str(ROOT / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"))
print(f"[wmvel][LOCKED] SUPERMIMIC_HA_ALIGN={os.environ.get('SUPERMIMIC_HA_ALIGN')}", flush=True)

from diff_vla_wbc.isaac_joint import load_isaac_bundle, flat_state_tuple  # noqa: E402
from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource  # noqa: E402
from diff_vla_wbc.rollout import _flat_history, compute_joint_loss, imagine_with_tokens, LossWeights  # noqa: E402
from trackerLab.kinematics.supermimic.rotations import so3_log  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500_wmvel_pi.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vel-lambda", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_tokens/packed.pt",
                    help="参考库。hold/命令流路线要指 runs/isaac_motion/ha_cmd/packed.pt，否则"
                         "标定出来的 WM 只对执行轨迹口径成立（S3 contrast 会掉到 0）")
    ap.add_argument("--npz-dir", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--port-dir", type=Path, default=None)
    ap.add_argument("--token-space", choices=("auto", "label", "port"), default="auto")
    ap.add_argument("--no-explicit-integration", action="store_true",
                    help="保持旧结构（整步增量）；默认启用 mjlab 的半隐式显式积分")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(a.device)
    bundle = load_isaac_bundle(a.ckpt, device=dev,
                               npz=sorted(Path(a.npz_dir).glob("*.npz")),
                               packed_motion=Path(a.packed),
                               packed_dir=None, obs_mode=None, hist=10, token_dim=64)
    # 结构：显式物理积分（mjlab parity）。必须在加载后、训练前设定，并写进 ckpt。
    bundle.wm.explicit_integration = not a.no_explicit_integration
    bundle.wm.dt = 0.02
    wm_integration = ("explicit_dt%g" % bundle.wm.dt) if bundle.wm.explicit_integration else "residual"
    print(f"[structure] ws_integration={wm_integration}", flush=True)
    lib = bundle.library
    rec = RecordedTokenSource(Path(a.npz_dir), lib, token_horizon=40, token_dim=64, device=dev,
                              port_dir=a.port_dir, token_space=a.token_space)
    prov = MotionBatchProvider(bundle, lib, batch_size=a.batch, horizon=40, token_source=None,
                               token_dim=64, token_horizon=40, seed=12345, hand_dir=None)
    wm = bundle.wm
    xi = torch.zeros(a.batch, 33, device=dev)

    def one_batch():
        b = prov.sample()
        tok = rec({"frames": b.frames})
        obs = _flat_history(bundle, b.history)
        with torch.no_grad():
            act = bundle.act(obs, tok[:, 0], ref=b.references[0])
        S = flat_state_tuple(b.start_state)
        S1 = flat_state_tuple(b.teacher_states[:, 0])
        return S, act, S1

    def eval_wm(n=8):
        S, act, S1 = one_batch()
        with torch.no_grad():
            out = wm.step(S, act, xi[: act.shape[0]])
        corr = float(((out[5] - out[5].mean()) * (S1[5] - S1[5].mean())).mean()
                     / (out[5].std() * S1[5].std() + 1e-9))
        return {"qd_corr": corr,
                "qd_ratio": float(out[5].abs().mean() / (S1[5].abs().mean() + 1e-9)),
                "joint_mse": float((out[4] - S1[4]).square().mean()),
                "pos_mse": float((out[0] - S1[0]).square().mean())}

    def policy_eval():   # 真正关心的指标：策略 rollout
        b = prov.sample()
        tok = rec({"frames": b.frames})
        W = LossWeights(); W.state_vel = 1.0
        with torch.no_grad():
            ro = imagine_with_tokens(bundle, tokens=tok, state=b.start_state, obs_hist=b.obs_hist,
                                     history=b.history, xi=b.xi, references=b.references,
                                     prev_action=b.prev_action, horizon=8, obs_noise=0.0)
            _, t = compute_joint_loss(ro, teacher_states=b.teacher_states, gamma=0.99, weights=W)
        return float(t["state"]), float(ro.rewards.mean())

    print("[before wm] " + json.dumps({k: round(v, 4) for k, v in eval_wm().items()}))
    se, rw = policy_eval()
    print(f"[before policy] state_term={se:.4f} reward={rw:+.5f}")
    params = list(wm.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    for it in range(1, a.steps + 1):
        S, act, S1 = one_batch()
        n = act.shape[0]
        out = wm.step(S, act, xi[:n])
        pos = (out[0] - S1[0]).square().mean() + so3_log(out[1].transpose(-1, -2) @ S1[1]).square().mean() \
            + (out[4] - S1[4]).square().mean()
        sv = S1[2].std(0).mean() + 1e-6
        sw = S1[3].std(0).mean() + 1e-6
        sq = S1[5].std(0).mean() + 1e-6
        vel = ((out[2] - S1[2]).square().mean() / sv**2
               + (out[3] - S1[3]).square().mean() / sw**2
               + (out[5] - S1[5]).square().mean() / sq**2)
        loss = pos + a.vel_lambda * vel
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if it % max(1, a.steps // 4) == 0:
            se, rw = policy_eval()
            print(f"  [{it:>4}] wm={json.dumps({k: round(v,4) for k,v in eval_wm(4).items()})} "
                  f"policy state={se:.3f} reward={rw:+.4f}", flush=True)
    se, rw = policy_eval()
    print(f"[after policy] state_term={se:.4f} reward={rw:+.5f}")
    ck = dict(torch.load(a.ckpt, map_location="cpu", weights_only=False))
    ck["wm"] = {k: v.detach().cpu() for k, v in wm.state_dict().items()}
    ck["wm_integration"] = wm_integration
    torch.save(ck, a.out)
    print(f"[saved] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
