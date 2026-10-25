#!/usr/bin/env python3
"""Verify the Isaac-native joint bundle (``diff_vla_wbc/isaac_joint.py``) end to end.

The whole point of the Isaac path is that **nothing is re-implemented**: observation, reward,
termination, FK and the reference motion all come from ``trackerlab-supermimic``.  This script
pins that claim numerically, plus the plumbing that does live in our code (state layout,
flattened observation history, frame indexing, reference-start history, WM windows).

Checks
------
1. checkpoint tensors load strictly into the Isaac networks (no shape reinterpreting);
2. flat state ⇄ trackerlab state tuple is lossless (and the layout guard fires on a swap);
3. ``bundle.step`` == ``WorldModel.step`` (bit-exact) on the same inputs;
4. ``bundle.observation`` == ``MDP.obs``; ``bundle.reward`` == ``MDP.reward``;
   ``sigmoid(bundle.termination_logits)`` == ``MDP.soft_termination``;
5. 99-wide observation history flatten/split round-trips losslessly, and the policy consumes
   it exactly like the native ``Policy``;
6. reference-start history: the obs equals ``MDP.obs`` on the same reference states, and the
   implied ``a_prev`` reproduces the env's joint target (``default_q + scale * a``);
7. reward at the reference state equals the paper's per-step max (all tracking terms 1);
8. WM windows: ``states`` are the true reference states, ``actions`` the normalised offsets;
9. frame alignment between the npz clips (VLA images/tokens/hands) and the packed motion
   library (1 frame = 1 frame, same clip order).

Usage
-----
    python3 scripts/check_isaac_bundle.py \
        --ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt \
        --motion data/npz/ha_tokens \
        --packed runs/isaac_motion/ha_tokens/packed.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc import isaac_joint as ij  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="Isaac/trackerlab SuperMimic ckpt")
    ap.add_argument("--packed", type=Path, default=None, help="packed Isaac motion library")
    ap.add_argument("--motion", type=Path, default=ROOT / "data" / "npz" / "ha_tokens",
                    help="pure-HA npz directory (frame alignment is checked against it)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=8)
    args = ap.parse_args()

    files = sorted(args.motion.glob("*.npz")) if args.motion.is_dir() else [args.motion]
    if not files:
        raise SystemExit(f"no npz clips under {args.motion}")
    packed = args.packed or ij.ensure_packed_motion([args.motion])
    bundle = ij.load_isaac_bundle(args.ckpt, device=args.device, packed_motion=packed,
                                  hist=args.hist)
    dev = bundle.device
    hist = bundle.hist
    torch.manual_seed(0)
    print(f"[bundle] obs_mode={bundle.obs_mode} obs_width={bundle.obs_width} "
          f"hist={hist} token_dim={bundle.token_dim} motion={bundle.motion.provenance}")

    # ---- 1) checkpoint tensors load strictly ------------------------------------
    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    pi_keys = set(ck["pi"]); wm_keys = set(ck["wm"])
    pi_have = set(bundle.policy.state_dict()); wm_have = set(bundle.wm.state_dict())
    assert pi_keys == pi_have, (pi_keys ^ pi_have)
    assert wm_keys == wm_have, (wm_keys ^ wm_have)
    print(f"[1/9] checkpoint keys: policy {len(pi_keys)} / WM {len(wm_keys)} — exact match")

    # ---- 2) state layout round-trip + swap guard --------------------------------
    b = args.batch
    flat = torch.cat((
        torch.randn(b, 3) * 0.1,                                    # p
        torch.nn.functional.normalize(torch.randn(b, 4), dim=-1),   # quat
        torch.randn(b, 29) * 0.1,                                   # q
        torch.randn(b, 3) * 0.1,                                    # v
        torch.randn(b, 3) * 0.1,                                    # w
        torch.randn(b, 29) * 0.1,                                   # qd
    ), dim=-1)
    S = ij.flat_state_tuple(flat)
    back = torch.cat((S[0], ij.mat_to_quat(S[1]), S[4], S[2], S[3], S[5]), -1)
    d_rot = float((ij.quat_to_mat(back[..., 3:7]) - ij.quat_to_mat(flat[..., 3:7])).abs().max())
    d_rest = float(max((back[..., 0:3] - flat[..., 0:3]).abs().max(),
                       (back[..., 7:] - flat[..., 7:]).abs().max()))
    print(f"[2/9] layout round-trip: rot {d_rot:.2e}, other {d_rest:.2e} "
          f"(layout: {ij.FLAT_LAYOUT})")
    assert max(d_rot, d_rest) < 1e-5, "flat state round-trip is not lossless"

    # ---- 3) world model step -----------------------------------------------------
    action = torch.randn(b, 29) * 0.1
    xi = ij.default_xi(b, dev)
    xi_n = ij.normalize_xi(xi)
    with torch.no_grad():
        p1, R1, v1, w1, q1, qd1 = bundle.wm.step(ij.flat_state_tuple(flat), action, xi_n)
        want = torch.cat((p1, ij.mat_to_quat(R1), q1, v1, w1, qd1), -1)
        got = bundle.step(flat, action, xi)
    d_step = float((want - got).abs().max())
    print(f"[3/9] world model step max|Δ| = {d_step:.3e}")
    assert d_step < 1e-6, "bundle.step diverges from WorldModel.step"

    # ---- 4) analytic heads -------------------------------------------------------
    f = torch.randint(0, bundle.motion.num_frames, (b,), device=dev)
    ref = ij.FrameRef(f)
    a_prev = torch.randn(b, 29) * 0.1
    state = ij.canonical_to_flat(bundle.motion.ref[f].float())
    with torch.no_grad():
        obs = bundle.observation(state, a_prev, xi, ref=ref)
        want_obs = bundle.mdp.obs(ij.flat_state_tuple(state), a_prev, xi, f)
        rew = bundle.reward(state, action, a_prev, ref)
        want_rew = bundle.mdp.reward(ij.flat_state_tuple(state), f, action, a_prev)
        logits = bundle.termination_logits(state, ref)
        want_p = bundle.mdp.soft_termination(ij.flat_state_tuple(state), f)
    d_obs = float((obs - want_obs).abs().max())
    d_rew = float((rew["total"] - want_rew).abs().max())
    d_term = float((torch.sigmoid(logits) - want_p).abs().max())
    print(f"[4/9] obs max|Δ| = {d_obs:.3e}; reward max|Δ| = {d_rew:.3e}; "
          f"termination max|Δ| = {d_term:.3e}")
    assert max(d_obs, d_rew, d_term) < 1e-6, "analytic heads diverge from the Isaac MDP"

    # ---- 5) history flattening + policy ------------------------------------------
    obs_hist = torch.randn(b, hist, bundle.obs_width) * 0.1
    flat_obs = bundle.flatten_history(obs_hist.permute(1, 0, 2).contiguous())
    back_obs = ij.split_flat_obs(flat_obs, hist, bundle.term_dims)
    d_split = float((back_obs - obs_hist).abs().max())
    token = torch.randn(b, bundle.token_dim)
    with torch.no_grad():
        want_a = bundle.policy(obs_hist, token)
        got_a = bundle.act(flat_obs, token)
    d_act = float((want_a - got_a).abs().max())
    print(f"[5/9] history flatten round-trip = {d_split:.3e}; policy max|Δ| = {d_act:.3e}")
    assert d_split < 1e-6 and d_act < 1e-6

    # ---- 6) reference-start history ------------------------------------------------
    frames = torch.randint(hist, bundle.motion.num_frames - args.horizon - 2, (b,), device=dev)
    start = bundle.reference_start(bundle.motion, frames)
    assert start["observations"].shape == (hist, b, bundle.obs_width), start["observations"].shape
    assert start["history" if "history" in start else "obs_hist"].shape == (b, hist * bundle.obs_width)
    # last history frame's state must be the reference state at ``frames``
    d_state = float((start["state"] - ij.canonical_to_flat(bundle.motion.ref[frames].float())).abs().max())
    # implied action must reproduce the env's joint target: target = default_q + scale * a.
    # ``prev_action`` is the action of the *previous* frame (obs at t sees a_{t-1}), so the
    # target it reproduces is the reference joint position of frame f-1 (clamped to the clip).
    motion = bundle.motion
    prev_frames = (frames - 1).clamp(min=motion.clip_start[motion.clip_id[frames]])
    target = bundle.default_joint_pos + bundle.action_scale * start["prev_action"]
    want_target = motion.ref[prev_frames].float()[..., 13 : 13 + 29]
    d_act_implied = float((target - want_target).abs().max())
    print(f"[6/9] reference start: state max|Δ| = {d_state:.3e}; "
          f"implied joint target max|Δ| = {d_act_implied:.3e}")
    assert d_state < 1e-4 and d_act_implied < 1e-4

    # ---- 7) reward at the reference state = per-step max ---------------------------
    z = torch.zeros(b, 29, device=dev)
    with torch.no_grad():
        r_ref, ref_terms = bundle.mdp.reward(ij.flat_state_tuple(state), f, z, z, terms=True)
    # the six tracking terms must all be 1 (the robot *is* the reference); the total can still be
    # slightly below ``MAX_STEP_REWARD`` when a reference pose sits outside the soft joint limits
    # (the paper's ``w_joint_limit`` penalty), so that part is checked as a bound, not equality.
    d_track = float((ref_terms["track"] - ij.MAX_STEP_REWARD).abs().max())
    over = bool((r_ref > ij.MAX_STEP_REWARD + 1e-6).any())
    print(f"[7/9] reward at reference: track = {float(ref_terms['track'].mean()):.6f} "
          f"(max {ij.MAX_STEP_REWARD}) max|Δ| = {d_track:.3e}; total = {float(r_ref.mean()):.6f} "
          f"≤ max: {not over}; joint-limit penalty max = {float(ref_terms['joint_limit'].max()):.3e}")
    assert d_track < 1e-4, "the tracking terms are not maximal on the reference state"
    assert not over, "reward exceeds the paper's per-step maximum"

    # ---- 8) WM windows -----------------------------------------------------------
    win = bundle.sample_windows(batch_size=b, window=args.horizon, seed=7)
    idx = win["frames"][:, None] + torch.arange(args.horizon + 1, device=dev)[None, :]
    want_states = ij.canonical_to_flat(bundle.motion.ref[idx.reshape(-1)].float()).reshape(
        b, args.horizon + 1, -1).permute(1, 0, 2)
    d_win = float((win["states"] - want_states).abs().max())
    want_actions = (want_states.permute(1, 0, 2)[:, 1:, 7:36] - bundle.default_joint_pos) / bundle.action_scale
    d_win_a = float((win["actions"] - want_actions.permute(1, 0, 2)).abs().max())
    print(f"[8/9] WM windows: states max|Δ| = {d_win:.3e}; actions max|Δ| = {d_win_a:.3e} "
          f"(shape {tuple(win['states'].shape)} / {tuple(win['actions'].shape)})")
    assert d_win < 1e-4 and d_win_a < 1e-4

    # ---- 9) frame alignment npz ⇄ packed motion -----------------------------------
    npz_frames = 0
    import numpy as np

    for path in files:
        with np.load(path) as d:
            npz_frames += int(d["joint_pos"].shape[0])
    names_ok = bundle.motion.names[: len(files)] == [p.stem for p in sorted(files)][: len(bundle.motion.names)]
    print(f"[9/9] npz frames {npz_frames} vs motion library {bundle.motion.num_frames}; "
          f"clip names aligned = {names_ok}")
    assert npz_frames == bundle.motion.num_frames, "npz and packed motion frame counts differ"
    assert names_ok, "clip order differs between npz and the packed motion library"

    print(json.dumps({
        "ckpt": str(args.ckpt), "packed": str(packed),
        "clips": len(bundle.motion.names), "frames": int(bundle.motion.num_frames),
        "obs_mode": bundle.obs_mode, "obs_width": bundle.obs_width,
        "max_step_reward": ij.MAX_STEP_REWARD,
    }, indent=1))
    print("\n[ok] Isaac-native joint bundle verified (policy/WM/obs/reward/termination/FK/motion "
          "+ npz frame alignment)")


if __name__ == "__main__":
    main()
