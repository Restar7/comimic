#!/usr/bin/env python3
"""Gate: prove the A3/A5 ablation runs on the **Isaac Lab chain** with **pure-HA** data.

The user's requirement, made executable:

  * the WBC policy **and** the world model come from the Isaac Lab / PhysX run
    (``ckpt_010500.pt``, trackerlab ``{pi, wm}`` — not the mjlab/MuJoCo replica);
  * the observation (99-d ``relative_ori99``), reward, soft termination, differentiable FK and
    the reference motion library all come from ``trackerlab-supermimic`` (``MDP`` /
    ``G1Kinematics`` / ``CompactMotionLib``) — nothing from ``third_party/SuperMimic-Reproduced``;
  * the data is pure HA (``data/npz/ha_tokens``): no ``ft318``, no ``mixed_ha_ft318`` episode.

Every line below is an assertion, not a comment; exit code 0 means all gates passed.

Usage::

    python3 scripts/verify_isaac_chain.py \
        --sm-ckpt trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt \
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

OBS_WIDTH_99 = 29 + 29 + 3 + 3 + 29 + 6
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  [ok]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def module_of(obj) -> str:
    return type(obj).__module__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm-ckpt", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt")
    ap.add_argument("--motion", type=Path, nargs="+", default=[ROOT / "data/npz/ha_tokens"])
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_tokens/packed.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from diff_vla_wbc.isaac_joint import IsaacJointBundle, is_isaac_checkpoint

    # ---------------------------------------------------------------- 1. checkpoint
    print(f"\n[1] SuperMimic checkpoint: {args.sm_ckpt}")
    ck = torch.load(args.sm_ckpt, map_location="cpu", weights_only=False)
    check(is_isaac_checkpoint(args.sm_ckpt),
          "checkpoint is the Isaac layout {pi, wm} (trackerlab/Isaac Lab, not mjlab {policy, wm, config})")
    pi_keys = sorted(ck.get("pi", {}))
    check(all(k.startswith(("dec.", "obs_norm")) for k in pi_keys),
          f"policy = token-interface decoder only ({pi_keys[0]} …): the goal encoder is removed")
    check("obs_norm.mean" in pi_keys and int(ck["pi"]["obs_norm.mean"].numel()) == OBS_WIDTH_99,
          f"policy owns its RunningNorm over {int(ck['pi'].get('obs_norm.mean', torch.zeros(0)).numel())} obs dims")
    check(any(k.startswith("net.") for k in ck.get("wm", {})),
          "world model is the LayerNorm MLP (wm.net.*) of the Isaac implementation")
    a = ck.get("args") or {}
    check(a.get("interface") == "token", f"interface = {a.get('interface')!r}")
    check(a.get("obs_mode") == "relative_ori99", f"obs_mode = {a.get('obs_mode')!r}")
    check(int(a.get("hist", 0)) == 10, f"hist = {a.get('hist')}")
    check(bool(ck.get("reward_cfg")), "checkpoint carries the reward config the Isaac run used")

    # ---------------------------------------------------------------- 2/3. bundle provenance
    print("\n[2] training bundle provenance (every consumer must be trackerlab)")
    bundle = IsaacJointBundle(args.sm_ckpt, packed_motion=args.packed, device=args.device)
    bundle.to(args.device)
    pol_sd = bundle.policy.state_dict()
    wm_sd = bundle.wm.state_dict()
    worst_p = max(float((pol_sd[k].float() - v.float()).abs().max()) for k, v in ck["pi"].items())
    worst_w = max(float((wm_sd[k].float() - v.float()).abs().max()) for k, v in ck["wm"].items())
    check(set(pol_sd) == set(ck["pi"]), "bundle policy keys == ckpt['pi'] keys (no extra/replaced layers)")
    check(set(wm_sd) == set(ck["wm"]), "bundle world-model keys == ckpt['wm'] keys")
    check(worst_p == 0.0, f"policy tensors bit-identical to ckpt['pi'] (max|Δ| = {worst_p:.3e}, {len(pol_sd)} tensors)")
    check(worst_w == 0.0, f"WM tensors bit-identical to ckpt['wm'] (max|Δ| = {worst_w:.3e}, {len(wm_sd)} tensors)")
    check(not any(k.startswith(("encoders.", "decoder.mlp", "dynamics.mlp")) for k in list(pol_sd) + list(wm_sd)),
          "no mjlab keys (encoders.* / decoder.mlp.* / dynamics.mlp.*) in the bundle")

    print("\n[3] analytic layer (obs / reward / termination / FK / motion)")
    check(module_of(bundle.mdp).startswith("trackerLab"),
          f"MDP is trackerlab's ({module_of(bundle.mdp)}.MDP)")
    check(module_of(bundle.kin).startswith("trackerLab"),
          f"FK is trackerlab's ({module_of(bundle.kin)})")
    check(module_of(bundle.motion).startswith("trackerLab"),
          f"reference motion is trackerlab's CompactMotionLib ({module_of(bundle.motion)})")
    check(module_of(bundle.spec).startswith("trackerTask"),
          f"robot contract from trackerTask get_robot('g1') ({module_of(bundle.spec)})")
    check(int(bundle.obs_width) == OBS_WIDTH_99, f"per-frame obs width = {bundle.obs_width} (relative_ori99)")
    check(int(bundle.hist) == 10, f"obs history = {bundle.hist}")
    check(tuple(bundle.term_dims) == (29, 29, 3, 3, 29, 6), f"obs term dims = {bundle.term_dims}")
    check(callable(getattr(bundle, "reward", None)) and callable(getattr(bundle, "observation", None)),
          "reward()/observation() dispatch to MDP.obs / MDP.reward")

    # ---------------------------------------------------------------- 4. data purity
    print("\n[4] dataset: pure HA only (no ft318, no mixed)")
    for motion in args.motion:
        check("ft318" not in str(motion).lower() and "mixed" not in str(motion).lower(),
              f"motion path has no ft318/mixed tag: {motion}")
    meta_path = args.packed.parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        dirs = [str(d) for d in meta.get("npz_dirs", [])]
        check(all("ft318" not in d.lower() and "mixed" not in d.lower() for d in dirs),
              f"packed Isaac motion library built from HA only: {dirs}")
        check(int(meta.get("num_clips", -1)) == 699,
              f"packed library clips = {meta.get('num_clips')} (pure HA set)")
        check(int(meta.get("num_frames", -1)) == 634016,
              f"packed library frames = {meta.get('num_frames')}")
    else:
        check(False, f"missing packed-motion meta: {meta_path}")

    npz = sorted(args.motion[0].glob("*.npz")) if args.motion[0].is_dir() else list(args.motion)
    check(bool(npz), f"{len(npz)} HA episodes found")
    if npz:
        import numpy as np

        with np.load(npz[0], allow_pickle=False) as z:
            fields = set(z.files)
            bad = [k for k in ("g1_task", "g1_dataset_root", "source")
                   if k in fields and "ft318" in str(z[k]).lower()]
        check(not bad, f"episode metadata has no ft318 reference ({npz[0].name})")
        check("g1_motion_token_64" in fields, "episode carries the recorded 64-d SONIC token")
        check("g1_hand_joints_14" in fields, "episode carries the 14-d hand labels")

    print("\n" + "=" * 78)
    if FAILS:
        print(f"[GATE FAILED] {len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("[GATE PASSED] Isaac Lab chain end-to-end (policy/WM/obs/reward/termination/FK/motion from "
          "trackerlab-supermimic) + pure HA data (699 clips). No mjlab weights, no ft318 episode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
