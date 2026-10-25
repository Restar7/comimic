#!/usr/bin/env python3
"""Audit / convert a trackerlab-style SuperMimic checkpoint into our bundle's layout.

Two SuperMimic implementations coexist:

* **trackerlab-supermimic** (Isaac Lab / PhysX, trained in the other container) saves
  ``{"pi": {...}, "wm": {...}, "critic": ..., "args": ..., "reward_cfg": ...}``
  with module names ``enc.*`` / ``dec.*`` and an MLP+LayerNorm world model ``net.*``.
* **third_party/SuperMimic-Reproduced** (mjlab / MuJoCo-Warp, what our joint trainer loads)
  saves ``{"policy": {...}, "wm": {...}}`` with names ``encoders.motion.mlp.*`` /
  ``decoder.mlp.*`` / ``dynamics.mlp.*`` and a plain-MLP world model.

This script never trains and never writes into the source checkpoint.  It maps layer by
layer **and verifies every shape**; anything that does not map exactly is reported and the
conversion is refused (``--allow-partial`` overrides, for inspection only).

Usage::

    python3 scripts/adapt_trackerlab_ckpt.py \
        --ckpt ../trackerlab-supermimic/logs/bones_seed_token_2x8192/ckpt_000500.pt \
        --reference third_party/SuperMimic-Reproduced/logs/e9_bones_seed_token_h200x2/ckpt_0024000.pt \
        --report runs/adapt_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

#: trackerlab group -> our group
GROUP_MAP = {"pi": "policy", "wm": "wm"}

#: trackerlab key prefix -> our key prefix (the hidden-layer numbering is identical)
PREFIX_MAP = {
    "enc.": "encoders.motion.mlp.",      # token/motion encoder tower
    "dec.": "decoder.mlp.",              # action decoder tower
}

#: trackerlab world-model linear layers -> our index (theirs interleaves LayerNorm)
WM_LINEAR_INDEX = {0: 0, 3: 2, 6: 4, 9: 6}


def _shapes(sd) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}


def audit(ckpt_path: Path, reference_path: Path) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ref = torch.load(reference_path, map_location="cpu", weights_only=False)

    report: dict = {
        "ckpt": str(ckpt_path),
        "reference": str(reference_path),
        "interface": (ck.get("args") or {}).get("interface"),
        "iteration": ck.get("iteration"),
        "groups": {},
        "mapped": [],
        "shape_mismatch": [],
        "unmapped_source": [],
        "unmapped_reference": [],
        "notes": [],
    }

    for src_group, dst_group in GROUP_MAP.items():
        if src_group not in ck or dst_group not in ref:
            report["notes"].append(f"missing group: {src_group} / {dst_group}")
            continue
        src = _shapes(ck[src_group])
        dst = _shapes(ref[dst_group])
        used_dst: set[str] = set()
        info = {"source_keys": len(src), "reference_keys": len(dst), "layers": []}

        for sk, sshape in sorted(src.items()):
            dk = None
            if src_group == "pi":
                for pre, dpre in PREFIX_MAP.items():
                    if sk.startswith(pre):
                        # our towers carry an extra ``net.`` level: enc.0.weight
                        # -> encoders.motion.mlp.net.0.weight
                        dk = dpre + "net." + sk[len(pre):]
                        break
            else:  # world model: net.<idx>.*, idx 0/3/6/9 are the linears
                if sk.startswith("net."):
                    parts = sk.split(".")
                    if parts[1].isdigit() and int(parts[1]) in WM_LINEAR_INDEX:
                        dk = "dynamics.mlp.net." + str(WM_LINEAR_INDEX[int(parts[1])]) + "." + parts[2]
                elif sk.startswith("in_norm") or sk.startswith("out_norm"):
                    dk = "dynamics." + sk  # reported, not mapped (we normalise elsewhere)
            if dk is None:
                if "norm" in sk:
                    report["notes"].append(
                        f"{src_group}.{sk}: their normalisation stats; ours lives in "
                        f"policy.decoder.scale / encoders.motion.scale (not a 1:1 tensor)")
                else:
                    report["unmapped_source"].append(f"{src_group}.{sk}")
                continue
            dshape = dst.get(dk)
            row = {"source": f"{src_group}.{sk}", "target": f"{dst_group}.{dk}", "source_shape": sshape}
            if dshape is None:
                row["status"] = "target-missing"
                report["shape_mismatch"].append(row)
            elif dshape != sshape:
                row["status"] = "shape-mismatch"
                row["target_shape"] = dshape
                report["shape_mismatch"].append(row)
            else:
                row["status"] = "ok"
                used_dst.add(dk)
                report["mapped"].append(row)
            info["layers"].append(row)

        for dk in sorted(dst):
            if dk not in used_dst:
                report["unmapped_reference"].append(f"{dst_group}.{dk}")
        report["groups"][f"{src_group}->{dst_group}"] = info

    # Structural facts worth printing even when everything maps.
    pi_src = _shapes(ck.get("pi", {}))
    if "enc.0.weight" in pi_src:
        report["notes"].append(
            f"policy encoder input width {pi_src['enc.0.weight'][1]} "
            f"(token interface removes the encoder entirely: encoder input becomes 64)")
    if "dec.0.weight" in pi_src:
        report["notes"].append(
            f"policy decoder input width {pi_src['dec.0.weight'][1]} "
            f"(= latent + obs history; ours is 64 token + 930 obs)")
    wm_src = _shapes(ck.get("wm", {}))
    ln = [k for k, v in wm_src.items() if k.split(".")[-2] == "1" or k.split(".")[-2] == "4"
          or k.split(".")[-2] == "7"]
    if ln:
        report["notes"].append(
            f"world model uses LayerNorm ({len(ln)} params: {ln[:4]}); our dynamics MLP has none")
    return report


def convert(ckpt_path: Path, reference_path: Path, out_path: Path, allow_partial: bool) -> dict:
    """Write our-format weights for every layer that maps 1:1. Refuses on any mismatch."""
    rep = audit(ckpt_path, reference_path)
    if rep["shape_mismatch"] and not allow_partial:
        print("[refuse] shape mismatches found; rerun with --allow-partial to inspect only",
              file=sys.stderr)
        return rep
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    new = {"policy": {}, "wm": {}}
    for row in rep["mapped"]:
        sg, sk = row["source"].split(".", 1)
        dg, dk = row["target"].split(".", 1)
        new[dg][dk] = ck[sg][sk].clone()
    torch.save(
        {
            "policy": new["policy"],
            "wm": new["wm"],
            "adapted_from": str(ckpt_path),
            "adapted_iteration": ck.get("iteration"),
            "args": ck.get("args"),
            "reward_cfg": ck.get("reward_cfg"),
            "partial": bool(rep["shape_mismatch"]),
        },
        out_path,
    )
    print(f"[written] {out_path} ({len(new['policy'])} policy + {len(new['wm'])} wm tensors)")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="trackerlab ckpt_*.pt / last.pt")
    ap.add_argument("--reference", type=Path,
                    default=ROOT / "third_party/SuperMimic-Reproduced/logs/"
                                   "e9_bones_seed_token_h200x2/ckpt_0024000.pt",
                    help="our ckpt whose layout we target (only used to read shapes)")
    ap.add_argument("--out", type=Path, default=None, help="write a converted ckpt here")
    ap.add_argument("--report", type=Path, default=None, help="write the audit JSON here")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    rep = audit(args.ckpt, args.reference)
    ok = len(rep["mapped"])
    bad = len(rep["shape_mismatch"])

    print(f"interface: {rep['interface']}  iteration: {rep['iteration']}")
    print(f"mapped 1:1 : {ok}")
    print(f"mismatched : {bad}")
    for row in rep["shape_mismatch"][:12]:
        print(f"  ! {row['source']} {row['source_shape']} vs {row.get('target_shape')}")
    print(f"unmapped source ({len(rep['unmapped_source'])}): {rep['unmapped_source'][:8]}")
    print(f"unmapped ref    ({len(rep['unmapped_reference'])}): {rep['unmapped_reference'][:8]}")
    for note in rep["notes"]:
        print(f"  note: {note}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=1))
        print(f"[report] {args.report}")
    if args.out:
        convert(args.ckpt, args.reference, args.out, args.allow_partial)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
