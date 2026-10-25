#!/usr/bin/env python3
"""Recompute the 64-d SONIC token for every HA clip *in the WBC's space* (port assembly).

Why: the label stored in the HA npz (`g1_motion_token_64`) comes from the deployment
`encoder_input (1762)` front-end; the trackerlab WBC is conditioned on the 640-d port
assembly (10 future reference frames + anchor). Measured with
`scripts/probe_token_shift.py` / `probe_port_tokens.py`, the port tokens drive the bundle
~2x better than the stored labels once the action-scale口径 is fixed (0.070/2.87 vs
0.052/6.48). This script writes the port tokens as a sidecar so ``RecordedTokenSource``,
the eval ``token_mse`` and the joint finetune all see the same space.

Output: ``data/port_tokens/ha_tokens/<clip>.npy``  (T, 64) float32
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "trackerlab-supermimic/source/trackerLab"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from trackerLab.algorithms.supermimic.sonic_token import SonicTokenBuilder  # noqa: E402


class ShimLib:
    def __init__(self, jp, jv, quat):
        self.q, self.qd, self._quat, self.n = jp, jv, quat, jp.shape[0]
    def last_frame_of(self, frame):
        return torch.full_like(frame, self.n - 1)
    def root_quat(self, frames):
        return self._quat[frames]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--out", type=Path, default=ROOT / "data/port_tokens/ha_tokens")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.src.glob("*.npz"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        raise SystemExit(f"no npz under {a.src}")
    print(f"[port] {len(files)} clips -> {a.out}")
    for i, path in enumerate(files):
        dst = a.out / f"{path.stem}.npy"
        if dst.exists():
            continue
        with np.load(path) as d:
            jp = torch.from_numpy(np.asarray(d["joint_pos"], dtype=np.float32))
            jv = torch.from_numpy(np.asarray(d["joint_vel"], dtype=np.float32))
            quat = torch.from_numpy(np.asarray(d["body_quat_w"], dtype=np.float32)[:, 0])
        builder = SonicTokenBuilder(ShimLib(jp, jv, quat), a.device, num_future=10)
        T = jp.shape[0]
        with torch.no_grad():
            tok = builder(torch.arange(T), robot_quat=quat)
        np.save(dst, tok.float().cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    print(f"[port] done -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
