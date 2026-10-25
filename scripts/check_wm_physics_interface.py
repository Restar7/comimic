#!/usr/bin/env python3
"""Physics-interface checks for the SuperMimic world model's integration.

Three properties, all independent of the learned weights (the last MLP layer is
zero-initialised, so a fresh ``WorldModel`` outputs exactly zero and the step reduces to the
state prior):

1. ``targets`` and ``step`` are exact inverses: integrating ``targets(S, S1)`` returns S1.
2. Zero output ⇒ constant-velocity extrapolation with **world-frame** root velocity:
   ``p' = p + dt*v``, ``R' = exp(dt*[w]) @ R``, ``q' = q + dt*qd``.
   (The state's ``w`` is ``data.root_link_ang_vel_w`` — see ``tracker_env/supermimic/env.py``.)
3. The body-frame convention is the one mjlab uses (``root_ang_vel_b``): rotating ``w`` into
   the body frame before the right-multiplied exponential must be what makes (2) hold;
   the naive ``R @ exp(dt*w)`` must *fail* the check with a visible error.

Usage:
    python3 scripts/check_wm_physics_interface.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "trackerlab-supermimic/source/trackerLab",):
    sys.path.insert(0, str(p))

from trackerLab.algorithms.supermimic.networks import WorldModel  # noqa: E402
from trackerLab.kinematics.supermimic.rotations import so3_exp, so3_log  # noqa: E402

NJ = 29
B = 8
DT = 0.02


def rand_state(seed: int):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(B, 3, generator=g)
    rot = torch.randn(B, 3, generator=g) * 0.8
    R = so3_exp(rot)
    v = torch.randn(B, 3, generator=g)
    w = torch.randn(B, 3, generator=g) * 2.0
    q = torch.randn(B, NJ, generator=g) * 0.5
    qd = torch.randn(B, NJ, generator=g)
    return (p, R, v, w, q, qd)


def integrate(S, d):
    """Apply the (unscaled) delta exactly the way ``WorldModel.step`` does."""
    p, R, v, w, q, qd = S
    dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NJ, NJ), -1)
    v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
    w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
    qd1 = qd + dqd
    from trackerLab.kinematics.supermimic.rotations import yaw_mat

    H = yaw_mat(R)
    p1 = p + DT * v1 + (H @ dp.unsqueeze(-1)).squeeze(-1)
    w1_b = (R.transpose(-1, -2) @ w1.unsqueeze(-1)).squeeze(-1)
    R1 = R @ so3_exp(DT * w1_b + dth)
    return (p1, R1, v1, w1, q + DT * qd1 + dq, qd1)


def maxdiff(A, B):
    return max(float((a - b).abs().max()) for a, b in zip(A, B))


def main() -> int:
    wm = WorldModel(hidden=[2048, 2048, 2048], num_joints=NJ)
    wm.explicit_integration = True
    wm.dt = DT
    # a fresh WorldModel has a zero-initialised last layer and a zero-mean RunningNorm
    wm.eval()

    S = rand_state(0)
    S1 = rand_state(1)

    # ---- 1) targets/step inverse -------------------------------------------------
    t = wm.targets(S, S1)
    S_hat = integrate(S, t)
    inv = maxdiff(S_hat, S1)

    # ---- 2) zero output = constant-velocity prior (world w, left multiplication) --
    a = torch.zeros(B, NJ)
    xi = torch.zeros(B, 33)
    with torch.no_grad():
        out = wm.step(S, a, xi)
    p, R, v, w, q, qd = S
    want_p = p + DT * v
    want_R = so3_exp(DT * w) @ R          # world angular velocity -> left multiplication
    want_q = q + DT * qd
    prior = maxdiff((out[0], out[1], out[4]), (want_p, want_R, want_q))

    # the naive (body-frame) composition must NOT satisfy it — proves the check bites
    naive_R = R @ so3_exp(DT * w)
    naive = float((naive_R - want_R).abs().max())

    # ---- 3) state_error is zero on identical states ------------------------------
    err0 = float(wm.state_error(S, S).abs().max())

    # ---- 4) world->body round trip used by the fix -------------------------------
    w_b = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
    roundtrip = float((R @ w_b.unsqueeze(-1)).squeeze(-1).sub(w).abs().max())

    rows = [
        ("targets→integrate == S1 (exact inverse)", inv, 1e-5),
        ("zero output == world-frame constant-velocity prior", prior, 1e-5),
        ("naive R@exp(dt*w) differs (must be > 1e-2)", naive, None),
        ("state_error(S,S) == 0", err0, 1e-6),
        ("R @ (R^T w) == w", roundtrip, 1e-6),
    ]
    ok = True
    print(f"dt={DT}  B={B}  structure=explicit")
    for name, val, tol in rows:
        if tol is None:
            good = val > 1e-2
        else:
            good = val < tol
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {val:.3e}"
              + (f" (tol {tol:g})" if tol is not None else ""))
    print(("[PASS]" if ok else "[FAIL]") + " world-model physics interface")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
