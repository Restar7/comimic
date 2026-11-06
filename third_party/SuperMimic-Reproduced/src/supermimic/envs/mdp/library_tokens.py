"""Recorded SONIC token source: read the dataset's own 64-d tokens per frame.

Why this exists (comimic, 2026-10-28)
------------------------------------
Our 318-clip real-robot G1 dataset ships the **deployment** token for every frame:
``data/npz/g1_tasks/*.npz::g1_motion_token_64`` is bit-identical to the parquet column
``action.motion_token`` (measured: 2993/2993 frames, max|Δ| = 0, all values on the FSQ grid
``k/16``).  Re-computing the token online would need the planner *command stream*, which this
dataset does not contain (only ``action.wbc`` + the executed trajectory), so an online builder
could only approximate it (mean|Δ| = 0.39 rad, best lag +8 frames).  Reading the recorded token
is therefore both exact and cheaper.

Not a "port sidecar": the retired sidecar was a *re-derived* token in a different space
(corr 0.057 vs the labels, see ``scripts/train_joint_v2.py`` S0.4 note).  This module reads the
dataset's label itself — the same space the downstream VLA is supervised in.

File layout (aligned with ``MotionLibrary``'s global frame axis, produced by
``scripts/build_mjlab_tokens.py``)::

    {"tokens": (N, 64) fp16, "names": [clip, ...], "starts": [int], "lengths": [int]}

Signature is kept compatible with ``SonicTokenBuilder`` so ``Trainer.act`` and
``imagination/rollout.py`` can call either source interchangeably.
"""

from __future__ import annotations

from pathlib import Path

import torch


class LibraryTokens:
    """``__call__(frame, ...) -> (N, 64)`` from a per-frame token table."""

    def __init__(self, path: str | Path, device: torch.device, num_future: int = 10):
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        tok = d.get("tokens")
        if tok is None or tok.ndim != 2:
            raise ValueError(f"{path}: expected a 2-D 'tokens' tensor, got {type(tok)}")
        self.path = Path(path)
        self.tokens = tok.to(device=device, dtype=torch.float32).contiguous()
        self.n = int(self.tokens.shape[0])
        self.dim = int(self.tokens.shape[1])
        self.names = d.get("names")
        self.space = str(d.get("space", "label"))
        #: kept for parity with ``SonicTokenBuilder`` (unused here)
        self.num_future = int(num_future)

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor, ref=None, state=None, robot_quat=None) -> torch.Tensor:
        """Return the recorded token of each frame.  ``ref`` / ``state`` / ``robot_quat`` are ignored."""
        idx = frame.to(self.tokens.device, dtype=torch.long)
        return self.tokens[idx]

    def audit(self) -> str:
        return (
            f"[tokens] recorded dataset tokens: {self.path} | {self.n} frames x {self.dim} dim "
            f"| space={self.space} (clips={len(self.names) if self.names else '?'})"
        )


__all__ = ["LibraryTokens"]
