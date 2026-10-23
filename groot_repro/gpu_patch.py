"""Run Gr00tPolicy with a GPU/CPU split.

The checkpoint is 6.29 GiB in bf16 (1.52 B backbone + 1.62 B action head) while
this machine only has ~4 GiB of VRAM free next to the desktop.  `device_map`
lets Transformers place as many blocks as fit on the GPU and keep the rest in
host RAM; activations hop between devices through accelerate hooks.  The
`model.to(...)` that `Gr00tPolicy` performs after loading is neutralised
because dispatched models must not be moved.
"""

from __future__ import annotations

import torch
from transformers import AutoModel


def install(max_gpu_gib: str = "3.4GiB", max_cpu_gib: str = "22GiB") -> None:
    if getattr(AutoModel.from_pretrained, "_groot_device_map", False):
        return

    original = AutoModel.from_pretrained

    def from_pretrained(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("device_map", "auto")
        kwargs.setdefault(
            "max_memory", {0: max_gpu_gib, "cpu": max_cpu_gib}
        )
        kwargs.setdefault("dtype", torch.bfloat16)
        model = original(*args, **kwargs)
        model.to = lambda *a, **k: model  # dispatched models must not be moved
        return model

    from_pretrained._groot_device_map = True  # type: ignore[attr-defined]
    AutoModel.from_pretrained = from_pretrained
