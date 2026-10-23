"""Memory-lean fallback for the gated Cosmos-Reason2-2B backbone.

The published GR00T checkpoint stores every backbone weight in its own
safetensors shards, so the backbone never needs to be downloaded.  The stock
loader, however, first materialises a random-init copy of the backbone from
`config.model_name` before the GR00T shards overwrite it.  That costs an extra
~3 GB of RAM/VRAM, which does not fit next to the 6.3 GB of shards on a small
workstation.  Building that throw-away copy on the meta device is numerically
irrelevant (the weights are overwritten) and keeps memory at the final size.
"""

from __future__ import annotations

import torch
import transformers


def install() -> None:
    cls = transformers.Qwen3VLForConditionalGeneration
    if getattr(cls.from_pretrained, "_groot_meta_patch", False):
        return

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):  # type: ignore[no-untyped-def]
        cfg_kwargs = {
            k: kwargs[k]
            for k in ("trust_remote_code", "revision", "cache_dir", "token")
            if k in kwargs
        }
        cfg = transformers.Qwen3VLConfig.from_pretrained(
            pretrained_model_name_or_path, **cfg_kwargs
        )
        with torch.device("meta"):
            model = cls(cfg)
        return model.eval()

    from_pretrained._groot_meta_patch = True  # type: ignore[attr-defined]
    cls.from_pretrained = from_pretrained
