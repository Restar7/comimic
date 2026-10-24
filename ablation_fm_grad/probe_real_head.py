#!/usr/bin/env python3
"""Memory/time probe of FM gradient backends on the *real* GR00T N1.7 action head.

Loads only ``action_head.*`` weights from the official checkpoint (no 3B VLM),
builds dummy backbone/state features, and compares pathwise vs checkpointed
sampling with the same initial noise.

    python -m ablation_fm_grad.probe_real_head --batch 1 --steps 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "Isaac-GR00T") not in sys.path:
    sys.path.insert(0, str(ROOT / "Isaac-GR00T"))

from safetensors import safe_open  # noqa: E402
from transformers.feature_extraction_utils import BatchFeature  # noqa: E402

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config  # noqa: E402
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead  # noqa: E402

from diff_vla_wbc.gr00t_fm import sample_fm_from_features  # noqa: E402

DEFAULT_SNAPSHOT = (
    "/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface/hub/"
    "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
)


def load_head(snapshot: Path, device: str, dtype: torch.dtype) -> Gr00tN1d7ActionHead:
    raw = json.loads((snapshot / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(Gr00tN1d7Config)}
    kwargs = {k: v for k, v in raw.items() if k in fields and v is not None}
    cfg = Gr00tN1d7Config(**kwargs)
    head = Gr00tN1d7ActionHead(cfg)

    state = {}
    for shard in sorted(snapshot.glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith("action_head."):
                    state[key[len("action_head.") :]] = f.get_tensor(key)
    missing, unexpected = head.load_state_dict(state, strict=False)
    print(f"[load] action-head tensors={len(state)} missing={len(missing)} unexpected={len(unexpected)}")
    head = head.to(device=device, dtype=dtype).eval()
    head.requires_grad_(True)
    return head


def build_features(head, batch: int, seq_len: int, device: str, dtype: torch.dtype):
    cfg = head.config
    vl = torch.randn(batch, seq_len, cfg.backbone_embedding_dim, device=device, dtype=dtype)
    image_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)
    image_mask[:, : max(1, seq_len // 2)] = True
    attn = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
    backbone = BatchFeature(
        data={"backbone_features": vl, "image_mask": image_mask, "backbone_attention_mask": attn}
    )
    action_input = BatchFeature(
        data={
            "state": torch.randn(batch, 1, cfg.max_state_dim, device=device, dtype=dtype),
            "embodiment_id": torch.zeros(batch, dtype=torch.long, device=device),
        }
    )
    with torch.no_grad():
        feats = head._encode_features(backbone, action_input)
    return backbone, action_input, feats


def run_once(head, feats, backbone, action_input, steps: int, noise, checkpoint_steps: bool):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    out = sample_fm_from_features(
        head,
        feats.backbone_features,
        feats.state_features,
        action_input.embodiment_id,
        backbone_output=backbone,
        initial_noise=noise,
        num_steps=steps,
        checkpoint_steps=checkpoint_steps,
    )
    loss = (out.float() ** 2).mean()
    loss.backward()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / (1024**2)
    grads = {n: p.grad.detach().clone() for n, p in head.named_parameters() if p.grad is not None}
    return float(loss.detach()), mem, dt, out.detach(), grads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=Path(DEFAULT_SNAPSHOT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    head = load_head(args.snapshot, args.device, dtype)
    backbone, action_input, feats = build_features(head, args.batch, args.seq_len, args.device, dtype)
    noise = torch.randn(args.batch, head.config.action_horizon, head.config.max_action_dim,
                        device=args.device, dtype=dtype)

    print(f"[probe] batch={args.batch} seq_len={args.seq_len} K={args.steps} dtype={args.dtype}")
    # warm up kernels / allocator for both paths, then measure the second run
    for ckpt in (False, True):
        head.zero_grad(set_to_none=True)
        run_once(head, feats, backbone, action_input, args.steps, noise, ckpt)
        torch.cuda.empty_cache()
    rows = []
    for ckpt in (False, True):
        head.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        loss, mem, dt, out, grads = run_once(head, feats, backbone, action_input, args.steps, noise, ckpt)
        rows.append((ckpt, loss, mem, dt, out, grads))
        print(f"  checkpoint={ckpt!s:5s} loss={loss:.4f} peak={mem:8.1f} MB time={dt:.3f}s")

    ref = rows[0][5]
    test = rows[1][5]
    max_diff = max((test[n] - ref[n]).abs().max().item() for n in ref)
    out_diff = (rows[0][4] - rows[1][4]).abs().max().item()
    print(f"[check] grad max|diff|={max_diff:.3e}  token max|diff|={out_diff:.3e}")
    print(f"[check] memory saving = {(1 - rows[1][2] / rows[0][2]) * 100:.1f}%  "
          f"time overhead = {rows[1][3] / rows[0][3]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
