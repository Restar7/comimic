#!/usr/bin/env python3
"""Parity check: async prefetch / step memo must reproduce ``processed_batch`` bit-exactly.

Runs on CPU (no GPU needed), so it can be executed while the card is busy:
    python3 scripts/check_prefetch_parity.py --frames 2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diff_vla_wbc.prefetch import BatchPrefetcher, StepMemo  # noqa: E402


def compare(a, b, path="") -> int:
    n = 0
    if hasattr(a, "keys") and hasattr(b, "keys"):   # dict / transformers.BatchFeature / UserDict
        ka, kb = list(a.keys()), list(b.keys())
        assert set(ka) == set(kb), f"key mismatch at {path}: {set(ka) ^ set(kb)}"
        for k in ka:
            n += compare(a[k], b[k], f"{path}.{k}")
    elif torch.is_tensor(a):
        assert torch.equal(a, b), f"tensor mismatch at {path}: max|d|={float((a-b).abs().max())}"
        n += 1
    elif hasattr(a, "__len__") and not isinstance(a, str):
        assert len(a) == len(b), f"length mismatch at {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            n += compare(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"value mismatch at {path}"
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--vlm-path", type=Path, default=ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights")
    a = ap.parse_args()

    files = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    from diff_vla_wbc.g1_tasks import (TaskFrameIndex, TaskObservationBuilder,
                                       load_gr00t_processor, processed_batch)
    index = TaskFrameIndex(files)
    builder = TaskObservationBuilder(index)
    processor = load_gr00t_processor(a.ckpt, vlm_path=a.vlm_path)
    frames = torch.arange(1000, 1000 + a.frames * 97, 97)[: a.frames]
    print(f"[parity] frames={frames.tolist()}")

    ref = processed_batch(processor, builder, frames)
    memo = StepMemo(lambda f: processed_batch(processor, builder, f))
    got_memo = memo.get(frames)
    n1 = compare(ref, got_memo)
    print(f"[parity] StepMemo: {n1} fields identical")

    pre = BatchPrefetcher(lambda f: processed_batch(processor, builder, f))
    pre.submit(frames)
    got_pre = pre.get(frames, lambda: processed_batch(processor, builder, frames))
    n2 = compare(ref, got_pre)
    print(f"[parity] BatchPrefetcher: {n2} fields identical  stats={pre.stats()}")
    pre.close()

    # 同一组 frames 第二次取：必须命中缓存，且仍然相同
    got_again = memo.get(frames)
    compare(ref, got_again)
    print(f"[parity] memo hit twice OK  stats={memo.stats()}")
    print("[parity] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
