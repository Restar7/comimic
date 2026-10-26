"""可微的动作编解码（GR00T min-max 归一化 ↔ 原始 token/手指空间）。

背景（2026-10-26 复核 J2）：`unitree_g1_sonic` 的 action 三个组都没有
`mean_std_embedding_keys`，因此走 **min-max 归一到 [-1,1]**（`use_percentiles=True`）。
官方推理必须 `decode_action` 才能拿回原始空间；但
`scripts/ha_vla_latent64_server.py` 与 `scripts/train_joint.py` 都直接把 head 输出当
token/手指用 —— token 被缩放+偏移，手指阈值 `>1.70` 在 [-1,1] 里永远不成立。

本模块用 torch 复刻同一条 min-max 逆变换（可微，能进训练图），并提供单测对拍工具。
"""
from __future__ import annotations

import numpy as np
import torch

KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")
DEFAULT_TAG = "unitree_g1_sonic"


class ActionCodec:
    """``decode``: 归一化空间 → 原始空间；``encode``: 原始 → 归一化（clamp 到 [-1,1]）。"""

    def __init__(self, processor, tag: str = DEFAULT_TAG, device: str | torch.device = "cpu"):
        sap = processor.state_action_processor
        cfg = processor.modality_configs[tag]["action"]
        ms_keys = getattr(cfg, "mean_std_embedding_keys", None)
        if ms_keys:
            raise NotImplementedError("action 使用 mean-std 归一化，需要另写分支")
        params = sap.norm_params[tag]["action"]
        lo, hi = [], []
        for k in KEYS:
            p = params[k]
            lo.append(np.asarray(p["min"], dtype=np.float32).reshape(-1))
            hi.append(np.asarray(p["max"], dtype=np.float32).reshape(-1))
        lo = np.concatenate(lo)
        hi = np.concatenate(hi)
        self.lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.rng = torch.tensor(np.maximum(hi - lo, 1e-8), dtype=torch.float32, device=device)
        self.dim = int(self.lo.numel())
        self.tag = tag
        self.device = device
        # 供审计用
        self.raw_min = lo
        self.raw_max = hi

    # ------------------------------------------------------------------ API
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., >=dim)`` 归一化 → 原始空间（不 clamp，保留梯度；越界由调用方记录）。"""
        y = x[..., : self.dim].float()
        return (y + 1.0) * 0.5 * self.rng.to(y.device) + self.lo.to(y.device)

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """``(..., dim)`` 原始 → [-1, 1]（训练目标用，clamp 与 GR00T `clip_outliers` 一致）。"""
        return (2.0 * (y.float() - self.lo.to(y.device)) / self.rng.to(y.device) - 1.0).clamp(-1.0, 1.0)

    def out_of_range_frac(self, x: torch.Tensor) -> float:
        y = x[..., : self.dim].float()
        return float(((y < -1.0) | (y > 1.0)).float().mean())

    def audit(self) -> str:
        return (f"[AUDIT][codec] tag={self.tag} dim={self.dim} "
                f"raw_min[{self.raw_min.min():.3f},{self.raw_max.max():.3f}] "
                f"rng_mean={float(self.rng.mean()):.4f}")


def main() -> int:
    """单测：codec.decode 与 GR00T 官方 decode_action 逐元素对拍 + encode/decode 往返。"""
    import argparse
    from pathlib import Path
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--vlm-path", default="groot_repro/nvidia/Cosmos-Reason2-2B-weights")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(".").resolve()))
    from diff_vla_wbc.g1_tasks import load_gr00t_processor
    from gr00t.data.embodiment_tags import EmbodimentTag

    proc = load_gr00t_processor(a.ckpt, vlm_path=a.vlm_path)
    codec = ActionCodec(proc)
    print(codec.audit())

    torch.manual_seed(0)
    x = (torch.rand(2, 40, codec.dim) * 2 - 1)
    y = codec.decode(x)

    ref = proc.decode_action(x.numpy(), EmbodimentTag(DEFAULT_TAG))
    ref_cat = np.concatenate([np.asarray(ref[k], dtype=np.float32) for k in KEYS], axis=-1)
    err = float(np.abs(y.numpy() - ref_cat).max())
    print(f"[codec-test] decode vs processor.decode_action: max|Δ| = {err:.2e}  "
          f"{'PASS' if err < 1e-4 else 'FAIL'}  shapes={y.shape} vs {ref_cat.shape}")

    rt = codec.encode(codec.decode(x))
    err2 = float((rt - x.clamp(-1, 1)).abs().max())
    print(f"[codec-test] encode(decode(x)) vs clamp(x): max|Δ| = {err2:.2e}  "
          f"{'PASS' if err2 < 1e-5 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
