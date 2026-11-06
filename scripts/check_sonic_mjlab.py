#!/usr/bin/env python3
"""Self-check for :mod:`diff_vla_wbc.sonic_mjlab` (the SONIC executor of the mjlab joint FT).

四项检查，全过才算接线正确：

1. **decoder 图对拍**：``SonicDecoderTorch`` vs onnxruntime，``max|Δ|`` 应 ~1e-6；
2. **观测拼接**：``obs930_to_obs93`` + 通道重排，与"直接用逐帧 93 维块拼 994"逐位相等；
3. **动作空间往返**：``ha_to_std(std_to_ha(x)) == x``，且不改变数值；
4. **可微性**：``∂a/∂token ≠ 0``（联合微调的命门；原地赋值会静默断图）。

用法::

    third_party/SuperMimic-Reproduced/.venv/bin/python scripts/check_sonic_mjlab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diff_vla_wbc.mjlab_joint import TERM_DIMS  # noqa: E402
from diff_vla_wbc.sonic_decoder import build_obs_994, supermimic_obs_to_blocks  # noqa: E402
from diff_vla_wbc.sonic_mjlab import (  # noqa: E402
    DEFAULT_DECODER_ONNX,
    SonicExecutor,
    ha_to_std,
    obs930_to_obs93,
    std_to_ha,
)

H = 10
OBS = 93


def _fake_obs(b: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    obs93 = torch.randn(b, H, OBS, generator=g) * 0.1
    token = (torch.randint(-16, 16, (b, 64), generator=g).float() / 16.0)
    # term-major flatten (same as supermimic.world_model.heads.flatten_history_term_major)
    chunks, start = [], 0
    for w in TERM_DIMS:
        chunks.append(obs93[:, :, start:start + w].permute(0, 1, 2).reshape(b, H * w))
        start += w
    return torch.cat(chunks, dim=-1), token


def main() -> int:
    ok = True
    obs_flat, token = _fake_obs()

    # ---------------------------------------------------------------- 1 parity
    ex = SonicExecutor(DEFAULT_DECODER_ONNX)
    with torch.no_grad():
        obs994 = ex.obs994(obs_flat, torch.zeros_like(token))
        delta = ex.decoder.check_parity(obs994, DEFAULT_DECODER_ONNX)
    print(f"[1] decoder torch vs ORT max|Δ| = {delta:.3e}")
    ok &= delta < 1e-4

    # ---------------------------------------------------------------- 2 obs build
    obs93 = obs930_to_obs93(obs_flat)
    direct = supermimic_obs_to_blocks(obs93)
    idx = torch.tensor(std_to_ha())
    manual = build_obs_994(
        torch.zeros_like(token),
        direct["ang_vel"],
        direct["joint_pos"].index_select(-1, idx),
        direct["joint_vel"].index_select(-1, idx),
        direct["last_action"].index_select(-1, idx),
        direct["gravity"],
    )
    d2 = float((manual - obs994).abs().max())
    print(f"[2] obs930→994 vs manual build max|Δ| = {d2:.3e}")
    ok &= d2 < 1e-6
    # 语义核对：994 里的 q 块末帧 == 输入的 q-q0（未经置换时）
    qblk = obs994[:, 64 + 30:64 + 30 + 290].reshape(4, H, 29)
    back = qblk.index_select(-1, torch.tensor(ha_to_std()))
    d3 = float((back[:, -1] - direct["joint_pos"][:, -1]).abs().max())
    print(f"[2b] q 块往返（末帧）max|Δ| = {d3:.3e}")
    ok &= d3 < 1e-6

    # ---------------------------------------------------------------- 3 perm
    x = torch.arange(29.0)
    rt = x[torch.tensor(std_to_ha())][torch.tensor(ha_to_std())]
    d4 = float((rt - x).abs().max())
    print(f"[3] 关节序往返 max|Δ| = {d4:.3e}  perm(std→ha)={std_to_ha()[:6]}…")
    ok &= d4 == 0.0

    # ---------------------------------------------------------------- 4 grad
    tok = token.clone().requires_grad_(True)
    a = ex.act(tok, obs_flat)
    a.sum().backward()
    g = tok.grad
    print(f"[4] action shape {tuple(a.shape)}  |∂a/∂token| mean={float(g.abs().mean()):.3e} max={float(g.abs().max()):.3e}")
    ok &= bool(torch.isfinite(a).all()) and float(g.abs().max()) > 0.0

    # ---------------------------------------------------------------- 5 与 ORT 端到端
    import onnxruntime as ort

    sess = ort.InferenceSession(str(DEFAULT_DECODER_ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    obs_with_tok = obs994.clone()
    obs_with_tok[:, :64] = token
    with torch.no_grad():
        ref = np.concatenate([
            sess.run(None, {name: obs_with_tok[i:i + 1].numpy()})[0]
            for i in range(obs_with_tok.shape[0])
        ])
    ref = ref[:, ha_to_std()]   # ORT 是 HA 序，比较前换回 bundle 的 STANDARD 序
    d5 = float(np.abs(ref - a.detach().numpy()).max())
    print(f"[5] 端到端 action vs ORT max|Δ| = {d5:.3e}")
    ok &= d5 < 1e-4

    print("\nALL CHECKS PASS" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
