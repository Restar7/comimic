#!/usr/bin/env python3
"""部署契约端到端自检：**真 ckpt** → `load_delta_token_v5_from_checkpoint` → `delta_from_chunk_full_v5`。

检查（全绿才算 v5 checkpoint 可交付）：
1. ckpt 里的 spec 带 `T/K/H`、`state_joint_order=ha_v31`、`hist_joint_order=ha_v31`、
   两个 default（`hist_default_joint_pos` 与 `default_joint_pos`）、`out_mode=per_frame`；
2. 用**部署序（HA_V31）**的 state64 / state_hist_10 调用 `delta_from_chunk_full_v5`，
   输出 `(40,64)`、`t≥K` 恒 0、`‖Δ‖≤cap`；
3. 同一份量换成训练序（standard）再调用，输出必须**逐位相同**（证明换序是唯一的差异）；
4. `apply_delta_per_frame` 只在 `t<K` 上加；`t≥K` 非零会抛。

用法::

    source scripts/harness03_env.sh
    python3 scripts/verify_v5_deploy_contract.py --ckpt runs/.../checkpoint_actor_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diff_vla_wbc.delta_token_io_v5 import (                    # noqa: E402
    DEPLOY_HIST_JOINT_ORDER, DEPLOY_STATE_JOINT_ORDER, TRAIN_HIST_JOINT_ORDER,
    TRAIN_STATE_JOINT_ORDER, apply_delta_per_frame, delta_from_chunk_full_v5,
    joint_permutation, load_delta_token_v5_from_checkpoint)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(int(a.seed))
    obj = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    net, spec = load_delta_token_v5_from_checkpoint(obj, device="cpu")
    if net is None:
        raise SystemExit("[verify] 不是 v5 checkpoint")
    print(f"[verify] ckpt={a.ckpt.name} params={net.n_params:,}")
    print(f"[verify] spec: T/K/H={spec.T}/{spec.K}/{spec.H} cap={spec.cap} "
          f"out_mode={spec.out_mode} state_order={spec.state_joint_order} "
          f"hist_order={spec.hist_joint_order} hist_clock={spec.hist_clock}")
    assert spec.out_mode == "per_frame"
    assert spec.state_joint_order == DEPLOY_STATE_JOINT_ORDER
    assert spec.hist_joint_order == DEPLOY_HIST_JOINT_ORDER
    assert "hist_default_joint_pos" in spec.meta and "default_joint_pos" in spec.meta
    assert float(spec.cap) > 0

    T, K = int(spec.T), int(spec.K)
    Z = rng.normal(size=(T, 64)).astype(np.float32)
    # 部署序（HA_V31）的 state64 与 hist10
    q_std = rng.normal(size=29).astype(np.float32) * 0.2
    qd_std = rng.normal(size=29).astype(np.float32) * 0.2
    p_s2h = joint_permutation(TRAIN_STATE_JOINT_ORDER, DEPLOY_STATE_JOINT_ORDER)
    h_s2d = joint_permutation(TRAIN_HIST_JOINT_ORDER, DEPLOY_HIST_JOINT_ORDER)
    state64_deploy = np.concatenate([rng.normal(size=6).astype(np.float32),
                                     q_std[p_s2h], qd_std[p_s2h]]).astype(np.float32)
    hist_q = rng.normal(size=(10, 29)).astype(np.float32) * 0.1
    hist_qd = rng.normal(size=(10, 29)).astype(np.float32) * 0.1
    hist10_deploy = np.concatenate([hist_q[:, h_s2d], hist_qd[:, h_s2d]], -1).astype(np.float32)

    d_deploy = delta_from_chunk_full_v5(Z, state64_deploy, hist10_deploy, net, spec)
    # 期望值：把**训练序**的张量直接喂网络（不经过契约层的换序）——两者必须逐位相同，
    # 这就是"契约层只做换序、不做任何其它事情"的证明
    state64_train = np.concatenate([state64_deploy[:6], q_std, qd_std]).astype(np.float32)
    hist10_train = np.concatenate([hist_q, hist_qd], -1).astype(np.float32)
    mask = torch.zeros(1, T)
    mask[:, :K] = 1.0
    with torch.no_grad():
        d_ref = net(torch.as_tensor(Z)[None], torch.as_tensor(state64_train)[None],
                    torch.as_tensor(hist10_train)[None], mask, cap=float(spec.cap))[0].numpy()
    print(f"[verify] Δ deploy-order vs 直连训练序: max|diff|={float(np.abs(d_deploy-d_ref).max()):.3e}")
    assert float(np.abs(d_deploy - d_ref).max()) <= 1e-6, "换序口径不一致"

    assert d_deploy.shape == (T, 64)
    assert float(np.abs(d_deploy[K:]).max()) == 0.0
    n = float(np.linalg.norm(d_deploy, axis=-1).max())
    assert n <= float(spec.cap) + 1e-6, n
    chunk = rng.normal(size=(T, 64)).astype(np.float32) * 10.0
    out = apply_delta_per_frame(chunk, d_deploy, spec)
    assert np.allclose(out[K:], chunk[K:]), "t≥K 必须保持不变"
    assert np.allclose(out[:K], chunk[:K] + d_deploy[:K])
    bad = d_deploy.copy()
    bad[K + 3, 0] = 1e-3
    try:
        apply_delta_per_frame(chunk, bad, spec)
    except Exception as exc:  # noqa: BLE001
        print(f"[verify] t≥K 非零 ⇒ 正确抛错：{type(exc).__name__}")
    else:  # pragma: no cover
        raise SystemExit("[verify] FAIL：t≥K 非零没有抛错")
    print(f"[verify] PASS：‖Δ‖max={n:.4f} ≤ cap={spec.cap}，t≥K 恒 0，换序一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
