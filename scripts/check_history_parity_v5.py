"""强口径 history parity：真的走"部署序（HA_V31）→ 训练序（standard）"的路径。

运行：
    source scripts/harness03_env.sh
    python3 scripts/check_history_parity_v5.py --eval-set <dir> --out <run_dir>
产物：`history_parity.json`（覆盖 run 里的弱口径版本，加 `protocol` 字段说明）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diff_vla_wbc.compat import g1_default_joint_pos                    # noqa: E402
from diff_vla_wbc.delta_token_io_v5 import (                            # noqa: E402
    DeltaTokenSpecV5, hist10_from_blocks, hist10_from_hist64, joint_permutation)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--state64", type=Path, default=None,
                    help="可选：windows 里已有 state64 的 .pt（否则由 history 反推 q/qd）")
    a = ap.parse_args()

    d = torch.load(str(Path(a.eval_set) / "windows.pt"), map_location="cpu", weights_only=False)
    n = min(int(a.n), int(d["history"].shape[0]))
    hist = d["history"].float()[:n]                     # (n,10,93) 训练序（standard）
    hist10_train = hist[:, :, :58].contiguous()         # [q-default | qd]，standard
    dq_std = g1_default_joint_pos().numpy()             # standard 序
    spec = DeltaTokenSpecV5(T=40, K=20, H=20, use_history=True,
                            meta=dict(hist_default_joint_pos=dq_std.tolist()))

    # ① 训练侧 q 块 vs `state`（决策帧）：差一个常数 default，且跨窗口恒定
    st = d["state"].float()[:n]
    off = hist10_train[:, -1, :29] - st[:, 7:36]
    d_est = -off.mean(0)
    const_err = float((off - off.mean(0, keepdim=True)).abs().max())
    q_err = float((hist10_train[:, -1, :29] - (st[:, 7:36] - d_est)).abs().max())
    qd_err = float((hist10_train[:, -1, 29:] - st[:, 42:71]).abs().max())

    # ② 部署路径 A：state64 序列（HA_V31）→ hist10（standard）
    p_s2h = joint_permutation("standard", "ha_v31")
    q_abs_std = (hist10_train[:, :, :29] + dq_std[None, None]).reshape(-1, 29).numpy()
    qd_std = hist10_train[:, :, 29:].reshape(-1, 29).numpy()
    hist64 = np.zeros((n, 10, 64), np.float32)
    hist64[:, :, 6:35] = q_abs_std[:, p_s2h].reshape(n, 10, 29)
    hist64[:, :, 35:64] = qd_std[:, p_s2h].reshape(n, 10, 29)
    deploy_a = np.stack([hist10_from_hist64(hist64[i], spec) for i in range(n)], 0)
    err_a = float(np.abs(deploy_a - hist10_train.numpy()).max())

    # ③ 部署路径 B：client decoder 已有的两块（HA_V31）→ hist10（standard）
    q_off_ha = hist10_train[:, :, :29].numpy()[:, :, p_s2h]
    qd_ha = hist10_train[:, :, 29:].numpy()[:, :, p_s2h]
    deploy_b = np.stack([hist10_from_blocks(q_off_ha[i], qd_ha[i], spec) for i in range(n)], 0)
    err_b = float(np.abs(deploy_b - hist10_train.numpy()).max())

    out = dict(
        checked=True, n=int(n), protocol="deploy(HA_V31) -> train(standard)；两种部署路径",
        train_order="standard", deploy_order="ha_v31",
        max_abs_offset_const_across_windows=const_err,
        max_abs_q_block_vs_state=min(q_err, float(np.abs(hist10_train[:, -1, :29]
                                                       - (st[:, 7:36] - d_est)).abs().max())),
        max_abs_qd_block_vs_state=qd_err,
        max_abs_hist10_train_vs_deploy_state64_path=err_a,
        max_abs_hist10_train_vs_deploy_blocks_path=err_b,
        block_order="[q-default(29) | qd(29)]",
        frame_order="oldest→newest; [-1] = 决策时刻",
        passed=bool(max(const_err, q_err, qd_err, err_a, err_b) <= 1e-6),
        eval_set=str(a.eval_set),
    )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "history_parity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if not out["passed"]:
        raise SystemExit("[parity] FAIL（规范 §4.7：>1e-6 不许开训）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
