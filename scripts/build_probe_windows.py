#!/usr/bin/env python3
"""把**配对探针**的 shard 切成 20 步窗口（供 WM 多步训练，补"换动作后的 20 步响应"）。

为什么需要
----------
`finetune_wm_gain.py` 的阶段 2 原来只用 A' 窗口，那是"**沿录制动作**走的 20 步"；
联合微调的梯度问的是"**同一段换个动作序列**，20 步会差多少"。探针数据每个环境都从
同一个起点、用不同的动作噪声跑满 20 步 ⇒ 正好是这部分真值。

做法
----
按 `env_ids` 把每个 env 的行串成轨迹，取**帧号严格连续（+1）**的 21 帧窗口
（段重置处帧号会跳变，自动断开），输出：

    states   (N, k+1, 71)
    actions  (N, k,   29)

用法::

    python3 scripts/build_probe_windows.py --dir runs/paired_probe2 --k 20 \
        --out runs/paired_probe2/probe_windows.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "runs/paired_probe2")
    ap.add_argument("--pattern", default="*_off*.pt")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-windows", type=int, default=200000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    files = sorted(a.dir.glob(a.pattern))
    if not files:
        raise SystemExit(f"[pw] {a.dir} 下没有 {a.pattern}")
    out = a.out or (a.dir / "probe_windows.pt")
    K = int(a.k)

    S_list, A_list = [], []
    n_win = 0
    for f in files:
        z = torch.load(f, map_location="cpu", weights_only=False)
        if "env_ids" not in z:
            raise SystemExit(f"[pw][FAIL] {f.name} 没有 env_ids —— 需要用新采集脚本重采")
        S = z["states"].float()
        A = z["actions"].float()
        S1 = z["next_states"].float()
        F = z["frames"].long()
        E = z["env_ids"].long()
        n_env = int(E.max()) + 1
        # 按 env 串轨迹（同一 env 的行按出现顺序即时间顺序）
        for e in range(n_env):
            m = (E == e).nonzero().flatten()
            if m.numel() < K + 1:
                continue
            # 用 frames 的连续性判断是否同段（重置处帧号跳变）
            fr = F[m]
            st = S[m]
            ac = A[m]
            i = 0
            while i + K < m.numel():
                if int(fr[i + K] - fr[i]) == K and (fr[i + 1:i + K + 1] - fr[i:i + K] == 1).all():
                    S_list.append(st[i:i + K + 1])
                    A_list.append(ac[i:i + K])
                    n_win += 1
                    i += int(a.stride)
                    if n_win >= a.max_windows:
                        break
                else:
                    i += 1
            if n_win >= a.max_windows:
                break
        if n_win >= a.max_windows:
            break

    if not S_list:
        raise SystemExit("[pw][FAIL] 一个窗口都没切出来（检查 env_ids / frames 连续性 / --k）")
    SS = torch.stack(S_list)
    AA = torch.stack(A_list)
    print(f"[pw] {len(files)} 个 shard → {n_win} 个 {K} 步窗口", flush=True)
    print(f"[pw] 动作幅度 |a| 均值 {float(AA.abs().mean()):.3f}", flush=True)
    print(f"[pw] 相邻帧动作变化 |Δa| 均值 {float((AA[:,1:]-AA[:,:-1]).abs().mean()):.3f}", flush=True)
    torch.save({"states": SS, "actions": AA, "k": K,
                "meta": {"src": [f.name for f in files], "k": K, "kind": "paired_probe_windows"}},
               out)
    print(f"[pw] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
