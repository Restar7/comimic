#!/usr/bin/env python3
"""把**配对探针**的产物整理成"同状态 × 不同动作"的真配对文件（供 WM 增益损失用）。

输入：`runs/paired_probe/*_off*.pt`（`run_paired_probe.sh` 产物，
      每 K 个 env 共用同一窗口 + `--segment` 周期重置 ⇒ 重置点上状态逐位相同）。

筛选规则（严格）：
  * 按 (`clip_ids`, `frames`) 分组；
  * 组内**状态两两最大距离 < 1e-5**（真同状态）；
  * 组内**动作标准差 > 0.05**（动作确实不同）；
  * 组大小 = `--k`（默认 4）。

输出 `gain_pairs.pt`：
  ``s (M,71)``、``a (M,K,29)``、``s1 (M,K,71)``、``meta``

用法::

    python3 scripts/build_gain_pairs.py --dir runs/paired_probe --k 4 --out runs/paired_probe/gain_pairs.pt
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "runs/paired_probe")
    ap.add_argument("--pattern", default="*_off*.pt")
    ap.add_argument("--k", type=int, default=4, help="每组 env 数（= 采集时的 --plan-repeat-k）")
    ap.add_argument("--state-tol", type=float, default=1e-5)
    ap.add_argument("--action-min-std", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    files = sorted(a.dir.glob(a.pattern))
    if not files:
        raise SystemExit(f"[pairs] {a.dir} 下没有 {a.pattern}")
    out = a.out or (a.dir / "gain_pairs.pt")

    Ps, Pa, Ps1 = [], [], []
    n_files = 0
    for f in files:
        z = torch.load(f, map_location="cpu", weights_only=False)
        S = z["states"].float()
        A = z["actions"].float()
        S1 = z["next_states"].float()
        FR = z["frames"].long()
        CL = z["clip_ids"].long()
        n_files += 1
        # ★ 2026-11-05 关键修复：**按文件分组**。原来把所有 shard 拼起来用
        #   (clip_id, frame) 做键，不同 launch 的同键组会合并成 8/16/32 行，
        #   被"必须正好 k 行"的过滤全丢掉（16 个 shard 只剩 156 组）。
        key = CL * 100000 + FR
        order = torch.argsort(key)
        key, S, A, S1 = key[order], S[order], A[order], S1[order]
        uniq, counts = torch.unique_consecutive(key, return_counts=True)
        starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)[:-1]])
        for st, n in zip(starts.tolist(), counts.tolist()):
            if n < a.k:
                continue
            s_blk = S[st:st + n]
            a_blk = A[st:st + n]
            s1_blk = S1[st:st + n]
            # 组内按"状态几乎相同"聚类（同一帧里既有重置点、也有段中帧）
            used = torch.zeros(n, dtype=torch.bool)
            for i in range(n):
                if used[i]:
                    continue
                same = ((s_blk - s_blk[i]).norm(dim=-1) <= a.state_tol) & (~used)
                idx = same.nonzero().flatten()
                if idx.numel() < a.k:
                    continue
                idx = idx[: a.k]
                used[idx] = True
                if float(a_blk[idx].std(dim=0).mean()) <= a.action_min_std:
                    continue
                Ps.append(s_blk[i])
                Pa.append(a_blk[idx])
                Ps1.append(s1_blk[idx])
    if not Ps:
        raise SystemExit("[pairs][FAIL] 一组真配对都没找到 —— 检查 --plan-repeat-k / --segment / noise")
    print(f"[pairs] 扫了 {n_files} 个 shard", flush=True)

    P_s = torch.stack(Ps)
    P_a = torch.stack(Pa)
    P_s1 = torch.stack(Ps1)
    print(f"[pairs] 真配对组 {P_s.shape[0]} 组 × {a.k} 个动作 "
          f"（动作组内 std 均值 {float(P_a.std(dim=1).mean()):.3f}）", flush=True)
    print(f"[pairs] 下一步差 p50 = {float((P_s1 - P_s1[:, :1]).norm(dim=-1).median()):.4f}"
          f"（同一状态、不同动作导致的位移，应当 > 0）", flush=True)
    torch.save({"s": P_s, "a": P_a, "s1": P_s1, "k": int(a.k),
                "meta": {"src": [str(f) for f in files], "k": int(a.k)}}, out)
    print(f"[pairs] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
