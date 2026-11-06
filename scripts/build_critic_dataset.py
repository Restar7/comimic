#!/usr/bin/env python3
"""把 A' 语料 + 当前口径的奖励，做成 `fit_critic_real.py` 能吃的数据集（2026-11-05）。

为什么需要这一步
----------------
`fit_critic_real.py` 要 `{states, ref_rows, ret, clip_ids}`，其中 `ret` 是
**return-to-go**（从该状态往后的折扣累计奖励）。A' 语料只有逐行奖励，没有 `ret`，
也没有 `ref_rows`（参考帧）。这个脚本补这两样。

两个必须小心的点（都做了自检）
------------------------------
1. **行是跨环境交错的**：同一环境相邻两行相隔 `stride` 行（stride ≈ 并行环境数）。
   直接按 `i+1` 往后看会串到别的机器人身上。所以先**用数据推断 stride**
   （对候选值算 `frames[i+s] == frames[i]+1` 的比例，取最高），并把比例打印出来。
2. **窗口内必须连续**：`frames` 要逐帧 +1、且 `clip_ids` 不变，否则截断
   （重置/换片段处不能把未来的奖励算进来）。

口径提醒：奖励由 `compute_collect_reward.py --from-args <那一轮 args.json>` 生成，
与训练同一个 `bundle.reward` 实现；本脚本只做"累计 + 打包"，不改奖励。

用法::

    python3 scripts/build_critic_dataset.py \
        --shards runs/a_prime_sonic_leash \
        --rewards runs/a_prime_sonic_leash/rewards_current \
        --out runs/wm_real/transitions_current_g095.pt --gamma 0.95 --window 20
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

STRIDE_CANDIDATES = (1, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)


def infer_stride(frames: torch.Tensor, clip: torch.Tensor) -> tuple[int, float, dict]:
    """返回 (stride, 连续率, 各候选的得分)。"""
    scores = {}
    for s in STRIDE_CANDIDATES:
        if s >= frames.shape[0]:
            continue
        ok = (frames[s:] == frames[:-s] + 1) & (clip[s:] == clip[:-s])
        scores[s] = float(ok.float().mean())
    if not scores:
        return 1, 0.0, {}
    best = max(scores, key=scores.get)
    return best, scores[best], scores


def build(shards: list[Path], rewards_dir: Path, packed: Path, gamma: float, window: int,
          min_cont: float = 0.5):
    from diff_vla_wbc.isaac_joint import canonical_to_flat  # noqa: PLC0415

    pk = torch.load(str(packed), map_location="cpu", weights_only=False)
    ref = canonical_to_flat(torch.as_tensor(pk["ref"]).float())

    states, refs, frames, clips, rews, rets = [], [], [], [], [], []
    skipped: list[tuple[str, float]] = []
    offset = 0
    for sh in shards:
        d = torch.load(str(sh), map_location="cpu", weights_only=False)
        rp = rewards_dir / f"{sh.stem}_r.pt"
        if not rp.exists():
            raise SystemExit(f"[critic-ds] 缺奖励文件 {rp}（先跑 compute_collect_reward.py）")
        r = torch.load(str(rp), map_location="cpu", weights_only=False)["rewards"].float()
        S = d["states"].float()
        F = d["frames"].long()
        C = d["clip_ids"].long()
        if r.shape[0] != S.shape[0]:
            raise SystemExit(f"[critic-ds] {sh.name}: 奖励 {r.shape[0]} 行 ≠ 状态 {S.shape[0]} 行")
        stride, frac, scores = infer_stride(F, C)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print(f"[critic-ds] {sh.name:<26} n={S.shape[0]:6d} stride={stride:<4d} 连续率={frac:.3f} "
              f"候选前三={[(k, round(v, 3)) for k, v in top]}", flush=True)
        if frac < min_cont:
            # 不留情面地丢：这些分片的行序不是"同环境等步长交错"，按任何 stride 往后看都会
            # 串到别的机器人 ⇒ ret 只剩 1 步，和正常行的 20 步不同量级，混进去会毁掉 critic。
            skipped.append((sh.name, frac))
            print(f"[critic-ds]   ✗ 跳过（连续率 {frac:.3f} < {min_cont}）：ret 不可信", flush=True)
            continue
        # ---- return-to-go：沿 stride 往后看 window 步，遇到断帧/换片段就停 ----
        n = S.shape[0]
        idx = torch.arange(n)
        acc = torch.zeros(n)
        alive = torch.ones(n, dtype=torch.bool)
        for k in range(window):
            j = idx + k * stride
            inb = j < n
            jj = torch.where(inb, j, torch.zeros_like(j))
            if k > 0:
                prev = torch.where((idx + (k - 1) * stride) < n, idx + (k - 1) * stride,
                                   torch.zeros_like(idx))
                cont = (F[jj] == F[prev] + 1) & (C[jj] == C[prev])
                alive = alive & inb & cont
            acc = acc + (gamma ** k) * torch.where(alive, r[jj], torch.zeros_like(r))
        states.append(S)
        refs.append(ref[F])
        frames.append(F)
        clips.append(C + offset)
        rews.append(r)
        rets.append(acc)
        offset = int(max(offset, int(C.max()) + 1))

    S = torch.cat(states)
    R = torch.cat(refs)
    F = torch.cat(frames)
    C = torch.cat(clips)
    RW = torch.cat(rews)
    RET = torch.cat(rets)
    print(f"\n[critic-ds] 采用 {len(shards) - len(skipped)}/{len(shards)} 个分片"
          f"（跳过 {[n for n, _ in skipped]}）")
    print(f"[critic-ds] 合计 {S.shape[0]} 行 / {int(C.max()) + 1} 个片段")
    print(f"[critic-ds] 奖励 mean={float(RW.mean()):+.5f} std={float(RW.std()):.5f}")
    print(f"[critic-ds] ret  mean={float(RET.mean()):+.4f} std={float(RET.std()):.4f} "
          f"p1={float(RET.quantile(0.01)):+.3f} p99={float(RET.quantile(0.99)):+.3f}")
    return {"states": S, "ref_rows": R, "ret": RET, "clip_ids": C,
            "frames": F, "rewards": RW,
            "meta": {"gamma": gamma, "window": window,
                     "skipped_shards": [n for n, _ in skipped], "min_continuity": min_cont,
                     "source": "a_prime_sonic_leash + compute_collect_reward --from-args",
                     "n": int(S.shape[0])}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=ROOT / "runs/a_prime_sonic_leash")
    ap.add_argument("--rewards", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/rewards_current")
    ap.add_argument("--packed", type=Path,
                    default=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs/wm_real/transitions_current_g095.pt")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--min-continuity", type=float, default=0.5,
                    help="某分片的 stride 连续率低于它就整片丢弃（ret 只剩 1 步，会污染 critic）")
    a = ap.parse_args()

    shards = sorted(Path(p) for p in glob.glob(str(a.shards / "*_off*.pt")))
    if not shards:
        raise SystemExit(f"[critic-ds] 没有分片：{a.shards}")
    print(f"[critic-ds] 分片 {len(shards)} 个；γ={a.gamma} 窗口={a.window}")
    d = build(shards, a.rewards, a.packed, a.gamma, a.window,
              min_cont=float(a.min_continuity))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, str(a.out))
    print(f"[critic-ds] 写好 → {a.out}  ({a.out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
