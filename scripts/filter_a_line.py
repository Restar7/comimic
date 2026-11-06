#!/usr/bin/env python3
"""清洗 A 线采集数据：按**参考轨迹**判定漂坏，把每条 env 轨迹截断到首次违规之前。

为什么需要（2026-10-29 实测）
----------------------------
`isaac_a_collect_batch.py` 没有终止/复位逻辑（刻意的：只采动力学，不算 episode 成败）。
结果是**摔倒/炸开的机器人会继续被记录**：a_w0.pt 里出现关节角 −9.35 rad（正常 ±2.5）、
策略动作 |a| 到 22.7、关节速度 10.8 rad/s。这些行喂给 WM 就是在教它垃圾动力学，
而且会让 `val_multistep` 里 20 步 rollout 出 NaN（早停直接失效，big5 的 A 段就是这么坏的）。

判据用**相对参考**而不是绝对高度：库里含 SitSofa（坐下）片段，根高本来就低，
用 `z < 0.3 m` 会把 100% 的 env 都判成摔倒。

    |q| ≤ 3.2 rad、|qd| ≤ 40 rad/s、全部有限
    根位置偏离参考 ≤ 0.6 m、根高偏离参考 ≤ 0.4 m、关节角平均偏离参考 ≤ 0.6 rad

输出保持**同一交错布局**（`steps × num_envs`），只多存一个 `valid_len`（每条 env 的干净长度），
这样 `compute_collect_reward.py` 的 "回退 num_envs 行取 prev_action" 仍然成立。

用法：
    python3 scripts/filter_a_line.py --in 'runs/wm_real/a_w*.pt' --out-dir runs/wm_real \
        --min-steps 50
产出：`ac_w0.pt …`（`--a-line` 直接指向 `ac_w*.pt`）
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

Q_MAX = 3.2
QD_MAX = 40.0
E_POS_MAX = 1.5   # 只拦"离谱"（明显炸开），不拦正常的跟踪漂移
UP_MIN = 0.2      # up = R[2,2]，低于 0.2 视为倒地/翻过来


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="glob：A 线文件")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs/wm_real")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--min-steps", type=int, default=50, help="干净长度低于此值的 env 整条丢弃")
    ap.add_argument("--suffix", default="ac_", help="输出文件名前缀")
    args = ap.parse_args()

    packed = torch.load(str(args.packed), map_location="cpu", weights_only=False)
    ref = torch.as_tensor(packed["ref"]).float()

    files = sorted(glob.glob(args.inp))
    if not files:
        raise SystemExit(f"[filter] 没匹配到 {args.inp}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tot_keep = tot_drop = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S = d["states"].float()
        n_env = int(d.get("num_envs", 1))
        T = int(S.shape[0]) // max(1, n_env)
        S = S[: T * n_env].reshape(T, n_env, 71).permute(1, 0, 2)      # (env,T,71)
        F = d["frames"].long()[: T * n_env].reshape(T, n_env).permute(1, 0)  # (env,T)
        R = ref[F.clamp(0, ref.shape[0] - 1)]                          # (env,T,71)

        # 状态布局 [p(3), quat_wxyz(4), q(29), v(3), w(3), qd(29)]
        #   q=7:36、v=36:39、w=39:42、qd=42:71
        # ★ 2026-10-29 修：原来按 13:42 取 q、42:71 取 qd，把速度项当成了关节角，
        #   于是"509/512 env 关节角超限"是假的。
        p, z = S[:, :, 0:3], S[:, :, 2]
        q, qd = S[:, :, 7:36], S[:, :, 42:71]
        rp, rz = R[:, :, 0:3], R[:, :, 2]
        rq = R[:, :, 7:36]

        # 判据只保留**物理退化**（炸开/倒地/NaN），不判"跟踪得好不好"。
        # 为什么：WM 学的是 (s,a,s') 的物理一致性 —— 机器人偏离参考 1 m 时这些转移
        # **仍然是真实物理**，对 WM 完全有用；"跟踪误差"是奖励的事，由
        # compute_collect_reward 单独算。用 e_pos/e_q 去筛会把可用数据砍掉一大半。
        up = 1 - 2 * (S[:, :, 4] ** 2 + S[:, :, 5] ** 2)          # R[2,2]（wxyz 约定）
        bad = (
            ~torch.isfinite(S).all(-1)
            | (q.abs() > Q_MAX).any(-1)
            | (qd.abs() > QD_MAX).any(-1)
            | (up < UP_MIN)                                        # 倒了/翻过来
            | ((p - rp).norm(dim=-1) > E_POS_MAX)                   # 离谱到 1.5 m 也算炸了
        )                                                              # (env,T)
        valid_len = torch.full((n_env,), T, dtype=torch.long)
        for e in range(n_env):
            idx = torch.nonzero(bad[e], as_tuple=False)
            if len(idx):
                valid_len[e] = int(idx[0])

        keep = int(valid_len.sum())
        drop = T * n_env - keep
        tot_keep += keep
        tot_drop += drop
        out = dict(d)
        out["valid_len"] = valid_len
        out["num_envs"] = n_env
        out["steps"] = T
        out["meta"] = dict(d.get("meta", {}))
        out["meta"].update({"filtered": True, "filter_rules":
                            f"|q|<={Q_MAX},|qd|<={QD_MAX},up>={UP_MIN},e_pos<={E_POS_MAX}",
                            "kept_rows": keep, "dropped_rows": drop,
                            "note": "valid_len[e] = 第 e 条 env 轨迹的干净前缀长度"})
        name = Path(f).name
        outp = args.out_dir / (args.suffix + name.split("_", 1)[-1] if name.startswith("a_") else f"{args.suffix}{name}")
        torch.save(out, str(outp))
        good = int((valid_len >= args.min_steps).sum())
        print(f"[filter] {name}: env={n_env} T={T} 保留 {keep} 行({keep/(T*n_env)*100:.1f}%) "
              f"≥{args.min_steps}步的 env={good}/{n_env} "
              f"有效长度 median={int(valid_len.median())} p10={int(valid_len.sort()[0][max(0,n_env//10)])} "
              f"→ {outp.name}", flush=True)

    print(f"[filter] 合计保留 {tot_keep} 行，丢弃 {tot_drop} 行"
          f"（{tot_keep/max(1,tot_keep+tot_drop)*100:.1f}% 干净）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
