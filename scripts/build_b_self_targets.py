#!/usr/bin/env python3
"""给 B 线 dump 打上"**真实闭环结果**"标签（2026-10-29）。

为什么需要它
------------
上一晚的结论是：所有"往参考 token 拉"的监督（锚定 hinge / token BC / 参考 token）都会
让闭环变差，因为**参考 token 对"已经漂了的状态"不是纠偏方向**。
而奖励项靠 WM 想象，WM 20 步误差 0.18 rad ≈ 被裁判的量，也不可信。

**但 dump 里其实已经有真值**：`tokens (T,64)` 是策略当时**真正下发**的 token，
`states (T,71)` + `frames (T,)` 记录机器人**真正走到了哪**。
所以可以对每个重规划时刻 t 算出"接下来 H 步实际漂了多少"，这就是**闭环真值**：

    realized_err[t] = mean_{k<t+H} | states[k].root_xyz − ref[frames[k]].root_xyz |

有了它就能做**按结果筛选的自模仿**：只在"实际走得好"的时刻上，让 head 复现
**它自己当时那个 token**（不是参考 token）。既不拉向错误方向，也不依赖 WM。

用法：
    PYTHONPATH=. python3 scripts/build_b_self_targets.py \
        --dump 'runs/B_r*/vla_*_f*_r*/dump/*.pt' \
        --out  runs/wm_real/b_self_targets.pt --horizon 20

输出：``{name: {"tok": (T,64), "err": (T,), "frames": (T,)}, ..., "_meta": {...}}``
其中 ``name`` 是 dump 的文件名（b_start 按文件名匹配）。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="glob：B 线 dump")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/b_self_targets.pt")
    # ★ 默认必须指 318 mjlab 库：dump 里的 frames 是**这个库**的行号。
    #   2026-10-29 的默认值还是 Isaac 的 ha_cmd/packed.pt（634,016 帧），行号被解释到**另一个库**上，
    #   算出来的 realized_err 全是错的（而且不报错，只是悄悄偏移）。
    ap.add_argument("--ref", type=Path, default=ROOT / "data/library/ft318_compact.pt")
    ap.add_argument("--horizon", type=int, default=20, help="看接下来多少步的实际漂移")
    a = ap.parse_args()

    lib = torch.load(str(a.ref), map_location="cpu", weights_only=False)["ref"].float()
    n_lib = int(lib.shape[0])
    files = sorted(glob.glob(a.dump))
    if not files:
        raise SystemExit(f"[self-tgt] 没匹配到 dump: {a.dump}")

    out: dict = {}
    stats = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        S = d.get("states")
        T = d.get("tokens")
        F = d.get("frames")
        if S is None or T is None or F is None:
            print(f"[self-tgt] SKIP {Path(f).name}: 缺 states/tokens/frames")
            continue
        S = S.float()
        F = F.long()
        if int(F.max()) >= n_lib:
            raise SystemExit(
                f"[self-tgt] {Path(f).name}: frames 最大 {int(F.max())} ≥ 参考库帧数 {n_lib} "
                f"⇒ --ref 指错了库（应该是 dump 对应的那个 motion library）")
        T_len = int(S.shape[0])
        if int(F.numel()) != T_len:
            # vla 臂只在**重规划时刻**记 frames（execute_steps=20 tick 一个，600 tick ⇒ 30 个），
            # 而 states/tokens 是每 tick 的（600）⇒ 把这 30 个参考帧按段展开到 600 个 tick。
            n_f = int(F.numel())
            stride = max(1, (T_len + n_f - 1) // n_f)
            plan_of_tick = torch.clamp(torch.arange(T_len) // stride, max=n_f - 1)
            F = F[plan_of_tick]
        e = (S[:, :3] - lib[F][:, :3]).norm(dim=-1)           # (T,) 每步 root 误差
        n = int(e.numel())
        H = int(a.horizon)
        # 每个 t：接下来 min(H, T-t) 步的平均误差（t 越靠后窗口越短，用平均消掉长度偏差）
        cs = torch.cumsum(torch.cat([torch.zeros(1), e]), 0)   # (T+1,)
        idx = torch.arange(n)
        hi = torch.clamp(idx + H, max=n)
        err = (cs[hi] - cs[idx]) / (hi - idx).clamp(min=1).float()
        # ★ 必须用"末两级路径"做 key：43 个 dump 只有 6 个不同文件名（每任务一集，同名跨目录）,
        #   用 basename 会互相覆盖，而且长度不一致会让 b_start 索引越界（2026-10-29 实测）。
        _key = str(Path(f).resolve())   # 绝对路径：B_r2~B_r5 用了同一行号，目录名会撞
        out[_key] = {"tok": T.float(), "err": err, "e_inst": e, "frames": F}
        stats.append((_key, float(err.mean()), float(e.mean()), n))

    if not out:
        raise SystemExit("[self-tgt] 没有任何可用 dump")

    allerr = torch.cat([v["err"] for v in out.values()])
    q = [float(torch.quantile(allerr, x)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)]
    meta = {
        "horizon": int(a.horizon),
        "n_dumps": len(out),
        "n_rows": int(allerr.numel()),
        "err_quantiles": {"p10": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4]},
        "err_mean": float(allerr.mean()),
        "glob": a.dump,
    }
    out["_meta"] = meta
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)

    print(f"[self-tgt] {len(stats)} 条 dump / {meta['n_rows']} 个时刻 → {a.out}")
    # ★ 2026-10-30 修：`out` 里已经塞了 `_meta`（没有 e_inst），直接遍历 out.values() 会 KeyError
    _inst_all = torch.cat([v["e_inst"] for k, v in out.items()
                           if k != "_meta" and v.get("e_inst") is not None])
    print(f"[self-tgt] 瞬时 e_root 分位：p10={float(torch.quantile(_inst_all, 0.1)):.3f} "
          f"p50={float(torch.quantile(_inst_all, 0.5)):.3f}")
    print(f"[self-tgt] realized_err 分位：p10={q[0]:.3f} p25={q[1]:.3f} p50={q[2]:.3f} "
          f"p75={q[3]:.3f} p90={q[4]:.3f} (mean {meta['err_mean']:.3f} m)")
    print("[self-tgt] 建议筛选阈值：keep p25 ⇒ --b-self-err-max "
          f"{q[1]:.3f}；keep p50 ⇒ {q[2]:.3f}")
    print("  （per-dump 误差均值最大的 5 条）")
    for name, m, inst, n in sorted(stats, key=lambda x: -x[1])[:5]:
        print(f"    {name:52s} realized={m:.3f}  瞬时={inst:.3f}  T={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
