#!/usr/bin/env python3
"""P1-2：用 **on-policy 真轨迹** 的 MC return-to-go 做 critic 的训练目标。

为什么不是"给 DR 扰动起点编目标"：扰动起点上**没有真值**（从没从那个状态执行过）。
唯一干净的做法是拿我们控制器真走过的轨迹，沿时间轴算折扣 return-to-go：

    ret[f] = Σ_{k>=f} γ^{k-f} · r_k        （r_k 是**真物理跟踪奖励**，由
                                            scripts/compute_collect_reward.py 用训练同一个
                                            bundle.reward 算出）

输入：`runs/isaac_replay_*/dump/*_r.pt`（含 `states`/`frames`/`rewards`）
输出：`runs/wm_real/critic_mc_targets.pt`，字段
    states (N,71) / ref_rows (N,71) / frames (N,) / ret (N,) / meta

注意：不做 per-clip 去均值——策略侧的 bootstrap 用 `V(s_H) − V(ref_H)` 的相对形式，
常数偏置自动消掉，所以这里给**原始** MC 目标即可。

用法：
    python3 scripts/build_critic_return_targets.py \
        --inputs 'runs/isaac_replay_20261027_095852/dump/*_r.pt' \
        --out runs/wm_real/critic_mc_targets.pt --gamma 0.99
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=str, required=True, help="glob，元素需含 states/frames/rewards")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/critic_mc_targets.pt")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--episode-len", type=int, default=0,
                    help=">0 时按这个长度切分（同一进程多集时用）；0=每个文件一集")
    ap.add_argument("--centered", action="store_true", default=False,
                    help="可选：再按 episode 去均值（默认不去，配合 V(s_H)-V(ref_H) 相对形式）")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import canonical_to_flat  # noqa: PLC0415

    files = sorted(glob.glob(a.inputs))
    if not files:
        raise SystemExit(f"[mc] 没有匹配 {a.inputs}")
    packed = torch.load(str(a.packed), map_location="cpu", weights_only=False)
    ref_all = torch.as_tensor(packed["ref"]).float()

    S, R, F_out, RET = [], [], [], []
    ep = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if "rewards" not in d:
            print(f"[mc] skip {Path(f).name}（没有 rewards；先跑 compute_collect_reward）", flush=True)
            continue
        S_e, F_e, r_e = d["states"].float(), d["frames"].long(), d["rewards"].float()
        n = int(S_e.shape[0])
        # 按环境交错存（每步 N 个 env）时，一"集"= 每 env 一串；这里按 frame 排序后逐 env 还原
        n_env = int(d.get("num_envs", 1) or 1)
        steps = n // max(1, n_env)
        for e in range(min(n_env, 1) if a.episode_len == 0 else n_env):
            pass  # 保留结构：默认只处理 env0 的时间轴（同一时刻各 env 同步）
        # 取 env0 的时间序列：索引 0, n_env, 2*n_env, ...
        idx = torch.arange(0, n, max(1, n_env))
        if a.episode_len > 0:
            idx = idx[: a.episode_len]
        s_seq = S_e[idx]
        f_seq = F_e[idx]
        r_seq = r_e[idx]
        g = torch.zeros_like(r_seq)
        acc = torch.zeros((), dtype=r_seq.dtype)
        for t in range(int(r_seq.numel()) - 1, -1, -1):
            acc = r_seq[t] + a.gamma * acc
            g[t] = acc
        if a.centered and r_seq.numel() > 1:
            g = g - g.mean()
        S.append(s_seq); R.append(ref_all[f_seq.long()]); F_out.append(f_seq); RET.append(g)
        print(f"[mc] {Path(f).name}: steps={int(r_seq.numel())} r_mean={float(r_seq.mean()):+.5f} "
              f"G[0]={float(g[0]):+.4f} G[-1]={float(g[-1]):+.4f}", flush=True)
        ep += 1

    if not S:
        raise SystemExit("[mc] 没有可用输入")
    states = torch.cat(S); ref_rows = torch.cat(R); frames = torch.cat(F_out); ret = torch.cat(RET)
    out = {
        "states": states, "ref_rows": ref_rows, "frames": frames, "ret": ret,
        "meta": {"gamma": a.gamma, "episodes": ep, "n": int(states.shape[0]),
                 "source": "on-policy Isaac rollouts of our own WBC (A line)",
                 "reward": "trackerlab bundle.reward (same fn as training)",
                 "centered": bool(a.centered),
                 "note": "P1-2: MC return-to-go; bootstrap 用 V(s_H)-V(ref_H) 相对形式"},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(a.out))
    print(f"[mc] saved {a.out}  n={int(states.shape[0])} "
          f"ret mean={float(ret.mean()):+.4f} std={float(ret.std()):.4f}", flush=True)
    a.out.with_suffix(".json").write_text(json.dumps(out["meta"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
