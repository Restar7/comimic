#!/usr/bin/env python3
"""在 **mjlab / MuJoCo-Warp 环境里**大规模采集真转移（给 WM 保鲜 / critic / b_start）。

为什么不用 CPU 闭环（`closed_loop_vla_supermimic.py --dump`）：
* CPU 闭环一次只有 1 个 env，600 tick 要 6–10 s，而且**没有帧号**（自由跑，不跟参考），
  所以算不出奖励、也不能当 b_start；
* 这里是**向量化**的（2048 envs 一次推进），参考驱动 ⇒ **帧号/奖励/终止全都是真的**，
  2048×2000 步 ≈ 4M 条转移，几十秒跑完（对比 CPU 闭环 5 条 episode）。

产出（键与 `diff_vla_wbc.wm_refresh.refresh_wm` / `trainer.attach_real_returns` 对齐）::

    states (N,71) / next_states (N,71) / actions (N,29)      # actions = **executed**（带动作噪声）
    clean_actions (N,29) / xi (N,33) / rewards (N,)
    frames (N,) / frame_map (num_frames,) / ret (N,)         # ret 按 episode 倒序折扣
    meta {...}

用法::

    .venv/bin/python scripts/collect_mjlab.py \\
        --ckpt logs/wbc_ft318_mjlab/ckpt_0007000.pt \\
        --library data/library/ft318_compact.pt --tokens data/library/ft318_tokens.pt \\
        --num-envs 2048 --steps 2000 --out runs/wm_real/mjlab_transitions.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="mjlab WBC ckpt（token 条件化）")
    ap.add_argument("--library", type=Path, required=True, help="compact 参考库（或给 --npz-dir）")
    ap.add_argument("--npz-dir", type=Path, default=None, help="npz 参考库（与 --library 二选一）")
    ap.add_argument("--tokens", type=Path, required=True, help="录制 token 包（conditioning）")
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=4000, help="每个 env 推进的控制步数")
    ap.add_argument("--replay-capacity", type=int, default=4000,
                    help="每个 env 的环形缓冲长度；能存住的转移数 ≈ num_envs × min(steps, capacity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--action-noise-std", type=float, default=0.1,
                    help="与训练一致（0 = 采干净动作，WM 会学不到噪声分布）")
    ap.add_argument("--obs-noise", dest="obs_noise", action="store_true", default=False)
    ap.add_argument("--reset-noise", dest="reset_noise", action="store_true", default=False)
    ap.add_argument("--sampling-mode", choices=("start", "return", "uniform"), default="start")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/mjlab_transitions.pt")
    ap.add_argument("--max-rows", type=int, default=0, help=">0 时随机下采样到这么多行")
    args = ap.parse_args()

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.library_tokens import LibraryTokens
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    dev = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    src = str(args.library) if args.npz_dir is None else str(args.npz_dir)
    cfg = SuperMimicEnvCfgArgs(
        npz_files=[src],
        num_envs=int(args.num_envs),
        future_frames=10,
        episode_length_s=1.0e5,          # 让 clip 结束决定 episode
        action_noise_std=float(args.action_noise_std),
        obs_noise=bool(args.obs_noise),
        reset_noise=bool(args.reset_noise),
        domain_randomization=False,
        sampling_mode=str(args.sampling_mode),
        record_transitions=True,
        compact_library=str(src).endswith(".pt"),
        replay_capacity=int(args.replay_capacity),
        seed=int(args.seed),
        anchor_obs=False,
    )
    env = make_env(cfg, device=str(dev))
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    recorder = env.recorder_manager.get_term("replay")
    buf = recorder.buffer
    policy, ck_cfg = load_policy_from_checkpoint(str(args.ckpt), str(dev))
    tok = LibraryTokens(str(args.tokens), device=dev)
    print(f"[collect] envs={args.num_envs} steps={args.steps} "
          f"clips={lib.num_clips} frames={lib.num_frames} | {tok.audit()}", flush=True)
    if tok.n != int(lib.num_frames):
        raise SystemExit(f"[fail] token 包 {tok.n} != 库 {int(lib.num_frames)} 帧")

    obs, _ = env.reset()
    with torch.no_grad():
        for t in range(int(args.steps)):
            token = tok(cmd.frame)
            act = policy(obs["policy"], token, interface="token")
            obs, _r, _term, _tout, _extra = env.step(act)
            if (t + 1) % 200 == 0:
                print(f"  [{t + 1}/{args.steps}] rows={buf.total_rows}", flush=True)

    E, T = int(args.num_envs), int(buf.T)
    nt = (torch.arange(T, device=buf.s.device) + 1) % T
    nxt_ep = buf.ep_id[:, nt]
    mask = (buf.valid & ~buf.terminal & buf.valid[:, nt] & (nxt_ep == buf.ep_id))
    e, t = mask.nonzero(as_tuple=True)
    n = int(e.numel())
    if n == 0:
        raise SystemExit("[fail] 没有可用转移（steps 太少？）")
    idx = torch.arange(n, device=e.device)
    if args.max_rows > 0 and n > args.max_rows:
        sel = torch.randperm(n, device=e.device)[: args.max_rows]
        e, t, idx, n = e[sel], t[sel], torch.arange(args.max_rows, device=e.device), args.max_rows
    states = buf.s[e, t].float().cpu()
    next_states = buf.s[e, nt[t]].float().cpu()
    actions = buf.a_exec[e, t].float().cpu()
    clean = buf.a[e, t].float().cpu()
    xi = buf.xi[e, t].float().cpu()
    rewards = buf.r[e, t].float().cpu()
    frames = buf.frame[e, t].long().cpu()
    ep_ids = buf.ep_id[e, t].long().cpu()

    # return-to-go（同 episode 倒序折扣；ep_id 变了就重置）
    ret = torch.zeros(n)
    g = 0.0
    for i in range(n - 1, -1, -1):
        if i + 1 < n and int(ep_ids[i + 1]) == int(ep_ids[i]):
            g = float(rewards[i]) + float(args.gamma) * g
        else:
            g = float(rewards[i])
        ret[i] = g

    # 帧 → 行映射（同一帧取第一行；缺帧用前一帧，与 train_joint_v2 同口径）
    frame_map = torch.full((int(lib.num_frames),), -1, dtype=torch.long)
    fr = frames.clamp(0, int(lib.num_frames) - 1)
    frame_map[fr] = torch.arange(n, dtype=torch.long)
    miss = (frame_map < 0).nonzero(as_tuple=False).flatten()
    if miss.numel():
        frame_map[miss] = frame_map[torch.clamp(miss - 1, min=0)]
    if int((frame_map < 0).sum()):
        frame_map[frame_map < 0] = 0

    # critic 拟合（scripts/fit_critic_real.py）要求：states / ref_rows / ret / clip_ids
    ref_rows = lib.ref71(frames).float()
    clip_ids = buf.clip[e, t].long().cpu()
    payload = {
        "states": states, "next_states": next_states,
        "ref_rows": ref_rows, "clip_ids": clip_ids,
        "actions": actions, "clean_actions": clean, "xi": xi,
        "rewards": rewards, "ret": ret, "frames": frames,
        "ep_ids": ep_ids, "frame_map": frame_map,
        "meta": {
            "source": "collect_mjlab (vectorised mjlab env)",
            "ckpt": str(args.ckpt), "library": src, "tokens": str(args.tokens),
            "num_envs": E, "steps": int(args.steps), "rows": n,
            "action_noise_std": float(args.action_noise_std),
            "sampling_mode": str(args.sampling_mode),
            "coverage": int(torch.unique(buf.clip[e, t]).numel()),
            "replay_capacity": int(buf.T),
            "gamma": float(args.gamma),
            "state_layout": "[p(3), quat(4), q(29), v_w(3), w_b(3), qd(29)]",
            "joint_order": "standard",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out))
    print(json.dumps({k: v for k, v in payload["meta"].items() if k != "source"}, indent=1), flush=True)
    print(f"[saved] {out}  rows={n}  ({n / max(1, int(args.steps)):.0f} 行/步)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
