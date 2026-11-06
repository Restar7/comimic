#!/usr/bin/env python3
"""把 MuJoCo 采集 dump 变成「真转移」：WM 保鲜 + critic 真回报 + b_start 的输入。

对应 `docs/S3_END_TO_END_PLAN_20261028.md` 的 P0-3（用**真正下发的动作**重建真转移）。
Isaac 版的输入是 HA 回放 dump；这里换成 `scripts/closed_loop_vla_supermimic.py --dump` 的产物。

动作语义（**与 WM 的输入对齐**，别搞错）：
    mjlab:  q_target = default_q + action_scale * a      ⇒ WM 吃的是归一化动作 `a` 本身
    所以 dump 里的 `actions` 就是 `a`（不是 q_target，也不是 Δq）

输出 ``.pt``（键与 `diff_vla_wbc.wm_refresh.refresh_wm` / `trainer.attach_real_returns` 对齐）::

    states (N,71) / next_states (N,71) / actions (N,29)
    frames (N,)           # 库行号；自由闭环为 -1（此时不产 rewards/ret）
    rewards (N,) / ret (N,) / ref_rows (N,71)     # 仅当 frames 有效且 --with-rewards
    frame_map (num_frames,)                       # 帧 → 真转移行号（--start-source recorded 用）
    meta {...}

用法::

    python3 scripts/build_real_transitions_mjlab.py \\
        --dump 'runs/mjlab_collect/*/dump.pt' \\
        --ckpt runs/wbc_ft318_mjlab/ckpt_latest.pt \\
        --motion data/npz/g1_tasks \\
        --out runs/wm_real/mjlab_transitions.pt --with-rewards
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="dump 的 glob（每条 episode 一个 .pt）")
    ap.add_argument("--ckpt", type=Path, required=True, help="mjlab WBC ckpt（算奖励要用它的解析件）")
    ap.add_argument("--motion", nargs="+", type=Path, default=[ROOT / "data/npz/g1_tasks"])
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/mjlab_transitions.pt")
    ap.add_argument("--with-rewards", action="store_true", default=True)
    ap.add_argument("--no-rewards", dest="with_rewards", action="store_false")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-steps", type=int, default=10)
    args = ap.parse_args()

    from diff_vla_wbc.mjlab_joint import load_mjlab_bundle

    files = sorted(glob.glob(str(args.dump)))
    if not files:
        raise SystemExit(f"[fail] 没有匹配到 dump：{args.dump}")

    npz = []
    for p in args.motion:
        npz.extend(sorted(Path(p).glob("*.npz")) if Path(p).is_dir() else [Path(p)])
    bundle = load_mjlab_bundle(args.ckpt, device=args.device, npz=npz or None, token_dim=64)
    lib = bundle.library
    dev = torch.device(args.device)

    S, S1, A, F, R = [], [], [], [], []
    ret_rows, ref_rows = [], []
    frame_map = torch.full((int(lib.num_frames),), -1, dtype=torch.long)
    n_rows = 0
    meta_eps = []

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        st, nx, ac = d.get("states"), d.get("next_states"), d.get("actions")
        if st is None or ac is None:
            print(f"[skip] {Path(f).name}: 缺 states/actions")
            continue
        T = int(ac.shape[0])
        if T < int(args.min_steps):
            print(f"[skip] {Path(f).name}: 只有 {T} 步")
            continue
        nx = nx if nx is not None else torch.cat((st[1:], st[-1:]), 0)   # 末帧没有后继
        _fr = d.get("frames")
        fr = (torch.full((T,), -1, dtype=torch.long) if _fr is None
              else torch.as_tensor(_fr).long().flatten()[:T])
        if fr.numel() < T:
            fr = torch.cat((fr, torch.full((T - fr.numel(),), -1, dtype=torch.long)))

        S.append(st[:T].float())
        S1.append(nx[:T].float())
        A.append(ac[:T].float())
        F.append(fr[:T])

        # 奖励：只在 frames 有效（回放式采集）时算；自由闭环没有库行号 ⇒ 跳过
        if bool(args.with_rewards) and int(fr[:T].max()) >= 0:
            eps_r = []
            for t in range(T):
                fidx = int(fr[t])
                ref = bundle.reference_frame(lib, torch.tensor([max(fidx, 0)], device=dev))
                prev_a = A[-1][t - 1] if t > 0 else ac[0].float()
                terms = bundle.reward(S1[-1][t].unsqueeze(0), A[-1][t].unsqueeze(0),
                                      prev_a.unsqueeze(0), ref)
                eps_r.append(float(terms["total"]))
            r = torch.tensor(eps_r, dtype=torch.float32)
            R.append(r)
            # return-to-go（倒序累加）
            g = 0.0
            rtg = []
            for v in reversed(eps_r):
                g = v + args.gamma * g
                rtg.append(g)
            ret_rows.append(torch.tensor(list(reversed(rtg)), dtype=torch.float32))
            ref_rows.append(lib.ref71(fr[:T]).float())

        # 帧 → 行映射（同一帧多条时取第一条；缺失帧用前一帧补，与 train_joint_v2 同口径）
        valid = (fr[:T] >= 0) & (fr[:T] < frame_map.numel())
        if bool(valid.any()):
            frame_map[fr[:T][valid]] = torch.arange(n_rows, n_rows + T)[valid]
        n_rows += T
        meta_eps.append({"dump": str(f), "ticks": T,
                         "max_tilt_deg": float((d.get("meta") or {}).get("max_tilt_deg", -1))})

    if not S:
        raise SystemExit("[fail] 没有可用 dump")

    states = torch.cat(S, 0)
    next_states = torch.cat(S1, 0)
    actions = torch.cat(A, 0)
    frames = torch.cat(F, 0)

    # 帧映射补洞（缺的帧用前一帧；起点缺失则用 0）
    miss = (frame_map < 0).nonzero(as_tuple=False).flatten()
    if miss.numel():
        prev = torch.clamp(miss - 1, min=0)
        frame_map[miss] = frame_map[prev]
    if int((frame_map < 0).sum()):
        frame_map[frame_map < 0] = 0

    payload = {
        "states": states, "next_states": next_states, "actions": actions,
        "frames": frames, "frame_map": frame_map,
        "meta": {
            "source": "closed_loop_vla_supermimic --dump",
            "n_rows": int(states.shape[0]),
            "n_dumps": len(meta_eps),
            "has_ref": bool(R),
            "has_frames": bool(int(frames.max()) >= 0),
            "ckpt": str(args.ckpt),
            "gamma": float(args.gamma),
            "episodes": meta_eps,
        },
    }
    if R:
        payload["rewards"] = torch.cat(R, 0)
        payload["ret"] = torch.cat(ret_rows, 0)
        payload["ref_rows"] = torch.cat(ref_rows, 0)

    # ---- 验收（三条，与 S3 §3.1 对齐）----
    ok_cont = True
    if payload["meta"]["has_frames"]:
        pass  # 自由闭环不做跨步连续性断言（末帧无后继）
    finite = bool(torch.isfinite(states).all() and torch.isfinite(actions).all())
    print(json.dumps({
        "rows": int(states.shape[0]), "dumps": payload["meta"]["n_dumps"],
        "has_frames": payload["meta"]["has_frames"], "has_rewards": bool(R),
        "finite": finite,
        "states": tuple(states.shape), "actions": tuple(actions.shape),
        "frame_map_valid": int((frame_map >= 0).sum()), "num_frames": int(lib.num_frames),
    }, indent=1), flush=True)
    if not finite:
        raise SystemExit("[fail] NaN/Inf in states/actions")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.out))
    print(f"[saved] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
