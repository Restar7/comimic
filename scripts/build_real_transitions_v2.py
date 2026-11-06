#!/usr/bin/env python3
"""P0-3：用**真正下发的动作**重建真转移，一次产出 WM 保鲜数据 + critic 监督目标。

为什么重写（2026-10-28）：
旧 `scripts/build_real_transitions.py` 的动作是 `a_t = (q_cmd(t+1) − default_q)/scale`
（**命令反推**），与 WBC 真正下发到 PD 的目标有系统偏差 ⇒ WM 保鲜时等于在错的动力学上监督
（评审 ③；上次保鲜把 `qd_corr` 从 0.752 打到 0.589 就是这个原因）。

现在的动作来源（正确）：
* 录制侧：`runs/wm_real/ref_action_lib.pt` 的 `actions`（= raw npz 的 `decoder_target_action`，
  asset→canonical 已转，覆盖 634,016 帧 100%）；
* on-policy 侧（可选 `--extra-dumps`）：B 线 dump 里的 `q_target_applied`（真正下发、已 clip），
  归一化动作 `a = (q_target_applied − default_q) / action_scale`（**不是** `−default_q`；
  实测 `action_scale` mean≈0.379，若按 1.0 算，WM 的动作语义会差 ~2.6×）。

奖励：与训练/评测**同一个函数** `bundle.reward`，参考用 `FrameRef(真实库行号)`。

产出（`--out`）：
    states (N,71) / actions (N,29) / next_states (N,71) / frames (N,) / clip_ids (N,)
    rewards (N,) / ret (N,) / ref_rows (N,71)          ← ret/ref_rows 给 critic 用
    meta{...}

用法：
    python3 scripts/build_real_transitions_v2.py \
        --out runs/wm_real/real_transitions_v2.pt \
        --extra-dumps 'runs/B_r1_*/dump/*.pt'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diff_vla_wbc.isaac_joint import FrameRef, canonical_to_flat, load_isaac_bundle  # noqa: E402


def _discounted_return_to_go(r: torch.Tensor, clip_id: torch.Tensor, gamma: float) -> torch.Tensor:
    """每个 clip 内部从后往前的折扣 return-to-go（不跨 clip）。"""
    out = torch.zeros_like(r)
    n = r.shape[0]
    run = 0.0
    prev_clip = None
    for i in range(n - 1, -1, -1):
        c = int(clip_id[i])
        if prev_clip is not None and c != prev_clip:
            run = 0.0
        run = float(r[i]) + gamma * run
        out[i] = run
        prev_clip = c
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--ref-action-lib", type=Path, default=ROOT / "runs/wm_real/ref_action_lib.pt")
    ap.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--extra-dumps", type=str, default="",
                    help="glob：B 线 dump（含 states/frames/q_target_applied），追加为 on-policy 数据")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wm_real/real_transitions_v2.pt")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    dev = torch.device(a.device)
    npz = sorted(a.npz_dir.glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz, packed_motion=a.packed,
                               obs_mode=None, hist=None, token_dim=64)
    lib = bundle.motion
    ref = lib.ref.float().cpu()                      # canonical (N,71): p,quat,v,w,q,qd
    states = canonical_to_flat(ref)                  # flat (N,71):     p,quat,q,v,w,qd
    n_ref = int(ref.shape[0])

    d = torch.load(str(a.ref_action_lib), map_location="cpu", weights_only=False)
    acts = d["actions"].float()
    valid = d["valid"].bool() if "valid" in d else torch.ones(n_ref, dtype=torch.bool)
    clip_id = torch.zeros(n_ref, dtype=torch.long)
    clip_start = torch.zeros(n_ref, dtype=torch.bool)
    clip_end = torch.zeros(n_ref, dtype=torch.bool)
    for i, c in enumerate(lib.clips):
        s, l = int(c.start), int(c.length)
        clip_id[s:s + l] = i
        clip_start[s] = True
        clip_end[s + l - 1] = True
    print(f"[trans v2] library rows={n_ref} clips={len(lib.clips)} "
          f"action coverage={float(valid.float().mean())*100:.2f}%", flush=True)

    # prev_action：clip 内前一步（clip 首帧用自身，与 compute_collect_reward 口径一致）
    prev = acts.clone()
    prev[1:] = acts[:-1]
    prev[clip_start] = acts[clip_start]

    frames = torch.arange(n_ref, dtype=torch.long)
    r = torch.zeros(n_ref)
    for lo in range(0, n_ref, a.batch):
        sl = slice(lo, min(lo + a.batch, n_ref))
        terms = bundle.reward(states[sl].to(dev), acts[sl].to(dev), prev[sl].to(dev),
                              FrameRef(frames[sl]))
        r[sl] = terms["total"].detach().cpu()
        if (lo // a.batch) % 40 == 0:
            print(f"[trans v2] reward {lo}/{n_ref}", flush=True)

    nxt = states.clone()
    nxt[:-1] = states[1:]
    nxt[clip_end] = states[clip_end]                 # clip 末帧的 next = 自身

    # ---- 可选：追加 B 线 on-policy 数据（动作 = q_target_applied − default_q）----
    extra = {"states": [], "actions": [], "next_states": [], "frames": [], "rewards": [],
             "ret": [], "ref_rows": [], "clip_ids": []}
    if a.extra_dumps:
        import glob  # noqa: PLC0415
        dflt = bundle.default_joint_pos.float().cpu()
        scale = bundle.action_scale.float().cpu()
        # ★ WM 的动作语义 = 策略原始输出 a_norm；dump 里直接有 `actions`（provider 记的就是它），
        #   优先用它。不要拿 bundle.action_scale 反推 —— 那个是标量 1.0，而 per-joint scale
        #   的均值是 0.379（ckpt 的 robot_spec），反推会差 2.6×（2026-10-28 发现）。
        _ck = torch.load(str(a.wbc_ckpt), map_location="cpu", weights_only=False)
        _spec_scale = torch.as_tensor(_ck["robot_spec"]["action_scale"], dtype=torch.float32)
        files = sorted(glob.glob(a.extra_dumps))
        print(f"[trans v2] extra dumps: {len(files)} 个  "
              f"action_scale mean={float(scale.mean()):.4f} max={float(scale.max()):.4f}", flush=True)
        for f in files:
            e = torch.load(f, map_location="cpu", weights_only=False)
            if "q_target_applied" not in e:
                print(f"[trans v2] SKIP {Path(f).name}（没有 q_target_applied，旧 dump）", flush=True)
                continue
            from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415
            import joint_contract as JC  # noqa: PLC0415
            _perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)
            s = state_to_standard(e["states"].float(), _perm)   # asset -> standard
            q_tgt = e["q_target_applied"].float()
            # 归一化动作：WM/act 的动作语义是 default + scale * a；不是 default + a。
            # WM 的动作语义（**实测**）：``q_target_applied − default_q``（弧度），
            # 即 ``bundle.action_scale`` 是标量 1.0；dump 的 ``actions`` 是策略原始输出 a_norm，
            # 两者差一个 per-joint scale（mean 0.379），**不能**直接混用（2026-10-28 核对）。
            act = (q_tgt - dflt.unsqueeze(0)) / scale.unsqueeze(0)
            fr = e["frames"].long()
            rew = e.get("rewards")
            if rew is None:
                print(f"[trans v2] SKIP {Path(f).name}（缺 rewards；先跑 compute_collect_reward）", flush=True)
                continue
            n = s.shape[0]
            extra["states"].append(s)
            extra["actions"].append(act)
            ns = torch.cat((s[1:], s[-1:]), dim=0)
            extra["next_states"].append(ns)
            extra["frames"].append(fr)
            extra["rewards"].append(rew.float())
            _cid = torch.full((n,), n_ref + len(extra["states"]) - 1, dtype=torch.long)
            extra["clip_ids"].append(_cid)
            # critic 侧要 (states, ret, ref_rows)：B 线也要算 MC return-to-go 和参考行
            extra["ret"].append(_discounted_return_to_go(rew.float(), _cid, a.gamma))
            _rr = ref[fr.clamp(0, n_ref - 1)].float()
            extra["ref_rows"].append(_rr)
            print(f"[trans v2]   + {Path(f).name}: {n} 帧 r_mean={float(rew.mean()):+.5f}", flush=True)

    out = {
        "states": states,
        "actions": acts,
        "next_states": nxt,
        "frames": frames,
        "clip_ids": clip_id,
        "rewards": r,
        "ret": _discounted_return_to_go(r, clip_id, a.gamma),
        "ref_rows": ref,                              # canonical 行，critic 侧会用 canonical_to_flat
        "valid": valid,
        "meta": {
            "source": "ref_action_lib (decoder_target_action, canonical) + bundle.reward",
            "gamma": a.gamma,
            "n_ref": n_ref,
            "action_scale_is_one": False,
            "action_scale_mean": float(bundle.action_scale.mean()),
            "action_scale_max": float(bundle.action_scale.max()),
            "extra_action_normalization": "(q_target_applied - default_q) / action_scale",
            "reward_fn": "bundle.reward(... FrameRef(library row))",
            "note": "P0-3：动作标签 = 录制时真正下发的动作；不是 q_cmd 反推",
        },
    }
    if extra["states"]:
        out["extra"] = {k: torch.cat(v) for k, v in extra.items()}
        out["meta"]["n_extra"] = int(sum(x.shape[0] for x in extra["states"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"[trans v2] saved {a.out}")
    print(f"[trans v2]   r_mean={float(r.mean()):+.5f}  ret_mean={float(out['ret'].mean()):+.5f}")
    if "extra" in out:
        print(f"[trans v2]   extra={out['meta']['n_extra']} 帧 "
              f"r_mean={float(out['extra']['rewards'].mean()):+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
