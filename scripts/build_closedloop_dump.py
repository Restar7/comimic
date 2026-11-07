#!/usr/bin/env python3
"""把闭环 `dump/steps.jsonl` 拼成 `probe_wm_gradient_truth.py` 能吃的 dump，
并**自检口径**（WM 一步预测误差）。2026-11-06。

为什么要这一步：`probe_wm_gradient_truth.py plan` 需要
``states (T,71, ha_v31 序)`` + ``q_target_applied``（**归一化动作、standard 序**）+ ``frames``。
闭环 dump 里 q/applied 是 **asset(HA) 序**原始关节目标，必须先按 `joint_contract` 置换、
再按 ``(q_tgt − default_q)/action_scale`` 归一化 —— 与
`scripts/build_real_transitions_v2.py` 完全同一套口径（WM 就是在这套口径上训的）。

自检（必须先过，否则后面的探针数字没有意义）：
    在 `runs/a_prime_sonic_exc/all_transitions.pt`（WM 训练分布）上量 WM 一步误差当**基线**，
    再在本 dump 上量同样的量；两者同量级 ⇒ 口径对。

用法::

    PYTHONPATH=. python3 scripts/build_closedloop_dump.py \
        --runs 'runs/closedloop_r3/*/dump/steps.jsonl' --out runs/probe_truth/closedloop_dump.pt \
        --ckpt runs/a_prime_sonic_exc/wm_gain2.pt --device cuda:0
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

STATE_KEYS = ["frame", "q", "qd", "root_pos", "root_quat_wxyz",
              "root_lin_vel_b", "root_ang_vel_b", "applied"]


def load_run(p: Path) -> dict[str, torch.Tensor] | None:
    buf = {k: [] for k in STATE_KEYS}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if not all(k in d for k in STATE_KEYS):
            continue
        for k in STATE_KEYS:
            buf[k].append(d[k])
    if len(buf["q"]) < 50:
        return None
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in buf.items()}


def to_flat71(s: dict[str, torch.Tensor]) -> torch.Tensor:
    """[p(3), quat(4), q(29), v(3), w(3), qd(29)] —— asset(HA) 序。"""
    return torch.cat([s["root_pos"], s["root_quat_wxyz"], s["q"],
                      s["root_lin_vel_b"], s["root_ang_vel_b"], s["qd"]], dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/closedloop_r3/*/dump/steps.jsonl")
    ap.add_argument("--out", default="runs/probe_truth/closedloop_dump.pt")
    ap.add_argument("--wbc-ckpt", default="runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--ckpt", default="runs/a_prime_sonic_exc/wm_gain2.pt",
                    help="**WM 覆盖** ckpt（只有 wm 张量）；底座用 --wbc-ckpt")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check-n", type=int, default=512)
    a = ap.parse_args()

    import joint_contract as JC  # noqa: PLC0415
    from diff_vla_wbc.isaac_joint import default_xi, load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    files = sorted(glob.glob(a.runs))
    if a.max_runs:
        files = files[: a.max_runs]
    S_h, A_std, FR = [], [], []
    skipped = 0
    for f in files:
        s = load_run(Path(f))
        if s is None:
            skipped += 1
            continue
        S_h.append(to_flat71(s))
        A_std.append(s["applied"])
        FR.append(s["frame"].long())
    if not S_h:
        raise SystemExit("没有可用 dump")
    S_h = torch.cat(S_h, 0)
    A_ha = torch.cat(A_std, 0)
    FR = torch.cat(FR, 0)
    print(f"[dump] runs={len(S_h) and len(files) - skipped} skipped={skipped} 帧={S_h.shape[0]}")

    bundle = load_isaac_bundle(
        a.wbc_ckpt, device=dev,
        npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
        packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
        obs_mode=None, hist=None, token_dim=64,
    )
    _wd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    bundle.wm.load_state_dict(_wd.get("wm", _wd), strict=True)
    print(f"[wm] swapped from {a.ckpt}", flush=True)
    perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)

    # ---- 动作：HA 序 → standard 序 → 按 (q_tgt − default)/scale 归一化 ----------
    dflt = bundle.default_joint_pos.float().cpu().reshape(-1)
    scale = bundle.action_scale.float().cpu().reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand_as(dflt)
    q_std = A_ha[:, perm]
    act = (q_std - dflt.unsqueeze(0)) / scale.unsqueeze(0)
    print(f"[act] default_q[:4]={dflt[:4].tolist()}  scale mean={float(scale.mean()):.4f} "
          f"|a| mean={float(act.abs().mean()):.4f} max={float(act.abs().max()):.3f}")

    # ---- 自检 1：WM 在训练分布（all_transitions）上的一步误差 --------------------
    def one_step_err(states_std, acts, tag):
        n = min(int(a.check_n), states_std.shape[0] - 1)
        idx = torch.linspace(0, states_std.shape[0] - 2, n).long()
        s = states_std[idx].to(dev)
        nx = states_std[idx + 1].to(dev)
        ac = acts[idx].to(dev)
        xi = default_xi(n, dev)
        with torch.no_grad():
            pred = bundle.step(s, ac, xi, step_idx=1)
        e_p = (pred[:, 0:3] - nx[:, 0:3]).norm(dim=-1)
        e_q = (pred[:, 7:36] - nx[:, 7:36]).norm(dim=-1)
        e_qd = (pred[:, 42:71] - nx[:, 42:71]).norm(dim=-1)
        print(f"[check] {tag:<22} n={n} 一步误差 p50: pos={float(e_p.median())*100:.2f} cm  "
              f"q={float(e_q.median()):.4f} rad  qd={float(e_qd.median()):.3f} rad/s")

    tr = torch.load(ROOT / "runs/a_prime_sonic_exc/all_transitions.pt",
                    map_location="cpu", weights_only=False)
    _idx = torch.randperm(tr["states"].shape[0])[:4000]
    one_step_err(tr["states"].float()[_idx], tr["actions"].float()[_idx], "训练分布(exc)")

    # ---- 自检 2：本闭环 dump 上同一量 -----------------------------------------
    from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415
    S_std = state_to_standard(S_h, perm)
    one_step_err(S_std, act, "闭环 dump")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": S_h, "q_target_applied": act, "frames": FR,
                "meta": {"src": a.runs, "n": int(S_h.shape[0]),
                         "action": "(q_tgt_std - default_q)/action_scale",
                         "state_order": "ha_v31 (asset)"}}, out)
    print(f"[ok] 写出 {out}  states={tuple(S_h.shape)} acts={tuple(act.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
