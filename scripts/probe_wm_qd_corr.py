#!/usr/bin/env python3
"""WM 速度通道验收门（问题 F）：``qd_corr`` / 幅值比 / 位置误差。

对应 `docs/HANDOVER_SUPERMIMIC_FINETUNE.md` §2.4 的验收线：
    corr(pred_qd, ref_qd) > 0.9，|pred_qd|/|ref_qd| ∈ [0.9, 1.1]
并额外报位置/姿态/关节误差（mjlab 的 WM 是**仿真转移**训的，量级与 Isaac 不同）。

口径（**必须用 policy 自己的动作**，不能用手算 Δq —— 历史踩坑，见交接文档 §1③）：
    a_t = bundle.act(flatten_history(obs_hist_t), token_t)
    s_{t+1} = bundle.step(s_t, a_t, xi)
    pred_qd = s_{t+1}[..., 42:71]   vs   teacher_qd = teacher_states[..., 42:71]

用法::

    python3 scripts/probe_wm_qd_corr.py \\
        --ckpt runs/wbc_ft318_mjlab/ckpt_latest.pt \\
        --motion data/npz/g1_tasks --tokens data/library/ft318_tokens.pt \\
        --n 64 --h 8 --out runs/gates/wm_qd_corr.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--motion", nargs="+", type=Path, default=[ROOT / "data/npz/g1_tasks"])
    ap.add_argument("--tokens", type=Path, default=ROOT / "data/library/ft318_tokens.pt")
    ap.add_argument("--transitions", type=Path, default=None,
                    help="**正确口径**：用仿真真转移（states/actions/next_states）做一步预测对比。"
                         "不给则退化成『WM vs 参考库』的旧口径——那比的是**参考的平滑度**，不是 WM 精度")
    # 2026-10-29：n=256 的抽样噪声约 ±0.01（同一 ckpt 能测出 1.0944=PASS 与 1.1056=FAIL 两种
    # 相反结论）⇒ 这里强制下限 2000（噪声 ~0.002，代价 ~10 s），防止"运气过门"。
    ap.add_argument("--n", type=int, default=64, help="采样窗口数（实际取 max(n,2000)）")
    ap.add_argument("--h", type=int, default=8, help="想象步数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    # 2026-10-29 用户口径：验收必须用 qd_corr 的严标准（0.95），而不是 0.90 的宽标准。
    # 之前出过的事故是"corr 0.9586 但 ratio 1.26"，所以 corr 与 ratio **两个都要过**：
    #   pass = (qd_corr >= pass_corr) and (pass_ratio_lo <= qd_ratio <= pass_ratio_hi)
    ap.add_argument("--pass-corr", type=float, default=0.95)
    ap.add_argument("--pass-ratio-lo", type=float, default=0.90)
    ap.add_argument("--pass-ratio-hi", type=float, default=1.10)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/gates/wm_qd_corr.json")
    args = ap.parse_args()

    from diff_vla_wbc.mjlab_joint import default_xi, load_mjlab_bundle

    npz = []
    for p in args.motion:
        npz.extend(sorted(Path(p).glob("*.npz")) if Path(p).is_dir() else [Path(p)])
    bundle = load_mjlab_bundle(args.ckpt, device=args.device, npz=npz, token_dim=64)
    lib = bundle.library
    dev = torch.device(args.device)

    pack = torch.load(str(args.tokens), map_location="cpu", weights_only=False)["tokens"].float()
    if int(pack.shape[0]) != int(lib.num_frames):
        raise SystemExit(f"[fail] token 包 {pack.shape[0]} != 库 {int(lib.num_frames)}")

    # ---------------- 正确口径：一步预测 vs 仿真真值 ----------------
    if args.transitions is not None:
        d = torch.load(str(args.transitions), map_location="cpu", weights_only=False)
        n_all = int(d["actions"].shape[0])
        g = torch.Generator().manual_seed(int(args.seed))
        idx = torch.randperm(n_all, generator=g)[: int(max(int(args.n), 2000))]
        st = d["states"][idx].to(dev).float()
        ac = d["actions"][idx].to(dev).float()
        nx = d["next_states"][idx].to(dev).float()
        xi = default_xi(st.shape[0], dev)
        with torch.no_grad():
            pred = bundle.step(st, ac, xi)
        pq, rq = pred[..., 42:71], nx[..., 42:71]
        qd_corr = float(((pq - pq.mean(0)) * (rq - rq.mean(0))).mean()
                        / (pq.std() * rq.std() + 1e-9))
        out = {
            "ckpt": str(args.ckpt), "mode": "sim_transitions",
            "transitions": str(args.transitions), "n": int(idx.numel()),
            "qd_corr": qd_corr,
            "qd_ratio": float(pq.abs().mean() / rq.abs().mean().clamp_min(1e-9)),
            "qd_abs_pred": float(pq.abs().mean()), "qd_abs_ref": float(rq.abs().mean()),
            "joint_rmse_rad": float((pred[..., 7:36] - nx[..., 7:36]).pow(2).mean().sqrt()),
            "pos_err_m": float((pred[..., 0:3] - nx[..., 0:3]).pow(2).sum(-1).mean().sqrt()),
        }
        ok = (out["qd_corr"] >= args.pass_corr
              and args.pass_ratio_lo <= out["qd_ratio"] <= args.pass_ratio_hi)
        out["pass"] = bool(ok)
        print("[GATE][wm_qd] " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                             for k, v in out.items()}), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
        return 0 if ok else 1

    g = torch.Generator().manual_seed(int(args.seed))
    frames = torch.randint(10, int(lib.num_frames) - int(args.h) - 20, (int(args.n),), generator=g)
    start = bundle.reference_start(lib, frames)
    tgt = bundle.teacher_states(lib, frames, int(args.h))

    state, hist = start["state"], start["observations"]            # (B,71) / (H,B,93)
    a_prev = start["prev_action"]
    preds = []
    for t in range(int(args.h)):
        obs_flat = bundle.flatten_history(hist)
        f = torch.clamp(frames + t, max=pack.shape[0] - 1)
        token = pack[f].to(dev)
        a = bundle.act(obs_flat, token)
        state = bundle.step(state, a, start["xi"])
        obs = bundle.observation(state, a, start["xi"])
        hist = torch.cat((hist[1:], obs.unsqueeze(0)), dim=0)
        preds.append(state)
        a_prev = a
    pred = torch.stack(preds, 1)                                    # (B,H,71)

    def corr(x, y):
        x = x.reshape(-1).double()
        y = y.reshape(-1).double()
        x = x - x.mean()
        y = y - y.mean()
        d = (x.norm() * y.norm()).clamp_min(1e-12)
        return float((x @ y) / d)

    pred_qd, ref_qd = pred[..., 42:71], tgt[..., 42:71]
    pred_q, ref_q = pred[..., 7:36], tgt[..., 7:36]
    pred_p, ref_p = pred[..., 0:3], tgt[..., 0:3]
    ratio = float(pred_qd.abs().mean() / ref_qd.abs().mean().clamp_min(1e-9))
    out = {
        "ckpt": str(args.ckpt), "n": int(args.n), "h": int(args.h),
        "qd_corr": corr(pred_qd, ref_qd),
        "qd_ratio": ratio,
        "qd_abs_pred": float(pred_qd.abs().mean()),
        "qd_abs_ref": float(ref_qd.abs().mean()),
        "joint_rmse_rad": float((pred_q - ref_q).pow(2).mean().sqrt()),
        "pos_err_m": float((pred_p - ref_p).pow(2).sum(-1).mean().sqrt()),
        "ref_qd_abs": float(ref_qd.abs().mean()),
    }
    ok = (out["qd_corr"] >= args.pass_corr
          and args.pass_ratio_lo <= out["qd_ratio"] <= args.pass_ratio_hi)
    out["mode"] = "library_reference(诊断用，不是验收口径)"
    out["pass"] = bool(ok)
    out["thresholds"] = {"corr": args.pass_corr,
                         "ratio": [args.pass_ratio_lo, args.pass_ratio_hi]}
    # 备注：318 数据集的**根**速度恒为 0（prepare_g1_tasks_npz.py:215），所以这里 qd 只对
    # 29 个关节速度有意义；根速度维度不参与判据。
    print("[GATE][wm_qd] " + json.dumps({k: round(v, 6) if isinstance(v, float) else v
                                         for k, v in out.items()}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
