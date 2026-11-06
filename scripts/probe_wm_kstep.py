#!/usr/bin/env python3
"""WM 的 k 步自回归误差：关节角 / 关节速度 / 根位置 / 根朝向，逐 k 报。

为什么需要（和门禁的分工）：
    ``probe_wm_qd_corr.py`` 量的是**一步**预测（qd_corr / qd_ratio），它证明
    "WM 的下一步是对的"，但训练用的是 **H=8 的想象 rollout**——一步准 ≠ 8 步准。
    这里从 318 库里的真实帧出发，自回归滚 K 步（状态自己喂自己），逐步比真值，
    量的就是"想象会飘多远"。

三种动作口径，别混：
    real    —— a_k 取 B 线 dump 里**真实下发**的 ``actions``（**推荐**，唯一诚实的口径）。
               从连续轨迹的 states[t] 出发，用真实的 a[t..t+K-1]，比 states[t+1..t+K]。
    teacher —— a_k = (q_ref[frame+k+1] − default_q) / action_scale
               理想动作（直接指向参考下一帧）。**只配做相对对比**：真机上 q_target
               领先实际 q（PD 跟随滞后），所以这个动作不是仿真会下发的动作，
               绝对误差会被系统性放大（实测 qd_ratio ≈ 4.3 ⇒ 关节速度被喂大）。
    policy  —— a_k = policy(obs_hist, token)，token 取库里的 label/hold。
               训练真实看到的想象轨迹，含策略自身的偏差。

动作语义已实测校验：``q_target_applied = default_q + actions * action_scale``
（在 round_1 B 线 dump 上 max|diff| = 0.000e+00）。

用法：
    python3 scripts/probe_wm_kstep.py --ckpt runs/j1_scene2/round_1/train/joint_final.pt \
        --k 20 --n 256 --device cpu --mode both
    # 多 ckpt 对比（同一批起点、同一 seed，逐 ckpt 打印）
    python3 scripts/probe_wm_kstep.py --ckpt a.pt b.pt --k 20 --n 256

口径提示：state 71 = [p(3), quat(4,wxyz), q(29), v_w(3), w_w(3), qd(29)]，
          q 在 [7:36]，qd 在 [42:71]。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATE_DIM = 71
ACTION_DIM = 29
Q_SLICE = slice(7, 36)
QD_SLICE = slice(42, 71)
SHOW_K = (1, 2, 4, 8, 12, 16, 20)


def _quat_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """两个 (w,x,y,z) 四元数之间的最短夹角（度）。"""
    d = (q1 * q2).sum(-1).abs().clamp(max=1.0 - 1e-7)
    return torch.acos(d) * (360.0 / math.pi)


def _teacher_actions(tgt: torch.Tensor, dflt: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(B,K,71) 真值状态 → (K,B,29) 理想归一化动作（指向下一帧的 q）。"""
    q = tgt[..., Q_SLICE]                                   # (B,K,29)
    return ((q - dflt) / scale).permute(1, 0, 2).contiguous()


def _roll(bundle, start, acts, horizon: int, *, policy: bool, token: torch.Tensor | None):
    """自回归滚 horizon 步，返回 (H,B,71) 预测状态。

    policy=True 时 acts 被忽略，动作由 policy 现算（obs 历史随想象更新）。
    """
    xi = start["xi"]
    state = start["state"]
    hist = start.get("observations") if policy else None    # (H_obs,B,93) rollout 顺序
    preds = []
    with torch.no_grad():
        for k in range(horizon):
            if policy:
                a = bundle.act(bundle.flatten_history(hist), token)
            else:
                a = acts[k]
            state = bundle.step(state, a, xi)
            preds.append(state)
            if policy and k + 1 < horizon:
                obs = bundle.observation(state, a, xi)      # (B,93)
                hist = torch.cat([hist[1:], obs.unsqueeze(0)], dim=0)
    return torch.stack(preds, dim=0)                        # (H,B,71)


def _report(pred: torch.Tensor, tgt: torch.Tensor, ks) -> list[dict]:
    rows = []
    for k in ks:
        if k < 1 or k > pred.shape[0]:
            continue
        p, t = pred[k - 1], tgt[:, k - 1]                   # (B,71)
        eq = (p[..., Q_SLICE] - t[..., Q_SLICE]).abs()      # (B,29) rad
        e_deg = eq * (180.0 / math.pi)
        per_frame = e_deg.mean(-1)                          # (B,) 每帧平均关节误差
        vp, vt = p[..., QD_SLICE], t[..., QD_SLICE]
        vp_c = vp - vp.mean(0)
        vt_c = vt - vt.mean(0)
        rows.append({
            "k": int(k),
            "joint_mae_deg": float(e_deg.mean()),
            "joint_p90_deg": float(per_frame.quantile(0.9)),
            "joint_max_deg": float(e_deg.max()),
            "joint_rmse_rad": float(eq.pow(2).mean().sqrt()),
            "qd_mae": float((vp - vt).abs().mean()),
            "qd_corr": float((vp_c * vt_c).mean() / (vp.std() * vt.std() + 1e-9)),
            "qd_ratio": float(vp.abs().mean() / vt.abs().mean().clamp_min(1e-9)),
            "root_pos_cm": float((p[..., 0:3] - t[..., 0:3]).pow(2).sum(-1).sqrt().mean() * 100),
            "root_ori_deg": float(_quat_angle_deg(p[..., 3:7], t[..., 3:7]).mean()),
        })
    return rows


def _load_dumps(pattern: str, k: int, stride: int, max_files: int = 0):
    """把 B 线 dump 切成 (起点状态, K 步真实动作, K 步真值状态)。

    返回 S (B,71) / A (K,B,29) / G (B,K,71)。
    """
    import glob  # noqa: PLC0415

    files = sorted(glob.glob(pattern))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"[fail] --dump 没匹配到文件：{pattern}")
    S, A, G = [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        st, ac = d["states"].float(), d["actions"].float()
        T = int(st.shape[0])
        if T < k + 1:
            continue
        s = torch.arange(0, T - k, stride)
        S.append(st[s])
        A.append(torch.stack([ac[s + j] for j in range(k)], dim=1))       # (n,K,29)
        G.append(torch.stack([st[s + j + 1] for j in range(k)], dim=1))   # (n,K,71)
    if not S:
        raise SystemExit(f"[fail] {len(files)} 个 dump 都短于 k+1={k + 1} 帧")
    return torch.cat(S), torch.cat(A).permute(1, 0, 2).contiguous(), torch.cat(G)


def _print_table(tag: str, rows: list[dict]) -> None:
    print(f"\n[{tag}]  k   joint_mae  p90   max   qd_corr  qd_ratio  root_pos  root_ori")
    print("            (deg)    (deg) (deg)           (×)       (cm)     (deg)")
    for r in rows:
        print(f"          {r['k']:2d}   {r['joint_mae_deg']:7.2f}  {r['joint_p90_deg']:5.1f} "
              f"{r['joint_max_deg']:5.1f}   {r['qd_corr']:6.3f}   {r['qd_ratio']:5.3f}   "
              f"{r['root_pos_cm']:6.2f}   {r['root_ori_deg']:6.2f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--motion", nargs="+", type=Path, default=[ROOT / "data/npz/g1_tasks"])
    ap.add_argument("--tokens", type=Path, default=ROOT / "data/library/ft318_tokens.pt")
    ap.add_argument("--k", type=int, default=20, help="滚多少步（训练 H=8，门禁一步）")
    ap.add_argument("--n", type=int, default=256, help="起点窗口数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("real", "teacher", "policy", "both", "all"), default=None,
                    help="real=dump 里真实下发的动作（推荐）; teacher/policy=库里的理想/策略动作; "
                         "both=teacher+policy; all=三种全跑。不给时：有 --dump 就只跑 real")
    ap.add_argument("--dump", default=None, help="glob，B 线 dump（含 states/actions，需连续）")
    ap.add_argument("--stride", type=int, default=20, help="dump 里每隔多少 tick 取一个起点")
    ap.add_argument("--max-files", type=int, default=0, help="最多用几个 dump（0=全部）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None, help="可选：json 输出")
    args = ap.parse_args()

    from diff_vla_wbc.mjlab_joint import load_mjlab_bundle  # noqa: PLC0415

    npz: list[Path] = []
    for p in args.motion:
        npz.extend(sorted(Path(p).glob("*.npz")) if Path(p).is_dir() else [Path(p)])

    dev = torch.device(args.device)
    modes = args.mode or ("real" if args.dump else "both")
    if modes == "all":
        modes = ["real", "teacher", "policy"]
    elif modes == "both":
        modes = ["teacher", "policy"]
    else:
        modes = [modes]
    if "real" in modes and not args.dump:
        raise SystemExit("[fail] --mode real 需要 --dump")
    print(f"[probe] ckpt={len(args.ckpt)} k={args.k} n={args.n} modes={modes} device={args.device}",
          flush=True)

    # real 模式的数据与 ckpt 无关，先切好
    S = A = G = None
    if "real" in modes:
        S, A, G = _load_dumps(args.dump, int(args.k), int(args.stride), int(args.max_files))
        G = G.to(dev)
        print(f"[probe] dump 起点 {S.shape[0]} 个（stride={args.stride}, k={args.k}）", flush=True)

    pack = None
    need_policy = "policy" in modes
    if need_policy:
        pack = torch.load(str(args.tokens), map_location="cpu", weights_only=False)["tokens"].float()

    want = sorted({k for k in (1, 2, 4, 8, 12, 16, 20, args.k) if k <= args.k})
    all_rows: dict[str, dict] = {}

    for ck in args.ckpt:
        need_lib = ("teacher" in modes) or need_policy
        if need_lib and not npz:
            raise SystemExit(f"[fail] teacher/policy 模式需要 --motion 的 npz（库里的参考帧）")
        bundle = load_mjlab_bundle(ck, device=dev, npz=npz if need_lib else None, token_dim=64)
        if need_lib:
            lib = bundle.library
            g = torch.Generator().manual_seed(int(args.seed))
            frames = torch.randint(10, int(lib.num_frames) - args.k - 20,
                                   (int(args.n),), generator=g).to(dev)
            start = bundle.reference_start(lib, frames)
            tgt = bundle.teacher_states(lib, frames, int(args.k))        # (B,K,71)
            dflt = bundle.default_joint_pos.float().to(dev)
            scale = bundle.action_scale.float().to(dev)
            acts = _teacher_actions(tgt, dflt, scale)

        entry: dict = {"ckpt": str(ck), "k": int(args.k), "n": int(args.n), "runs": {}}
        if "real" in modes:
            from diff_vla_wbc.mjlab_joint import default_xi  # noqa: PLC0415
            Sd, Ad = S.to(dev), A.to(dev)
            pred = _roll(bundle, {"state": Sd, "xi": default_xi(Sd.shape[0], dev)},
                         Ad, int(args.k), policy=False, token=None)
            rows = _report(pred, G, want)
            _print_table(f"real     {Path(ck).parent.parent.name}/{Path(ck).name}", rows)
            entry["runs"]["real"] = rows
        if "teacher" in modes:
            pred = _roll(bundle, start, acts, int(args.k), policy=False, token=None)
            rows = _report(pred, tgt, want)
            _print_table(f"teacher  {Path(ck).parent.parent.name}/{Path(ck).name}", rows)
            entry["runs"]["teacher"] = rows
        if need_policy:
            token = pack[frames.cpu()].to(dev)
            pred = _roll(bundle, start, None, int(args.k), policy=True, token=token)
            rows = _report(pred, tgt, want)
            _print_table(f"policy   {Path(ck).parent.parent.name}/{Path(ck).name}", rows)
            entry["runs"]["policy"] = rows

        for mode, rows in entry["runs"].items():
            r = next((x for x in rows if x["k"] == args.k), None)
            if r:
                print(f"[summary] {Path(ck).name} · {mode} · k={args.k}: "
                      f"joint_mae={r['joint_mae_deg']:.2f}° p90={r['joint_p90_deg']:.1f}° "
                      f"qd_corr={r['qd_corr']:.3f} ratio={r['qd_ratio']:.3f} "
                      f"root_pos={r['root_pos_cm']:.2f}cm root_ori={r['root_ori_deg']:.2f}°", flush=True)
        all_rows[str(ck)] = entry

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(all_rows, indent=1))
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
