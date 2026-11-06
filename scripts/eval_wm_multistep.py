#!/usr/bin/env python3
"""O2：WM 的**多步**留出验收（不是一个步的 qd_corr）。

为什么需要：
    ``eval_wm_velocity`` 量的是**一步**预测（qd_corr / pos_err），而训练用的是 H=20 的
    **多步**想象。一步准 ≠ 20 步准；"想象涨、真机不动"正是多步误差造成的。
    这里从 B 线真实走过的**重规划状态**出发，用**真实下发的动作**滚 H 步，
    和真实状态逐步比 —— 这就是 WM 在"策略自己会去的地方"的误差。

做法：
    s_t = B dump 的 states（asset 序 → standard 序）
    a_t = q_target_applied[t] − default_q      （WM 的动作语义，弧度）
    ŝ_{t+k} = WM(ŝ_{t+k-1}, a_{t+k-1})，k=1..H（动作用真的，状态自回归）
    误差：关节 RMSE / 根位置 / 根朝向 / 速度相关，按 k 报

判据（G0-b 的第二步）：
    同一批 B 片段上，新版 WM 的 20 步累计误差不高于上一版 5%。

用法：
    PYTHONPATH=. python3 scripts/eval_wm_multistep.py \
        --ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt runs/S1_handover/WBC_A24000_wmvel_v2.pt \
        --dump 'runs/B_*/dump/*.pt' --horizon 20 --stride 20
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _load_dumps(pattern: str, perm: torch.Tensor, device):
    from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[multistep] 没有匹配 {pattern}")
    rows = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        # ★★ 2026-10-31 修（**仪器 bug，与 A10 同类**）：原来用 `q_target_applied − default_q`
        #   当动作喂 WM。但采集里 `q_target = clip(q0 + a*scale)`，`q0` 是**当时的关节位置**
        #   而不是 `default_q` ⇒ `q_target − default_q = (q0 − default_q) + a`，
        #   实测 `|a|` 均值 0.89 vs `|q_target−default_q|` 0.23、max 差 9.5 rad，
        #   即喂进去的**不是策略输出的动作**。而 WM 训练用的是 `dump["actions"]`。
        #   现在优先用 `actions`（与训练/部署同一语义）；缺失才回退并**明确标注**。
        if "actions" in d:
            A = d["actions"].float().reshape(-1, 29)
            src = "actions(训练同源)"
        elif "q_target_applied" in d:
            A = d["q_target_applied"].float().reshape(-1, 29)
            src = "q_target-default(回退，非训练语义)"
            print(f"[multistep] WARN {Path(f).name}：没有 actions，回退 q_target_applied"
                  f"（动作语义与训练不同，结果仅供参考）", flush=True)
        else:
            print(f"[multistep] SKIP {Path(f).name}（既没有 actions 也没有 q_target_applied）", flush=True)
            continue
        S = state_to_standard(d["states"].float(), perm)          # asset -> standard
        rows.append({"name": Path(f).name, "states": S, "q_tgt": A,
                     "frames": d.get("frames"), "action_source": src})
    if not rows:
        raise SystemExit("[multistep] 没有可用 dump")
    return rows


@torch.no_grad()
def _roll(bundle, S0: torch.Tensor, acts: torch.Tensor, horizon: int,
          step0: int = 1) -> torch.Tensor:
    """从 S0 出发，用给定动作滚 horizon 步 → (horizon+1, N, 71)。

    ``step0``：第一步的 ``step_idx``。热启动（先喂 W 步真实历史填满隐状态）后要从 W+1 起，
    否则 ``IsaacJointBundle.step`` 会在 step_idx=1 处把隐状态清零（GRU 的记忆就白填了）。
    """
    xi = (bundle.default_xi(S0.shape[0], S0.device)
          if hasattr(bundle, "default_xi") else None)
    if xi is None:
        from diff_vla_wbc.isaac_joint import default_xi  # noqa: PLC0415
        xi = default_xi(S0.shape[0], S0.device)
    out = [S0]
    s = S0
    for k in range(horizon):
        s = bundle.step(s, acts[:, k], xi, step_idx=int(step0) + k)
        out.append(s)
    return torch.stack(out, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--dump", required=True, help="glob，元素含 states/q_target_applied")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-starts", type=int, default=0, help="0=全部重规划时刻")
    ap.add_argument("--start-lo", type=int, default=0, help="只用这个帧号之后的起点（留出段评测）")
    ap.add_argument("--start-hi", type=int, default=0, help="只用这个帧号之前的起点（0=不限制）")
    ap.add_argument("--warmup", type=int, default=0,
                    help="记忆热启动步数 W：从 S[idx-W] 起用**真实动作**滚 W+horizon 步，"
                         "只在最后 horizon 步上量误差。W=0（默认）与历史口径逐位相同；"
                         "W>0 让 GRU 的隐状态先被历史填满（见 --warmup-mode）")
    ap.add_argument("--warmup-mode", choices=("teacher", "free"), default="teacher",
                    help="teacher（默认）= 预热 W 步**喂真实状态**（只填隐状态，不累积漂移）⇒ "
                         "对 MLP 是**逐位零变化**的完美对照；free = 预热也自回归（含漂移误差）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None, help="可选：json 输出")
    a = ap.parse_args()

    import joint_contract as JC  # noqa: PLC0415
    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)
    dumps = _load_dumps(a.dump, perm, dev)
    print(f"[multistep] action_source = {dumps[0].get('action_source', '?')}"
          f"  [data_source={a.dump}]", flush=True)
    keys = (1, max(1, a.horizon // 4), max(1, a.horizon // 2), a.horizon)
    keys = tuple(sorted(set(k for k in keys if k <= a.horizon)))

    results = {}
    for ck in a.ckpt:
        bundle = load_isaac_bundle(ck, device=dev,
                                   npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
                                   packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                                   obs_mode=None, hist=None, token_dim=64)
        dflt = bundle.default_joint_pos.float().to(dev)
        per_k = {k: {"joint": [], "root": [], "rot": [], "qd_corr": [],
                     "root_w_corr": [], "root_w_ratio": []} for k in keys}
        for d in dumps:
            S, Qt = d["states"].to(dev), d["q_tgt"].to(dev)
            T = int(S.shape[0])
            W = max(0, int(a.warmup))          # 记忆热启动：起点前移 W 步（W=0 ⇒ 历史口径）
            # ★ 起点集合**不能**因为 W 改变：否则比的是两批不同的窗口，MLP 对照也会"变"。
            #   只保证预热起点不越界（idx-W >= 10 的那条"静置区"约束）。
            lo = max(10, int(a.start_lo), W)
            hi = (int(a.start_hi) if int(a.start_hi) > 0 else T)
            starts = list(range(lo, min(hi, T - a.horizon - 1), max(1, int(a.stride))))
            if a.max_starts:
                starts = starts[: int(a.max_starts)]
            if not starts:
                continue
            idx = torch.tensor(starts, device=dev)
            if W > 0 and a.warmup_mode == "teacher":
                # 只为了填隐状态：喂真实状态 W 步（不参与测量）⇒ MLP 上逐位等价于 W=0
                from diff_vla_wbc.isaac_joint import default_xi as _dxi  # noqa: PLC0415
                xi = (bundle.default_xi(int(idx.shape[0]), dev)
                      if hasattr(bundle, "default_xi") else _dxi(int(idx.shape[0]), dev))
                for _k in range(W):
                    bundle.step(S[idx - W + _k], Qt[idx - W + _k], xi, step_idx=_k + 1)
                acts = torch.stack([Qt[idx + k] for k in range(int(a.horizon))], 1)
                pred = _roll(bundle, S[idx], acts, int(a.horizon), step0=W + 1)
                p_at = lambda k: pred[k].float()          # noqa: E731  第 k 步就是 pred[k]
            else:
                S0 = S[idx - W]
                acts = torch.stack([Qt[idx - W + k] for k in range(W + int(a.horizon))], 1)
                pred = _roll(bundle, S0, acts, W + int(a.horizon))
                p_at = lambda k: pred[W + k].float()      # noqa: E731
            for k in keys:
                p = p_at(k)
                t = S[idx + k].float()
                qe = (p[:, 7:36] - t[:, 7:36]).pow(2).mean(-1).sqrt()
                per_k[k]["joint"].append(qe.cpu())
                per_k[k]["root"].append((p[:, :3] - t[:, :3]).norm(dim=-1).cpu())
                Rp = _quat_to_mat(p[:, 3:7])
                Rt = _quat_to_mat(t[:, 3:7])
                # ★ 2026-10-31 修（A10，硬 bug）：`acos(tr)` 是错的。
                #   旋转角 θ 满足 tr(R_pᵀ R_t) = 1 + 2cosθ ⇒ cosθ = (tr−1)/2。
                #   对 tr≈3 的小角，`acos(clamp(tr,−1,1))` 会 clamp 成 1、acos(1)=0
                #   ⇒ **rot_err 恒为 0.0°**（三个模型、k=1..20 全是 0 就是这个原因；
                #   实测两个纯 yaw 差 10° 也被算成 0.0000°）。
                #   修正后：k=20 的真实姿态误差是 **2~10°**（手工 roll 已核对）。
                tr = (Rp.transpose(-1, -2) @ Rt).diagonal(dim1=-2, dim2=-1).sum(-1)
                cos_t = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
                per_k[k]["rot"].append(torch.rad2deg(torch.acos(cos_t)).cpu())
                vp = (p[:, 42:71] - p[:, 42:71].mean(0))
                vt = (t[:, 42:71] - t[:, 42:71].mean(0))
                corr = (vp * vt).mean() / (vp.std() * vt.std() + 1e-9)
                per_k[k]["qd_corr"].append(torch.as_tensor(float(corr)))
                # ★ 2026-10-31：**in-distribution 的 root 角速度质量**。
                #   奖励里的 yaw_rate 项吃的是这个量，而它本来只在 OOD(录制语料) 上量过——
                #   那是错分布。这里补上，让 yaw_rate 的准入判据和奖励用同一个分布。
                wp, wt = p[:, 39:42], t[:, 39:42]
                _wp = wp - wp.mean(0)
                _wt = wt - wt.mean(0)
                per_k[k]["root_w_corr"].append(torch.as_tensor(float(
                    (_wp * _wt).mean() / (_wp.std() * _wt.std() + 1e-9))))
                per_k[k]["root_w_ratio"].append(torch.as_tensor(float(
                    wp.abs().mean() / (wt.abs().mean() + 1e-9))))
        row = {}
        for k in keys:
            row[k] = ({m: float(torch.cat([x.reshape(-1) for x in v]).mean())
                       for m, v in per_k[k].items()} if per_k[k]["joint"] else {})
        results[ck] = row
        # A12：标明这是 **in-distribution**（本轮 B 线留出）的多步仪器
        print(f"[multistep] {Path(ck).name}  [data_source=B-line holdout (in-distribution), "
              f"instrument=multi-step k={a.horizon} warmup={a.warmup}]", flush=True)
        for k in keys:
            r = row[k]
            if r:
                print(f"   k={k:>2}  joint_rmse={r['joint']:.3f} rad  root_err={r['root'] * 100:.1f} cm  "
                      f"rot_err={r['rot']:.1f}°  qd_corr={r['qd_corr']:.3f}  "
                      f"root_w_corr={r.get('root_w_corr', float('nan')):.3f}  "
                      f"root_w_ratio={r.get('root_w_ratio', float('nan')):.3f}", flush=True)

    if len(a.ckpt) > 1:
        base = results[a.ckpt[0]]
        print("[multistep] 相对第一个 ckpt：", flush=True)
        for ck in a.ckpt[1:]:
            for k in keys:
                if base[k] and results[ck][k]:
                    dj = (results[ck][k]["joint"] / max(base[k]["joint"], 1e-9) - 1) * 100
                    dr = (results[ck][k]["root"] / max(base[k]["root"], 1e-9) - 1) * 100
                    print(f"   {Path(ck).name} k={k}: joint {dj:+.1f}%  root {dr:+.1f}%", flush=True)
        gate = (results[a.ckpt[-1]][a.horizon]["joint"]
                <= 1.05 * results[a.ckpt[0]][a.horizon]["joint"])
        print(f"[multistep] G0-b(多步) 20 步关节误差 ≤ 1.05× 上一版：{'PASS' if gate else 'FAIL'}",
              flush=True)

    if a.out:
        import json  # noqa: PLC0415
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"ckpts": a.ckpt, "horizon": a.horizon,
                                     "warmup": a.warmup,
                                     "keys": list(keys), "results": results}, indent=1, default=float))
        print(f"[multistep] saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
