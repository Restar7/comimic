#!/usr/bin/env python3
"""**对齐性实测**：联合微调让 WM 预测的状态，落在 WM 训练分布内吗？

问题（用户 2026-11-05 提出）：联合微调的 rollout 起点是 A'（录制、漂移 ≤10 cm），
WM 的训练窗口也来自 A'。但 rollout 一旦往前滚 20 步，状态就离开起点了 ——
**它在不在 WM 见过的分布里？** 这是"能不能用这个 WM 算奖励"的前提。

做法（不猜，量出来）：
 1. 从 A' 抽 N 个窗口起点 s0（与联合微调同一个采样器口径：连续帧窗口的首帧）；
 2. 用 WM **自回归滚 20 步**，记下每一步的预测状态（= 联合微调时 WM 必须预测的状态）；
 3. 把 A' 的全部窗口状态当作"训练分布"参考集；
 4. 在**标准化空间**（逐维除以训练集 std）里，算每个预测状态到参考集的最近邻距离；
    同时给出参考集自己内部的最近邻距离（= 分布内的"地板值"），用来做对照。

判据：
  * 预测状态 NN 距离 ≈ 地板值（≤2×）⇒ 在分布内，奖励可信；
  * 预测状态 NN 距离 ≫ 地板值（≥5×）⇒ 已经跑到分布外，那个奖励是外推。

用法::

    python3 scripts/probe_rollout_vs_train_dist.py \
        --ckpt runs/a_prime_sonic_exc/wm_scratch_ddp.pt \
        --windows runs/a_prime_sonic_exc/roll_windows_s1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wbc-ckpt", type=Path, default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "runs/a_prime_sonic_exc/wm_scratch_ddp.pt")
    ap.add_argument("--windows", type=Path,
                    default=ROOT / "runs/a_prime_sonic_exc/roll_windows_s1.pt")
    ap.add_argument("--n", type=int, default=128, help="rollout 条数")
    ap.add_argument("--ref", type=int, default=200_000, help="参考集最多抽多少个状态")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    b = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                          packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                          obs_mode=None, hist=None, token_dim=64)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    b.wm.load_state_dict(ck.get("wm", ck), strict=True)
    b.wm.requires_grad_(False)
    b.wm.eval()
    print(f"[align] WM = {a.ckpt.name}", flush=True)

    w = torch.load(a.windows, map_location="cpu", weights_only=False)
    S, A = w["states"].float(), w["actions"].float()          # (N,k+1,71) / (N,k,29)
    K = int(a.k)

    # ---- 训练分布参考集（A' 的所有窗口状态）----
    flat = S.reshape(-1, S.shape[-1])
    g = torch.Generator().manual_seed(0)
    if flat.shape[0] > a.ref:
        ref = flat[torch.randperm(flat.shape[0], generator=g)[: a.ref]]
    else:
        ref = flat
    mu = ref.mean(0)
    sd = ref.std(0).clamp_min(1e-6)

    def nn_dist(x: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """x: (M,71) → 到 ref 的最近邻距离（标准化空间）。"""
        xs = ((x - mu) / sd)
        rs = ((ref - mu) / sd).to(dev)
        out = []
        for i in range(0, xs.shape[0], chunk):
            d = torch.cdist(xs[i:i + chunk].to(dev), rs).min(dim=1).values
            out.append(d.cpu())
        return torch.cat(out)

    with torch.no_grad():
        # 地板值：参考集自己（抽 512 个）到参考集的 NN（排除自身用第 2 近）
        sub = ref[torch.randperm(ref.shape[0], generator=g)[:512]]
        rs = ((ref - mu) / sd).to(dev)
        q = ((sub - mu) / sd).to(dev)
        d_self = torch.cdist(q, rs).kthvalue(2, dim=1).values
        floor = float(d_self.median())
        print(f"[align] 分布内地板值（参考集自 NN, 中位）= {floor:.3f}（标准化单位）", flush=True)

        sel = torch.randperm(S.shape[0], generator=g)[: a.n]
        s = S[sel, 0].to(dev)
        acts = A[sel, :K].to(dev)
        xi = b.default_xi(s.shape[0], dev)
        traj = [s.cpu()]
        for t in range(K):
            out = b.step(s, acts[:, t], xi)
            s = out[0] if isinstance(out, (tuple, list)) else out
            traj.append(s.cpu())
        traj = torch.stack(traj, 1)          # (n, K+1, 71)

        print(f"[align] WM 自回归 {K} 步（喂**录制动作**，n={a.n}）后，"
              f"预测状态到训练分布的最近邻距离：", flush=True)
        print(f"    {'t':>4}{'NN p50':>10}{'NN p90':>10}{'÷地板':>9}", flush=True)
        rows = []
        for t in (1, 2, 5, 10, 15, 20):
            d = nn_dist(traj[:, t])
            p50, p90 = float(d.median()), float(d.quantile(0.9))
            rows.append((t, p50, p90))
            print(f"    {t:>4}{p50:>10.3f}{p90:>10.3f}{p50 / floor:>9.2f}", flush=True)

    print()
    print("[align] 判读：'÷地板' ≤2 ⇒ 该步的预测仍在训练分布内；≥5 ⇒ 已在分布外（奖励是外推）")
    bad = [t for t, p50, _ in rows if p50 / floor >= 5]
    if bad:
        print(f"[align] ⚠ 从第 {bad[0]} 步起已出分布")
    else:
        print("[align] ✅ 20 步内仍在训练分布内")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
