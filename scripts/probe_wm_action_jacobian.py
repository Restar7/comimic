#!/usr/bin/env python3
"""**想象梯度对不对**：WM 的 ∂q/∂a 雅可比探针（29×29）。

为什么需要它：
    k 步状态误差只回答"预测得准不准"，不回答"**动作的因果方向对不对**"。
    而这条路线要的是 `tokens → WBC → WM → 奖励 → 反传`，梯度里唯一带"因果"的就是
    WM 的 ∂state/∂action。如果 WM 把"动作 j"的效应算到了别的关节上（错位/换序/串味），
    k=1 误差可以很好看，而**想象梯度会指向错误的关节** —— 这正是 I5（想象回报不能当裁决器）
    最可能的机制。

探针做法（不需要反事实数据）：
    q̂_{t+k} = WM^k(s_t, a_t + ε·e_j)；J_k[i, j] = ∂q_i(t+k)/∂a_j(t)
    动作语义是 `a = q_target − default_q`（绝对目标偏移），所以物理上 J 应该**对角占优**：
    抬动作 j ⇒ 关节 j 的位置目标抬高 ⇒ q_i 只在 i=j 上明显响应。
    报告：
      * diag / off-diag 均值、对角比
      * **对角命中率**（每行 |J| 最大的列是不是自己）—— 1.0 = 完全对齐
      * 若命中率低，打印"每行最强列"的映射，便于看出是哪一种换序

用法：
    PYTHONPATH=. python3 scripts/probe_wm_action_jacobian.py \
        --ckpt runs/S1_handover/wm_d3.pt runs/S1_handover/wm_d5.pt \
        --dump "runs/wm_real/traj_wmuse2_val/v_11*.pt" --horizon 1 --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.05, help="动作扰动幅度（rad）")
    ap.add_argument("--starts", type=int, default=24, help="抽多少个起点")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import joint_contract as JC  # noqa: PLC0415
    from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415
    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)
    d = torch.load(sorted(glob.glob(a.dump))[0], map_location="cpu", weights_only=False)
    S = state_to_standard(d["states"].float(), perm)
    A = (d["actions"] if "actions" in d else d["q_target_applied"]).float().reshape(-1, 29)
    starts = list(range(10, int(S.shape[0]) - a.horizon - 2, max(1, int(a.stride))))[: int(a.starts)]
    idx = torch.tensor(starts, dtype=torch.long)
    S0, A0 = S[idx].to(dev), A[idx].to(dev)

    def q_at(bundle, s0, acts, h):
        xi = bundle.default_xi(int(s0.shape[0]), s0.device)
        s = s0
        for k in range(h):
            ak = acts[:, k] if acts.ndim == 3 else acts
            s = bundle.step(s, ak, xi, step_idx=k + 1)
        return s[:, 7:36]

    eye = torch.eye(29, device=dev)
    print(f"[jac] dump={Path(sorted(glob.glob(a.dump))[0]).name} starts={len(starts)} "
          f"eps={a.eps} horizon={a.horizon}", flush=True)
    for ck in a.ckpt:
        bundle = load_isaac_bundle(ck, device=dev,
                                   npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
                                   packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                                   obs_mode=None, hist=None, token_dim=64)
        arch = getattr(bundle, "wm_arch", "mlp")
        with torch.no_grad():
            base = q_at(bundle, S0, A0, int(a.horizon))
            cols = []
            for j in range(29):
                A1 = A0 + float(a.eps) * eye[j][None, :]
                cols.append((q_at(bundle, S0, A1, int(a.horizon)) - base) / float(a.eps))
            J = torch.stack(cols, -1).mean(0)              # (29,29) [i,j] = ∂q_i/∂a_j
        diag = J.diagonal()
        mask = ~torch.eye(29, dtype=torch.bool, device=dev)
        off_m = J[mask]
        dom = J.abs().argmax(-1)                            # 每行最强的动作列
        hit = float((dom == torch.arange(29, device=dev)).float().mean())
        print(f"\n[{Path(ck).name}]  arch={arch}")
        print(f"  diag  mean={float(diag.mean()):+.4f}  (|.| {float(diag.abs().mean()):.4f})")
        print(f"  off   mean={float(off_m.mean()):+.4f}  (|.| {float(off_m.abs().mean()):.4f})")
        print(f"  对角比 |diag| / |off| = {float(diag.abs().mean() / (off_m.abs().mean() + 1e-12)):.2f}")
        print(f"  **对角命中率** = {hit:.3f}（1.0 = 动作 j 只驱动关节 j）")
        if hit < 0.95:
            m = [int(x) for x in dom.tolist()]
            print(f"  每行最强动作列 = {m}")
            print(f"  是恒等映射的置换吗 = {sorted(m) == list(range(29))}（是 ⇒ 整体换序；否 ⇒ 混乱）")
        # 参考：数据里"动作 vs 关节偏移"的真实对齐（同一批样本的自相关）
        print(f"  [ref] 训练目标口径：J 的对角应显著 > 0（抬动作 j ⇒ 关节 j 上移）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
