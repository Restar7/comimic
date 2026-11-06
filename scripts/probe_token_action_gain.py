#!/usr/bin/env python3
"""**链路最弱的一环**：token → action 的灵敏度（策略到底有多大权限改变行为？）

链路是：`token(64) --VLA--? 不，是 WBC 解码--> action(29) --物理/WM--> state --> 奖励`
我们已验证 `∂J/∂a`（真物理 98–100% 可信），但**从没量过 `∂a/∂token`**。
如果 WBC 对 token 不敏感，那不管 WM 多准、梯度多对，**策略根本没有权限改变行为** ——
这正是"可得收益小"最可能的物理原因。

本探针直接算两个雅可比的范数比：
    g_a   = ∂J/∂a          （autograd，经过 WM + 奖励）
    g_tok = ∂J/∂token      （autograd，再穿过 WBC 解码）
    ⇒ **token 上要走多远，才等价于 action 上走 δ=0.03 rad？**
      把它和"数据里 token 的自然幅度"比：
        · 若需要的 token 步长 ≲ 数据幅度  ⇒ 权限足够
        · 若远大于数据幅度              ⇒ 权限不足（梯度再准也没用）

用法：
    PYTHONPATH=. python3 scripts/probe_token_action_gain.py \
        --ckpt runs/S1_handover/WBC_A24000_wmvel_v6.pt \
        --dump "runs/wm_real/traj_wmuse2_val/v_11*.pt" --windows 64 --horizon 1 --device cuda:1
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/S1_handover/WBC_A24000_wmvel_v6.pt",
                    help="WBC ckpt（策略解码器 + WM 都在里面）")
    ap.add_argument("--dump", default="runs/wm_real/traj_wmuse2_val/v_11*.pt")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()

    import joint_contract as JC  # noqa: PLC0415
    from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415
    from diff_vla_wbc.isaac_joint import FrameRef, default_xi, load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)
    bundle = load_isaac_bundle(a.ckpt, device=dev,
                               npz=sorted((ROOT / "data/npz/ha_tokens").glob("*.npz")),
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    hist_n = int(getattr(bundle, "hist", 10))
    H = int(a.horizon)

    # ---- 取窗口：状态/动作/token/参考帧 ----
    d = torch.load(sorted(glob.glob(a.dump))[0], map_location="cpu", weights_only=False)
    S = state_to_standard(d["states"].float(), perm)
    A = (d.get("actions") if d.get("actions") is not None else d["q_target_applied"]).float().reshape(-1, 29)
    T = d.get("tokens")
    if T is None:
        raise SystemExit("[gain] dump 里没有 tokens")
    T = T.float().reshape(-1, 64)
    FR = d.get("frames")
    FR = FR.long() if torch.is_tensor(FR) else None
    starts = list(range(hist_n + 1, int(S.shape[0]) - H - 1, max(1, int(a.stride))))[: int(a.windows)]
    if not starts:
        raise SystemExit("[gain] 没取到窗口")
    idx = torch.tensor(starts, dtype=torch.long)
    xb = int(idx.numel())
    xi = default_xi(xb, dev)

    # ---- 用**真实历史**搭 obs 历史（与部署/训练同口径）----
    obs_hist = []
    for t in range(hist_n):
        s_t = S[(idx - hist_n + t)].to(dev)
        a_t = A[(idx - hist_n + t)].to(dev)
        f_t = (FR[idx - hist_n + t] if FR is not None else torch.zeros_like(idx)).to(dev)
        obs_hist.append(bundle.observation(s_t, a_t, xi, FrameRef(f_t)))
    # ★ 扁平化必须用 bundle.flatten_history（term-major），不能简单 flatten：
    #   split_flat_obs 的布局是"每个 term 的 hist*dim 连续"，顺序错了 batch 维会错位。
    obs_flat = bundle.flatten_history(torch.stack(obs_hist, 0))     # (hist,B,W) -> (B, term-major)

    s0 = S[idx].to(dev)
    tok0 = T[idx].to(dev)
    fr0 = (FR[idx] if FR is not None else torch.zeros_like(idx)).to(dev)
    prev = A[(idx - 1)].to(dev)

    def j_of_action(acts):
        s = s0
        tot = 0.0
        a_prev = prev
        for t in range(H):
            act = acts[:, t] if acts.ndim == 3 else acts
            s = bundle.step(s, act, xi, step_idx=t + 1)
            tot = tot + bundle.reward(s, act, a_prev, FrameRef(fr0 + t))["total"]
            a_prev = act
        return tot

    # ---- 动作空间参考：∂J/∂a ----
    acts = A[idx].to(dev)[:, None, :].repeat(1, H, 1).requires_grad_(True)
    ja = j_of_action(acts).sum()
    ga, = torch.autograd.grad(ja, acts)
    ga_rms = float(ga.pow(2).mean().sqrt())

    # ---- token 空间：∂J/∂token（穿过 WBC 解码）----
    tok = tok0.clone().requires_grad_(True)
    with torch.enable_grad():
        a_tok = bundle.act(obs_flat, tok)                        # (B,29)
        acts2 = a_tok[:, None, :].repeat(1, H, 1).reshape(xb, H, -1)
        jt = j_of_action(acts2).sum()
        gt, = torch.autograd.grad(jt, tok)

    # ---- 基线 token 的自然幅度（作为"现实中能动的范围"参照）----
    tok_scale = float((T - T.mean(0)).pow(2).mean().sqrt())
    gt_rms = float(gt.pow(2).mean().sqrt())
    act_per_tok = gt_rms / (ga_rms + 1e-30)                      # 每单位 token 步 ≈ 多少 action 步

    print("\n" + "=" * 78)
    print(f"token→action 灵敏度（H={H}, N={xb}, token_scale(数据内 std)={tok_scale:.3f}）")
    print("=" * 78)
    print(f"  |∂J/∂a|   RMS = {ga_rms:.6e}")
    print(f"  |∂J/∂tok| RMS = {gt_rms:.6e}")
    print(f"  ⇒ 每单位 token 步 ≈ {act_per_tok:.4f} 个 action 步（RMS 比）")
    if act_per_tok > 1e-30:
        need = 0.03 / act_per_tok
        print(f"  ⇒ 要在 action 上达到 δ=0.03 rad，token 步长需 ≈ {need:.3f}")
        print(f"  ⇒ 相当于数据内 token 自然幅度（std={tok_scale:.3f}）的 {need / (tok_scale + 1e-30):.2f} 倍")
    print("\n判读：")
    print("  · '需要倍数' ≲1   ⇒ 策略权限足够（梯度能在数据分布内驱动行为）")
    print("  · '需要倍数' ≫1   ⇒ **策略权限不足**：token 这个小接口改变不了多少行为，")
    print("                       那 WM 再准、梯度再对，闭环也不会动。对策：解冻/适配 WBC 解码器。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
