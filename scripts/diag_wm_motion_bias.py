#!/usr/bin/env python3
"""为什么「把每步漂移压下去」反而让整体漂移变大？——量 WM 的**运动偏置**。

假设（2026-11-04）
------------------
联合微调的奖励是"WM 想象出来的 20 步跟踪误差"。如果这个 surrogate 的误差**随运动幅度增长**
（走得越快/关节动得越大，WM 外推越不准），那么优化器看到的最优解就是**少动**：
少动 → 想象误差小 → 奖励高。而在真实系统里少动 = 追不上参考 = 整体漂移变大（甚至站桩）。

于是出现用户看到的现象：
    WM 里的 reward_norm 降 66%  ↔  真机净位移从 1.006 m 掉到 0.111 m

本脚本用 `roll_windows.pt`（真实连续窗口，k=20）做三件事：
 1. 同一批窗口，喂**真值动作**跑 20 步，把 WM 的误差按"窗口运动幅度"分箱 —— 看误差是否随运动增长；
 2. 同一批窗口，对比三种动作序列在 **WM 里**的预测误差：真值 / 缩半 / 冻结（重复首帧动作）
    —— 看 surrogate 是不是把"少动"排在"照着做"前面；
 3. 同时给出**真实**的 20 步漂移（用窗口真值算），对照 surrogate 的名次。

用法：
    python3 scripts/diag_wm_motion_bias.py --ckpt runs/a_prime_sonic_leash/wm_ssm_s0_more.pt
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
                    default=ROOT / "runs/a_prime_sonic_leash/wm_ssm_s0_more.pt")
    ap.add_argument("--windows", type=Path,
                    default=ROOT / "runs/a_prime_sonic_leash/roll_windows.pt")
    ap.add_argument("--n", type=int, default=768)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz,
                               packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                               obs_mode=None, hist=None, token_dim=64)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    bundle.wm.load_state_dict(ck.get("wm", ck), strict=True)
    wm = bundle.wm
    wm.eval()
    print(f"[bias] WM = {a.ckpt}", flush=True)

    w = torch.load(a.windows, map_location="cpu", weights_only=False)
    S, A = w["states"].float(), w["actions"].float()          # (N,k+1,71) / (N,k,29)
    n = min(a.n, int(S.shape[0]))
    g = torch.Generator().manual_seed(int(a.seed))
    idx = torch.randperm(int(S.shape[0]), generator=g)[:n]
    K = int(a.k)
    S, A = S[idx], A[idx]
    s0 = S[:, 0].to(dev)
    acts = A[:, :K].to(dev)
    ref = S[:, : K + 1].to(dev)                                # 真值轨迹（当作"参考"）
    xi = bundle.default_xi(s0.shape[0], dev)

    def rollout(act_seq: torch.Tensor) -> torch.Tensor:
        """返回 (n, k+1, 71) 的想象轨迹（第 0 帧 = 真实起点）。"""
        traj = [s0]
        s = s0.clone()
        for t in range(K):
            out = bundle.step(s, act_seq[:, t], xi)
            s = out[0] if isinstance(out, (tuple, list)) else out
            traj.append(s)
        return torch.stack(traj, 1)

    def pose_err(traj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """和真的沿用同一口径：每步根位置误差 / 关节平均绝对误差，取 20 步均值。"""
        e_root = (traj[:, 1:, 0:3] - ref[:, 1:, 0:3]).norm(dim=-1).mean(1)     # (n,)
        e_q = (traj[:, 1:, 7:36] - ref[:, 1:, 7:36]).abs().mean(dim=(1, 2))    # (n,)
        return e_root, e_q

    with torch.no_grad():
        traj_gt = rollout(acts)
        traj_half = rollout(acts * 0.5)
        traj_frz = rollout(acts[:, :1].expand(-1, K, -1).contiguous())
        e_gt = pose_err(traj_gt)
        e_half = pose_err(traj_half)
        e_frz = pose_err(traj_frz)

        # 窗口的"真实运动幅度"：真值 q 走 20 步的平均每步变化 + 根位移
        mq = (ref[:, 1:, 7:36] - ref[:, :-1, 7:36]).abs().mean(dim=(1, 2))
        mv = (ref[:, 1:, 0:3] - ref[:, :-1, 0:3]).norm(dim=-1).mean(1)

    # ---- 1) WM 误差 vs 运动幅度（分箱）---------------------------------------
    order = torch.argsort(mq)
    nb = 5
    print("\n[1] 同样喂**真值动作**，WM 的 20 步平均误差按窗口运动幅度分箱：")
    print(f"    {'运动档':<8}{'|dq|/步':>10}{'|dp|/步 m':>11}{'WM e_q':>10}{'WM e_root cm':>14}")
    for b in range(nb):
        sl = order[b * n // nb: (b + 1) * n // nb]
        print(f"    {b + 1}/{nb:<6}{float(mq[sl].mean()):>10.4f}{float(mv[sl].mean()):>11.4f}"
              f"{float(e_gt[1][sl].mean()):>10.4f}{float(e_gt[0][sl].mean()) * 100:>14.2f}")
    cq = float(torch.corrcoef(torch.stack([mq, e_gt[1]]))[0, 1])
    cr = float(torch.corrcoef(torch.stack([mv, e_gt[0]]))[0, 1])
    print(f"    corr(运动幅度, WM 关节误差) = {cq:+.3f}   corr(根位移, WM 根误差) = {cr:+.3f}")

    # ---- 2) 三种动作在 surrogate 里的名次 ------------------------------------
    print("\n[2] 同一批窗口，三种动作序列在 WM 里的 20 步误差（越小=surrogate 认为越好）：")
    print(f"    {'动作':<14}{'e_q':>10}{'e_root cm':>12}{'比真值':>10}")
    base_q = float(e_gt[1].mean())
    base_r = float(e_gt[0].mean())
    for name, (er, eq) in (("真值动作", e_gt), ("动作×0.5", e_half), ("冻结(重复首帧)", e_frz)):
        print(f"    {name:<14}{float(eq.mean()):>10.4f}{float(er.mean()) * 100:>12.2f}"
              f"{float(eq.mean()) / max(base_q, 1e-9):>10.2f}x")
    win_q = int((e_frz[1] < e_gt[1]).sum())
    win_r = int((e_frz[0] < e_gt[0]).sum())
    print(f"    冻结比真值更优的窗口：关节 {win_q}/{n} ({100.0 * win_q / n:.1f}%)，"
          f"根位置 {win_r}/{n} ({100.0 * win_r / n:.1f}%)")

    # ---- 3) 真实世界里"少动"到底更差多少 ------------------------------------
    # 冻结/半速时，真值那条轨迹本身就是要追的参考 ⇒ 真实漂移就是"参考走了多远"。
    print("\n[3] 真实世界（参考轨迹自己走了多远 = 站桩会有的漂移）：")
    real_root = (ref[:, -1, 0:3] - ref[:, 0, 0:3]).norm(dim=-1)
    real_q = (ref[:, -1, 7:36] - ref[:, 0, 7:36]).abs().mean(-1)
    print(f"    20 步参考根位移 p50 {float(real_root.median()) * 100:.2f} cm，"
          f"关节位移 p50 {float(real_q.median()):.4f} rad")
    print(f"    ⇒ surrogate 认为冻结更好（{100.0 * win_q / n:.0f}% 窗口），"
          f"但真机上冻结必然产生 {float(real_root.median()) * 100:.1f} cm / "
          f"{float(real_q.median()):.3f} rad 的漂移")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
