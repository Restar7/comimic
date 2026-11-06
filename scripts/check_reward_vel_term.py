#!/usr/bin/env python3
"""契约检查：`REWARD_VEL_SCALE` 在 `huber_err` 模式下**真的**把速度项接进奖励了吗？

背景（2026-11-05）：`huber_err` 分支原来只算 pos/ori，**完全没有速度项**
（旧注释写着 "速度项不在此分支"）。而 base 的症结是平均速度只有录制的 1/2，
这个亏欠在 0.4 s 里只积累 2 cm，**速度却是即时可见的**。
补上速度项后必须有一条自动检查，防止"改了但没接上/被别的分支吃掉"。

做法：同一批窗口、同样 5 步 rollout，只改 `reward_vel_scale`（0 vs 1），
奖励必须有可测差异且全部有限。

用法::

    python3 scripts/check_reward_vel_term.py            # 期望 exit 0
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def main() -> int:
    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415
    from diff_vla_wbc.rollout import imagine_with_tokens  # noqa: PLC0415

    dev = torch.device("cuda:0")
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    b = load_isaac_bundle(ROOT / "runs/S1_handover/WBC_A24000_wmvel_cmd.pt", device=dev, npz=npz,
                          packed_motion=ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                          obs_mode=None, hist=None, token_dim=64)
    b.wm.requires_grad_(False)
    b.wm.eval()

    w = torch.load(ROOT / "runs/a_prime_sonic_leash/roll_windows.pt", map_location="cpu",
                   weights_only=False)
    S, A = w["states"].float(), w["actions"].float()
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(int(S.shape[0]), generator=g)[:4]
    # rollout 只要 5 步；token 用零（只测奖励分支，不关心 token 好坏）
    toks = torch.zeros(4, 5, 64, device=dev)
    s0 = S[sel, 0].to(dev)
    hist = torch.zeros(4, 1, 1, device=dev)  # 占位；下面用 bundle 自己的 history 构造
    xi = b.default_xi(4, dev)

    # references 走 bundle 的 teacher_references（与训练完全同一路）
    fr = torch.tensor([int(w["frames"][int(i)]) if "frames" in w else 0 for i in sel])
    refs = b.teacher_references(b.library, fr.to(dev), 5)

    def run(vs: float):
        setattr(b, "reward_vel_scale", float(vs))
        setattr(b, "reward_mode", "huber_err")
        setattr(b, "root_ori_mode", "split_huber")
        setattr(b, "huber_err_delta_pos", 0.05)
        setattr(b, "huber_err_delta_ori", 0.05)
        setattr(b, "huber_err_d_max", 0.30)
        with torch.no_grad():
            r = imagine_with_tokens(b, tokens=toks, state=s0, obs_hist=None, history=None,
                                    xi=xi, references=refs, horizon=5)
        return r

    r0 = run(0.0)
    r1 = run(1.0)
    rew0 = float(r0.rewards.sum())
    rew1 = float(r1.rewards.sum())
    finite = bool(torch.isfinite(r1.rewards).all())
    print(f"[vel] reward_vel_scale=0 → 累计奖励 {rew0:.6f}")
    print(f"[vel] reward_vel_scale=1 → 累计奖励 {rew1:.6f}")
    print(f"[vel] 差值 {rew1 - rew0:+.6f}   finite={finite}")
    if not finite:
        print("[vel][FAIL] 打开速度项后出现非有限值")
        return 1
    if abs(rew1 - rew0) < 1e-9:
        print("[vel][FAIL] 打开速度项后奖励**完全没变** ⇒ 速度项没接进 huber_err（改动无效）")
        return 1
    print("[vel][PASS] 速度项已接进 huber_err，且数值有限")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
