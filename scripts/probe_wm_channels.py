#!/usr/bin/env python3
"""WM 各通道（位置/朝向/线速度/角速度/q/qd）在 20 步上的**可分辨性**。

用途：判断"能不能直接用**速度项**当奖励"。
  * 位置误差要**积累**才看得见（base 慢 2 倍，20 步只差 2.1 cm，WM 自身误差 1.1 cm）；
  * 速度误差是**即时**的 ⇒ 不需要长视界。前提是 WM 的速度通道够准。

输出每个块的 p50 误差，并和"参考本身的波动(std)"对比：
    ratio = 误差 / 数据 std ⇒ ratio ≪ 1 才说明这个通道能用来做奖励。

用法::

    python3 scripts/probe_wm_channels.py --ckpt runs/a_prime_sonic_leash/wm_ssm_s0_gain.pt
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
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs/a_prime_sonic_leash/wm_ssm_s0_gain.pt")
    ap.add_argument("--windows", type=Path, default=ROOT / "runs/a_prime_sonic_leash/roll_windows.pt")
    ap.add_argument("--n", type=int, default=512)
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
    b.wm.eval()
    w = torch.load(a.windows, map_location="cpu", weights_only=False)
    S, A = w["states"].float(), w["actions"].float()
    K = int(a.k)
    g = torch.Generator().manual_seed(0)
    sel = torch.randperm(int(S.shape[0]), generator=g)[: a.n]
    s0, acts, real = S[sel, 0].to(dev), A[sel, :K].to(dev), S[sel, K].to(dev)
    xi = b.default_xi(s0.shape[0], dev)
    s = s0
    with torch.no_grad():
        for t in range(K):
            out = b.step(s, acts[:, t], xi)
            s = out[0] if isinstance(out, (tuple, list)) else out

    # 状态布局 [p(0:3) quat(3:7) q(7:36) v(36:39) w(39:42) qd(42:71)]
    blocks = {"pos(cm)": (0, 3, 100.0), "quat": (3, 7, 1.0), "q(rad)": (7, 36, 1.0),
              "linvel(m/s)": (36, 39, 1.0), "angvel(rad/s)": (39, 42, 1.0), "qd(rad/s)": (42, 71, 1.0)}
    print(f"[ch] WM={a.ckpt.name}  k={K}  n={int(s0.shape[0])}")
    print(f"    {'通道':<14}{'20步误差 p50':>14}{'参考自身幅度 p50':>18}{'误差/幅度':>12}")
    for name, (i, j, sc) in blocks.items():
        e = (s[:, i:j] - real[:, i:j]).abs().mean(-1) * sc
        amp = (real[:, i:j] - real[:, i:j].mean(0)).abs().mean(-1) * sc
        # 一帧内该通道的变化幅度更有意义：用相邻帧差
        chg = (S[sel, 1:K].to(dev)[:, :, i:j] - S[sel, :K - 1].to(dev)[:, :, i:j]).abs().mean(dim=(1, 2)) * sc
        r = float(e.median()) / max(float(chg.median()), 1e-9)
        print(f"    {name:<14}{float(e.median()):>14.4f}{float(chg.median()):>18.4f}{r:>12.2f}")
    print("\n[ch] 判读：'误差/单步变化' < 0.5 的通道才适合直接做奖励（信噪比 > 2）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
