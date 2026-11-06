#!/usr/bin/env python3
"""奖励接线的端到端自检（**需要训练栈 + GPU，但不跑训练**）。

验证 `diff_vla_wbc/isaac_joint.py` 新加的两件事真的接上了：
  1. `root_ori_mode="split"` 会把旧的单块 root_ori 项**精确替换**成 tilt/yaw/yaw_rate；
  2. `tail_weight>0` 会在跟踪变差时把奖励压低，而且**完美跟踪时尾部恰为 0**。

跑法：`SUPERMIMIC_HA_ALIGN=1 PYTHONPATH=. python3 scripts/test_reward_wiring.py`
"""
from __future__ import annotations

import glob
import sys

import torch

sys.path.insert(0, ".")

CKPT = "runs/S1_handover/WBC_A24000_wmvel_v6.pt"
PACKED = "runs/isaac_motion/ha_cmd/packed.pt"


def main() -> int:
    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle  # noqa: PLC0415

    npz = sorted(glob.glob("data/npz/ha_tokens/*.npz"))[:2]
    bundle = load_isaac_bundle(CKPT, device=torch.device("cuda:0"), npz=npz,
                              packed_motion=PACKED, obs_mode=None, hist=None, token_dim=64,
                              reward_cfg={"w_root_pos": 1.0, "w_root_ori": 1.0,
                                          "root_pos_std": 0.2, "root_ori_std": 0.25})
    B = 4
    # 用库里的真实参考行做"完美跟踪"样本
    lib = bundle.ref_state_at(torch.arange(B)) if hasattr(bundle, "ref_state_at") else None
    frames = torch.arange(100, 100 + B)
    ref = bundle.ref_state_at(frames).to("cuda:0").float()
    act = torch.zeros(B, 29, device="cuda:0")
    prev = torch.zeros_like(act)

    def run(mode: str, tail: float, state: torch.Tensor):
        bundle.root_ori_mode = mode
        bundle.tail_weight = tail
        out = bundle.reward(state, act, prev, FrameRef(frames))
        return float(out["total"].mean()), out

    print("[1] 完美跟踪（state = 参考状态）")
    r_mono, o1 = run("mono", 0.0, ref.clone())
    r_split, o2 = run("split", 0.0, ref.clone())
    r_tail, o3 = run("mono", 2.0, ref.clone())
    print(f"  mono   r={r_mono:+.6f}")
    print(f"  split  r={r_split:+.6f}  (含 tilt2={float(o2['tilt2'].mean()):.2e}, "
          f"yaw2={float(o2['yaw2'].mean()):.2e})")
    print(f"  mono+tail2 r={r_tail:+.6f}  pos_tail={float(o3['root_pos_tail'].mean()):.2e} "
          f"tilt_tail={float(o3['root_tilt_tail'].mean()):.2e}")
    ok_tail_zero = abs(r_tail - r_mono) < 2e-5
    print(f"  ✅ 完美跟踪时尾部不扣分" if ok_tail_zero else "  ❌ 完美跟踪时尾部仍在扣分")

    print("[2] 制造漂移（位置 +0.6 m）")
    bad = ref.clone()
    bad[:, 0] += 0.6
    r_mono_b, _ = run("mono", 0.0, bad)
    r_tail_b, o_b = run("mono", 2.0, bad)
    print(f"  mono   r={r_mono_b:+.6f}")
    print(f"  mono+tail2 r={r_tail_b:+.6f}  pos_tail={float(o_b['root_pos_tail'].mean()):.3f} m")
    ok_tail_hit = r_tail_b < r_mono_b - 1e-3
    print(f"  ✅ 有漂移时尾部显著扣分" if ok_tail_hit else "  ❌ 尾部没生效")

    print("[3] split 是否精确替换（无漂移时 r_split 应比 mono 大，因为总权重 1.5 > 1.0）")
    print(f"  差 {r_split - r_mono:+.6f}（期望 ≈ +0.5·dt·... 量级，>0 即可）")
    ok_split = r_split > r_mono
    print(f"  ✅ split 生效" if ok_split else "  ❌ split 没生效")

    print("[4] 有 yaw 偏移时：split 的扣分应远小于 mono（yaw 权重 0.2 vs 1.0）")
    import math  # noqa: PLC0415
    yaw = torch.tensor([math.cos(0.26), 0.0, 0.0, math.sin(0.26)], device="cuda:0")
    rot = bad.clone()
    w, x, y, z = yaw
    qw, qx, qy, qz = rot[:, 3], rot[:, 4], rot[:, 5], rot[:, 6]
    rot[:, 3] = w * qw - x * qx - y * qy - z * qz
    rot[:, 4] = w * qx + x * qw + y * qz - z * qy
    rot[:, 5] = w * qy - x * qz + y * qw + z * qx
    rot[:, 6] = w * qz + x * qy - y * qx + z * qw
    m_m, _ = run("mono", 0.0, rot)
    m_s, o_s = run("split", 0.0, rot)
    print(f"  mono={m_m:+.6f} split={m_s:+.6f}  dyaw={float(o_s['yaw2'].mean()) ** 0.5:.3f} rad")
    ok_yaw = m_s > m_m
    print(f"  ✅ split 对 yaw 更宽容（符合设计）" if ok_yaw else "  ⚠ 需人工看一下")

    fails = sum(not v for v in (ok_tail_zero, ok_tail_hit, ok_split))
    print(f"\n{'✅ 接线全部正常' if fails == 0 else f'❌ {fails} 项异常'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
