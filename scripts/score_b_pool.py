#!/usr/bin/env python3
"""给整池 B 线 dump **一次性**补算奖励（bundle 只加载一次，按集分别落盘）。

为什么不用 `compute_collect_reward.py` 的批量模式
------------------------------------------------
它的多输入模式会把输出命名为 ``<stem>_r.pt`` 放在同一个 ``--out`` 目录里，
而我们的 dump 跨 B_r2/B_r3/B_r4/B_r5 **文件名相同**（同一任务同一行号、不同轮次采的）
⇒ 会互相覆盖。而这个脚本给每个 dump 写在它自己目录下的 ``scored.pt``，不会撞。

产物：``<episode_dir>/scored.pt``（= 原 dump + 标准序 states + next_states + rewards + reward_terms），
正是 ``build_real_transitions_v2.py --extra-dumps`` 需要的格式。

用法：
    PYTHONPATH=. SUPERMIMIC_HA_ALIGN=1 python3 scripts/score_b_pool.py \
        --dump 'runs/B_*/vla_*_f*_r*/dump/*.pt' --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="glob：B 线 dump")
    ap.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A24000_wmvel_v6.pt")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--reward-mode", default="saturating", choices=("saturating", "linear"))
    ap.add_argument("--reward-vel-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.dump))
    if not files:
        raise SystemExit(f"[score] 没匹配到: {a.dump}")

    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415
    from compute_collect_reward import _one  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz, packed_motion=a.packed,
                               obs_mode=None, hist=None, token_dim=64)
    bundle.reward_mode = str(a.reward_mode)
    bundle.reward_vel_scale = float(a.reward_vel_scale)

    print(f"[score] {len(files)} 个 dump，bundle 只加载这一次", flush=True)
    done = skipped = failed = 0
    for f in files:
        p = Path(f)
        if p.name.endswith("_r.pt"):
            continue
        out_path = p.parent.parent / "scored.pt"     # <episode>/dump/x.pt -> <episode>/scored.pt
        if a.skip_existing and out_path.exists():
            skipped += 1
            continue
        try:
            _one(p, out_path, a, bundle, dev)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[score][FAIL] {p}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[score] 完成：新算 {done}，已有 {skipped}，失败 {failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
