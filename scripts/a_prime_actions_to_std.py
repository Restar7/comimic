#!/usr/bin/env python3
"""把 A' 的**动作**从 SONIC/HA 序换成 STANDARD 序（2026-11-04）

为什么：SONIC 执行器（`diff_vla_wbc/sonic_isaac.SonicIsaacBundle`）返回的是
**STANDARD 序**动作（与 mjlab 侧口径一致），而 A' 采集时存的 `actions` 是
**decoder 原始输出 = SONIC/HA 序**。WM 训练数据里的动作序必须与 rollout 时一致，
否则 WM 学到的"动作通道"和微调用的不是一套（数值顺序不同 = 同一个网络输入被乱序）。

用法：
    python3 scripts/a_prime_actions_to_std.py --src runs/a_prime_sonic_leash/filtered.pt \
            --out runs/a_prime_sonic_leash/filtered_std.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1"))
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from diff_vla_wbc.sonic_mjlab import ha_to_std
    perm = torch.tensor(ha_to_std(), dtype=torch.long)
    d = torch.load(a.src, map_location="cpu", weights_only=False)
    act = d["actions"]
    if act.shape[-1] != perm.numel():
        raise SystemExit(f"[std-actions] 动作维度 {act.shape[-1]} != {perm.numel()}")
    d2 = dict(d)
    d2["actions_sonic_order"] = act.clone()          # 保留原始序，便于回溯
    d2["actions"] = act.index_select(-1, perm)       # HA(SONIC) -> STANDARD
    d2.setdefault("meta", {})
    d2["meta"] = dict(d2["meta"])
    d2["meta"]["action_order"] = "STANDARD_29 (permuted from SONIC/HA by scripts/a_prime_actions_to_std.py)"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d2, a.out)
    print(f"[std-actions] {a.src.name}: {tuple(act.shape)} -> {a.out.name}（动作已换到 STANDARD 序）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
