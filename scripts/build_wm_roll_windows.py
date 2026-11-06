#!/usr/bin/env python3
"""从 A' 分片抽 **k=20 连续窗口**，喂 WM 的阶段 2b（多步自回归）。

为什么需要（2026-11-04 实测）：
    只训单步的 WM，20 步想象的漂移是 **28.6 cm**，而真实只有 **6.8 cm**（4.2 倍），
    与真实漂移的相关只有 **0.205** ⇒ 拿它当可微 plant 去微调，优化的是一套夸张的动力学。
    单步损失里 state 永远是真实值，模型从没被约束过"用自己输出继续滚"。阶段 2b 就是治这个的。

两个必须避开的坑（都有实测依据）：
    ① **不能取相邻行**：分片里的行是 `row i = env(i % num_envs)` 交错存的，
        `st[t:t+21]` 是 21 个不同 env 的状态 ⇒ 假窗口。
    ② **不能跨段**：段边界是一次瞬移（物理上不连续）⇒ 假转移。
       本脚本用"帧号连续 + 同 clip + 段边界检测"来保证。
       段边界检测：重置后 `|q − q_ref|` 会突然掉到 ~0.02 rad，据此切段。

输出：``{states:(N,k+1,71), actions:(N,k,29), clips:(N,), frames:(N,), meta}``
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1"))

import torch  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, nargs="+",
                    default=[ROOT / "runs/a_prime_sonic_leash", ROOT / "runs/a_prime_sonic"])
    ap.add_argument("--pattern", default="*_off*.pt")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--stride", type=int, default=4, help="段内滑窗步长（1=最密）")
    ap.add_argument("--max-windows", type=int, default=120000)
    ap.add_argument("--reset-eps", type=float, default=0.05,
                    help="段边界检测：|q−q_ref| 低于该值视为段首（rad）")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/a_prime_sonic_leash/roll_windows.pt")
    a = ap.parse_args()

    from action_provider.action_provider_sonic import SONIC_ISAACLAB_JOINT_ORDER as SO  # noqa: PLC0415

    ck = torch.load(ROOT / "runs/S1_handover/exp_WBC_exec_haonly_best.pt",
                    map_location="cpu", weights_only=False)
    jn = list(ck["robot_spec"]["joint_names"])
    B = torch.tensor([SO.index(nm) for nm in jn])          # 用 ref[B] 把 SONIC 序转 standard 序
    pk = torch.load(ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                    map_location="cpu", weights_only=False)
    ref = torch.as_tensor(pk["ref"]).float()
    starts = torch.tensor(pk["starts"]).long()
    W = 780                                                # 采集的枚举跨度（只用于算 segpos 日志）

    files: list[Path] = []
    for d in a.dir:
        files += [Path(p) for p in sorted(glob.glob(str(Path(d) / a.pattern)))]
    if not files:
        raise SystemExit(f"[roll-win] 没有分片：{a.dir}")

    K = int(a.k)
    S_all, A_all, C_all, F_all = [], [], [], []
    n_seg = 0
    n_win = 0
    for fp in files:
        if n_win >= a.max_windows:
            break
        d = torch.load(fp, map_location="cpu", weights_only=False)
        S = d["states"].float(); A = d["actions"].float()
        F = d["frames"].long(); C = d["clip_ids"].long()
        # 按 clip 分组，组内按帧号排序
        for ci in torch.unique(C).tolist():
            m = C == ci
            f = F[m]; s = S[m]; act = A[m]
            o = torch.argsort(f)
            f, s, act = f[o], s[o], act[o]
            if f.numel() <= K:
                continue
            # 帧号连续（差 1）的片段切出来；不连续的断开
            brk = (f[1:] - f[:-1]) != 1
            seg_id = torch.cat([torch.zeros(1, dtype=torch.long), brk.long().cumsum(0)])
            for si in torch.unique(seg_id).tolist():
                mm = seg_id == si
                fs, ss, aa = f[mm], s[mm], act[mm]
                if fs.numel() <= K:
                    continue
                n_seg += 1
                # 段边界检测：|q − q_ref| 突然变小 ⇒ 该处是（重）置后的段首
                qs = ss[:, 7:36]
                qr = ref[fs][:, 13:42][:, B]
                err = (qs - qr).abs().mean(dim=-1)
                is_reset = err < float(a.reset_eps)
                # 只保留"不跨越重置点"的窗口
                cum = is_reset.long().cumsum(0)
                for t0 in range(0, fs.numel() - K, max(1, int(a.stride))):
                    if cum[t0 + K] != cum[t0]:      # 窗口内出现了新的重置点 ⇒ 跨段
                        continue
                    S_all.append(ss[t0:t0 + K + 1])
                    A_all.append(aa[t0:t0 + K])
                    C_all.append(ci)
                    F_all.append(int(fs[t0]))
                    n_win += 1
                    if n_win >= a.max_windows:
                        break
                if n_win >= a.max_windows:
                    break
            if n_win >= a.max_windows:
                break
        print(f"[roll-win] {fp.name:28s} 累计窗口 {n_win}", flush=True)

    if not S_all:
        raise SystemExit("[roll-win] 一个窗口都没抽到")
    S = torch.stack(S_all); A = torch.stack(A_all)
    out = {"states": S, "actions": A,
           "clips": torch.tensor(C_all), "frames": torch.tensor(F_all),
           "meta": {"k": K, "stride": int(a.stride), "n_files": len(files),
                    "n_segments": n_seg, "n_windows": int(S.shape[0]),
                    "dirs": [str(p) for p in a.dir], "enum_span": W}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"[roll-win] {int(S.shape[0])} 个窗口（k={K}）→ {a.out}  "
          f"states{tuple(S.shape)} actions{tuple(A.shape)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
