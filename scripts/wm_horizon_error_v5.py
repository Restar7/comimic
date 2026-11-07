#!/usr/bin/env python3
"""§5.5.5：WM 开环 40 步误差测量（B 路线 `K=40` 的**唯一前置门槛**）。

做法（与规范 §5.5.5 的伪代码一致）：同一 clip 的相邻帧就是连续控制步。
对 N 个起点 ``f``：

    s = recorded_states[f]                          # 真实状态
    for k in 1..40:
        s = bundle.step(s, recorded_actions[f+k-1], xi, step_idx=k)
        err[k] = ‖s.pos − recorded_states[f+k].pos‖ …

**动作来源（诚实说明，与 §5.5.5 伪代码的偏差）**：`all_transitions.pt` 只有**稀疏采样**的
行（实测"同 clip 且帧号 +1"的相邻行只占 **0.05%**，最长连续段 85 行只有 2 段 ⇒ **无法**用它
回放 40 个连续控制步）。所以这里的动作改用**参考命令代理**：

    a_ha(k) = (q_ref[f+k] − bundle.default_joint_pos) / bundle.action_scale

（= burn-in 里 `a_hist` 的同一换算；`bundle.action_scale` 实测全 1.0）。这与历史
"喂录制动作"的 0.89 cm 不是同一口径，**不能直接对比**，只用于回答
"WM 滚 40 步还稳不稳"这个**相对**问题。

**动作顺序**：`bundle.step()` 吃 **STANDARD 序**、内部换回 SONIC/HA 给 WM（`sonic_isaac.py:168`）。
`q_ref` 是 HA_V31 序 ⇒ 先按 ``ha_v31 → standard`` 换序再进 `step()`。

**状态比较**：从**参考状态** `bundle.ref_state_at(f)` 起步，逐步与 `ref_state_at(f+k)` 比
（起点误差 0，量到的是**纯 WM 漂移**，不含 recorded-vs-reference 的 5 cm 级跟踪差）。

验收阈值（规范 §5.5.5）：

| 指标 | 上限 |
|---|---|
| `pos` p50 @ k=20 | ≤ 1.5 cm |
| `pos` p50 @ k=40 | ≤ 3 cm |
| `err[40]/err[20]` | ≤ 2.5× |

不过阈值 ⇒ **B 直接 NOT_RUN**，本版继续走 A（写进风险表）。

用法::

    source scripts/harness03_env.sh
    python3 scripts/wm_horizon_error_v5.py --out runs/temporal_ac_bptt_v5/main \
        --n 256 --horizon 40 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ.get("COMIMIC_MAIN", str(ROOT)))
for _p in (str(MAIN), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(MAIN / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"))

import diag_bptt_learnability as d03                              # noqa: E402
from diff_vla_wbc.compat import quat_error_magnitude              # noqa: E402
from diff_vla_wbc.delta_token_io import reorder_joints            # noqa: E402

THRESHOLDS = dict(pos_p50_mm_k20=15.0, pos_p50_mm_k40=30.0, growth_40_over_20=2.5)
CHECKPOINTS = (1, 5, 10, 20, 40)


def _quant(x: torch.Tensor) -> dict:
    if int(x.numel()) == 0:
        return dict(p50=float("nan"), p90=float("nan"), p99=float("nan"), mean=float("nan"))
    q = torch.quantile(x.float(), torch.tensor([0.5, 0.9, 0.99], device=x.device))
    return dict(p50=float(q[0]), p90=float(q[1]), p99=float(q[2]), mean=float(x.float().mean()))


def main() -> int:
    ap = argparse.ArgumentParser("wm_horizon_error_v5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bptt-args", type=Path, default=MAIN / "runs/bptt1/args.json")
    ap.add_argument("--transitions", type=Path, default=None)
    ap.add_argument("--n", type=int, default=256, help="起点数")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-order", default="ha_v31", choices=["ha_v31", "standard"])
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(int(a.seed))
    env = d03.load_env(a.device, a.bptt_args)
    lib = env.lib
    n_frames = int(lib.num_frames)
    Hm = int(a.horizon)
    cid = torch.as_tensor(lib.clip_id).detach().cpu().long()
    clip_end = torch.as_tensor(lib.clip_end).detach().cpu().long()[cid]
    ar = torch.arange(n_frames, dtype=torch.long)
    ok = (ar >= 10) & (ar + Hm <= clip_end)
    pool = ok.nonzero(as_tuple=False).flatten()
    if int(pool.numel()) == 0:
        raise SystemExit("[wm-h] 没有 f+horizon 仍在 clip 内的起点")
    pick = pool[torch.randperm(int(pool.numel()))[: int(a.n)]]
    d03.log(f"[wm-h] 起点 {int(pool.numel())} 可用，取 {int(pick.numel())}；"
            f"动作来源=参考命令代理（transitions 无连续帧，0.05% 相邻）")

    def _q_to_action(q_ha: torch.Tensor) -> torch.Tensor:
        """参考关节角（HA_V31）→ STANDARD 序的归一化动作（`step()` 的输入）。"""
        dq = env.bundle.default_joint_pos.to(q_ha.device).float()
        sc = env.bundle.action_scale.to(q_ha.device).float()
        a_ha = (q_ha.float() - dq) / sc.clamp_min(1e-6)
        if str(a.action_order) == "standard":
            return a_ha
        out = np.stack([reorder_joints(a_ha[i].cpu().numpy(), "ha_v31", "standard")
                        for i in range(a_ha.shape[0])], 0)
        return torch.from_numpy(out).to(q_ha.device)

    errs: dict[int, dict[str, list]] = {k: dict(pos=[], ori=[], q=[], qd=[]) for k in
                                        range(1, Hm + 1)}
    t0 = time.time()
    for s0 in range(0, int(pick.numel()), int(a.batch)):
        fr_b = pick[s0:s0 + int(a.batch)]
        B = int(fr_b.numel())
        with torch.no_grad():
            s = env.bundle.ref_state_at(fr_b.to(dev)).float()
            xi = env.bundle.default_xi(B, dev)
        batch_err = {k: dict(pos=[], ori=[], q=[], qd=[]) for k in range(1, Hm + 1)}
        for k in range(1, Hm + 1):
            q_cmd = env.bundle.ref_joint_pos((fr_b + k).to(dev))
            act = _q_to_action(q_cmd)
            with torch.no_grad():
                s = env.bundle.step(s, act, xi, step_idx=k)
                tgt = env.bundle.ref_state_at((fr_b + k).to(dev)).float()
            d = (s - tgt).float()
            batch_err[k]["pos"].append(d[:, 0:3].norm(dim=-1).detach().cpu())
            batch_err[k]["ori"].append(quat_error_magnitude(s[:, 3:7], tgt[:, 3:7]).detach().cpu())
            batch_err[k]["q"].append(d[:, 7:36].norm(dim=-1).detach().cpu())
            batch_err[k]["qd"].append(d[:, 42:71].norm(dim=-1).detach().cpu())
        for k in range(1, Hm + 1):
            for key in ("pos", "ori", "q", "qd"):
                errs[k][key].extend(batch_err[k][key])
        d03.log(f"[wm-h] {s0 + B}/{int(pick.numel())} ({time.time()-t0:.0f}s)")

    table = {}
    for k in range(1, Hm + 1):
        table[k] = {key: _quant(torch.cat(v)) for key, v in errs[k].items()}
    pos20 = table[min(20, Hm)]["pos"]["p50"]
    pos40 = table[Hm]["pos"]["p50"]
    ratio = (pos40 / pos20) if pos20 and pos20 > 0 else float("nan")
    verdict = dict(
        pos_p50_mm_k20=pos20 * 1e3, pos_p50_mm_k40=pos40 * 1e3,
        growth_40_over_20=ratio,
        pass_k20=bool(pos20 * 1e3 <= THRESHOLDS["pos_p50_mm_k20"]),
        pass_k40=bool(pos40 * 1e3 <= THRESHOLDS["pos_p50_mm_k40"]),
        pass_growth=bool(ratio <= THRESHOLDS["growth_40_over_20"]),
    )
    verdict["B_route_allowed"] = bool(verdict["pass_k20"] and verdict["pass_k40"]
                                      and verdict["pass_growth"])
    # CSV
    with (a.out / "wm_horizon_error.csv").open("w") as f:
        f.write("k,pos_p50_mm,pos_p90_mm,pos_p99_mm,ori_p50_rad,q_p50_rad,qd_p50_radps\n")
        for k in range(1, Hm + 1):
            r = table[k]
            f.write(f"{k},{r['pos']['p50']*1e3:.4f},{r['pos']['p90']*1e3:.4f},"
                    f"{r['pos']['p99']*1e3:.4f},{r['ori']['p50']:.6f},"
                    f"{r['q']['p50']:.6f},{r['qd']['p50']:.6f}\n")
    obj = dict(
        n_starts=int(pick.numel()), horizon=Hm, action_order=str(a.action_order),
        action_source="reference command proxy: (q_ref − default)/scale（HA→STANDARD 换序）",
        state_source="bundle.ref_state_at(f) → ref_state_at(f+k)；起点误差 0",
        thresholds=THRESHOLDS, verdict=verdict,
        checkpoints={int(k): table[k] for k in CHECKPOINTS if int(k) <= Hm},
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        note=("B 路线（K=40）唯一前置门槛；不通过 ⇒ B 标 NOT_RUN，本版继续走 A（规范 §5.5.5）"))
    (a.out / "wm_horizon_error.json").write_text(json.dumps(obj, indent=1))
    d03.log(f"[wm-h] pos p50 @20={verdict['pos_p50_mm_k20']:.3f}mm "
            f"@40={verdict['pos_p50_mm_k40']:.3f}mm ratio={verdict['growth_40_over_20']:.2f} "
            f"=> B_route_allowed={verdict['B_route_allowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
