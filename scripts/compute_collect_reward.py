#!/usr/bin/env python3
"""给 `isaac_g1_collect.py` 采到的转移补算奖励（在**系统 python** 里做，口径不重写）。

为什么分两步：
  Isaac 进程里不能 import 我们的训练栈（`diff_vla_wbc`/trackerLab MDP 会连锁依赖 MuJoCo 常量，
  在 Isaac venv 里炸），所以 `isaac_g1_collect.py` 只存 `(s, a, s', frame)`。
  奖励只依赖 `(state, action, prev_action, frame)`，全部在盘上 ⇒ 这里用**训练同一个**
  `bundle.reward(...)` 补算，保证口径与训练完全一致。

用法：
    python3 scripts/compute_collect_reward.py --in /tmp/collect_smoke/pt.pt \
        --out runs/wm_real/onpolicy_gt_doubledesk.pt
    # 可选：附加"命令派生动作"口径的奖励（诊断两种动作口径的差异）
    python3 scripts/compute_collect_reward.py --in ... --out ... --also-command-reward
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, nargs="+", required=True,
                    help="一个或多个 dump（bundle 只加载一次，批量补算）")
    ap.add_argument("--out", type=Path, required=True,
                    help="单个输入时是输出文件；多个输入时是输出**目录**")
    ap.add_argument("--wbc-ckpt", type=Path,
                    default=ROOT / "runs/S1_handover/WBC_A20000_wmvel_cmd.pt")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_cmd/packed.pt")
    ap.add_argument("--reward-mode", default="saturating", choices=("saturating", "linear", "huber", "huber_err"))
    ap.add_argument("--huber-delta", type=float, default=0.02)
    ap.add_argument("--huber-cap", type=float, default=0.15)
    ap.add_argument("--reward-delta-pos", type=float, default=0.05)
    ap.add_argument("--reward-delta-ori", type=float, default=0.05)
    ap.add_argument("--root-ori-mode", choices=("mono", "split", "split_huber"), default="split_huber")
    ap.add_argument("--reward-root-pos-std", type=float, default=None,
                    help="与训练一致：新线 0.05（5cm）")
    ap.add_argument("--reward-root-ori-std", type=float, default=None,
                    help="与训练一致：新线 0.10（~5.7°）")
    ap.add_argument("--reward-body-pos-std", type=float, default=None,
                    help="与训练一致：新线 0.05")
    ap.add_argument("--reward-body-ori-std", type=float, default=None,
                    help="与训练一致：新线 0.10")
    ap.add_argument("--reward-vel-scale", type=float, default=1.0)
    ap.add_argument("--from-args", type=Path, default=None,
                    help="★ 直接从某一轮的 args.json 继承**完整奖励口径**（推荐）。"
                         "走 diff_vla_wbc.reward_cfg.apply_reward_cfg —— 与训练同一实现，"
                         "保证'补算的奖励 == 那轮训练真正用的奖励'。给了它就忽略下面的奖励开关。")
    ap.add_argument("--state-order", choices=("ha_v31", "standard"), default="ha_v31",
                    help="输入 dump 的 q/qd 关节序。ha_v31=闭环 provider dump(默认,历史行为);"
                         "standard=A' 采集分片(isaac_a_collect_batch 产出)。"
                         "★ 2026-11-05 实测:A' 分片**已是 standard**，用 ha_v31 会多转一次，"
                         "body_ori 误差 0.84→1.17、关节极限惩罚 0.0002→0.798、总奖励 +0.028→−0.008")
    ap.add_argument("--also-command-reward", action="store_true",
                    help="另外算一遍\"命令派生动作\"口径的奖励（诊断用）")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from diff_vla_wbc.isaac_joint import FrameRef, load_isaac_bundle  # noqa: PLC0415

    dev = torch.device(a.device)
    npz = sorted((ROOT / "data/npz/ha_tokens").glob("*.npz"))
    _rc = {}
    if a.reward_root_pos_std is not None:
        _rc["root_pos_std"] = float(a.reward_root_pos_std)
    if a.reward_root_ori_std is not None:
        _rc["root_ori_std"] = float(a.reward_root_ori_std)
    if a.reward_body_pos_std is not None:
        _rc["body_pos_std"] = float(a.reward_body_pos_std)
    if a.reward_body_ori_std is not None:
        _rc["body_ori_std"] = float(a.reward_body_ori_std)
    bundle = load_isaac_bundle(a.wbc_ckpt, device=dev, npz=npz, packed_motion=a.packed,
                               obs_mode=None, hist=None, token_dim=64,
                               reward_cfg=(_rc or None))
    from diff_vla_wbc.reward_cfg import apply_reward_cfg, reward_cfg_summary  # noqa: PLC0415
    if a.from_args is not None:
        import json  # noqa: PLC0415
        _cfg = json.loads(a.from_args.read_text())
        apply_reward_cfg(bundle, _cfg)
        print(f"[reward] 口径继承自 {a.from_args}", flush=True)
    else:
        bundle.reward_mode = str(a.reward_mode)
        bundle.reward_vel_scale = float(a.reward_vel_scale)
        bundle.huber_delta = float(getattr(a, "huber_delta", 0.02))
        bundle.huber_cap = float(getattr(a, "huber_cap", 0.15))
        bundle.huber_err_delta_pos = float(getattr(a, "reward_delta_pos", 0.05))
        bundle.huber_err_delta_ori = float(getattr(a, "reward_delta_ori", 0.05))
        bundle.root_ori_mode = str(getattr(a, "root_ori_mode", "split_huber"))
    print(f"[reward][canon] {reward_cfg_summary(bundle)}", flush=True)
    multi = len(a.inp) > 1
    if multi:
        a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for inp in a.inp:
        out_path = (a.out / (inp.stem + "_r.pt")) if multi else a.out
        rows.append(_one(inp, out_path, a, bundle, dev))
    if multi:
        print("\n[reward] ===== 汇总 =====", flush=True)
        for k, v in rows:
            print(f"[reward] {k:34s} r_mean={v:+.5f}", flush=True)
    return 0


def _to_standard(states: torch.Tensor) -> torch.Tensor:
    """dump 的 71 维（q/qd 为 asset/HA_V31 序）→ standard 序。"""
    from diff_vla_wbc.b_start import state_to_standard  # noqa: PLC0415

    import joint_contract as JC  # noqa: PLC0415

    perm = torch.as_tensor(JC.permutation("ha_v31", "standard"), dtype=torch.long)
    return state_to_standard(states.float(), perm)


def _one(inp: Path, out_path: Path, a, bundle, dev) -> tuple[str, float]:
    from diff_vla_wbc.isaac_joint import FrameRef  # noqa: PLC0415

    d = torch.load(str(inp), map_location="cpu", weights_only=False)
    S, A, F = d["states"], d["actions"], d["frames"]
    # ★ provider 的 dump 是 **asset(HA_V31) 序**（`_record_state_dump` 注释写明），bundle 用
    #   standard 序；不转就是拿错位的关节去算奖励（2026-10-28 发现）。actions 本来就是 standard。
    _order = str(getattr(a, "state_order", "ha_v31"))
    if _order == "ha_v31":
        S = _to_standard(S)
    print(f"[reward] state_order={_order}（ha_v31 ⇒ 转 standard；standard ⇒ 原样使用）", flush=True)
    # 奖励只依赖 (state, action, prev_action, frame)；next_states 只是附带字段，
    # 官方闭环的 state dump 没有它（也不需要它）⇒ 缺失时按帧位移补上，不影响奖励。
    S2 = d.get("next_states")
    if S2 is None:
        S2 = torch.cat((S[1:], S[-1:]), dim=0)
    elif _order == "ha_v31":
        S2 = _to_standard(S2)
    print(f"[reward] in={inp} n={S.shape[0]} clip={d.get('clip')} "
          f"envs={d.get('num_envs')} steps={d.get('steps')}", flush=True)

    n = int(S.shape[0])
    bs = 4096
    out = {"rewards": torch.zeros(n), "reward_terms": {}}
    terms_acc: dict[str, list] = {}
    for lo in range(0, n, bs):
        sl = slice(lo, min(lo + bs, n))
        st = S[sl].to(dev).float()
        ac = A[sl].to(dev).float()
        fr = F[sl].to(dev).long()
        # prev_action：同一 env 的上一行（按环境交错，故用"回退 num_envs 行"近似；
        # 第一帧用自身）
        prev = A[torch.clamp(torch.arange(lo, min(lo + bs, n)) - int(d.get("num_envs", 1)),
                             min=0)].to(dev).float()
        terms = bundle.reward(st, ac, prev, FrameRef(fr), bodies=None)
        out["rewards"][sl] = terms["total"].detach().cpu()
        for k, v in terms.items():
            if torch.is_tensor(v) and v.shape == fr.shape:
                terms_acc.setdefault(k, []).append(v.detach().cpu())
    for k, vs in terms_acc.items():
        out["reward_terms"][k] = torch.cat(vs)
    print(f"[reward] r mean={float(out['rewards'].mean()):+.5f} "
          f"std={float(out['rewards'].std()):.5f} "
          f"min={float(out['rewards'].min()):+.4f} max={float(out['rewards'].max()):+.4f}", flush=True)
    # ---- 相对漂移（2026-10-28）：消除"env 起点 ≠ 库起点"的常数偏移 ----
    # 为什么需要：B 线（http）模式机器人从 env 的 init pose 出生，和参考库 clip 起点不同，
    # 绝对 e_root_pos 会有常数偏移（实测前 200 步就 0.37 m），跨臂不可比。
    # e_root_rel[t] = || (p_t − p_0) − (ref_t − ref_0) ||_xy  —— 只反映"走了多偏"，与出生点无关。
    try:
        _ref = bundle.motion.ref.float().cpu()
        _fr = F.long().cpu()
        _p = S[:, :2].float().cpu()
        _rp = _ref[_fr][:, :2]
        _rel = ((_p - _p[0]) - (_rp - _rp[0])).norm(dim=1)
        out["reward_terms"]["e_root_rel"] = _rel
        print(f"[reward]   e_root_rel      mean={float(_rel.mean()):.4f} "
              f"end={float(_rel[-1]):.4f} (相对起点漂移，跨臂可比)", flush=True)
    except Exception as _exc:  # noqa: BLE001
        print(f"[reward] WARN e_root_rel 计算失败: {_exc}", flush=True)

    for k in ("track", "e_body_pos", "e_body_ori", "e_body_linvel", "e_body_angvel",
              "e_root_pos", "e_root_ori", "action_rate", "joint_limit"):
        if k in out["reward_terms"]:
            t = out["reward_terms"][k]
            print(f"[reward]   {k:15s} mean={float(t.mean()):+.6f}", flush=True)

    payload = dict(d)
    # 2026-10-28 fix: the dump's states are asset(HA_V31) order; reward uses the converted S.
    # The downstream transfer/WM scripts read ``scored.pt`` and expect **standard** states, so
    # save the converted tensors -- otherwise the 29 joints are silently permuted.
    payload["states"] = S
    payload["next_states"] = S2
    payload["rewards"] = out["rewards"]
    payload["reward_terms"] = out["reward_terms"]
    payload.setdefault("meta", {})
    payload["meta"].update({"reward_mode": a.reward_mode,
                            "root_pos_std": a.reward_root_pos_std,
                            "root_ori_std": a.reward_root_ori_std,
                            "body_pos_std": a.reward_body_pos_std,
                            "body_ori_std": a.reward_body_ori_std,
                            "huber_delta": getattr(a, "huber_delta", None),
                            "huber_cap": getattr(a, "huber_cap", None),
                            "reward_vel_scale": float(a.reward_vel_scale),
                            "reward_fn": "trackerlab mdp.reward via bundle.reward (同一函数)",
                            "reward_computed_by": "scripts/compute_collect_reward.py"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    stats = {"n": n, "r_mean": float(out["rewards"].mean()),
             "r_std": float(out["rewards"].std()), "meta": payload["meta"]}
    out_path.with_suffix(".json").write_text(json.dumps(stats, indent=1, default=str))
    print(f"[reward] saved {out_path}", flush=True)
    return (inp.stem, float(out["rewards"].mean()))


if __name__ == "__main__":
    raise SystemExit(main())
