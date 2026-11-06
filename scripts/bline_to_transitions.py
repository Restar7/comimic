#!/usr/bin/env python3
"""把**B 线闭环 dump** 转成 WM 训练用的 transitions 格式，并可与仿真采集拼接。

为什么需要（2026-10-29，用户指出）：WM 现在只用 `collect_mjlab.py` 的 `transitions.pt` 训练，
而那份数据是 **平地、无桌子/物体、WBC 自己跑（无 VLA）** 的；部署却在 **任务场景、VLA 在环**。
于是 WM 从没学过"物体接触 / VLA 分布"，多步预测 `qd_ratio` 只有 0.84（低估速度 16%）。
B 线 dump 里有 `states / next_states / actions / rewards`，本身就是完整的 (s, a, s′, r)，
只是字段名/形状不同 —— 转换后就能混进 WM 训练。

    # 单轮：B 线 → transitions 格式
    python3 scripts/bline_to_transitions.py \
        --dump 'runs/j1_rounds_v4/round_1/bline/b_*.pt' \
        --library data/library/ft318_compact.pt \
        --out runs/j1_rounds_v4/round_1/bline_transitions.pt

    # 与仿真采集按比例混合（B 线权重默认 0.25）
    python3 scripts/make_wm_mix.py \
        --sim runs/j1_rounds_v4/round_1/transitions.pt \
        --bline runs/j1_rounds_v4/round_1/bline_transitions.pt \
        --ratio 0.25 --out runs/j1_rounds_v4/round_1/wm_mix.pt
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def per_tick_frames(frames: torch.Tensor, n_tick: int) -> torch.Tensor:
    """vla 臂只在重规划时刻记 frames（600 tick → 30 个）⇒ 按段展开到每 tick。"""
    if int(frames.numel()) == n_tick:
        return frames
    n_f = int(frames.numel())
    stride = max(1, (n_tick + n_f - 1) // n_f)
    plan_of_tick = torch.clamp(torch.arange(n_tick) // stride, max=n_f - 1)
    return frames[plan_of_tick]


def discounted_rtg(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """return-to-go（从每步往后的折扣累计），与 collect_mjlab 的 `ret` 同义。"""
    out = torch.zeros_like(rewards)
    run = 0.0
    for t in range(int(rewards.numel()) - 1, -1, -1):
        run = float(rewards[t]) + gamma * run
        out[t] = run
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="glob：B 线 dump")
    ap.add_argument("--library", type=Path, default=ROOT / "data/library/ft318_compact.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--no-filter", action="store_true", help="关掉坏集过滤（不建议）")
    ap.add_argument("--max-tilt", type=float, default=45.0, help="单集最大倾角上限（度）")
    ap.add_argument("--scene-task", default="", help="task1..4：从该场景读真实摩擦/COM 填 xi")
    ap.add_argument("--scene-dir", type=Path, default=Path(
        "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/SONIC_DATA/deployment/"
        "gear_sonic/data/robot_model/model_data/g1"))
    args = ap.parse_args()

    lib = torch.load(str(args.library), map_location="cpu", weights_only=False)
    starts, lengths = lib["starts"], lib["lengths"]
    n_clip = len(starts)

    files = sorted(glob.glob(args.dump))
    if not files:
        raise SystemExit(f"[b2t] 没匹配到 dump: {args.dump}")

    # 坏集过滤（2026-10-29）：摔倒 / 大倾角的 episode **不许进 wm_mix**。
    # 否则 WM 把"摔"学成分布内，下一轮又在附近拿奖励（R3 起摔倒 2→4/80 的正反馈就是这个）。
    if not args.no_filter:
        sys.path.insert(0, str(ROOT / "scripts"))
        from bline_health import select_dumps  # noqa: PLC0415
        good, rep = select_dumps(files, max_tilt_deg=args.max_tilt)
        print(f"[b2t] 坏集过滤: {rep['n_good']}/{rep['n_total']} 可用"
              f"（剔除 {rep['n_dropped']}，摔倒 {rep['fell_total']}）", flush=True)
        for d in rep["dropped"][:5]:
            print(f"[b2t]   剔除 {Path(d['path']).name}: {d['why']}", flush=True)
        if not good:
            raise SystemExit("[b2t][fail] 过滤后没有可用 episode")
        files = good

    # 库行号 -> clip_id（二分或线性，473k 帧足够快）
    starts_t = torch.tensor(starts, dtype=torch.long)
    clip_of_row = torch.searchsorted(starts_t, torch.arange(int(sum(lengths)), dtype=torch.long)) - 1
    clip_of_row = clip_of_row.clamp(0, n_clip - 1)

    acc: dict[str, list[torch.Tensor]] = {k: [] for k in (
        "states", "next_states", "actions", "rewards", "ret", "frames", "clip_ids", "ep_ids")}
    used = 0
    for ei, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        S, Sn, A, R, F = (d.get("states"), d.get("next_states"), d.get("actions"),
                          d.get("rewards"), d.get("frames"))
        if any(x is None for x in (S, Sn, A, R, F)):
            print(f"[b2t] SKIP {Path(f).name}: 缺 states/next_states/actions/rewards/frames")
            continue
        S, Sn, A, R = S.float(), Sn.float(), A.float(), R.float()
        T = int(S.shape[0])
        F = per_tick_frames(F.long().clamp(0, int(sum(lengths)) - 1), T)
        acc["states"].append(S)
        acc["next_states"].append(Sn)
        acc["actions"].append(A)
        acc["rewards"].append(R)
        acc["ret"].append(discounted_rtg(R, args.gamma))
        acc["frames"].append(F)
        acc["clip_ids"].append(clip_of_row[F])
        acc["ep_ids"].append(torch.full((T,), ei, dtype=torch.long))
        used += 1

    if not used:
        raise SystemExit("[b2t] 没有可用 dump")
    pack = {k: torch.cat(v) for k, v in acc.items()}
    n = int(pack["states"].shape[0])
    pack["clean_actions"] = pack["actions"].clone()
    # xi（33 维扰动编码）= [foot_friction(1), encoder_bias(29), torso_COM_offset(3)]
    # 2026-10-29：从**真实场景**读摩擦与 COM 偏移（而不是一律置零）。
    xi = torch.zeros(n, 33, dtype=torch.float32)
    if args.scene_task:
        import mujoco
        SC = {"task1": "scene_43dof_two_table.xml", "task2": "scene_43dof.xml",
              "task3": "scene_43dof_handover.xml", "task4": "scene_43dof_bend_and_pick.xml"}
        mp = args.scene_dir / SC[args.scene_task]
        m = mujoco.MjModel.from_xml_path(str(mp))
        foot = [m.geom(i).name for i in range(m.ngeom)
                if any(k in m.geom(i).name for k in ("foot", "ankle"))]
        fr = [float(m.geom_friction[m.geom(nm).id][0]) for nm in foot] if foot else [1.0]
        torso = m.body("torso_link").id if "torso_link" in [
            m.body(i).name for i in range(m.nbody)] else 0
        xi[:, 0] = sum(fr) / len(fr)
        # ★ xi 的语义是"相对标称的**偏移**"，不是绝对值：
        #   场景参数是固定的（没有 DR）⇒ COM 偏移 = 0、encoder bias = 0，
        #   只有足部摩擦可能是非 1.0。这样与 mjlab 场景采集得到的 xi 完全一致（实测均为 friction=1, 其余 0）。
        xi[:, 30:33] = 0.0
        print(f"[b2t] xi 真值: foot_friction={float(xi[0,0]):.3f}，"
              f"com_offset=0（xi 是相对标称偏移，场景固定参数 ⇒ 与 mjlab 采集一致）")
    pack["xi"] = xi
    # ref_rows：库在 frames 处的标准 71 维参考（WM 的 goal 用不到，但保持一致）
    ref = lib["ref"].float()
    pack["ref_rows"] = ref[pack["frames"]]
    pack["frame_map"] = torch.arange(n, dtype=torch.long)
    pack["meta"] = {
        "source": "bline_dump", "glob": args.dump, "n_dumps": used,
        "rows": n, "gamma": args.gamma,
        "note": "B 线闭环（VLA 在环、任务场景）转成 transitions；xi 置零",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, str(args.out))
    print(f"[b2t] {used} 条 dump → {n} 行 → {args.out}")
    print(f"[b2t] ret mean={float(pack['ret'].mean()):.3f} reward mean={float(pack['rewards'].mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
