#!/usr/bin/env python3
"""逐轮健康审计：一条命令看清每一轮的门、指标、80 集漂移。

为什么要它：`mean_reward` 从 R2 起**不可跨轮比较**（`std_root_ori` 换过），
单集 eval 的 p50 有 ~30% 噪声，而 80 集分布要现算很慢。所以判断"这轮行不行"
需要把三个来源拼起来看，手工做一次要十几分钟 —— 容易漏、也不可复现。

用法：
    python3 scripts/report_rounds.py                       # 快速（门 + eval + 采集奖励）
    python3 scripts/report_rounds.py --drift               # 加 80 集漂移（慢，会缓存）
    python3 scripts/report_rounds.py --drift --rounds 1 2 3
    python3 scripts/report_rounds.py --json runs/j1_scene2/report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fmt(v, spec: str, alt: str = "—") -> str:
    """None/NaN 都打成 alt，避免 'unsupported format string passed to NoneType'。"""
    try:
        if v is None:
            return alt
        f = float(v)
        if not math.isfinite(f):
            return alt
        return format(f, spec)
    except Exception:  # noqa: BLE001
        return alt


def _q(x, p):
    return float(x.quantile(p)) if len(x) else float("nan")


def round_drift(run_dir: Path, r: int, ref_path: Path | None) -> dict:
    """80 集采集 dump 的漂移 p50/p90（root_pos 用 cm，ori 用度）。"""
    import torch  # noqa: PLC0415

    from scripts.eval_realized_drift import per_tick_frames  # noqa: PLC0415

    fs = sorted(glob.glob(str(run_dir / f"round_{r}" / "bline" / "b_*.pt")))
    if not fs:
        return {}
    ref = torch.load(str(ref_path), map_location="cpu", weights_only=False)["ref"].float()
    pos, ori, bad = [], [], 0
    for f in fs:
        d = torch.load(f, map_location="cpu", weights_only=False)
        st = d["states"].float()
        r_rows = ref[per_tick_frames(d["frames"], st.shape[0])]
        e = (st[:, 0:3] - r_rows[:, 0:3]).norm(dim=-1)
        pos.append(e)
        if float(e.median()) > 0.10:
            bad += 1
        q1, q2 = st[:, 3:7], r_rows[:, 3:7]
        ori.append(torch.acos((q1 * q2).sum(-1).abs().clamp(max=1 - 1e-7)) * (360 / math.pi))
    P, O = torch.cat(pos), torch.cat(ori)
    return {"n_eps": len(fs), "n_tick": int(P.numel()),
            "pos_p50_cm": 100 * _q(P, 0.5), "pos_p90_cm": 100 * _q(P, 0.9),
            "ori_p50_deg": _q(O, 0.5), "ori_p90_deg": _q(O, 0.9), "bad_eps": bad}


def round_kstep(run_dir: Path, r: int, k: int = 20, stride: int = 60, device: str = "cpu") -> dict:
    """**同分布** k 步误差：第 r 轮的 WM 在**它自己那一轮**的 B 线 dump 上滚 k 步。

    这是判 qd_ratio 唯一有效的口径 —— 跨轮比较量的是策略演化，不是 WM 质量
    （详见 RUNBOOK §9：跨轮 1.25→1.39 看似退化，同分布其实是 0.99–1.00）。
    """
    import torch  # noqa: PLC0415

    from diff_vla_wbc.mjlab_joint import load_mjlab_bundle  # noqa: PLC0415
    from scripts.probe_wm_kstep import _load_dumps, _report, _roll  # noqa: PLC0415

    ck = run_dir / f"round_{r}" / "wm_pre.pt"
    if not ck.is_file():
        return {}
    dumps = str(run_dir / f"round_{r}" / "bline" / "b_*.pt")
    dev = torch.device(device)
    S, A, G = _load_dumps(dumps, k, stride)
    bundle = load_mjlab_bundle(ck, device=dev, npz=None, token_dim=64)
    from diff_vla_wbc.mjlab_joint import default_xi  # noqa: PLC0415
    pred = _roll(bundle, {"state": S.to(dev), "xi": default_xi(S.shape[0], dev)},
                 A.to(dev), k, policy=False, token=None)
    row = next(x for x in _report(pred, G, [k]) if x["k"] == k)
    return {"n_start": int(S.shape[0]), "joint_mae_deg": row["joint_mae_deg"],
            "qd_corr": row["qd_corr"], "qd_ratio": row["qd_ratio"],
            "root_pos_cm": row["root_pos_cm"], "root_ori_deg": row["root_ori_deg"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs/j1_scene2")
    ap.add_argument("--rounds", nargs="*", type=int, default=None)
    ap.add_argument("--ref", type=Path, default=ROOT / "data/library/ft318_compact_v8.pt")
    ap.add_argument("--drift", action="store_true", help="算 80 集漂移（慢，~1 min/轮）")
    ap.add_argument("--kstep", action="store_true",
                    help="每个 WM 在**自己那一轮**的数据上滚 k 步（同分布口径，慢）")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    rounds = args.rounds
    if rounds is None:
        rounds = sorted(int(p.name.split("_")[1]) for p in run_dir.glob("round_*")
                        if p.name.split("_")[1].isdigit())
    out = []
    for r in rounds:
        rd = run_dir / f"round_{r}"
        e: dict = {"round": r}
        for tag in ("sim", "bline"):
            f = rd / f"gate_wm_{tag}.json"
            if f.is_file():
                d = json.loads(f.read_text())
                e[f"gate_{tag}"] = {"corr": d.get("qd_corr"), "ratio": d.get("qd_ratio"),
                                    "pass": d.get("pass")}
        f = rd / "eval.json"
        if f.is_file():
            d = json.loads(f.read_text())
            mr = d.get("mean_reward")
            dump = rd / "eval_bline.pt"
            if mr is None and dump.is_file():
                # 9/9 的 mean_reward 不在 eval.json 里，从 dump 的逐步奖励现算（与链内口径一致）
                try:
                    import torch  # noqa: PLC0415
                    dd = torch.load(str(dump), map_location="cpu", weights_only=False)
                    mr = float(dd["rewards"].mean())
                except Exception:  # noqa: BLE001
                    mr = None
            e["eval"] = {"mean_reward": mr, "fell": d.get("fell"),
                         "max_tilt_deg": d.get("max_tilt_deg")}
        f = rd / "train" / "summary.json"
        if f.is_file():
            try:
                s = json.loads(f.read_text())
                e["train"] = {"steps": s.get("steps"), "wall_s": round(s.get("wall_s", 0))}
            except Exception:  # noqa: BLE001
                pass
        if args.drift:
            e["drift80"] = round_drift(run_dir, r, args.ref)
        if args.kstep:
            e["kstep"] = round_kstep(run_dir, r, k=int(args.k))
        out.append(e)

    print("# 注：round_N 的 80 集漂移刻画的是 **第 N-1 轮训出来的策略**（B 线是用上一轮产物采的）")
    print(f"{'轮':>3} {'sim corr/ratio':>16} {'bline corr/ratio':>18} {'eval reward':>12} "
          f"{'fell':>5} {'tilt':>6}" + (f" {'pos p50/p90 (cm)':>18} {'ori p50/p90 (°)':>17}" if args.drift else ""))
    for e in out:
        gs, gb = e.get("gate_sim", {}), e.get("gate_bline", {})
        ev = e.get("eval", {})
        s1 = f"{_fmt(gs.get('corr'), '.4f')}/{_fmt(gs.get('ratio'), '.4f')}" if gs else "—"
        s2 = f"{_fmt(gb.get('corr'), '.4f')}/{_fmt(gb.get('ratio'), '.4f')}" if gb else "—"
        s3 = _fmt(ev.get("mean_reward"), ".4f")
        s4 = str(ev.get("fell", "—"))
        s5 = _fmt(ev.get("max_tilt_deg"), ".1f")
        line = f"{e['round']:>3} {s1:>16} {s2:>18} {s3:>12} {s4:>5} {s5:>6}"
        d = e.get("drift80") or {}
        if args.drift:
            line += (f" {_fmt(d.get('pos_p50_cm'), '8.1f'):>8}/{_fmt(d.get('pos_p90_cm'), '8.1f'):>8}"
                     f" {_fmt(d.get('ori_p50_deg'), '7.1f'):>7}/{_fmt(d.get('ori_p90_deg'), '8.1f'):>8}")
        print(line)
    if args.kstep:
        print(f"\n# 同分布 k={args.k} 自回归（第 N 轮 WM 在**自己那一轮**的 dump 上）—— 判 qd_ratio 唯一有效口径")
        for e in out:
            kk = e.get("kstep") or {}
            if kk:
                print(f"  round_{e['round']}: joint={_fmt(kk.get('joint_mae_deg'), '.2f')}° "
                      f"qd_corr={_fmt(kk.get('qd_corr'), '.3f')} "
                      f"qd_ratio={_fmt(kk.get('qd_ratio'), '.3f')} "
                      f"root_pos={_fmt(kk.get('root_pos_cm'), '.2f')}cm "
                      f"root_ori={_fmt(kk.get('root_ori_deg'), '.2f')}°")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"[out] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
