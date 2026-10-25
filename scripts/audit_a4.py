#!/usr/bin/env python3
"""Audit an A4 (or any fixed-wall-clock ablation) grid: prove the arms differ only in the variable.

Checks per cell
---------------
* ``args.json`` exists (written by ``train_joint.py``) and the run kept the hard-coded protocol
  (Isaac checkpoint, pure-HA motion, packed Isaac library, seed, batch size, hand loss, VLM ckpt);
* two DDP ranks started, the wall-clock ``[budget]`` line is present, and no traceback / CUDA OOM /
  non-finite loss appears in the log;
* the eval curve has enough points, strictly increasing steps, a constant horizon, and no NaN;
* ``wm_live`` cells really updated the world model (``wm/`` lines > 0) and ``wm_frozen`` cells
  really did not (== 0);
* peak memory from the ``.mem`` sampler.

Checks across cells
-------------------
* every arm used the **same** eval windows: the step-0 ``token_mse`` is identical (the eval
  provider draws its frames with a horizon-independent bound, so a mismatch means the protocol
  regressed);
* the resolved argument sets differ in exactly the ablated keys.

Usage
-----
    python3 scripts/audit_a4.py --root runs/ablations/A4_wm_horizon --logs logs/a4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: keys that are allowed to differ between arms (the ablation itself + bookkeeping)
ALLOWED_DIFF = {
    "horizon", "eval_every", "wm_update_every", "wm_update_warmup", "eval_out", "out",
    "wm_update_batch", "wm_update_window", "log_every",
}
#: everything that must be identical
MUST_MATCH = (
    "supermimic_ckpt", "motion", "motion_packed", "vlm_ckpt", "vlm_path", "seed", "batch_size",
    "hand_loss_weight", "hand_grasp_boost", "vlm_steps", "vlm_dataset", "train_vlm",
    "eval_batches", "policy_lr", "fm_lr", "adapter_lr", "obs_mode", "hist", "steps",
    "time_budget_minutes", "fm_grad_backend",
)
FAIL, WARN, OK = "FAIL", "WARN", "ok"


def load_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def peak_mem(mem: Path) -> float | None:
    if not mem.exists():
        return None
    best = 0.0
    for line in mem.read_text().splitlines():
        parts = line.split()
        for i in range(1, len(parts) - 1, 2):
            try:
                best = max(best, float(parts[i + 1]))
            except ValueError:
                pass
    return best or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/ablations/A4_wm_horizon"))
    ap.add_argument("--logs", type=Path, default=Path("logs/a4"))
    ap.add_argument("--min-points", type=int, default=8)
    ap.add_argument("--strict", action="store_true", help="warnings become failures")
    args = ap.parse_args()

    cells = sorted(p.name for p in args.root.iterdir() if p.is_dir()) if args.root.exists() else []
    if not cells:
        raise SystemExit(f"no cells under {args.root}")

    findings: list[tuple[str, str, str]] = []   # (level, cell, message)

    def note(level: str, cell: str, msg: str) -> None:
        findings.append((level, cell, msg))

    fps: dict[str, dict] = {}
    for cell in cells:
        cdir = args.root / cell
        log = args.logs / f"{cell}.log"
        mem = args.logs / f"{cell}.mem"

        # ---- args fingerprint ------------------------------------------------
        fj = cdir / "args.json"
        if not fj.exists():
            note(WARN, cell, "没有 args.json（该格在加审计指纹之前启动）")
        else:
            fps[cell] = json.loads(fj.read_text())
            fp = fps[cell]
            for key, want in (
                ("motion", "ha_tokens"), ("motion_packed", "ha_tokens"), ("seed", 0),
                ("batch_size", 1), ("hand_loss_weight", 0.2), ("vlm_steps", 2),
            ):
                got = fp.get(key)
                if isinstance(got, list):
                    got = " ".join(str(g) for g in got)
                if str(want) not in str(got):
                    note(FAIL, cell, f"args.{key} = {got!r}，期望含 {want!r}")
            for key in ("supermimic_ckpt", "motion", "motion_packed"):
                text = str(fp.get(key, "")).lower()
                if "mixed" in text or "ft318" in text:
                    note(FAIL, cell, f"args.{key} 指向禁用数据集: {fp.get(key)}")
            if "trackerlab-supermimic" not in str(fp.get("supermimic_ckpt", "")):
                note(FAIL, cell, f"args.supermimic_ckpt 不是 Isaac ckpt: {fp.get('supermimic_ckpt')}")

        # ---- log -------------------------------------------------------------
        if not log.exists():
            note(FAIL, cell, f"缺训练日志 {log}")
        else:
            text = log.read_text(errors="replace")
            ranks = sum(1 for ln in text.splitlines() if ln.startswith("[dist] rank"))
            if ranks < 2:
                note(FAIL, cell, f"只看到 {ranks} 个 rank 启动行（期望 2）")
            if "[budget]" not in text:
                note(FAIL, cell, "日志里没有 [budget] 墙钟截断行（该格可能没跑满预算或被 kill）")
            for bad in ("Traceback", "OutOfMemoryError", "nonfinite", "NaN", "nan,"):
                if bad in text:
                    note(FAIL, cell, f"日志出现 {bad!r}")
            wm_lines = sum(1 for ln in text.splitlines() if '"wm/loss"' in ln or '"wm/state_loss"' in ln)
        wm_lines = locals().get("wm_lines", 0)
        live = "live" in cell
        if live and wm_lines == 0:
            note(FAIL, cell, "wm_live 臂一次都没更新 WM（等同于冻结臂）")
        if (not live) and wm_lines > 0:
            note(FAIL, cell, f"wm_frozen 臂竟然出现了 {wm_lines} 次 WM 更新")

        # ---- curve -----------------------------------------------------------
        rows = load_curve(cdir / "eval_curve.jsonl")
        if len(rows) < args.min_points:
            note(WARN if rows else FAIL, cell, f"评测点只有 {len(rows)} 个（期望 ≥ {args.min_points}）")
        steps = [r.get("step") for r in rows]
        if steps != sorted(steps) or len(set(steps)) != len(steps):
            note(FAIL, cell, "评测 step 不是严格递增")
        hs = {r.get("h") for r in rows}
        if len(hs) > 1:
            note(FAIL, cell, f"同一格内 horizon 不一致: {hs}")
        for key in ("rollout_reward", "state_term", "token_mse", "rollout_reward@8", "state_term@8"):
            vals = [r[key] for r in rows if key in r]
            if vals and any((v is None) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                            for v in vals):
                note(FAIL, cell, f"eval 曲线 {key} 含 NaN/Inf")

        # ---- memory ----------------------------------------------------------
        peak = peak_mem(mem)
        if peak is None:
            note(WARN, cell, "没有显存采样")
        elif peak > 45_000:
            note(WARN, cell, f"峰值显存 {peak:.0f} MiB > 45000（离 48 GB 上限太近）")

    # ---- 参数变化审计：frozen 臂 WM 必须**逐位**等于源 ckpt，live 臂必须真的变了 ----------
    import torch  # 仅此处需要（CPU 加载 100 MB 级 ckpt）

    deltas: dict[str, tuple[float, float]] = {}
    for cell in cells:
        joint = args.root / cell / "joint_latest.pt"
        if not joint.exists():
            note(WARN, cell, "没有 joint_latest.pt（无法做参数变化审计）")
            continue
        ck = torch.load(str(joint), map_location="cpu", weights_only=False)
        src_path = ck.get("args", {}).get("source_checkpoint")
        if not src_path or not Path(src_path).exists():
            note(WARN, cell, "joint_latest.pt 里没有可用的 source_checkpoint")
            continue
        src = torch.load(src_path, map_location="cpu", weights_only=False)
        d_wm = max(float((ck["wm"][k].float() - src["wm"][k].float()).abs().max())
                   for k in src["wm"] if k in ck["wm"])
        d_pi = max(float((ck["pi"][k].float() - src["pi"][k].float()).abs().max())
                   for k in src["pi"] if k in ck["pi"])
        deltas[cell] = (d_wm, d_pi)
        live = "live" in cell
        if live and d_wm <= 0.0:
            note(FAIL, cell, f"wm_live 臂的 WM 参数与源 ckpt 逐位相同 (max|Δ|={d_wm:.1e}) → 根本没更新")
        if (not live) and d_wm > 0.0:
            note(FAIL, cell, f"wm_frozen 臂的 WM 变了 (max|Δ|={d_wm:.3e}) → 冻结失败")
        if d_pi <= 0.0:
            note(FAIL, cell, "policy 参数没有变化 → 该格没在训练 policy")
        note(OK, cell, f"参数审计: max|ΔWM|={d_wm:.3e}, max|Δpolicy|={d_pi:.3e} "
                       f"(frozen 应 0 / live 应 >0)")

    # ---- cross-cell: same eval windows + same protocol ---------------------------
    base = cells[0]
    for cell in cells[1:]:
        a = (fps.get(base) or {})
        b = (fps.get(cell) or {})
        if not (a and b):
            continue
        diff = {k for k in set(a) | set(b)
                if k not in ALLOWED_DIFF and not k.startswith("_") and a.get(k) != b.get(k)}
        if diff:
            note(FAIL, cell, f"与 {base} 有非消融差异: {sorted(diff)}")
        for key in MUST_MATCH:
            if a.get(key) != b.get(key):
                note(FAIL, cell, f"{key} 与 {base} 不一致: {b.get(key)!r} vs {a.get(key)!r}")

    # 同一批评测窗口：step-0 的 token_mse 必须逐位相同
    step0 = {}
    for cell in cells:
        rows = load_curve(args.root / cell / "eval_curve.jsonl")
        zero = next((r for r in rows if r.get("step") == 0), None)
        if zero is not None and "token_mse" in zero:
            step0[cell] = float(zero["token_mse"])
    if len(step0) >= 2:
        vals = list(step0.values())
        if max(vals) - min(vals) > 1e-6:
            note(FAIL, "ALL", f"step-0 token_mse 不一致 → 各臂的评测窗口不同: {step0}")
        else:
            note(OK, "ALL", f"step-0 token_mse 完全一致 ({vals[0]:.6f}) → 同一批评测窗口")

    # ---- report ------------------------------------------------------------------
    order = {FAIL: 0, WARN: 1, OK: 2}
    findings.sort(key=lambda f: order[f[0]])
    n_fail = sum(1 for lvl, _, _ in findings if lvl == FAIL)
    n_warn = sum(1 for lvl, _, _ in findings if lvl == WARN)
    print(f"# A4 审计：{len(cells)} 格 | FAIL {n_fail} | WARN {n_warn}\n")
    for lvl, cell, msg in findings:
        icon = {FAIL: "❌", WARN: "⚠️", OK: "✅"}[lvl]
        print(f"{icon} [{cell}] {msg}")
    if not findings:
        print("✅ 没有任何发现")
    print()
    print("审计覆盖：Isaac ckpt / 纯 HA 数据 / 无 mixed·ft318 / 2 卡启动 / 墙钟截断 / 无 OOM·NaN /")
    print("          评测点数量与单调性 / horizon 恒定 / wm_live 真更新·wm_frozen 真冻结 /")
    print("          峰值显存 / 跨臂仅有消融差异 / 同一批评测窗口")
    if n_fail or (args.strict and n_warn):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
