#!/usr/bin/env python3
"""把每一轮训练出来的 checkpoint 快照到独立目录，防止被后续轮次覆盖/误删。

为什么需要：
    `run_j1_rounds.sh` 每轮写 `round_N/train/joint_final.pt` + `vla_final.pt` +
    `round_N/wm_pre.pt`，这些**是本轮唯一的产物**。风险有三个：
      1. 同一轮重跑（`REUSE=0` 或崩了重来）会**原地覆盖** `--out round_N/train`；
      2. `round_N/` 里还堆着 transitions/wm_mix（每轮 ~4 GB），空间紧张时容易被人手工清理；
      3. `ckpt_vla_rN` 是 7.1 GB 的目录，里面只有 2 个真文件，其余是软链——单独拷它不划算。

策略：
    只快照**不可再生**的东西（LoRA 增量 + WM + 指标 json），不拷大件数据。
    每轮约 400 MB：joint_final 68 MB + vla_final 256 MB + wm_pre 68 MB + 指标。
    `ckpt_vla_rN` 不拷，可用下面这条命令从快照还原：
        python3 scripts/export_vla_ckpt.py --base <上一轮 ckpt_vla> \\
            --vla snapshots/round_N/vla_final.pt --out ckpt_vla_rN --force

用法：
    python3 scripts/snapshot_round_ckpts.py                       # 补齐所有已完成轮次
    python3 scripts/snapshot_round_ckpts.py --rounds 2 3          # 只做指定轮
    python3 scripts/snapshot_round_ckpts.py --verify-load         # 额外 torch.load 校验
    python3 scripts/snapshot_round_ckpts.py --list                # 只看不拷

幂等：内容一样（sha256 相同）就跳过；内容不同则**另存**带时间戳的副本，
      绝不覆盖已有快照——这正是"避免丢失"的核心。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (源相对路径, 是否必需)。只收不可再生的产物。
FILES = (
    ("train/joint_final.pt", True),      # policy + WM（本轮唯一）
    ("train/vla_final.pt", True),        # LoRA 增量（ckpt_vla_rN 的原料）
    ("wm_pre.pt", True),                 # 本轮 WM 预训练结果（= joint_final 的 wm）
    ("train/summary.json", True),        # 跑完的标志 + 超参
    ("train/args.json", False),
    ("train/v2_config.json", False),
    ("train/v3_config.json", False),
    ("train/train_metrics.jsonl", False),  # 580 KB，后面画曲线要用
    ("eval.json", False),                # 闭环验收
    ("eval_bline.pt", False),
    ("gate_wm_sim.json", False),
    ("gate_wm_bline.json", False),
    ("gate_wm_bline_after.json", False),
    ("drift.json", False),
)


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def snapshot_round(run_dir: Path, dest: Path, r: int, *, verify_load: bool, dry: bool) -> dict:
    rd = run_dir / f"round_{r}"
    out = dest / f"round_{r}"
    manifest_path = out / "MANIFEST.json"
    old: dict = {}
    if manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text())
        except Exception:
            old = {}
    old_files: dict = old.get("files", {})

    entry: dict = {"round": int(r), "source": str(rd), "files": {}, "skipped": []}
    if not rd.is_dir():
        return {"round": r, "status": "no-dir"}

    for rel, required in FILES:
        src = rd / rel
        if not src.is_file() or src.stat().st_size == 0:
            if required:
                return {"round": r, "status": "incomplete", "missing": rel}
            continue
        h = sha256(src)
        prev = old_files.get(rel)
        # 内容没变 ⇒ 已经在快照里了，跳过
        if prev and prev.get("sha256") == h and (out / prev["stored_as"]).is_file():
            entry["files"][rel] = prev
            entry["skipped"].append(rel)
            continue
        stored_as = rel.replace("/", "__")
        # 内容变了（重跑过）⇒ 另存，绝不覆盖
        if (out / stored_as).is_file():
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(src.stat().st_mtime))
            stem, suf = os.path.splitext(stored_as)
            stored_as = f"{stem}.{stamp}{suf}"
        if dry:
            entry["files"][rel] = {"stored_as": stored_as, "sha256": h,
                                   "size": src.stat().st_size, "dry_run": True}
            continue
        copy_atomic(src, out / stored_as)
        got = sha256(out / stored_as)
        if got != h:
            raise SystemExit(f"[snapshot] 拷贝校验失败 {src} -> {out / stored_as}")
        rec = {"stored_as": stored_as, "sha256": h, "size": src.stat().st_size,
               "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(src.stat().st_mtime))}
        if verify_load and stored_as.endswith(".pt") and "vla_" not in stored_as:
            try:
                import torch  # noqa: PLC0415

                d = torch.load(str(out / stored_as), map_location="cpu", weights_only=False)
                rec["load_ok"] = True
                rec["load_keys"] = sorted(d.keys())[:12] if isinstance(d, dict) else None
            except Exception as exc:  # noqa: BLE001
                rec["load_ok"] = False
                rec["load_err"] = str(exc)[:200]
        entry["files"][rel] = rec

    entry["status"] = "ok"
    entry["ckpt_vla"] = str(run_dir / f"ckpt_vla_r{r}")
    entry["restore_ckpt_vla"] = (
        f"python3 scripts/export_vla_ckpt.py --base <prev ckpt_vla> "
        f"--vla snapshots/round_{r}/" + (entry["files"].get("train/vla_final.pt", {}).get("stored_as", "vla_final.pt"))
        + f" --out ckpt_vla_r{r} --force"
    )
    entry["snapshot_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not dry:
        out.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(entry, indent=1, ensure_ascii=False))
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs/j1_scene2")
    ap.add_argument("--dest", type=Path, default=None,
                    help="默认 <run-dir>/snapshots（独立于 round_N，不会被重跑覆盖）")
    ap.add_argument("--rounds", nargs="*", type=int, default=None, help="不给=扫描所有 round_*")
    ap.add_argument("--verify-load", action="store_true", help="对 .pt 做 torch.load 校验")
    ap.add_argument("--list", action="store_true", help="只列状态不拷贝")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    dest = (args.dest or (run_dir / "snapshots")).expanduser().resolve()
    rounds = args.rounds
    if rounds is None:
        rounds = sorted(int(p.name.split("_")[1]) for p in run_dir.glob("round_*")
                        if p.is_dir() and p.name.split("_")[1].isdigit())

    changed = 0
    for r in rounds:
        e = snapshot_round(run_dir, dest, r, verify_load=args.verify_load, dry=args.list)
        st = e.get("status")
        if st == "ok":
            n_new = len(e["files"]) - len(e["skipped"])
            tot = sum(v.get("size", 0) for v in e["files"].values()) / 1048576
            if n_new:
                changed += 1
            if not args.quiet:
                print(f"[snapshot] round_{r}: {st}  文件 {len(e['files'])} 个 "
                      f"({tot:.0f} MB)，本次新增 {n_new}", flush=True)
        elif st == "incomplete":
            if not args.quiet:
                print(f"[snapshot] round_{r}: 还没跑完（缺 {e.get('missing')}）⇒ 先跳过", flush=True)
        elif st == "no-dir":
            pass
    if not args.quiet:
        print(f"[snapshot] dest={dest}  本次更新 {changed} 轮", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
