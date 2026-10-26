#!/usr/bin/env python3
"""S2.1 — 把 HA 会话的 ``action.motion_token`` 从"部署前端标签"换成 **WBC 端口 token**。

设计文件：``docs/S2_BASE_ALIGNED_PLAN_20261026.md``（S2.1 / D2 / D3）。

为什么需要它
------------
`data/ha_sessions/*` 里的 ``action.motion_token`` 来自 raw npz 的 ``encoder_latent``，也就是
SONIC **部署前端**（HA 自己的 1762 维 encoder_input）的输出。但我们的 WBC 吃的是
**端口装配**出来的 64 维 token（``SonicTokenBuilder``：10 帧未来参考 + anchor），
两者逐元素相关只有 ~0.05–0.11（见计划文档 F2）。⇒ VLA 必须改说端口语言（路线 II）。

映射（699/699 已验证，见计划文档 F1）
------------------------------------
``data/ha_sessions/<task>__<op>/data/chunk-000/episode_%06d.parquet``
  ↔ ``sorted(data/ha_raw/<task>/<op>/*.npz)[ep]``
  ↔ ``data/port_tokens/ha_tokens/<task>__<op>__ep%06d.npy``

只改三样东西
------------
1. ``action.motion_token`` 列 → 端口 token（float64 fixed_size_list[64]，沿用源 schema）；
2. 目标树里 ``episode_index`` / ``frame_index``（=帧内序号）/ ``index``（=全树单调）**重新编号**
   （GR00T 的 loader 按 ``episode_index // chunk_size`` 找文件，不能留空洞）；
3. ``meta/{info,episodes,episodes_stats,stats}.json`` 里与上面两件事相关的字段；
   **只重算 `action.motion_token` 的 6 个统计量**，其它键逐字段照抄。

其它所有列（状态/手部/视频路径无关列）**值不变**；视频用硬链接（跨设备才 copy）。

输出
----
``<out>/train/<task>__<op>/{data,videos,meta}``、``<out>/val/...``、``<out>/roots_train.txt``、
``<out>/roots_val.txt``、``<out>/audit.json``。默认 ``--val-every 10``（每任务 ``ep % 10 == 9`` 留出）。

用法::

    python3 scripts/relabel_sessions_port.py --sessions 1          # 单 session 冒烟
    python3 scripts/relabel_sessions_port.py                        # 全量
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
TOKEN_KEY = "action.motion_token"
CARRY_FILES = ("modality.json", "relative_stats.json", "tasks.jsonl")
STATE_JCOL = "observation.state"
N_JOINTS = 29
N_HANDS = 14


# --------------------------------------------------------------------------- stats
def stats_block(values: np.ndarray) -> dict[str, list[float]]:
    """与 ``scripts/prepare_ha_sessions.py::_stats_block`` 完全同口径。"""
    v = np.asarray(values).reshape(values.shape[0], -1).astype(np.float64)
    return {
        "mean": v.mean(0).tolist(),
        "std": v.std(0).tolist(),
        "min": v.min(0).tolist(),
        "max": v.max(0).tolist(),
        "q01": np.percentile(v, 1, axis=0).tolist(),
        "q99": np.percentile(v, 99, axis=0).tolist(),
    }


# -------------------------------------------------------------------------- helpers
def _link(src: Path, dst: Path) -> str:
    """硬链接（跨设备回退 copy）。返回 'hardlink' / 'copy'。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _joints_from_state(state: np.ndarray) -> np.ndarray:
    """``observation.state``(43) = [joints 0:22, hands_l 7, joints 22:29, hands_r 7] -> joints(29)."""
    return np.concatenate([state[:, 0:22], state[:, 29:36]], axis=1)


def _replace_fixed_list(table, name: str, values: np.ndarray):
    """用 ``values`` (T, W) 替换 ``name`` 这一列，保留源 schema 的元素类型与宽度。"""
    import pyarrow as pa

    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise KeyError(f"{name} not in parquet schema")
    field = table.schema.field(idx)
    typ = field.type
    if not pa.types.is_fixed_size_list(typ):
        raise TypeError(f"{name} is not fixed_size_list: {typ}")
    width = typ.list_size
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 2 or vals.shape[1] != width:
        raise ValueError(f"{name}: values {vals.shape} vs schema width {width}")
    flat = np.ascontiguousarray(vals.reshape(-1)).astype(typ.value_type.to_pandas_dtype())
    col = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=typ.value_type), width)
    return table.set_column(idx, name, col)


def _replace_int_column(table, name: str, values: np.ndarray):
    import pyarrow as pa

    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise KeyError(f"{name} not in parquet schema")
    typ = table.schema.field(idx).type
    col = pa.array(np.asarray(values).astype(np.int64), type=typ)
    return table.set_column(idx, name, col)


# ------------------------------------------------------------------------- one episode
def relabel_episode(job: dict) -> dict:
    """改写一个 episode 的 parquet + 视频硬链接。返回审计信息。"""
    import pyarrow.parquet as pq

    src = Path(job["src_parquet"])
    dst = Path(job["dst_parquet"])
    port_path = Path(job["port_npy"])
    local_ep = int(job["local_ep"])
    index_offset = int(job["index_offset"])
    expect_len = int(job["length"])

    port = np.load(port_path)
    if port.ndim != 2 or port.shape[1] != 64:
        raise ValueError(f"{port_path}: port token shape {port.shape}")
    if not np.isfinite(port).all():
        raise ValueError(f"{port_path}: non-finite values")

    table = pq.read_table(src)
    n = table.num_rows
    if n != expect_len:
        raise ValueError(f"{src}: rows {n} != episodes.jsonl length {expect_len}")
    if port.shape[0] != n:
        raise ValueError(f"{port_path}: {port.shape[0]} frames vs parquet {n}")

    table = _replace_fixed_list(table, TOKEN_KEY, port)
    table = _replace_int_column(table, "episode_index", np.full(n, local_ep))
    if "frame_index" in table.schema.names:
        table = _replace_int_column(table, "frame_index", np.arange(n))
    if "index" in table.schema.names:
        table = _replace_int_column(table, "index", np.arange(index_offset, index_offset + n))

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    link = _link(Path(job["src_video"]), Path(job["dst_video"]))

    out = {
        "session": job["session"],
        "local_ep": local_ep,
        "src_ep": int(job["src_ep"]),
        "clip": job["clip"],
        "length": n,
        "link": link,
        "token_stats": stats_block(port),
    }

    # 抽查：其它列逐值不变 + 旧 token == raw encoder_latent + 关节列 == raw qpos
    if job.get("spot"):
        src_tbl = pq.read_table(src)
        for name in src_tbl.schema.names:
            if name in ("action.motion_token", "episode_index", "frame_index", "index"):
                continue
            a = src_tbl[name].to_pylist()
            b = table[name].to_pylist()
            if a != b:
                raise AssertionError(f"{job['clip']}: column {name} changed during relabel")
        with np.load(job["raw_npz"]) as z:
            lab = np.asarray(z["encoder_latent"], dtype=np.float64)
            qpos = np.asarray(z["robot_qpos_before_decimation"], dtype=np.float64)
        old_tok = np.asarray(src_tbl[TOKEN_KEY].to_pylist(), dtype=np.float64)
        state = np.asarray(src_tbl[STATE_JCOL].to_pylist(), dtype=np.float64)
        joints = _joints_from_state(state)
        out["spot"] = {
            "clip": job["clip"],
            "label_vs_encoder_latent_max_abs": float(np.abs(old_tok - lab).max()),
            "state_joints_vs_raw_qpos_max_abs": float(np.abs(joints - qpos).max()),
            "other_columns_equal": True,
        }
    return out


# ----------------------------------------------------------------------- one session
def process_session(job: dict) -> dict:
    t0 = time.time()
    src_root = Path(job["src_root"])
    out_root = Path(job["out_root"])
    raw_dir = Path(job["raw_dir"])
    port_dir = Path(job["port_dir"])
    session = job["session"]
    val_every = int(job["val_every"])
    force = bool(job["force"])

    episodes_meta = [
        json.loads(line)
        for line in (src_root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    src_stats = json.loads((src_root / "meta" / "stats.json").read_text())
    ep_stats_rows = [
        json.loads(line)
        for line in (src_root / "meta" / "episodes_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(ep_stats_rows) != len(episodes_meta):
        raise ValueError(f"{session}: episodes_stats rows != episodes rows")

    raws = sorted(raw_dir.glob("*.npz"))
    n_ep = len(episodes_meta)
    if len(raws) != n_ep:
        raise ValueError(f"{session}: raw npz {len(raws)} != episodes {n_ep}")

    # ---- 切分（D2：每任务 ep % val_every == val_every-1 留出）
    splits = {"train": [], "val": []}
    for ep in range(n_ep):
        split = "val" if (val_every > 0 and ep % val_every == val_every - 1) else "train"
        splits[split].append(ep)

    existing = [p for p in (out_root / "train" / session, out_root / "val" / session) if p.exists()]
    if existing and not force:
        return {"session": session, "skipped": True, "reason": f"{existing[0]} exists (use --force)"}

    results = []
    spot_eps = set(job.get("spot_eps") or [])
    for split, eps in splits.items():
        if not eps:
            continue
        dst_root = out_root / split / session
        lengths, tokens = [], []
        cursor = 0
        # 本地编号
        for local_ep, src_ep in enumerate(eps):
            meta = episodes_meta[src_ep]
            length = int(meta["length"])
            clip = f"{session}__ep{src_ep:06d}"
            res = relabel_episode(
                {
                    "session": session,
                    "src_ep": src_ep,
                    "local_ep": local_ep,
                    "clip": clip,
                    "length": length,
                    "index_offset": cursor,
                    "src_parquet": str(src_root / "data" / "chunk-000" / f"episode_{src_ep:06d}.parquet"),
                    "dst_parquet": str(dst_root / "data" / "chunk-000" / f"episode_{local_ep:06d}.parquet"),
                    "src_video": str(src_root / "videos" / "chunk-000" / "observation.images.ego_view"
                                     / f"episode_{src_ep:06d}.mp4"),
                    "dst_video": str(dst_root / "videos" / "chunk-000" / "observation.images.ego_view"
                                     / f"episode_{local_ep:06d}.mp4"),
                    "port_npy": str(port_dir / f"{clip}.npy"),
                    "raw_npz": str(raws[src_ep]),
                    "spot": split == "train" and src_ep in spot_eps,
                }
            )
            cursor += length
            lengths.append(length)
            tokens.append(np.load(port_dir / f"{clip}.npy"))
            res["split"] = split
            results.append(res)

        # ---- meta
        meta_dst = dst_root / "meta"
        meta_dst.mkdir(parents=True, exist_ok=True)
        for name in CARRY_FILES:
            shutil.copyfile(src_root / "meta" / name, meta_dst / name)

        info = json.loads((src_root / "meta" / "info.json").read_text())
        info["total_episodes"] = len(eps)
        info["total_frames"] = int(sum(lengths))
        info["total_videos"] = len(eps)
        info["splits"] = {"train": f"0:{len(eps)}"}
        (meta_dst / "info.json").write_text(json.dumps(info, indent=4))

        with (meta_dst / "episodes.jsonl").open("w") as fh:
            for local_ep, (src_ep, length) in enumerate(zip(eps, lengths)):
                row = dict(episodes_meta[src_ep])
                row["episode_index"] = local_ep
                row["length"] = int(length)
                fh.write(json.dumps(row) + "\n")

        # 只换 action.motion_token 的统计量，其余照抄源文件的对应 episode
        new_token_stats = stats_block(np.concatenate(tokens, axis=0))
        with (meta_dst / "episodes_stats.jsonl").open("w") as fh:
            for local_ep, src_ep in enumerate(eps):
                row = json.loads(json.dumps(ep_stats_rows[src_ep]))
                row["episode_index"] = local_ep
                row["stats"][TOKEN_KEY] = stats_block(tokens[local_ep])
                fh.write(json.dumps(row) + "\n")

        stats = json.loads(json.dumps(src_stats))
        stats[TOKEN_KEY] = new_token_stats
        (meta_dst / "stats.json").write_text(json.dumps(stats))

    return {
        "session": session,
        "skipped": False,
        "episodes": {k: len(v) for k, v in splits.items()},
        "frames": {
            k: int(sum(int(episodes_meta[e]["length"]) for e in v)) for k, v in splits.items()
        },
        "raws": len(raws),
        "seconds": round(time.time() - t0, 1),
        "results": results,
    }


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/ha_sessions")
    ap.add_argument("--raw", type=Path, default=ROOT / "data/ha_raw")
    ap.add_argument("--port", type=Path, default=ROOT / "data/port_tokens/ha_tokens")
    ap.add_argument("--out", type=Path, default=ROOT / "data/ha_sessions_port")
    ap.add_argument("--val-every", type=int, default=10,
                    help="每 N 个 episode 留 1 个做 held-out（0 = 不切分，全部进 train）")
    ap.add_argument("--sessions", nargs="*", default=None, help="只处理这些 session 名")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--spot", type=int, default=5, help="全量改写里抽查多少个 episode 逐列对拍")
    args = ap.parse_args()

    src_roots = [sorted(p for p in args.src.iterdir() if p.is_dir())]
    sessions = []
    for group in src_roots:
        for p in group:
            if args.sessions and p.name not in set(args.sessions):
                continue
            task, _, op = p.name.partition("__")
            raw_dir = args.raw / task / op
            if not raw_dir.is_dir():
                raise SystemExit(f"missing raw dir for {p.name}: {raw_dir}")
            sessions.append(
                {
                    "session": p.name,
                    "src_root": str(p),
                    "raw_dir": str(raw_dir),
                    "port_dir": str(args.port),
                    "out_root": str(args.out),
                    "val_every": args.val_every,
                    "force": args.force,
                }
            )
    if not sessions:
        raise SystemExit("no sessions selected")

    # 抽查 episode：按 task 轮转，前 N 个
    spot_pool = []
    for s in sessions:
        n = len([ln for ln in (Path(s["src_root"]) / "meta" / "episodes.jsonl").read_text().splitlines() if ln.strip()])
        spot_pool.append((s["session"], n))
    for i, s in enumerate(sessions):
        cand = [ep for ep in (3, 17, 41, 73, 91) if ep < spot_pool[i][1]]
        s["spot_eps"] = cand[: max(1, args.spot // max(1, len(sessions)))] if args.spot else []

    t0 = time.time()
    print(f"[relabel] sessions={len(sessions)} out={args.out} val_every={args.val_every} "
          f"workers={args.workers}", flush=True)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_session, s): s["session"] for s in sessions}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                rep = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[relabel][FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
                raise
            reports.append(rep)
            if rep.get("skipped"):
                print(f"[relabel] skip {name}: {rep['reason']}", flush=True)
            else:
                print(f"[relabel] {name}: train={rep['episodes']['train']} "
                      f"val={rep['episodes']['val']} frames={rep['frames']} "
                      f"({rep['seconds']}s)", flush=True)

    # ---- 汇总：roots + audit
    reports.sort(key=lambda r: r["session"])
    audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "src": str(args.src),
        "raw": str(args.raw),
        "port_tokens": str(args.port),
        "val_every": args.val_every,
        "seconds_total": round(time.time() - t0, 1),
        "sessions": [],
        "spot_checks": [],
    }
    for split in ("train", "val"):
        roots = [str(args.out / split / r["session"]) for r in reports if not r.get("skipped")]
        if roots:
            (args.out / f"roots_{split}.txt").write_text(":".join(roots) + "\n")
    totals = {"train": 0, "val": 0}
    frames = {"train": 0, "val": 0}
    all_tokens = []
    for rep in reports:
        if rep.get("skipped"):
            continue
        totals["train"] += rep["episodes"]["train"]
        totals["val"] += rep["episodes"]["val"]
        frames["train"] += rep["frames"]["train"]
        frames["val"] += rep["frames"]["val"]
        for res in rep["results"]:
            if res.get("spot"):
                audit["spot_checks"].append(res["spot"])
            # 汇总端口 token（用于全量统计量对表）
        audit["sessions"].append(
            {k: v for k, v in rep.items() if k not in ("results",)}
        )
    # 全量端口 token 统计量（与每棵树 stats.json 的加权结果对表用）
    for rep in reports:
        if rep.get("skipped"):
            continue
        for split in ("train", "val"):
            root = args.out / split / rep["session"]
            st = json.loads((root / "meta" / "stats.json").read_text())
            all_tokens.append((rep["session"], split, st[TOKEN_KEY]))

    audit["totals"] = {"episodes": totals, "frames": frames}
    (args.out / "audit.json").write_text(json.dumps(audit, indent=1))
    print(f"[relabel] totals episodes={totals} frames={frames} "
          f"({audit['seconds_total']}s) -> {args.out}", flush=True)
    print(f"[relabel] spot checks: "
          f"{json.dumps(audit['spot_checks'], ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
