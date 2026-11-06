#!/usr/bin/env python3
"""把多份 transitions 按行拼接（场景采集是"一任务一份"，要先合并成一个 WM 训练集）。

    python3 scripts/concat_transitions.py \
        --inputs 'round_1/transitions_task*.pt' --out round_1/transitions.pt

保留各文件的 `meta` 列表（便于追溯每个任务采了多少行）。
"""
from __future__ import annotations

import argparse
import glob

import torch

KEYS = ("states", "next_states", "actions", "clean_actions", "xi",
        "rewards", "ret", "frames", "ref_rows", "clip_ids", "ep_ids")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="glob")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(a.inputs))
    if not files:
        raise SystemExit(f"[concat] 没匹配到 {a.inputs}")

    acc: dict[str, list[torch.Tensor]] = {k: [] for k in KEYS}
    metas, rows = [], []
    ep_offset = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        n = int(d["states"].shape[0])
        for k in KEYS:
            if k not in d:
                raise SystemExit(f"[concat] {f} 缺 {k}")
            acc[k].append(d[k])
        # ep_ids 各文件都从 0 开始 → 加偏移避免跨文件混号
        acc["ep_ids"][-1] = acc["ep_ids"][-1] + ep_offset
        ep_offset += int(d["ep_ids"].max()) + 1
        metas.append({"file": f, "rows": n, "meta": d.get("meta")})
        rows.append(n)

    pack = {k: torch.cat(v) for k, v in acc.items()}
    pack["frame_map"] = torch.arange(int(pack["states"].shape[0]), dtype=torch.long)
    pack["meta"] = {"source": "concat", "files": metas, "rows_per_file": rows,
                    "rows": int(pack["states"].shape[0])}
    torch.save(pack, a.out)
    print(f"[concat] {len(files)} 份 → {int(pack['states'].shape[0])} 行 → {a.out}")
    for m, n in zip(metas, rows):
        print(f"  {m['file'].split('/')[-1]:40s} {n:8d} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
