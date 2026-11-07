#!/usr/bin/env python3
"""缓存 token chunk 的基本统计（报告用）：范数、VLA vs 录制、VLA 采样噪声。

不做任何近似：直接读 `windows.pt` / `vla_tokens.pt`。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main(out: Path) -> int:
    d = torch.load(str(out / "windows.pt"), map_location="cpu", weights_only=False)
    v = torch.load(str(out / "vla_tokens.pt"), map_location="cpu", weights_only=False)
    tl, tv, tv2 = d["tokens_label"].float(), v["tokens_vla"].float(), v["tokens_vla_rep2"].float()
    st = dict(
        n=int(d["state"].shape[0]),
        token0_norm_mean=float(tv[:, 0].norm(dim=-1).mean()),
        token0_norm_p50=float(tv[:, 0].norm(dim=-1).median()),
        label_token0_norm_mean=float(tl[:, 0].norm(dim=-1).mean()),
        vla_absmean=float(tv.abs().mean()),
        label_absmean=float(tl.abs().mean()),
        vla_vs_label=float((tv - tl).abs().mean()),
        vla_rep_diff=float((tv - tv2).abs().mean()),
        vla_rep_rel=float((tv - tv2).abs().mean() / tv.abs().mean()),
        fm_steps=int(v.get("fm_steps", 0)),
        vlm_ckpt=str(v.get("vlm_ckpt", "")),
    )
    (out / "token_stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    raise SystemExit(main(ap.parse_args().out))
