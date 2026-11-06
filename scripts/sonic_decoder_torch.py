#!/usr/bin/env python3
"""SONIC release decoder 自检：ONNX → 可微 torch 模块，与 onnxruntime 对拍。

实现已经搬进库 ``diff_vla_wbc/sonic_decoder.py``（训练/度量两边共用一份，避免漂移），
本脚本只保留 CLI：

    python3 scripts/sonic_decoder_torch.py --onnx <model_decoder.onnx> --check
    python3 scripts/sonic_decoder_torch.py --onnx ... --out runs/sonic_decoder_torch.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.sonic_decoder import load_decoder  # noqa: E402


def _ort_run(onnx_path: str, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x})[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/"
                                      "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/sonic_decoder_torch.pt"))
    a = ap.parse_args()

    mod = load_decoder(a.onnx).eval()
    n_par = sum(p.numel() for p in mod.parameters())
    print(f"[dec] layers={len(mod.nodes)} params={n_par/1e6:.2f}M")
    for k, v in mod.params.items():
        print(f"   {k}: {tuple(v.shape)}")

    if a.check:
        torch.manual_seed(0)
        x = torch.randn(8, 994)
        with torch.no_grad():
            y_t = mod(x).numpy()
        y_o = np.concatenate([_ort_run(a.onnx, x[i:i + 1].numpy()) for i in range(x.shape[0])])
        print(f"[check L1] torch vs ORT  max|Δ|={np.abs(y_t - y_o).max():.3e}  "
              f"rel={np.abs(y_t - y_o).max()/max(1e-9, np.abs(y_o).max()):.2e}")
    torch.save({"state_dict": mod.state_dict(), "src": a.onnx}, str(a.out))
    print(f"[dec] saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
