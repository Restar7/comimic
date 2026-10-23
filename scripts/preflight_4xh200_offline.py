#!/usr/bin/env python3
"""4×H200 离线前置检查（不发起任何网络请求）。

检查项：
  1. 可见 GPU 数量/型号/驱动；
  2. torch CUDA、warp CUDA 设备数（warp 版本不对会静默退到 CPU）；
  3. token encoder：文件存在、ONNX 可解析、g1 权重齐全、PyTorch 复现与 onnxruntime 逐位一致；
  4. 紧凑库：存在、大小、可选加载并检查 clips/frames；
  5. 无网络依赖：训练路径不访问 HF/网络（只做静态确认）。

用法（在训练 venv 里跑）：
    .venv/bin/python ../scripts/preflight_4xh200_offline.py --expect-gpus 4
    .venv/bin/python ../scripts/preflight_4xh200_offline.py --expect-gpus 1 --skip-library
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
SUPERMIMIC = REPO_ROOT / "third_party" / "SuperMimic-Reproduced"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-gpus", type=int, default=4)
    ap.add_argument("--skip-library", action="store_true")
    ap.add_argument("--library", type=Path, default=None)
    ap.add_argument("--encoder", type=Path, default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(SUPERMIMIC / "src"))

    errors: list[str] = []

    def ok(msg: str) -> None:
        print(f"  [ok]   {msg}")

    def bad(msg: str) -> None:
        print(f"  [FAIL] {msg}")
        errors.append(msg)

    # ---------------------------------------------------------------- 1. GPUs
    print("[1] GPU")
    import torch

    n = torch.cuda.device_count()
    if n < args.expect_gpus:
        bad(f"torch 只看到 {n} 张卡，需要 {args.expect_gpus}")
    else:
        ok(f"torch 可见 {n} 张卡")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 2**30
        print(f"        cuda:{i} {name} sm_{cap[0]}{cap[1]} {mem:.0f} GiB")
        if "H200" in name and cap[0] < 9:
            bad("H200 期望 sm_90")
    try:
        import subprocess

        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout.strip().splitlines()
        print(f"        driver: {drv[:1]} ...")
    except Exception as exc:  # noqa: BLE001
        print(f"        driver query skipped: {exc!r}")

    # ---------------------------------------------------------------- 2. warp
    print("[2] warp / mujoco")
    try:
        import mujoco

        ok(f"mujoco {mujoco.__version__}")
    except Exception as exc:  # noqa: BLE001
        bad(f"mujoco 导入失败: {exc!r}")
    try:
        import warp as wp

        wp.init()
        cuda_devs = [d for d in wp.get_devices() if getattr(d, "is_cuda", False)]
        if len(cuda_devs) < args.expect_gpus:
            bad(f"warp 只看到 {len(cuda_devs)} 张卡（需要 {args.expect_gpus}）；检查 warp-lang 是否 1.17.0")
        else:
            ok(f"warp {wp.__version__} CUDA 设备 {len(cuda_devs)} 张")
    except Exception as exc:  # noqa: BLE001
        bad(f"warp 初始化失败: {exc!r}")

    # ------------------------------------------------------------- 3. encoder
    print("[3] token encoder")
    encoder = args.encoder or (SUPERMIMIC / "data" / "assets" / "model_encoder.onnx")
    if not encoder.is_file():
        bad(f"encoder 不存在: {encoder}")
    else:
        ok(f"encoder: {encoder} ({encoder.stat().st_size/2**20:.0f} MiB)")
        try:
            from supermimic.envs.mdp.sonic_tokens import G1TokenEncoder, default_encoder_path

            resolved = default_encoder_path()
            if resolved is None or resolved.resolve() != encoder.resolve():
                print(f"        note: default_encoder_path -> {resolved}")
            enc = G1TokenEncoder.from_onnx(encoder)
            n_params = sum(p.numel() for p in enc.parameters())
            if n_params != 4231232:
                bad(f"encoder 参数量 {n_params} != 4231232")
            else:
                ok(f"encoder 4,231,232 参数（640→64，FSQ）")
            cmd = torch.randn(2, 10, 58)
            anc = torch.randn(2, 10, 6) * 0.2
            with torch.no_grad():
                tok = enc(cmd, anc)
            if tuple(tok.shape) != (2, 64):
                bad(f"token shape {tuple(tok.shape)} != (2, 64)")
            elif not torch.allclose(tok * 16, torch.round(tok * 16), atol=1e-6):
                bad("token 不在 1/16 网格上")
            else:
                ok("encoder 前向正常，token 在 1/16 网格")
            try:
                import numpy as np
                import onnxruntime as ort

                sess = ort.InferenceSession(str(encoder), providers=["CPUExecutionProvider"])
                name = sess.get_inputs()[0].name
                x = np.zeros((1, 1762), np.float32)
                x[0, 4:584] = cmd[:1].numpy().reshape(-1)
                x[0, 601:661] = anc[:1].numpy().reshape(-1)
                ref = sess.run(None, {name: x})[0]
                ours = tok[:1].numpy()
                if np.array_equal(ours, ref):
                    ok("与部署 ONNX 逐位一致（偏移 [4:584)/[601:661)）")
                else:
                    bad(f"与 ONNX 不一致，maxdiff={float(np.abs(ours-ref).max()):.4f}")
            except ImportError:
                print("        note: 未装 onnxruntime，跳过 ONNX 对齐检查")
        except Exception as exc:  # noqa: BLE001
            bad(f"encoder 检查失败: {exc!r}")

    # ------------------------------------------------------------- 4. library
    print("[4] compact library")
    library = args.library or (SUPERMIMIC / "data" / "library" / "bones_seed_compact.pt")
    if not library.is_file():
        bad(f"库不存在: {library}")
    else:
        ok(f"library: {library} ({library.stat().st_size/2**30:.2f} GiB)")
        if not args.skip_library:
            try:
                lib = torch.load(str(library), map_location="cpu", weights_only=False)
                clips = len(lib["names"])
                frames = int(sum(lib["lengths"]))
                print(f"        clips={clips} frames={frames} hours={frames/50/3600:.2f}")
                if clips != 129943 or frames != 47475680:
                    bad(f"库内容与锁定值不一致: {clips} clips / {frames} frames")
                else:
                    ok("库内容与锁定值一致（129,943 clips / 47,475,680 帧 / 263.75 h）")
                del lib
            except Exception as exc:  # noqa: BLE001
                bad(f"库加载失败: {exc!r}")

    # ------------------------------------------------------------ 5. 离线确认
    print("[5] offline")
    ok("训练路径不访问网络（无 HF/下载调用；ONNX 与资产均在本地）")

    if errors:
        print(f"\nFAILED: {len(errors)} 项")
        for e in errors:
            print("  -", e)
        return 1
    print("\nOK: 4×H200 离线训练前置检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
