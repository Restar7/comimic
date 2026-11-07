#!/usr/bin/env python3
"""Δtoken 部署口径验证：**真实 checkpoint + 真实线上 state64** 上的数值证据。

它回答四个问题（每一步都打印数字，不靠"应该没问题"）：

1. 真实 ckpt 的 ``feat_dim`` 到底是多少？（现在训练器**不写** metadata，只能反推）
2. 用**旧加载路径**（写死 35 维 / ``strict=False``）会发生什么？——必须失败，且失败信息可读。
3. 关节序置换对 Δ 的影响有多大？（cos(Δ_correct, Δ_wrong)、‖Δ‖ 比）
4. ``Δ=0`` 时服务端 chunk 是否与 base 逐位相同？（cap=0 且末层近零初始化时）

用法::

    python3 scripts/verify_delta_token_deploy.py \
        --arms bptt1=runs/bptt1/vla_000050.pt gt7fix=runs/gt7fix/vla_final.pt \
        --state64-npz data/ha_raw/HOI_double_desk/tw/<ep>.npz \
        --out runs/harness_01_deploy/delta_token_verify.json

``--state64-npz`` 缺省时自动挑一个 ``data/ha_raw`` 里的录制（其中的 ``vla_state``
就是**线上真实发出来的 64 维**）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
#: 数据/权重根：worktree 里没有（gitignore 的）runs/data，用 --root 指到原仓库
DATA_ROOT = Path(os.environ.get("COMIMIC_DATA_ROOT", str(ROOT)))

from diff_vla_wbc.delta_token import FEAT_DIM, STATE64_DIM, DeltaTokenNet, apply_norm_cap  # noqa: E402
from diff_vla_wbc import delta_token_io as DIO  # noqa: E402


def _pick_state64_npz(pattern: str, root: Path) -> Path | None:
    files = sorted(glob.glob(str(root / pattern), recursive=True))
    return Path(files[0]) if files else None


def _load_state64(path: Path | None, n: int, seed: int = 0) -> np.ndarray:
    if path is not None and path.exists():
        z = np.load(path, allow_pickle=True)
        s = np.asarray(z["vla_state"], dtype=np.float32)
        idx = np.linspace(0, len(s) - 1, min(n, len(s))).astype(int)
        return s[idx], {"source": str(path), "kind": "ha_raw.vla_state", "rows": int(len(s))}
    rng = np.random.default_rng(seed)
    s = np.concatenate([
        rng.normal(size=(n, 6)) * 0.3, rng.normal(size=(n, 29)) * 0.4,
        rng.normal(size=(n, 29)) * 0.8], axis=1).astype(np.float32)
    return s, {"source": "synthetic", "kind": "gaussian", "rows": int(n)}


def _old_server_path(sd: dict, meta: dict) -> tuple[str, str]:
    """复刻 2026-11-07 之前服务端/探针的加载写法，用来证明它确实会失败。"""
    try:
        net = DeltaTokenNet(hidden=int(meta.get("hidden", 512) or 512),
                            depth=int(meta.get("depth", 2) or 2))
        net.load_state_dict(sd, strict=False)
        return "LOADED", f"params={net.n_params}"
    except Exception as exc:  # noqa: BLE001
        return "FAILED", f"{type(exc).__name__}: {exc}"


def verify_arm(name: str, path: Path, state64: np.ndarray, *, token_seed: int = 0) -> dict:
    import torch

    d = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = d.get("delta_token_state")
    meta = d.get("delta_token_meta") or {}
    out: dict = {"arm": name, "ckpt": str(path), "ckpt_sha256": DIO.sha256_file(path)}
    if sd is None:
        out["status"] = "NO_DELTA"
        return out

    out["meta_raw"] = dict(meta)
    out["weights_sha256"] = DIO.ckpt_sha256_of_tensor_dict(sd)
    out["first_layer"], out["n_layers"] = list(sd["net.0.weight"].shape), len(
        [k for k in sd if k.endswith(".weight") and k.startswith("net.")])

    # ---- 1) 新加载器（严格）----
    net, spec = DIO.load_delta_token_from_checkpoint(d)
    out["spec"] = spec.as_dict()
    out["spec"]["n_params"] = net.n_params

    # ---- 2) 旧加载路径 ----
    st, msg = _old_server_path(sd, meta)
    out["old_loader"] = {"status": st, "detail": msg}

    # ---- 3) 特征 / Δ 数值 ----
    rng = np.random.default_rng(token_seed)
    toks = rng.normal(size=(len(state64), 64)).astype(np.float32)
    feats = np.stack([DIO.delta_features_from_state64(s, spec) for s in state64])
    deltas = np.stack([DIO.delta_from_token_chunk(toks[i], feats[i], net, spec)
                       for i in range(len(state64))])

    # 错误口径：直接把 state64 当特征（不置换）、以及旧的 35 维切片
    feats_wrong = state64[:, :64] if spec.feat_dim == 64 else np.concatenate(
        [state64[:, :6], state64[:, 35:64]], axis=1)
    deltas_wrong = np.stack([DIO.delta_from_token_chunk(toks[i], feats_wrong[i], net, spec)
                             for i in range(len(state64))])

    def _cos(a, b):
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        return (a * b).sum(1) / np.clip(na * nb, 1e-12, None)

    cos_w = _cos(deltas, deltas_wrong)
    feat_abs = np.abs(feats - feats_wrong)
    out["numeric"] = {
        "n_samples": int(len(state64)),
        "delta_norm_mean": float(np.linalg.norm(deltas, axis=1).mean()),
        "delta_norm_p95": float(np.percentile(np.linalg.norm(deltas, axis=1), 95)),
        "feat_abs_diff_max": float(feat_abs.max()),
        "feat_abs_diff_mean": float(feat_abs.mean()),
        "cos_correct_vs_wrong_order_mean": float(cos_w.mean()),
        "cos_correct_vs_wrong_order_p05": float(np.percentile(cos_w, 5)),
        "norms_ratio_mean": float((np.linalg.norm(deltas_wrong, axis=1)
                                   / np.clip(np.linalg.norm(deltas, axis=1), 1e-12, None)).mean()),
    }
    # 有限差分：Δ 对 token 的敏感度（同一 state、两个邻近 token）
    eps = 1e-3
    d1 = DIO.delta_from_token_chunk(toks[0], feats[0], net, spec)
    d2 = DIO.delta_from_token_chunk(toks[0] + eps, feats[0], net, spec)
    out["numeric"]["d_delta_d_token_rms"] = float(np.sqrt(((d2 - d1) ** 2).mean()) / eps)

    # ---- 4) Δ=0 ⇒ 逐位一致（只有 cap=0 且权重近零时成立，报告实际值）----
    chunk = toks[0].reshape(1, 64).repeat(40, 0).copy()
    applied = (chunk + deltas[0][None, :]).astype(np.float32)
    out["identity_when_zero_delta"] = {
        "chunk_absmax_change": float(np.abs(applied - chunk).max()),
        "exact_equal": bool(np.array_equal(applied, chunk)) if spec.cap == 0 else None,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="名字=ckpt 路径")
    ap.add_argument("--state64-npz", default=(
        "data/ha_raw/HOI_double_desk/*/Isaac-Move-PickPlace-DoubleDesk-*.npz"))
    ap.add_argument("--n-state", type=int, default=32)
    ap.add_argument("--out", default="runs/harness_01_deploy/delta_token_verify.json")
    ap.add_argument("--root", default=str(DATA_ROOT),
                    help="数据/权重根（worktree 里没有 gitignore 的 runs/data 时指到原仓库）")
    a = ap.parse_args()

    root = Path(a.root)
    npz = _pick_state64_npz(a.state64_npz, root)
    state64, state_meta = _load_state64(npz, a.n_state)
    print(f"[verify] state64 来源={state_meta['source']} rows={state_meta['rows']} "
          f"用 {len(state64)} 条", flush=True)
    print(f"[verify] state64[0] = {np.round(state64[0], 4).tolist()}", flush=True)

    results = []
    for spec in a.arms:
        name, _, rel = spec.partition("=")
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            print(f"[verify] {name}: MISSING {path}", flush=True)
            results.append({"arm": name, "ckpt": str(path), "status": "MISSING"})
            continue
        r = verify_arm(name, path, state64)
        results.append(r)
        if r.get("status") == "NO_DELTA":
            print(f"[verify] {name}: ckpt 里没有 delta_token_state", flush=True)
            continue
        s, o = r["spec"], r["old_loader"]
        print(f"\n[verify] === {name} ===\n  ckpt={r['ckpt']}\n  sha256={r['ckpt_sha256'][:32]}…"
              f"\n  delta_sha256={r['weights_sha256'][:32]}…\n  first_layer={r['first_layer']}"
              f" layers={r['n_layers']}", flush=True)
        print(f"  ⇒ 推断 feat_dim={s['feat_dim']} ({s['feat_layout']}) token_dim={s['token_dim']} "
              f"hidden={s['hidden']} depth={s['depth']} cap={s['cap']:g} "
              f"feat_joint_order={s['feat_joint_order']} params={s['n_params']}", flush=True)
        print(f"  sources={s['sources']}", flush=True)
        print(f"  旧加载路径（写死 35 维/默认 hidden）: {o['status']} — {o['detail']}", flush=True)
        n = r["numeric"]
        print(f"  ‖Δ‖ 均值={n['delta_norm_mean']:.5f} p95={n['delta_norm_p95']:.5f}", flush=True)
        print(f"  特征 |Δ|: max={n['feat_abs_diff_max']:.4f} mean={n['feat_abs_diff_mean']:.4f}"
              f"（correct vs 不置换）", flush=True)
        print(f"  cos(Δ_correct, Δ_wrong_order) = {n['cos_correct_vs_wrong_order_mean']:+.4f}"
              f"（p05={n['cos_correct_vs_wrong_order_p05']:+.4f}）", flush=True)
        print(f"  ‖Δ_wrong‖/‖Δ_correct‖ = {n['norms_ratio_mean']:.4f}", flush=True)
        print(f"  dΔ/dtoken RMS = {n['d_delta_d_token_rms']:.6f}", flush=True)

    outp = Path(a.out)
    if not outp.is_absolute():
        outp = ROOT / outp
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"state64": state_meta, "arms": results}, indent=1,
                               ensure_ascii=False), encoding="utf-8")
    print(f"\n[verify] wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
