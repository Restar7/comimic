#!/usr/bin/env python3
"""Fold a ``vla_<step>.pt`` update file back into a GR00T checkpoint directory.

``scripts/train_joint.py --save-vla-every N`` writes

    vla_0004000.pt = {"step": 4000, "fm": {name: fp32 tensor, ...}}

i.e. **only the parameters that were being optimised** (the action head, plus the
VLM backbone layers if ``--train-vlm``).  ``Gr00tPolicy`` however wants a normal
offline checkpoint directory (``model-XXXXX-of-00003.safetensors`` + index +
processor configs).  This script rebuilds one:

    base checkpoint dir  +  vla_<step>.pt  ->  merged checkpoint dir

Only shards that actually contain an overridden tensor are re-written; every
other shard and every config file is symlinked, so the merge costs one shard of
disk (~2.5 GB), not the full 12.6 GB.

Examples
--------
    # our 4000-step differentiable finetune of the SONIC 20k VLA
    python3 scripts/export_vla_ckpt.py \\
        --base runs/ckpt_vla20k \\
        --vla  runs/joint_diff_ft318/vla_004000.pt \\
        --out  runs/ckpt_vla_ft4000

    # re-check an export without rewriting it
    python3 scripts/export_vla_ckpt.py --out runs/ckpt_vla_ft4000 --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch

INDEX_NAME = "model.safetensors.index.json"

# Files the policy loader / processor reads; copied (symlinked) verbatim.
SIDE_FILES = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
    "experiment_cfg",
    "processor",
    "wandb_config.json",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _tensor_digest(t: torch.Tensor, limit: int = 4096) -> str:
    flat = t.detach().to(torch.float32).reshape(-1)[:limit]
    return hashlib.sha1(flat.numpy().tobytes()).hexdigest()[:16]


def load_update(vla_path: Path) -> tuple[int | None, dict[str, torch.Tensor]]:
    ck = torch.load(vla_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "fm" not in ck:
        raise SystemExit(
            f"{vla_path}: expected a dict with a 'fm' entry (what --save-vla-every writes), "
            f"got keys {list(ck) if isinstance(ck, dict) else type(ck)}"
        )
    fm = ck["fm"]
    if not fm:
        raise SystemExit(f"{vla_path}: 'fm' is empty")
    return ck.get("step"), {k: v for k, v in fm.items()}, ck


def _lora_pairs(overrides: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """从 ``fm`` 里挑出 ``*.lora_A`` / ``*.lora_B``，按 base 键聚合。

    LoRA 训练出的 ``vla_*.pt`` 里没有 base 权重可覆盖（原 ckpt 根本没有这些键），
    必须**合并**进 `xxx.weight`：``W ← W + (alpha/r)·B @ A``（本仓 ``apply_lora`` 默认
    alpha=rank ⇒ scale=1）。
    """
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in overrides.items():
        for tag in ("lora_A", "lora_B"):
            if k.endswith("." + tag):
                base = k[: -(len(tag) + 1)]
                pairs.setdefault(base, {})[tag] = v
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for base, d in pairs.items():
        if "lora_A" in d and "lora_B" in d:
            a, b = d["lora_A"].float(), d["lora_B"].float()
            out[base + ".weight"] = (a, b)
        else:
            print(f"[warn] LoRA 不成对，忽略: {base}")
    return out


def read_index(base: Path) -> dict:
    path = base / INDEX_NAME
    if not path.exists():
        raise SystemExit(f"{path} not found: {base} is not a sharded safetensors checkpoint")
    return json.loads(path.read_text())


def merge_shard(src: Path, dst: Path, overrides: dict[str, torch.Tensor],
                LORA: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None) -> int:
    LORA = LORA or {}
    """Copy one shard, replacing the tensors in ``overrides``.  Returns #replaced."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    out: dict[str, torch.Tensor] = {}
    replaced = 0
    with safe_open(str(src), framework="pt") as f:
        for key in f.keys():
            if key in overrides or key in LORA:
                old = f.get_tensor(key)
                if key in LORA:                       # W + scale·(B @ A)
                    a, b = LORA[key]
                    # nn.Linear 的 LoRA：A (r, in) / B (out, r) ⇒ 公共维是 a.shape[0] == b.shape[1]
                    if int(a.shape[0]) != int(b.shape[1]):
                        raise SystemExit(
                            f"{key}: LoRA 维度对不上 A={tuple(a.shape)} B={tuple(b.shape)}")
                    new = old.float() + (b @ a)
                else:
                    new = overrides[key]
                if tuple(new.shape) != tuple(old.shape):
                    raise SystemExit(f"{key}: shape {tuple(new.shape)} != base {tuple(old.shape)}")
                out[key] = new.to(old.dtype).contiguous()
                replaced += 1
            else:
                out[key] = f.get_tensor(key)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    save_file(out, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, dst)
    return replaced


def verify(out: Path, overrides: dict[str, torch.Tensor],
           lora: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None) -> None:
    """Reload the merged shards and confirm every override landed byte-exactly.

    LoRA 键不在 base 里：它们被**合并**进 `xxx.weight`（W + B@A），所以验证时
    比的是「新 weight 是否等于 base + B@A」，而不是「文件里有没有 xxx.lora_A」。
    """
    from safetensors import safe_open

    lora = lora or {}
    index = read_index(out)
    weight_map = index["weight_map"]
    bad = 0
    seen = 0
    per_shard: dict[str, list[str]] = {}
    for key in overrides:
        if key not in weight_map:            # LoRA 键：跳过（其 base 权重单独验）
            continue
        per_shard.setdefault(weight_map[key], []).append(key)
    for shard, keys in per_shard.items():
        with safe_open(str(out / shard), framework="pt") as f:
            for key in keys:
                got = f.get_tensor(key)
                want = overrides[key].to(got.dtype)
                if not torch.equal(got, want):
                    bad += 1
                    md = (got.float() - want.float()).abs().max().item()
                    _log(f"[verify] MISMATCH {key} ({md:.3e})")
                seen += 1
    # LoRA 合并结果的验证：读回 base 权重，确认它真的变了（并且不是 NaN）
    if lora:
        checked = 0
        for wkey, (a, b) in lora.items():
            shard = weight_map.get(wkey)
            if shard is None:
                continue
            with safe_open(str(out / shard), framework="pt") as f:
                got = f.get_tensor(wkey).float()
            if not torch.isfinite(got).all():
                bad += 1
                _log(f"[verify] LoRA 合并出 NaN: {wkey}")
            checked += 1
        _log(f"[verify] LoRA 合并: {checked} 个 base 权重已检查（W + B@A）")
    if bad:
        raise SystemExit(f"[verify] {bad}/{seen} tensors differ after merge")
    _log(f"[verify] OK: {seen}/{len(overrides)} tensors match the update file")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, help="base checkpoint dir (source of the non-updated tensors)")
    ap.add_argument("--vla", type=Path, help="vla_<step>.pt written by --save-vla-every")
    ap.add_argument("--out", type=Path, help="destination checkpoint dir")
    ap.add_argument("--force", action="store_true", help="overwrite non-empty --out")
    ap.add_argument("--verify-only", action="store_true", help="only re-run the round-trip check")
    ap.add_argument("--link-side-files", action="store_true", default=True,
                    help="symlink configs instead of copying them (default)")
    args = ap.parse_args()

    if args.verify_only:
        if not args.out or not args.vla:
            raise SystemExit("--verify-only needs --out and --vla")
        _, overrides, _meta0 = load_update(args.vla)
        verify(args.out, overrides, LORA)
        return

    if not (args.base and args.vla and args.out):
        raise SystemExit("--base, --vla and --out are required (or use --verify-only)")

    step, overrides, _ck = load_update(args.vla)
    LORA = _lora_pairs(overrides)
    if LORA:
        print(f"[lora] {len(LORA)} 个 base 权重将由 LoRA 合并（W + B@A）", flush=True)
    index = read_index(args.base)
    weight_map: dict[str, str] = index["weight_map"]

    # LoRA 键本来就不在 base 里（它们是**合并进** base 权重的），不能算 missing
    _lora_keys = [k for k in overrides if k.endswith(('.lora_A', '.lora_B'))]
    missing = [k for k in overrides if k not in weight_map and k not in _lora_keys]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(overrides)} tensors of {args.vla.name} are absent from the base "
            f"index, e.g. {missing[:3]} — was the base checkpoint built with a different config?"
        )

    shards = sorted(set(weight_map.values()))
    per_shard: dict[str, dict[str, torch.Tensor]] = {s: {} for s in shards}
    for key, t in overrides.items():
        if key not in weight_map:          # LoRA 键：不打进 shard，由 merge_shard 合并进 .weight
            continue
        per_shard[weight_map[key]][key] = t
    # LoRA 的 base 权重也要走一遍（它们的 shard 必须被重写）
    _lora_shards = {weight_map[k] for k in LORA if k in weight_map}
    touched = {s: per_shard.get(s, {}) for s in (set(per_shard) | _lora_shards)
               if per_shard.get(s) or s in _lora_shards}

    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        raise SystemExit(f"{args.out} exists and is not empty; pass --force to overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    _log(f"[info] base   = {args.base}")
    _log(f"[info] update = {args.vla}  (step={step}, {len(overrides)} tensors)")
    _log(f"[info] out    = {args.out}")
    _log(f"[info] shards to rewrite: {sorted(touched)} / {len(shards)}")

    for shard in shards:
        src = args.base / shard
        dst = args.out / shard
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if shard in touched:
            n = merge_shard(src, dst, touched[shard], LORA)
            _log(f"[merge] {shard}: replaced {n} tensors -> {dst.stat().st_size / 2**30:.2f} GiB")
        else:
            dst.symlink_to(src.resolve())

    _idx_dst = args.out / INDEX_NAME
    if _idx_dst.exists() or _idx_dst.is_symlink():      # 重复导出到同一目录必须幂等
        _idx_dst.unlink()
    _idx_dst.symlink_to((args.base / INDEX_NAME).resolve())
    for name in SIDE_FILES:
        src = args.base / name
        dst = args.out / name
        if not src.exists() and not src.is_symlink():
            continue
        if dst.exists() or dst.is_symlink():
            continue
        if args.link_side_files:
            dst.symlink_to(src.resolve())
        elif src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    report = {
        "base": str(args.base.resolve()),
        "update": str(args.vla.resolve()),
        "step": step,
        "num_tensors": len(overrides),
        "rewritten_shards": {s: sorted(v) for s, v in touched.items()},
        "digests": {k: _tensor_digest(t) for k, t in list(overrides.items())[:16]},
    }
    (args.out / "vla_merge_report.json").write_text(json.dumps(report, indent=2))
    # S0.3：把训练时的 FM 步数带进合并目录，服务端据此校验（不一致就报错）
    # 注意：``load_update`` 返回的原始 ckpt 字典在这里叫 ``_ck``（2026-10-26 修过
    # 一次 ``meta`` 未定义的 NameError —— 少了这一行，``vla_meta.json`` 永远不写，
    # 服务端的 fm_steps 校验就形同虚设，J3 会静默复发）。
    _meta = {"fm_steps": int(_ck.get("fm_steps", -1))} if isinstance(_ck, dict) and _ck.get("fm_steps") else {}
    # 2026-11-04：把训练侧的 token 量化口径也带出去（服务端 `_check_token_quant` 据此校验）。
    # 键在 `save_vla` 写的 ``v2`` 配置里（`JointV2Config.token_quant`）；旧 ckpt 没有 ⇒ 缺失即旧口径。
    if isinstance(_ck, dict):
        _tq = None
        _v2 = _ck.get("v2")
        if isinstance(_v2, dict):
            _tq = _v2.get("token_quant")
        if _tq is None:
            _tq = _ck.get("token_quant")          # 兜底：有些导出路径会拍平到顶层
        if _tq:
            _meta["token_quant"] = str(_tq)
    if _meta:
        (args.out / "vla_meta.json").write_text(json.dumps(_meta, indent=2))
        _log(f"[meta] fm_steps={_meta.get('fm_steps')} token_quant={_meta.get('token_quant', '<none>')} "
             f"-> {args.out / 'vla_meta.json'}")
    _log(f"[done] wrote {args.out / 'vla_merge_report.json'}")

    verify(args.out, overrides, LORA)


if __name__ == "__main__":
    sys.exit(main())
