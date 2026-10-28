#!/usr/bin/env python3
"""把一条 clip 的**三臂 token 流**落盘，供 Isaac 回放（真物理裁判）。

三个臂：
  * ``gt``     —— npz 的 ``g1_motion_token_64``（训练标签，S0.4 确认 = 录制 encoder_latent）
  * ``oracle`` —— ``runs/oracle_tokens/<clip>.npy``（在 WM 里对 token 做梯度上升得到的，已存在）
  * ``vla``    —— 用录制帧离线跑我们的 VLA（需要 ``--vla-ckpt`` + ``--head-ckpt``；纯推理）

产物：``<out>/<clip>/{gt,oracle,vla}.npy``（(T,64) float32）+ ``hands.npy``（(T,14)，三臂共用）
      + ``<arm>.json`` 元数据（来源、md5、长度、token 统计）。

设计要点（口径，不可变）：
  1. **手部三臂共用同一份**（录制 ``g1_hand_joints_14``）—— 否则比的是手，不是 token；
  2. token 一律取 64 维、按**帧序**排列（Isaac 里按 ``HA_VLA_EXEC_STEPS=20`` 推进光标）；
  3. 三个臂的数组长度必须相同（= clip 帧数），不等就报错而不是截断。

用法：
    python3 scripts/dump_arm_tokens.py \
        --clips HOI_double_desk__tw__ep000000 HOI_pp_box__yb__ep000000 \
        --out runs/arm_tokens
    # 只做 gt/oracle（不需要 VLA，先把 WM 排序的真物理判定跑起来）
    python3 scripts/dump_arm_tokens.py --arms gt oracle --clips ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _npz_path(clip: str) -> Path:
    p = ROOT / "data/npz/ha_tokens" / f"{clip}.npz"
    if not p.exists():
        raise SystemExit(f"[dump] 找不到 npz: {p}")
    return p


def _load_gt(clip: str) -> tuple[np.ndarray, np.ndarray]:
    p = _npz_path(clip)
    with np.load(p) as z:
        tok = np.asarray(z["g1_motion_token_64"], dtype=np.float32)
        hands = np.asarray(z["g1_hand_joints_14"], dtype=np.float32) if "g1_hand_joints_14" in z else None
    if tok.ndim != 2 or tok.shape[1] != 64:
        raise SystemExit(f"[dump] {clip}: g1_motion_token_64 形状异常 {tok.shape}")
    return tok, hands


def _load_oracle(clip: str, t_expect: int) -> np.ndarray:
    cand = [ROOT / "runs/oracle_tokens" / f"{clip}.npy"]
    for p in cand:
        if p.exists():
            a = np.load(p).astype(np.float32)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if a.shape[0] != t_expect:
                raise SystemExit(
                    f"[dump] {clip}: oracle 长度 {a.shape[0]} != 帧数 {t_expect}（不做截断，先查清）")
            if a.shape[1] != 64:
                raise SystemExit(f"[dump] {clip}: oracle token 维度 {a.shape[1]} != 64")
            return a
    raise SystemExit(f"[dump] {clip}: 没有 oracle token（先跑 scripts/build_oracle_tokens.py）")


def _load_vla(clip: str, t_expect: int, *, vlm_ckpt: Path,
              head_ckpt: Path | None, device: str) -> np.ndarray:
    """用录制帧离线跑一次 VLA，得到该 clip 每个录制帧的 64 维 token。"""
    import torch  # noqa: PLC0415

    from diff_vla_wbc.action_codec import ActionCodec  # noqa: PLC0415
    from diff_vla_wbc.g1_tasks import (  # noqa: PLC0415
        EgoViewImages, TaskFrameIndex, TaskObservationBuilder, load_gr00t_processor,
        processed_batch,
    )
    from diff_vla_wbc.gr00t_vlm import Gr00tVLM  # noqa: PLC0415

    npz_dir = ROOT / "data/npz/ha_tokens"
    index = TaskFrameIndex(sorted(npz_dir.glob("*.npz")))
    builder = TaskObservationBuilder(index, images=EgoViewImages())
    # 该 clip 在全局帧序里的起点/长度
    # TaskFrameIndex 的公开属性：files / starts / num_frames（见 diff_vla_wbc/g1_tasks.py）
    starts = np.asarray(getattr(index, "starts", getattr(index, "_starts", None)), dtype=np.int64)
    names = [Path(p).stem for p in index.files]
    if clip not in names:
        raise SystemExit(f"[dump] {clip} 不在索引里（{len(names)} 个片段）")
    ci = names.index(clip)
    f0 = int(starts[ci])
    frames = list(range(f0, f0 + t_expect))

    processor = load_gr00t_processor(vlm_ckpt, vlm_path=ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights")
    codec = ActionCodec(processor, device=device)
    vlm = Gr00tVLM.from_pretrained(
        ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights", checkpoint=vlm_ckpt,
        select_layer=16, device=device, action_horizon=40, max_state_dim=132,
        freeze_vlm=True, attn_implementation="sdpa")
    if head_ckpt is not None:
        # 覆盖 action head 权重（S3 训练产出的 joint_*.pt）；None = 用底模自带的那份
        _raw = torch.load(str(head_ckpt), map_location="cpu", weights_only=False)
        sd = _raw.get("head", _raw.get("model", _raw)) if isinstance(_raw, dict) else _raw
        missing, unexpected = vlm.action_head.load_state_dict(sd, strict=False)
        print(f"[dump] head override {head_ckpt.name}: missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
    else:
        print("[dump] head = 底模自带（未覆盖）", flush=True)
    vlm.action_head.eval()

    from diff_vla_wbc.fm_bc import sample_once_with_grad  # noqa: PLC0415
    out = []
    bs = 8
    for lo in range(0, len(frames), bs):
        chunk = frames[lo:lo + bs]
        proc = processed_batch(processor, builder, chunk)
        for _k in list(proc.keys()):        # 原地搬设备，保持 BatchFeature/自定义类型
            if torch.is_tensor(proc[_k]):
                proc[_k] = proc[_k].to(device)
        # 注意：不能包 torch.no_grad()——`sample_once_with_grad` 有显式守卫
        # （"called under torch.no_grad() … the joint training graph would be cut"）。
        # 这里是纯推理，梯度算完即丢；VLM backbone 是冻结的，显存开销可接受。
        bb = vlm.backbone(proc)
        x = sample_once_with_grad(
            vlm.action_head, bb, proc,
            initial_noise=torch.randn(len(chunk), vlm.action_head.action_horizon,
                                      vlm.action_head.action_dim, device=device,
                                      dtype=vlm.action_head.dtype),
            num_steps=4, grad_backend="pathwise", truncated_tail=1)
        with torch.no_grad():
            raw = codec.decode(x.detach())
        tok = raw[..., :64]
        # 每个样本的 action chunk 取第 0 帧（与该录制帧对齐）
        out.append(tok[:, 0].detach().float().cpu().numpy())
        print(f"[dump] {clip}: {lo + len(chunk)}/{len(frames)}", flush=True)
    a = np.concatenate(out, axis=0).astype(np.float32)
    if a.shape != (t_expect, 64):
        raise SystemExit(f"[dump] vla token 形状 {a.shape} != {(t_expect, 64)}")
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["gt", "oracle"], choices=["gt", "oracle", "vla"])
    ap.add_argument("--out", type=Path, default=ROOT / "runs/arm_tokens")
    ap.add_argument("--vla-ckpt", type=Path,
                    default=ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000")
    ap.add_argument("--head-ckpt", type=Path, default=None,
                    help="vla 臂用：训练产出的 vla_*.pt（action head 权重）")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for clip in a.clips:
        gt, hands = _load_gt(clip)
        T = int(gt.shape[0])
        d = a.out / clip
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "hands.npy", hands if hands is not None else np.zeros((T, 14), np.float32))
        info = {"clip": clip, "T": T, "arms": {}}
        for arm in a.arms:
            if arm == "gt":
                arr, src = gt, f"npz:{_npz_path(clip)}"
            elif arm == "oracle":
                arr, src = _load_oracle(clip, T), f"oracle_tokens/{clip}.npy"
            else:
                # head_ckpt = None 时用底模自带的 action head（P0-2 的"底模"对照臂）
                arr, src = _load_vla(clip, T, vlm_ckpt=a.vla_ckpt, head_ckpt=a.head_ckpt,
                                     device=a.device), f"vla:{a.vla_ckpt}+{a.head_ckpt}"
            if arr.shape != (T, 64):
                raise SystemExit(f"[dump] {clip}/{arm}: 形状 {arr.shape} != {(T, 64)}")
            np.save(d / f"{arm}.npy", arr)
            meta = {"arm": arm, "src": src, "shape": list(arr.shape),
                    "mean": float(arr.mean()), "std": float(arr.std()),
                    "absmax": float(np.abs(arr).max()),
                    "md5": hashlib.md5(arr.tobytes()).hexdigest()[:16]}
            (d / f"{arm}.json").write_text(json.dumps(meta, indent=1))
            info["arms"][arm] = meta
            print(f"[dump] {clip}/{arm}: {arr.shape} mean={meta['mean']:+.4f} "
                  f"std={meta['std']:.4f} |{meta['absmax']:.3f}|max md5={meta['md5']}", flush=True)
        # 臂间一致性检查（同帧同长度，只差 token）
        if "gt" in a.arms and "oracle" in a.arms:
            dlt = float(np.abs(np.load(d / "gt.npy") - np.load(d / "oracle.npy")).mean())
            info["gt_vs_oracle_mad"] = dlt
            print(f"[dump] {clip}: |gt−oracle| 平均 {dlt:.4f}", flush=True)
        manifest[clip] = info

    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[dump] manifest -> {a.out / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
