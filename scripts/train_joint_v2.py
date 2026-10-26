#!/usr/bin/env python3
"""S3 阶段 C：联合微调 **v2** 入口（旧入口 ``scripts/train_joint.py`` 保持不动，便于对照复现）。

对应 ``docs/JOINT_FINETUNE_REVIEW_REDESIGN_V2_20261026.md`` §4 的 S3.1–S3.7：

    S3.1  新入口就是本文件（旧脚本不删、不改）
    S3.2  WBC / WM 完全冻结（``diff_vla_wbc.train_v2.JointTrainerV2``）；保存时逐张量比对门禁
    S3.3  每步单次采样 + GR00T 原生 FM-BC（``diff_vla_wbc.fm_bc``）；无 token_anchor、无 hand-MSE
    S3.4  起点 = 扰动 + 冻结 WBC 的 burn-in（``diff_vla_wbc.burnin``）
    S3.5  H=16，回报按 ``H·MAX_STEP_REWARD`` 归一化
    S3.6  ``--batch-size 4 --grad-accum 4``，每 ``--grad-ratio-every`` 步打印梯度比
    S3.7  固定 64 窗口评测（L_fm(val) / 想象回报 / decode 后 MSE），闭环挑 ckpt 见
          ``scripts/mini_closedloop_v2.sh``

用法见 ``scripts/launch_joint_ha_v2.sh``。最小 smoke：

    python3 scripts/train_joint_v2.py --steps 3 --batch-size 2 --grad-accum 1 \
        --eval-every 0 --save-every 0 --out runs/s3_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# S0.1 / J1：执行器 spec 必须在**任何** diff_vla_wbc / trackerLab import 之前锁定。
# WBC v2 是 HA_ALIGN=1 训的（knee 0.420 / ankle −0.230 / elbow 0.870，scale 0.3789/0.75）；
# 不设这两个环境变量会退回 BeyondMimic，WM 想象里的 (q−default_q) 与 a_prev 全错。
# ---------------------------------------------------------------------------------------------
import os as _os  # noqa: E402

_os.environ.setdefault("SUPERMIMIC_HA_ALIGN", "1")
_os.environ.setdefault(
    "SUPERMIMIC_ROBOT_USD",
    str(Path(__file__).resolve().parents[1]
        / "trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"),
)

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.action_codec import ActionCodec  # noqa: E402
from diff_vla_wbc.data import RecordedTokenSource  # noqa: E402
from diff_vla_wbc.g1_tasks import (  # noqa: E402
    TaskFrameIndex,
    TaskObservationBuilder,
    load_gr00t_processor,
    processed_batch,
)
from diff_vla_wbc.prefetch import BatchPrefetcher, StepMemo  # noqa: E402
from diff_vla_wbc.train_v2 import (  # noqa: E402
    BurnInBatchProvider,
    JointTrainerV2,
    JointV2Config,
    assert_wbc_frozen,
    fixed_eval,
)

HA_MOTION_DIR = ROOT / "data/npz/ha_tokens"
HA_PORT_DIR = ROOT / "data/port_tokens/ha_tokens"
PACKED_MOTION = ROOT / "runs/isaac_motion/ha_tokens/packed.pt"      # 旧：执行轨迹（端口/future 时代）
PACKED_CMD_MOTION = ROOT / "runs/isaac_motion/ha_cmd/packed.pt"    # 新：命令流（hold/label 时代）
FORBIDDEN_DATA_TOKENS = ("mixed", "ft318")
#: S3 的默认执行器：WBC v2（obs93 + 混合库共训，见 docs/S1_WBC_NEXT_PLAN_20261026.md）
WBC_V2_CKPT = ROOT / "trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_016000_spec.pt"  # 端口/future（作废）
#: hold/label 空间的 WBC（S1 正在训；`logs/wbc_hold_x20/`）。存在就用它。
WBC_HOLD_GLOB = "trackerlab-supermimic/logs/wbc_hold*/ckpt_*.pt"
WBC_V1_CKPT = ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint_spec.pt"


def _forbid_legacy_dataset(paths, flag: str) -> None:
    for p in paths:
        text = str(p).lower()
        bad = [t for t in FORBIDDEN_DATA_TOKENS if t in text]
        if bad:
            raise SystemExit(f"{flag} {p}: 含禁用数据集标记 {bad}；本工程永久只走纯 HA 数据。")


def _sha256_head(path: Path, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def _discover_hold_wbc(root: Path) -> Path | None:
    """找最新的 hold 空间 WBC（`logs/wbc_hold*/ckpt_*.pt`，优先带 robot_spec 的）。"""
    cands = []
    for pat in ("trackerlab-supermimic/logs/wbc_hold*/ckpt_*.pt",):
        cands.extend(root.glob(pat))
    if not cands:
        return None

    def key(path: Path):
        stem = path.stem
        it = 0
        for tok in stem.split("_"):
            if tok.isdigit():
                it = max(it, int(tok))
        return (path.name.endswith("_spec.pt"), it, path.stat().st_mtime)

    return sorted(cands, key=key)[-1]


def _wbc_window_of(ckpt: Path) -> tuple[str, str]:
    """读 WBC ckpt 的 token 窗口口径 → (window, library)。没有显式字段时按库名推断。"""
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    prov = ck.get("provenance") or {}
    a = ck.get("args") or {}
    lib = str(prov.get("library") or a.get("motion_packed") or "")
    win = str(prov.get("token_window") or a.get("token_window") or "").lower()
    if win in ("hold", "future"):
        return win, lib
    low = lib.lower()
    if "ha_cmd" in low:
        return "hold", lib          # 命令流库 = hold 前端建的
    if "ha_tokens" in low or "bones_seed" in low:
        return "unknown", lib
    return "unknown", lib


def collect_npz(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.npz")))
        elif p.suffix == ".npz":
            out.append(p)
        else:
            raise SystemExit(f"not an npz file or directory: {p}")
    if not out:
        raise SystemExit("no .npz motion clips found")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="联合微调 v2（S3）")
    # ---------------- 数据 / 权重 ----------------
    ap.add_argument("--motion", nargs="+", type=Path, default=[HA_MOTION_DIR])
    ap.add_argument("--port-tokens", type=Path, default=HA_PORT_DIR,
                    help="端口 GT token 侧车目录（仅 wbc-token-space=port 时用到）")
    ap.add_argument("--bc-token-space", choices=("label", "port"), default="label",
                    help="BC 目标的 token 空间。S0.4：npz 的 g1_motion_token_64 与录制 "
                         "encoder_latent 逐位相等，是 VLA 的原生空间 -> 默认 label；"
                         "port 只用于复现旧行为")
    ap.add_argument("--wbc-token-space", choices=("auto", "label", "port"), default="auto",
                    help="冻结 WBC 的训练空间（burn-in 的 conditioning 必须用它）。当前 "
                         "WBC_v2 是端口空间；S1 用标签空间重训后改成 label")
    ap.add_argument("--motion-packed", type=Path, default=PACKED_CMD_MOTION,
                    help="WBC 的参考库。S3 必须用**命令流**库（ha_cmd）：npz 的 joint_pos 是执行轨迹，"
                         "与命令流 mean|Δ|=0.41、lag +8 帧（交接文档 §4）。旧库只用于复现旧口径")
    ap.add_argument("--wbc-ckpt", type=Path, default=None,
                    help="冻结的 WBC（必须带 robot_spec；默认 WBC_v2 = ckpt_016000_spec.pt）")
    ap.add_argument("--vlm-ckpt", type=Path, required=True,
                    help="起始 VLA ckpt（官方 checkpoint-8000，或 S2 的 base_aligned）")
    ap.add_argument("--vlm-path", type=Path, default=ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights")
    ap.add_argument("--vlm-select-layer", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("runs/joint_v2_smoke"))
    # ---------------- 训练规模 ----------------
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    # ---------------- 损失 ----------------
    ap.add_argument("--horizon", type=int, default=16, help="想象 rollout 步数（S3.5：H=16）")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lambda-bc", type=float, default=1.0, help="FM-BC 主项权重（恒定，不退火）")
    ap.add_argument("--lambda-r", type=float, default=0.3, help="奖励项权重（扫 0 / 0.1 / 0.3 / 1.0）")
    ap.add_argument("--allow-token-space-mismatch", action="store_true",
                    help="显式放行 bc!=wbc 的 token 空间不匹配（只用于诊断；这种臂的数字不能进表）")
    ap.add_argument("--lambda-s", type=float, default=0.0, help="可选 state 项权重（小权重）")
    ap.add_argument("--lambda-a", type=float, default=0.0, help="可选动作蒸馏权重（WBC 冻结）")
    # ---------------- v3：端到端（token 不做任何标签监督）----------------
    ap.add_argument("--v3", action="store_true",
                    help="启用 v3 训练步：token 只由奖励(+critic)监督，手部单独 FM 监督")
    ap.add_argument("--lambda-hand", type=float, default=0.2,
                    help="v3: 手部 hand-only FM 损失权重（token 那 64 维不监督）")
    ap.add_argument("--critic", dest="critic", action="store_true", default=True)
    ap.add_argument("--no-critic", dest="critic", action="store_false")
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--push-std", type=float, default=0.0,
                    help="rollout 内过程扰动：每步给 root 线/角速度加 U(-std,std)")
    ap.add_argument("--start-source", choices=("reference", "recorded"), default="reference",
                    help="burn-in 起点：recorded = 真录制机器人状态（v3 推荐）")
    ap.add_argument("--real-transitions", type=Path,
                    default=ROOT / "runs/wm_real/ha_transitions.pt",
                    help="真转移数据集（scripts/build_real_transitions.py 生成）")
    ap.add_argument("--wm-refresh-every", type=int, default=0,
                    help="每 N 步用真转移刷新 WM（0 = 不刷新）")
    ap.add_argument("--wm-refresh-steps", type=int, default=200)
    ap.add_argument("--wm-refresh-lr", type=float, default=1e-4)
    ap.add_argument("--wm-kl-anchor", type=float, default=1e-3)
    ap.add_argument("--grad-log-every", type=int, default=0,
                    help="每 N 步打印分项梯度范数（奖励 vs 手部）")
    ap.add_argument("--clip-head", type=float, default=0.5,
                    help="VLA head 的梯度范数裁剪阈值。1.0 只适合小 MLP；1.6B head 要先测 p99 再定")
    ap.add_argument("--state-vel-weight", type=float, default=0.05)
    ap.add_argument("--fm-lr", type=float, default=1e-5)
    ap.add_argument("--fm-clip", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--fm-steps", type=int, default=4, help="采样 Euler 步数 K（= 服务端 --fm-steps）")
    ap.add_argument("--fm-grad-backend", default="pathwise",
                    choices=["pathwise", "checkpointed", "truncated", "adjoint"])
    ap.add_argument("--fm-truncated-tail", type=int, default=1)
    ap.add_argument("--head-train-mode", action="store_true",
                    help="让 action head 处于 train()（含 state dropout 0.2）；默认 eval（部署口径一致）")
    ap.add_argument("--grad-ratio-every", type=int, default=50)
    # ---------------- burn-in 起点（S3.4）----------------
    ap.add_argument("--burnin-steps", type=int, default=10)
    ap.add_argument("--no-burnin", action="store_true", help="对照臂：起点 = 参考帧（旧口径）")
    ap.add_argument("--no-perturb", action="store_true", help="burn-in 不加扰动（诊断用）")
    ap.add_argument("--perturb-xy", type=float, default=0.05)
    ap.add_argument("--perturb-yaw", type=float, default=0.10)
    ap.add_argument("--perturb-dq", type=float, default=0.05)
    ap.add_argument("--perturb-dqd", type=float, default=0.30)
    # ---------------- 评测 / 存档 ----------------
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=16, help="固定评测批次（×batch = 窗口数，S3.7 要 64）")
    ap.add_argument("--eval-out", type=Path, default=None, help="评测曲线 JSONL")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-vla-every", type=int, default=500)
    ap.add_argument("--prefetch-vla", dest="prefetch_vla", action="store_true", default=True)
    ap.add_argument("--no-prefetch-vla", dest="prefetch_vla", action="store_false")
    ap.add_argument("--time-budget-minutes", type=float, default=0.0)
    # ---------------- 门禁（S3 四道自检，不训练）----------------
    ap.add_argument("--gate-only", action="store_true",
                    help="只跑 codec / 冻结 / burn-in / FM-BC 四道门禁（写 gates.json 后退出）")
    ap.add_argument("--gate-out", type=Path, default=None)
    ap.add_argument("--wbc-token-window", choices=("hold", "future"), default="hold",
                    help="WBC 的 token 窗口口径。hold = HA 录制/部署口径（交接文档 §2）；"
                         "future = 旧端口空间，只用于复现旧结果")
    ap.add_argument("--wbc-token-source", choices=("online", "sidecar"), default="online",
                    help="burn-in/GT 臂的 token 来源：online = hold builder + 当时仿真机器人 anchor"
                         "（B 模式，交接文档 §5.3）；sidecar = 预计算侧车（旧端口空间）")
    ap.add_argument("--allow-window-mismatch", action="store_true",
                    help="放行 WBC ckpt 的 token 窗口口径与 --wbc-token-window 不一致（仅诊断）")
    ap.add_argument("--allow-library-mismatch", action="store_true",
                    help="放行参考库不是命令流库（仅复现旧口径）")
    ap.add_argument("--gate-batches", type=int, default=8,
                    help="burn-in / token 对比度门禁的批次数（× batch = 窗口数；对比度要 ≥16 才有统计意义）")
    args = ap.parse_args()

    if args.wbc_ckpt is None:
        _h = _discover_hold_wbc(ROOT)
        if _h is None:
            print("[warn] 没找到 logs/wbc_hold*/ckpt_*.pt，退回旧端口 WBC（结果只能当冒烟）", flush=True)
            args.wbc_ckpt = WBC_V2_CKPT
        else:
            args.wbc_ckpt = _h
            print(f"[info] 自动选中 hold 空间 WBC: {_h}", flush=True)
    args.wbc_ckpt = Path(args.wbc_ckpt)

    _forbid_legacy_dataset(args.motion, "--motion")
    for p in (args.wbc_ckpt, args.vlm_ckpt, args.vlm_path):
        if not Path(p).exists():
            raise SystemExit(f"[fail] 缺 {p}")
    if args.wbc_ckpt == WBC_V1_CKPT:
        print("[warn] 用的是 WBC v1（obs99）；S3 的正式读数应当用 WBC_v2（ckpt_016000_spec.pt）",
              flush=True)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    files = collect_npz(args.motion)

    fingerprint = {
        k: (str(v) if isinstance(v, Path) else (list(map(str, v)) if isinstance(v, list) else v))
        for k, v in vars(args).items()
    }
    fingerprint["_wbc_sha256_1m"] = _sha256_head(args.wbc_ckpt)
    fingerprint["_n_clips"] = len(files)
    (out / "args.json").write_text(json.dumps(fingerprint, indent=1, default=str))
    print(f"[audit] 参数指纹 -> {out / 'args.json'}", flush=True)

    # ---------------------------------------------------------------- bundle（冻结执行器 + WM）
    from diff_vla_wbc.isaac_joint import is_isaac_checkpoint, load_isaac_bundle

    if not is_isaac_checkpoint(args.wbc_ckpt):
        raise SystemExit(f"{args.wbc_ckpt}: 只接受 Isaac/trackerlab WBC ckpt（含 pi/wm）")
    bundle = load_isaac_bundle(
        args.wbc_ckpt, device=device, npz=files,
        packed_motion=args.motion_packed, obs_mode=None, hist=None, token_dim=64,
    )
    bundle.reward_mode = "saturating"     # §5 第 8 条：不改奖励核
    bundle.posture_weight = 0.0
    print(f"[info] library = {bundle.library.summary() if hasattr(bundle.library, 'summary') else '?'}",
          flush=True)

    # ---------------------------------------------------------------- processor + codec（J2）
    processor = load_gr00t_processor(args.vlm_ckpt, vlm_path=args.vlm_path)
    codec = ActionCodec(processor, device=device)
    print(codec.audit(), flush=True)
    frame_index = TaskFrameIndex(files)
    if frame_index.num_frames != bundle.library.num_frames:
        raise SystemExit(
            f"npz 索引 {frame_index.num_frames} 帧 ≠ motion library {bundle.library.num_frames} 帧"
        )
    builder = TaskObservationBuilder(frame_index)

    # ---------------------------------------------------------------- VLA（backbone 冻结 + head 可训）
    from diff_vla_wbc.gr00t_vlm import Gr00tVLM

    vlm = Gr00tVLM.from_pretrained(
        args.vlm_path, checkpoint=args.vlm_ckpt, select_layer=args.vlm_select_layer,
        device=device, action_horizon=40, max_state_dim=132,
        freeze_vlm=True, attn_implementation="sdpa",
    )
    head_dim = int(vlm.action_head.action_dim)
    if int(vlm.action_head.action_horizon) < int(args.horizon):
        raise SystemExit(f"action_horizon {vlm.action_head.action_horizon} < H {args.horizon}")
    print(f"[info] VLA head dim={head_dim} horizon={vlm.action_head.action_horizon} "
          f"dtype={next(vlm.action_head.parameters()).dtype} "
          f"trainable={sum(p.numel() for p in vlm.action_head.parameters() if p.requires_grad):,}",
          flush=True)

    # ---------------------------------------------------------------- 端口 token 两个源
    lib = bundle.library
    # ---------------------------------------------------------------- v3：真转移 + 真录制起点
    _real = None
    if (args.start_source == "recorded" or args.wm_refresh_every > 0) and Path(args.real_transitions).exists():
        _real = torch.load(str(args.real_transitions), map_location="cpu", weights_only=False)
        print(f"[v3] real transitions: {tuple(_real['states'].shape)} from {args.real_transitions}", flush=True)
    elif args.start_source == "recorded" or args.wm_refresh_every > 0:
        raise SystemExit(f"[fail] 需要 {args.real_transitions}；先跑 scripts/build_real_transitions.py")
    _recorded_states = _real["states"].to(device) if _real is not None else None

    # ---------------------------------------------------------------- token 前端（交接文档 §5.1/§5.3）
    # hold 窗口 + 仿真机器人 anchor（B 模式）。BC 标签仍是 npz 的 g1_motion_token_64。
    from diff_vla_wbc.sonic_frontend import LibraryTokenBuilder, library_kind  # noqa: PLC0415

    # token 空间跟着窗口口径走：hold ⇒ label（HA 录制/部署口径），future ⇒ port（旧端口空间）
    if getattr(args, "wbc_token_space", "auto") == "auto":
        args.wbc_token_space = "label" if args.wbc_token_window == "hold" else "port"
    _lib_kind = library_kind(args.motion_packed)
    if args.wbc_token_window == "hold" and _lib_kind != "command" and not args.allow_library_mismatch:
        raise SystemExit(
            f"[fail] 参考库 {args.motion_packed} 不是命令流库（kind={_lib_kind}）。hold 口径必须配 "
            f"{PACKED_CMD_MOTION}（npz 的 joint_pos 是执行轨迹，与命令流 lag +8 帧）；"
            "只想复现旧口径请显式 --allow-library-mismatch"
        )
    _wbc_win, _wbc_lib = _wbc_window_of(args.wbc_ckpt)
    print(
        f"[AUDIT][token-frontend] wbc_ckpt_window={_wbc_win} wbc_lib={_wbc_lib or '?'} "
        f"requested_window={args.wbc_token_window} lib_kind={_lib_kind} "
        f"source={args.wbc_token_source} bc_token_space={args.bc_token_space}",
        flush=True,
    )
    if _wbc_win not in ("hold", "future"):
        print(f"[warn] WBC ckpt 没有 token_window 标记（provenance/args 都没有），按库名推断失败；"
              f"库={_wbc_lib or '?'}", flush=True)
    elif _wbc_win != args.wbc_token_window and not args.allow_window_mismatch:
        raise SystemExit(
            f"[fail] WBC ckpt 的 token 窗口是 {_wbc_win!r}，但 --wbc-token-window={args.wbc_token_window}："
            "burn-in 与奖励都会在另一个空间里算（交接文档 §5.1）。修法：换 hold 版 WBC"
            "（logs/wbc_hold_*/ckpt_*.pt），或显式 --allow-window-mismatch 只做冒烟"
        )
    online_tokens = None
    if args.wbc_token_source == "online":
        online_tokens = LibraryTokenBuilder(
            bundle.motion, device=device, window=args.wbc_token_window,
            library_path=args.motion_packed,
        )
        print(online_tokens.audit(), flush=True)

    _tdir = args.motion[0] if args.motion[0].is_dir() else args.motion[0].parent
    port_chunk_src = RecordedTokenSource(
        _tdir, lib, token_horizon=40, token_dim=64, device=device,
        port_dir=args.port_tokens, token_space=args.bc_token_space,
    )
    # WBC 空间的整段 GT（门禁要区分"奖励通道在 WBC 空间的信号"和"部署现实"）
    wbc_chunk_src = RecordedTokenSource(
        _tdir, lib, token_horizon=40, token_dim=64, device=device,
        port_dir=args.port_tokens, token_space=args.wbc_token_space,
    )
    port_frame_src = RecordedTokenSource(
        _tdir, lib, token_horizon=1, token_dim=64, device=device,
        port_dir=args.port_tokens, token_space=args.wbc_token_space,
    )
    _space_mismatch = args.bc_token_space != args.wbc_token_space
    print(
        f"[AUDIT][token-space] bc={args.bc_token_space} wbc={args.wbc_token_space} "
        f"mismatch={_space_mismatch} port_dir={args.port_tokens} "
        f"(S0.4: 标签==录制 encoder_latent；端口装配 corr=0.057 已退役)",
        flush=True,
    )
    if _space_mismatch:
        print(
            "[warn] BC 空间 != WBC 训练空间：奖励分支的梯度跨了两套 token 空间，物理意义不成立"
            "（W5）。本轮必须 lambda_r=0（纯 BC 对照臂），等 S1 用标签空间重训 WBC 后再开 lambda_r。",
            flush=True,
        )
        # 硬门禁（S2 复核，2026-10-26）：警告不足以挡住一次"看起来很忙但结论无效"的跑法——
        # 02:22 就起过一次 lam_r=0.3 + bc=label/wbc=port 的臂。这种臂的 rollout 数字不能进任何表。
        if float(getattr(args, "lambda_r", 0.0)) > 0 and not getattr(
            args, "allow_token_space_mismatch", False
        ):
            raise SystemExit(
                "[fail] token 空间不匹配 (bc=%s != wbc=%s) 却要求 lambda_r=%s：奖励梯度的物理意义"
                "不成立，结果不能当证据。要么 --lambda-r 0（BC 对照臂），要么等 S1 用标签空间重训 "
                "WBC 后 --wbc-token-space label；真的只做诊断请显式 --allow-token-space-mismatch。"
                % (args.bc_token_space, args.wbc_token_space, args.lambda_r)
            )

    def port_chunk(frames):  # noqa: ANN001  (B,) -> (B,40,64)
        return port_chunk_src({"frames": frames})

    def port_frame(frames):  # noqa: ANN001  (B,) -> (B,64)
        return port_frame_src({"frames": frames})[:, 0]

    def wbc_chunk(frames):  # noqa: ANN001  (B,) -> (B,40,64)  WBC 空间
        return wbc_chunk_src({"frames": frames})

    # ---------------------------------------------------------------- 训练 / 评测 provider
    provider = BurnInBatchProvider(
        bundle, lib, port_source=port_chunk, port_frame_source=port_frame,
        wbc_chunk_source=wbc_chunk, online_tokens=online_tokens,
        recorded_states=_recorded_states,
        batch_size=args.batch_size, horizon=args.horizon, token_horizon=40,
        seed=int(args.seed), hand_dir=args.motion[0],
        burnin_steps=int(args.burnin_steps), burnin=not args.no_burnin,
        perturb=not args.no_perturb,
        perturb_xy=args.perturb_xy, perturb_yaw=args.perturb_yaw,
        perturb_dq=args.perturb_dq, perturb_dqd=args.perturb_dqd,
    )
    eval_provider = BurnInBatchProvider(
        bundle, lib, port_source=port_chunk, port_frame_source=port_frame,
        wbc_chunk_source=wbc_chunk, online_tokens=online_tokens,
        recorded_states=_recorded_states,
        batch_size=args.batch_size, horizon=args.horizon, token_horizon=40,
        seed=20261026, hand_dir=args.motion[0],
        burnin_steps=int(args.burnin_steps), burnin=not args.no_burnin,
        perturb=not args.no_perturb,
        perturb_xy=args.perturb_xy, perturb_yaw=args.perturb_yaw,
        perturb_dq=args.perturb_dq, perturb_dqd=args.perturb_dqd,
    )

    # ---------------------------------------------------------------- CPU 预处理（缓存 + 预取）
    def _processed(frames):  # noqa: ANN001
        return processed_batch(processor, builder, frames)

    memo = StepMemo(_processed)
    prefetch = BatchPrefetcher(_processed, enabled=bool(args.prefetch_vla))
    print(f"[prefetch] enabled={prefetch.enabled}", flush=True)

    # ---------------------------------------------------------------- trainer
    cfg = JointV2Config(
        horizon=args.horizon, gamma=args.gamma,
        lambda_bc=args.lambda_bc, lambda_r=args.lambda_r,
        lambda_s=args.lambda_s, lambda_a=args.lambda_a,
        state_vel=args.state_vel_weight,
        fm_steps=args.fm_steps, fm_lr=args.fm_lr, fm_clip=args.fm_clip,
        weight_decay=args.weight_decay, grad_backend=args.fm_grad_backend,
        fm_truncated_tail=args.fm_truncated_tail,
        head_eval_mode=not args.head_train_mode,
        grad_accum=args.grad_accum, grad_ratio_every=args.grad_ratio_every,
    )
    if args.v3:
        from diff_vla_wbc.v3_step import JointTrainerV3, V3Config  # noqa: PLC0415

        v3cfg = V3Config(
            lambda_r=args.lambda_r, lambda_hand=args.lambda_hand, use_critic=args.critic,
            critic_lr=args.critic_lr, gamma=args.gamma, lam=args.lam,
            push_std=args.push_std, grad_log_every=args.grad_log_every,
            clip_head=args.clip_head,
        )
        trainer = JointTrainerV3(bundle, vlm, codec, cfg=cfg, v3=v3cfg, device=device)
        (out / "v3_config.json").write_text(json.dumps(v3cfg.__dict__, indent=1))
    else:
        trainer = JointTrainerV2(bundle, vlm, codec, cfg=cfg, device=device)
    (out / "v2_config.json").write_text(json.dumps(cfg.__dict__, indent=1))

    # 起点就做一次冻结门禁（S3.2）：之后每次保存都会再比一次
    assert_wbc_frozen(bundle, args.wbc_ckpt)

    # ---------------------------------------------------------------- 门禁模式（不训练）
    if args.gate_only:
        from diff_vla_wbc.v2_gates import run_gates  # noqa: PLC0415

        ref_provider = BurnInBatchProvider(
            bundle, lib, port_source=port_chunk, port_frame_source=port_frame,
            wbc_chunk_source=wbc_chunk, online_tokens=online_tokens,
            recorded_states=_recorded_states,
            batch_size=args.batch_size, horizon=args.horizon, token_horizon=40,
            seed=int(args.seed), hand_dir=args.motion[0],
            burnin=False,                    # 对照组：起点 = 参考帧（旧口径）
            perturb=False,
        )
        gates = run_gates(
            args, bundle, vlm, codec, trainer, provider, ref_provider, processor,
            args.gate_out or (out / "gates.json"),
            n_batches=int(args.gate_batches),
        )
        prefetch.close()
        return 0 if all(
            bool(v.get("pass", True)) for k, v in gates.items() if not k.startswith("_")
        ) else 1

    # ---------------------------------------------------------------- 固定评测窗口（S3.7）
    eval_batches = []
    if args.eval_every and args.eval_batches:
        t0 = time.time()
        eval_batches = [eval_provider.sample() for _ in range(args.eval_batches)]
        print(f"[eval] fixed {len(eval_batches)} x batch {args.batch_size} = "
              f"{len(eval_batches) * args.batch_size} 窗口（burn-in 起点），采样 {time.time() - t0:.1f}s",
              flush=True)
    eval_out = args.eval_out or (out / "eval_curve.jsonl")

    def run_eval(step: int) -> dict:
        for b in eval_batches:
            b.model_inputs = memo.get(b.frames)
        m = fixed_eval(trainer, eval_batches)
        m["h"] = int(args.horizon)
        m["lambda_r"] = float(args.lambda_r)
        m["fm_steps"] = int(args.fm_steps)
        with open(eval_out, "a") as f:
            f.write(json.dumps(m) + "\n")
        return m

    if eval_batches:
        run_eval(0)

    # ---------------------------------------------------------------- 训练循环
    def next_batch():
        b = provider.sample()
        if prefetch.enabled:
            prefetch.submit(b.frames)
        return b

    deadline = time.time() + args.time_budget_minutes * 60.0 if args.time_budget_minutes > 0 else None
    metrics_path = out / "train_metrics.jsonl"
    _t0 = time.time()
    batch = next_batch()
    step = 0
    for step in range(1, int(args.steps) + 1):
        nxt = next_batch()
        batch.model_inputs = prefetch.get(batch.frames, lambda: memo.get(batch.frames))
        metrics = trainer.train_step(batch)
        metrics["step"] = step
        metrics["s_per_step"] = (time.time() - _t0) / step
        # 每一步都落盘：grad_ratio 只在打印步产生，只写在打印分支里会丢
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if step % max(1, args.log_every) == 0 or step == 1:
            print(f"[v2 {step}/{args.steps}] "
                  + json.dumps({k: round(v, 6) for k, v in metrics.items()}), flush=True)
        if args.v3 and args.wm_refresh_every and step % args.wm_refresh_every == 0:
            try:
                rm = trainer.refresh_wm(
                    _real, steps=args.wm_refresh_steps, lr=args.wm_refresh_lr,
                    kl_anchor=args.wm_kl_anchor, log=print,
                )
                print(f"[wm-refresh {step}] " + json.dumps(
                    {k: v for k, v in rm.items() if k in ("steps", "loss_first", "loss_last")}), flush=True)
            except Exception as exc:  # 刷新失败不能挡住训练
                print(f"[wm-refresh][warn] {exc}", flush=True)
        if eval_batches and args.eval_every and step % args.eval_every == 0:
            run_eval(step)
        if args.save_every and step % args.save_every == 0:
            trainer.save_joint(out / f"joint_{step:06d}.pt")
            assert_wbc_frozen(bundle, args.wbc_ckpt)
            if args.save_vla_every and step % args.save_vla_every == 0:
                p = trainer.save_vla(out / f"vla_{step:06d}.pt")
                print(f"[saved] {p}", flush=True)
        batch = nxt
        memo.clear()
        if deadline is not None and time.time() >= deadline:
            print(f"[budget] {args.time_budget_minutes:g} min 到达（step {step}）", flush=True)
            break

    # ---------------------------------------------------------------- 收尾
    trainer.save_joint(out / "joint_final.pt")
    frozen_ok = assert_wbc_frozen(bundle, args.wbc_ckpt)
    vla_final = trainer.save_vla(out / "vla_final.pt")
    print(f"[saved] {vla_final}", flush=True)
    final_eval = run_eval(int(step)) if eval_batches and step else {}
    summary = {
        "steps": int(step),
        "out": str(out),
        "wbc_ckpt": str(args.wbc_ckpt),
        "wbc_sha256_1m": fingerprint["_wbc_sha256_1m"],
        "vlm_ckpt": str(args.vlm_ckpt),
        "freeze_gate": bool(frozen_ok),
        "fm_steps": int(args.fm_steps),
        "decode": "codec",
        "bc_token_space": args.bc_token_space,
        "wbc_token_space": args.wbc_token_space,
        "token_space_mismatch": bool(_space_mismatch),
        "v3": bool(args.v3), "lambda_hand": float(args.lambda_hand),
        "push_std": float(args.push_std), "start_source": str(args.start_source),
        "wm_refresh_every": int(args.wm_refresh_every),
        "wbc_token_window": str(args.wbc_token_window),
        "wbc_token_source": str(args.wbc_token_source),
        "motion_packed": str(args.motion_packed),
        "library_kind": str(_lib_kind),
        "reward_branch_valid": bool((not _space_mismatch) and float(args.lambda_r) > 0),
        "lambda_r": float(args.lambda_r),
        "lambda_bc": float(args.lambda_bc),
        "horizon": int(args.horizon),
        "grad_accum": int(args.grad_accum),
        "final_eval": final_eval,
        "wall_s": time.time() - _t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("[AUDIT][v2-summary] " + json.dumps(summary, default=str), flush=True)
    prefetch.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
