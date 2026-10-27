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
    ap.add_argument("--horizon", type=int, default=20,
                    help="想象 rollout 步数。**默认 20 = 部署口径**：sonic provider 用 "
                         "HA_VLA_EXEC_STEPS=20 做 receding horizon（2.5 Hz @ 50 Hz），一个 40 帧 "
                         "chunk 只执行前 20 帧，所以一步推理只对这 20 步负责；更长的信用走 "
                         "gamma^H*V(s_H) 的 critic bootstrap（计划文档 §12）")
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
                    help="rollout 内过程扰动：每步给 root **线速度**加 U(-std,std) m/s")
    ap.add_argument("--push-ang", type=float, default=0.0,
                    help="同上，root 角速度 rad/s（要小，否则软终止会打断想象）")
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
    # ---------------- DR（域随机化）+ 模型可信度 ----------------
    ap.add_argument("--dr", dest="dr", action="store_true", default=True,
                    help="域随机化：起点/过程扰动按 nominal/mid/hard 分层采样（默认开）")
    ap.add_argument("--no-dr", dest="dr", action="store_false")
    ap.add_argument("--dr-xy-max", type=float, default=0.20, help="hard 桶的 root xy 上限 (m)")
    ap.add_argument("--dr-yaw-max", type=float, default=0.25, help="hard 桶的 yaw 上限 (rad)")
    ap.add_argument("--dr-dq-max", type=float, default=0.05)
    ap.add_argument("--dr-dqd-max", type=float, default=0.30)
    ap.add_argument("--dr-p-nominal", type=float, default=0.25)
    ap.add_argument("--dr-p-mid", type=float, default=0.45)
    ap.add_argument("--anchor-weight", type=float, default=0.0,
                    help="先验锚定 λ_a：λ_a·‖token − 底模先验‖²（先验代理 = 标签 chunk；"
                         "底模输出与标签 MSE 只有 0.004，所以数值等价）。0 = 关。建议 0.05")

    ap.add_argument("--lora-rank", type=int, default=32,
                    help="动作头 LoRA 的秩（0=关，训全量 1.62B）。开 LoRA 后只有适配器可训，"
                         "batch 才能开大（对齐 SuperMimic 的大 batch 前提）")
    ap.add_argument("--amp", choices=("off", "bf16"), default="bf16",
                    help="混合精度：bf16 = GR00T ckpt 的原生 dtype（config.json dtype=bfloat16）；"
                         "只作用于 backbone 前向与 head 的 FM 采样/损失，主权重与 rollout 仍 fp32")
    ap.add_argument("--lora-alpha", type=float, default=None, help="LoRA 缩放 alpha（默认=rank）")
    ap.add_argument("--lora-targets", type=str, default="attn1,ff",
                    help="逗号分隔的层名后缀，只给这些 nn.Linear 挂 LoRA（默认 attention+FFN）")
    ap.add_argument("--lora-train-proj-out", action="store_true", default=True,
                    help="同时解冻输出投影 proj_out_*（默认开）")

    ap.add_argument("--trust-eps", type=float, default=0.0,
                    help="token trust region 容差（相对先验的 RMSE，归一化 token 空间）。"
                         "只在 RMSE>eps 时给力；0=关。建议 0.15（≈底模自身偏差 0.10 的 1.5×）")
    ap.add_argument("--trust-weight", type=float, default=0.0,
                    help="trust region hinge 权重（无量纲）。1.0 ≈ 越界即与 lambda_r*|J| 同量级")

    ap.add_argument("--trust-center", choices=("label", "base"), default="label",
                    help="动作 hinge 的中心：label=标签 rollout（旧/稳定）；"
                         "base=底模自己的输出（P1-4，无标签，但当前会 NaN）")
    ap.add_argument("--critic-grad-through-state", action="store_true", default=False,
                    help="P1-1 的梯度部分：V 对 s_H 保留梯度（SHAC 本意）。默认关（冒烟会 NaN）")
    ap.add_argument("--vla-state-from-burnin", action="store_true", default=False,
                    help="P1-6：把 burn-in 末状态的本体感觉写回 VLA 的 state 输入"
                         "（否则 VLA 看不到 DR 扰动，只能学常数偏置）")
    ap.add_argument("--act-eps", type=float, default=0.0,
                    help="动作空间 trust region 容差：(a−a_ref) 的逐关节 RMS。0=关。"
                         "动作才是 WBC/WM 的输入，比 token 空间更本质")
    ap.add_argument("--act-weight", type=float, default=0.0,
                    help="动作空间 hinge 权重（无量纲）。1.0 ≈ 越界即与 lambda_r*|J| 同量级")
    ap.add_argument("--critic-source", choices=("imagined", "real"), default="imagined",
                    help="critic 的监督来源：imagined=想象转移 GAE（旧）；real=真录制轨迹 return-to-go（grounded）")
    ap.add_argument("--critic-real", type=Path, default=ROOT / "runs/wm_real/real_returns.pt",
                    help="--critic-source real 用的真回报文件（scripts/build_real_returns.py 生成）")
    ap.add_argument("--reward-mode", choices=("saturating", "linear"), default="saturating")
    ap.add_argument("--reward-vel-scale", type=float, default=1.0,
                    help="奖励里速度两项的系数（0=关）。WM 的速度通道最不准（qd_corr~0.64）")
    ap.add_argument("--obs-noise", type=float, default=0.0,
                    help="观测层 DR：给 VLA 输入的**本体感觉**通道（腿/腰/臂的 q）加 ±noise rad 均匀噪声；"
                         "不动 hand / projected_gravity 通道。0 = 关")
    ap.add_argument("--clip-head", type=float, default=10.0,
                    help="VLA head 的梯度范数裁剪阈值（只做防尖峰；Adam/AdamW 下全局缩放近似无效果，"
                         "实测 pre-clip 范数 4–9，所以默认 10）")
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
    ap.add_argument("--eval-horizon", type=int, default=0,
                    help="评测用的想象步数（0 = 与 --horizon 相同）。**部署口径是 20**；"
                         "训练可以在短视界求梯度（WM 更可信），但评估必须按部署视界")
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
    bundle.reward_mode = str(args.reward_mode)
    bundle.reward_vel_scale = float(args.reward_vel_scale)
    print(f"[AUDIT][reward] mode={bundle.reward_mode} vel_scale={bundle.reward_vel_scale} "
          f"(速度通道 qd_corr≈0.64 不可信；0=精确扣掉两项)", flush=True)
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
    if int(args.lora_rank) > 0:
        from diff_vla_wbc.lora import apply_lora, lora_summary  # noqa: PLC0415

        n = apply_lora(
            vlm.action_head, rank=int(args.lora_rank), alpha=args.lora_alpha,
            targets=tuple(s for s in args.lora_targets.split(",") if s),
            also_train=(("proj_out",) if args.lora_train_proj_out else ()),
        )
        sm = lora_summary(vlm.action_head)
        print(f"[AUDIT][lora] rank={args.lora_rank} alpha={args.lora_alpha or args.lora_rank} "
              f"layers={n} trainable={sm['trainable']:,} / {sm['total']:,} "
              f"({100*sm['trainable_frac']:.3f}%) targets={args.lora_targets} "
              f"proj_out_train={args.lora_train_proj_out}", flush=True)
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
    _recorded_states = None
    if _real is not None:
        # ★ 帧号对齐（2026-10-26 修）：transitions 里一行 = 一个**转移**，每个 clip 的最后一帧
        # 没有后继转移，所以 634016 帧只有 633317 行。直接拿帧号去索引 → 越界
        # （CUDA device-side assert，A0/A1 在 step 140 就是这么崩的），而且不崩时也会读到
        # 别的 clip 的状态（起点是错的）。这里用 transitions 自带的 `frames` 建一张
        # 帧 → 行 的表，把缺失的那 699 帧（每 clip 末帧）用**同 clip 的上一帧**顶上。
        _n_frames = int(lib.num_frames)
        _fr = _real["frames"].long()
        if int(_fr.max()) >= _n_frames:
            raise SystemExit(f"[fail] transitions 的 frames 最大 {int(_fr.max())} ≥ 库帧数 {_n_frames}")
        _map = torch.full((_n_frames,), -1, dtype=torch.long)
        _map[_fr] = torch.arange(_fr.numel())
        _miss = (_map < 0).nonzero(as_tuple=False).flatten()
        if _miss.numel():
            _map[_miss] = _map[(_miss - 1).clamp(min=0)]
        if int((_map < 0).sum()):
            raise SystemExit(f"[fail] real transitions 有 {int((_map < 0).sum())} 帧无法对齐（frames 缺失）")
        _recorded_states = _real["states"][_map].to(device)
        print(
            f"[v3] recorded_states 按帧对齐: {tuple(_recorded_states.shape)} "
            f"（原始转移 {tuple(_real['states'].shape)}，补 {int(_miss.numel())} 帧 = 每 clip 末帧）",
            flush=True,
        )

    from diff_vla_wbc.dr import DRConfig, estimate_ranges  # noqa: PLC0415

    _dr_cfg = DRConfig(
        enabled=bool(args.dr), p_nominal=args.dr_p_nominal, p_mid=args.dr_p_mid,
        xy_max=args.dr_xy_max, yaw_max=args.dr_yaw_max,
        dq_max=args.dr_dq_max, dqd_max=args.dr_dqd_max,
        push_std=args.push_std, push_ang=args.push_ang,
    )
    print(f"[AUDIT][dr] enabled={_dr_cfg.enabled} p={_dr_cfg.p_nominal}/{_dr_cfg.p_mid}/rest "
          f"max(xy,yaw,dq,dqd)=({_dr_cfg.xy_max},{_dr_cfg.yaw_max},{_dr_cfg.dq_max},{_dr_cfg.dqd_max}) "
          f"push=({_dr_cfg.push_std},{_dr_cfg.push_ang})", flush=True)
    if _real is not None:
        print("[AUDIT][dr][data] " + json.dumps(estimate_ranges(_real)), flush=True)

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
        recorded_states=_recorded_states, dr=_dr_cfg,
        batch_size=args.batch_size, horizon=args.horizon, token_horizon=40,
        seed=int(args.seed), hand_dir=args.motion[0],
        burnin_steps=int(args.burnin_steps), burnin=not args.no_burnin,
        perturb=not args.no_perturb,
        perturb_xy=args.perturb_xy, perturb_yaw=args.perturb_yaw,
        perturb_dq=args.perturb_dq, perturb_dqd=args.perturb_dqd,
    )
    # 评测 provider 的**参考帧数**要够评测视界用（训练 5 步、评测 20 步是常态）；
    # 起点（burn-in 后的 state/history）与训练视界无关，所以两臂仍逐位可比。
    _h_eval_req = max(int(args.horizon), int(getattr(args, "eval_horizon", 0) or 0))
    eval_provider = BurnInBatchProvider(
        bundle, lib, port_source=port_chunk, port_frame_source=port_frame,
        wbc_chunk_source=wbc_chunk, online_tokens=online_tokens,
        recorded_states=_recorded_states, dr=_dr_cfg,
        batch_size=args.batch_size, horizon=_h_eval_req, token_horizon=40,
        seed=20261026, hand_dir=args.motion[0],
        burnin_steps=int(args.burnin_steps), burnin=not args.no_burnin,
        perturb=not args.no_perturb,
        perturb_xy=args.perturb_xy, perturb_yaw=args.perturb_yaw,
        perturb_dq=args.perturb_dq, perturb_dqd=args.perturb_dqd,
    )

    # ---------------------------------------------------------------- CPU 预处理（缓存 + 预取）
    class _NoisyBuilder:
        """观测层 DR：只给本体感觉通道加噪（rad），不动 hand / gravity。"""

        GROUPS = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")

        def __init__(self, inner, std: float, seed: int = 0):
            self.inner = inner
            self.std = float(std)
            self.g = torch.Generator(device="cpu")
            self.g.manual_seed(int(seed))

        def build(self, frames):  # noqa: ANN001
            obs = self.inner.build(frames)
            if self.std > 0:
                for k in self.GROUPS:
                    key = f"state.{k}"
                    if key in obs:
                        arr = obs[key]
                        n = torch.rand(arr.shape, generator=self.g) * 2.0 - 1.0
                        obs[key] = (arr + (n.numpy() * self.std)).astype(arr.dtype)
            return obs

    _obs_builder = _NoisyBuilder(builder, args.obs_noise, seed=int(args.seed)) if args.obs_noise > 0 else builder
    if args.obs_noise > 0:
        print(f"[AUDIT][dr][obs] += U(±{args.obs_noise}) rad on {list(_NoisyBuilder.GROUPS)} (裸 46 维状态)", flush=True)

    def _processed(frames):  # noqa: ANN001
        return processed_batch(processor, _obs_builder, frames)

    def _processed_clean(frames):  # noqa: ANN001
        """评测用干净观测（跨步可比；观测 DR 只作用于训练）。"""
        return processed_batch(processor, builder, frames)

    memo = StepMemo(_processed)
    memo_eval = StepMemo(_processed_clean)
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
        eval_horizon=int(args.eval_horizon), amp=str(args.amp),
    )
    if args.v3:
        from diff_vla_wbc.v3_step import JointTrainerV3, V3Config  # noqa: PLC0415

        v3cfg = V3Config(
            lambda_r=args.lambda_r, lambda_hand=args.lambda_hand, use_critic=args.critic,
            critic_lr=args.critic_lr, gamma=args.gamma, lam=args.lam,
            push_std=args.push_std, push_ang=args.push_ang,
            anchor_weight=args.anchor_weight,
            trust_eps=args.trust_eps, trust_weight=args.trust_weight,
            act_eps=args.act_eps, act_weight=args.act_weight,
            vla_state_from_burnin=bool(args.vla_state_from_burnin),
            critic_grad_through_state=bool(args.critic_grad_through_state),
            trust_center=str(args.trust_center),
            critic_source=args.critic_source,
            grad_log_every=args.grad_log_every,
            clip_head=args.clip_head,
        )
        trainer = JointTrainerV3(bundle, vlm, codec, cfg=cfg, v3=v3cfg, device=device)
        if args.critic_source == "real":
            if _real is None or _recorded_states is None:
                raise SystemExit("[fail] --critic-source real 需要 --start-source recorded 与真转移文件")
            _rrp = Path(args.critic_real)
            if not _rrp.exists():
                raise SystemExit(f"[fail] 缺真回报文件 {_rrp}；先跑 scripts/build_real_returns.py")
            _rr = torch.load(str(_rrp), map_location="cpu", weights_only=False)
            _rr_ret = _rr["ret"]
            if "states" in _rr and "ref_rows" in _rr:
                # P1-2：MC 目标文件自带 states/ref_rows（按 episode 采的 on-policy 状态）
                print(f"[P1-2] critic 目标来自 on-policy MC：n={int(_rr_ret.shape[0])} "
                      f"（{(_rr.get('meta') or {}).get('episodes', '?')} 集）", flush=True)
                trainer.attach_real_returns(_rr_ret, _rr["states"].float(), _rr["ref_rows"].float())
            elif int(_rr_ret.shape[0]) != int(_recorded_states.shape[0]):
                raise SystemExit(
                    f"[fail] real returns 长度 {int(_rr_ret.shape[0])} != recorded_states "
                    f"{int(_recorded_states.shape[0])}；用同一个 --reward-mode/--reward-vel-scale 重建")
                _rr_meta = _rr.get("meta") or {}
                for _k, _want in (("reward_mode", args.reward_mode),
                                  ("reward_vel_scale", float(args.reward_vel_scale))):
                    if _k in _rr_meta and _rr_meta[_k] != _want:
                        raise SystemExit(
                            f"[fail] real returns 的 {_k}={_rr_meta[_k]} 与本次训练 {_want} 不一致；"
                            "重建：python3 scripts/build_real_returns.py --reward-mode ... "
                            "--reward-vel-scale ...")
                trainer.attach_real_returns(_rr_ret, _recorded_states,
                                            bundle.motion.ref.detach().float())
        (out / "v3_config.json").write_text(json.dumps(v3cfg.__dict__, indent=1, default=str))
        # 口径的一行真相：token 监督 / 手部 / 锚定 / DR 分桶 / critic 自举 / WM 保鲜
        _pn, _pm = float(args.dr_p_nominal), float(args.dr_p_mid)
        print(
            f"[AUDIT][v3] token_BC=OFF hand_mask=64:78 lam_hand={args.lambda_hand} "
            f"lam_r={args.lambda_r} anchor={args.anchor_weight} "
            f"trust=({args.trust_eps},{args.trust_weight}) "
            f"dr={bool(args.dr)} buckets={_pn:.2f}/{_pm:.2f}/{max(0.0, 1-_pn-_pm):.2f} "
            f"dr_max(xy,yaw,dq,dqd)=({args.dr_xy_max},{args.dr_yaw_max},{args.dr_dq_max},{args.dr_dqd_max}) "
            f"start={args.start_source} burnin={0 if args.no_burnin else args.burnin_steps} "
            f"push=({args.push_std},{args.push_ang}) "
            f"critic={args.critic}/{args.critic_source} lam={args.lam} gamma^H={args.gamma ** args.horizon:.4f} "
            f"vla_state_burnin={args.vla_state_from_burnin} act=({args.act_eps},{args.act_weight}) reward=({args.reward_mode},{args.reward_vel_scale}) "
            f"wm_refresh_every={args.wm_refresh_every} clip_head={args.clip_head} "
            f"eval_H={int(args.eval_horizon) or int(args.horizon)} amp={args.amp} "
            f"H={args.horizon} K={args.fm_steps} batch={args.batch_size} accum={args.grad_accum}",
            flush=True,
        )
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

    # 固定 WM 快照：训练中会做 WM 保鲜，若评测用"活的 WM"，`imag_return_gt` 这类读数会跟着漂
    # （实测 ret_gt 0.449 → 0.155），跨步不可比。评测一律用加载时的快照。
    import copy as _copy

    _eval_wm = _copy.deepcopy(bundle.wm)

    def run_eval(step: int) -> dict:
        for b in eval_batches:
            b.model_inputs = memo_eval.get(b.frames)
        _live_wm = bundle.wm
        try:
            bundle.wm = _eval_wm
            m = fixed_eval(trainer, eval_batches)
        finally:
            bundle.wm = _live_wm
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
        "push_std": float(args.push_std), "push_ang": float(args.push_ang),
        "clip_head": float(args.clip_head), "obs_noise": float(args.obs_noise),
        "anchor_weight": float(args.anchor_weight), "start_source": str(args.start_source),
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
