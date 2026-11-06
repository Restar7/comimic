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


def _ddp_broadcast_trainables(trainer) -> None:
    """DDP：训练开始前把**所有会被保存/参与前向的参数**从 rank0 广播一遍。

    为什么需要（2026-11-01，8 卡）：模型构造发生在 ``torch.manual_seed(base+rank)`` 之后，
    LoRA 的 A 是 kaiming 随机初始化 ⇒ 各 rank 的初始权重不同。DDP 只同步**梯度**，
    不同初值 + 同一平均梯度会让 8 个 rank 越走越远，rank0 存的 ckpt 也不再是
    "全局 batch 训练出来的那个"。这里先在起点对齐一次，之后靠 all_reduce(AVG) 保持同步。
    """
    import torch.distributed as dist  # noqa: PLC0415

    if not (dist.is_available() and dist.is_initialized()):
        return
    params = list(getattr(trainer, "fm_parameters", []) or [])
    params += list(getattr(trainer, "wbc_parameters", []) or [])
    _se = getattr(trainer, "state_extra_scale", None)
    if _se is not None:
        params.append(_se)
    _critic = getattr(trainer, "critic", None)
    if _critic is not None:
        params += list(_critic.parameters())
    with torch.no_grad():
        for p in params:
            dist.broadcast(p.data, src=0)
    print(f"[ddp] broadcast {len(params)} 个参数张量 from rank0（对齐 LoRA/state_extra/critic 初值）",
          flush=True)


def _build_task_weights(lib: Any) -> torch.Tensor:
    """按 7 个任务均衡的每帧权重（每个任务总权重相等）。"""
    n = int(lib.num_frames)
    starts = [int(c.start) for c in lib.clips]
    names = [str(c.name) for c in lib.clips]
    ends = starts[1:] + [n]
    tasks = [nm.split("__")[0] for nm in names]
    uniq = sorted(set(tasks))
    w = torch.ones(n, dtype=torch.float32)
    for t in uniq:
        rows = [i for i, x in enumerate(tasks) if x == t]
        n_t = sum(ends[i] - starts[i] for i in rows)
        if n_t <= 0:
            continue
        scale = float(n) / (len(uniq) * n_t)
        for i in rows:
            w[starts[i]:ends[i]] = scale
    return w


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
    ap.add_argument("--sync-gradients", action="store_true",
                    help="DDP：torchrun 起多进程时打开（梯度 all_reduce；每 rank 的种子自动 +rank）")
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
    ap.add_argument("--train-wbc", action="store_true", default=False,
                    help="解冻执行器（WBC，~4.8M）一起训：'直接学动作'的对照臂。"
                         "WM 仍冻结，动作 hinge 的中心 = π_k 的 WBC 快照。默认关。")
    ap.add_argument("--wbc-lr", type=float, default=1e-5,
                    help="--train-wbc 时 WBC 的学习率（小于 head 的 fm-lr 更稳）")
    ap.add_argument("--wbc-unfreeze", choices=("all", "dec"), default="all",
                    help="解冻 WBC 的范围：all=全部；dec=只解冻 dec.*（Path B 用）")
    ap.add_argument("--wbc-kl-anchor", type=float, default=0.0,
                    help="Path B：WBC action KL 锚权重（同状态/token 下 MSE 到起点 WBC）；0=关")
    ap.add_argument("--vla-kl-anchor", type=float, default=0.0,
                    help="Path B：VLA token KL 锚权重（同 obs/noise 下 MSE 到起点 action head）；0=关")
    ap.add_argument("--v3", action="store_true",
                    help="启用 v3 训练步：token 只由奖励(+critic)监督，手部单独 FM 监督")
    ap.add_argument("--token-bc-weight", type=float, default=0.0,
                    help="v3 的 token BC 权重（2026-10-28 新增）：用 B dump 的命令流 token 当标签做 "
                         "GR00T 原生 FM 回归，只监督前 64 维。v3 原来完全没有 token 级监督，"
                         "导致奖励+hinge 是仅有的信号、训练只在 π₀ 附近震荡。0=关（旧行为）")
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
    ap.add_argument("--start-from-b", default="",
                    help="训练起点 = B 线真实重规划状态（glob，例如 'runs/B_*/dump/*.pt'）。"
                         "给了它就不做 burn-in、不叠 DR；要求 dump 是 online 模式（frames=真实库行号）")
    ap.add_argument("--start-from-b-stride", type=int, default=20,
                    help="每隔多少步算一个重规划时刻（部署 HA_VLA_EXEC_STEPS=20）")
    ap.add_argument("--start-from-b-order", default="ha_v31",
                    help="B dump 的 states 关节顺序（provider 的 asset 序，默认 ha_v31）")
    ap.add_argument("--b-self-targets", type=Path, default=None,
                    help="build_b_self_targets.py 的产物：采集时策略自己的 token + 实际漂移。"
                         "给了它才可能用 token_bc_source=self / --b-self-err-max 筛选")
    # ★ 2026-10-31：新一轮数据加权（混全池训练时让本轮 on-policy 数据被采得更勤）
    ap.add_argument("--b-fresh-glob", type=str, default="",
                    help="glob：匹配到的 B 线 episode 采样权重 = --b-fresh-weight（其余 1.0）")
    ap.add_argument("--b-fresh-weight", type=float, default=3.0)
    ap.add_argument("--yawrate-weight", type=float, default=-1.0,
                    help="yaw_rate 奖励权重；<0 = 用 reward_shape 里的常量。"
                         "pipeline 会在 WM gate 之后按 root_w_corr 自动决定（>=0.40 才开）")
    ap.add_argument("--adv-baseline", type=str, default="none", choices=("none", "self", "label"),
                    help="e2：用'真机执行过的 token'再跑同一条想象做控制变量（adv = ret − ret_exec）。"
                         "必须配 --adv-tau>0 才不是安慰剂（pathwise 梯度下减常数不改变梯度）")
    ap.add_argument("--adv-tau", type=float, default=0.02,
                    help="e2：advantage 的有界温度（`-tanh(adv/τ)`）；0=退回均值形式（仅诊断）")
    ap.add_argument("--state-extra", type=str, default="none", choices=("none", "rotlocal6", "abs6", "vel6", "ori6"),
                    help="e1：VLA 的 state 46→52 维（6 个可训练缩放，初值 0 ⇒ 第 0 步与上轮逐位相同）。"
                         "abs6=绝对朝向 rot6d(R)（**零采集器改动**，线上 state64[:6] 就是它，推荐）；"
                         "vel6=体坐标 [Rᵀv,Rᵀw]（闭环需改第三方客户端，暂未接）；"
                         "ori6=rot6d(RᵀR_ref)（需参考帧，闭环没有 ⇒ 仅离线诊断）")
    ap.add_argument("--b-start-max-err", type=float, default=None,
                    help="只保留**瞬时** root 误差 ≤ 该值的重规划起点（米）。"
                         "奖励核 exp(-e/std^2) 在 0.5 m 以上梯度就没了，所以训练起点必须筛")
    ap.add_argument("--b-self-err-max", type=float, default=None,
                    help="只保留'接下来 horizon 步实际漂移 ≤ 该值'的重规划时刻（米）。"
                         "这是**闭环真值**筛选，不是参考 token 筛选")
    ap.add_argument("--token-bc-span", type=int, default=0,
                    help="token BC 只监督前 N 步（0=全部）。source=self 时必须设 20："
                         "dump 的 tokens 是单步 token，只有前 20 步属于那一次观测")
    ap.add_argument("--token-bc-source", choices=("ref", "self"), default="ref",
                    help="token BC 目标：ref=参考命令流 token（实测有害）；self=策略自己当时的 token")
    ap.add_argument("--b-start-nonstrict", action="store_true", default=False,
                    help="允许非 online 模式的 dump（危险：frames 是步号，不是库行号）")
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

    # 2026-11-02：参考起点加权采样（phase/task）+ 覆盖审计
    ap.add_argument("--start-sampler", choices=("uniform", "weights", "phase", "task"),
                    default="uniform",
                    help="uniform=全库均匀；weights/phase=读 --start-weights；task=按 7 个任务均衡")
    # ★ 2026-11-04：起点的**漂移上限**（m）。用户提的关键问题：
    #   "从 recorded 起点出发，越到后面漂移越大，会不会到 1 m 就降不下去了？"
    #   回答分两半：
    #     (a) A' 每 20 步重置回参考 ⇒ 实测状态偏移 p50 2cm / p90 7cm / p99 16cm（max 53cm 是异常段），
    #         **不存在 1 m 的起点**；
    #     (b) 即便如此也要显式封顶：只从 |ξ| ≤ δ 的帧采样起点，
    #         保证每个窗口都还在"参考仍然有意义"的信任域里。
    #   与 `--start-source recorded` 配套使用（不封顶就用 recorded 会采到 50cm 的异常段）。
    ap.add_argument("--start-max-drift", type=float, default=0.0,
                    help="起点状态相对参考的 root 偏移上限(m)；>0 时按 A' 真实状态建帧权重。"
                         "配合 --start-source recorded 使用；推荐 0.10")
    ap.add_argument("--start-weights", type=Path, default=None,
                    help="每帧权重 .pt（reward_from_objects.py --weights-out 生成）；phase 默认 runs/wm_real/exp_start_weights.pt")
    ap.add_argument("--start-audit", action="store_true", default=False,
                    help="记录采样到的参考帧，训练结束写 start_coverage.pt/json")

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
    ap.add_argument("--init-vla", default="",
                    help="多轮闭环：从上一轮 save_vla 写的 vla_*.pt 初始化 head（= 本轮 π_k 起点）。"
                         "留空 = 从官方底模起（第 1 轮）。需要 --lora-rank > 0。")

    ap.add_argument("--trust-eps", type=float, default=0.0,
                    help="token trust region 容差（相对先验的 RMSE，归一化 token 空间）。"
                         "只在 RMSE>eps 时给力；0=关。建议 0.15（≈底模自身偏差 0.10 的 1.5×）")
    ap.add_argument("--trust-weight", type=float, default=0.0,
                    help="trust region hinge 权重（无量纲）。1.0 ≈ 越界即与 lambda_r*|J| 同量级")

    ap.add_argument("--critic-state-grad-scale", type=float, default=0.1,
                    help="P1-1：只缩放\"经 s_H\"的梯度（数值不变，straight-through）")
    ap.add_argument("--critic-rel-clip", type=float, default=5.0,
                    help="P1-1：V 相对值 clamp（超界梯度为 0）")
    ap.add_argument("--act-hinge-cap", type=float, default=2.0,
                    help="P1-4：hinge 上界 min(excess,cap)^2，防止一次爆掉")
    ap.add_argument("--trust-center", choices=("self", "label", "base", "prev", "hybrid"), default="label",
                    help="信任域中心（token hinge 与动作 hinge 共用）："
                         "**self = 采集那一刻策略自己真正下发的 token（B 线真值，推荐）**；"
                         "label = 录制标签 rollout（属于另一个执行器）；"
                         "base = 底模自己的输出（P1-4，无标签，当前会 NaN）；"
                         "prev = 本轮起点 π_k 的冻结快照（只限制不漂、不给方向）；"
                         "hybrid = 低漂移用 label、高漂移用 self（阈值 --hybrid-err）")
    ap.add_argument("--hybrid-err", type=float, default=0.5,
                    help="hybrid 中心阈值：realized_err > 该值（米）时用 self token，否则用 label")
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
    ap.add_argument("--action-target-weight", type=float, default=0.0,
                    help="★ P1（2026-11-05）：把更新放到**动作空间**。用 WM 梯度算目标动作 "
                         "A* = A + eta*ĝ，损失改成 动作空间回归。0=关（旧行为）")
    ap.add_argument("--action-target-steps", type=int, default=20,
                    help="动作空间目标回归多少步（默认 20 = 全视界；目标是一条完整计划，不是逐状态回归）")
    ap.add_argument("--action-target-eta", type=float, default=0.15,
                    help="动作空间的步长（SONIC 单位，与真机 ±δ 实验同口径）")
    ap.add_argument("--reward-horizon", type=int, default=0,
                    help="奖励只用想象轨迹的前 k 步（0=满 H，旧行为）。WM 单步准、多步靠积分，"
                         "把奖励压在最准的时间尺度上（2026-11-05）")
    ap.add_argument("--act-penalty", choices=("hinge", "l2"), default="hinge",
                    help="动作空间惩罚形状。hinge=历史默认（eps 以内梯度为 0，有死区）；"
                         "l2=(r/eps)^2 从 0 起就有梯度。2026-11-05：round-5 的动作漂移"
                         "滑到 0.80 就是死区造成的，改用 l2 + 收紧 act-eps")
    ap.add_argument("--critic-source", choices=("imagined", "real"), default="imagined",
                    help="critic 的监督来源：imagined=想象转移 GAE（旧）；real=真录制轨迹 return-to-go（grounded）")
    ap.add_argument("--critic-ckpt", type=Path, default=None,
                    help="预拟合的 critic（scripts/fit_critic_real.py 产物）；给了就先加载再继续训")
    ap.add_argument("--critic-use-diff", action="store_true", default=False,
                    help="critic 显式吃 s−ref；必须与 --critic-ckpt 的 meta 一致")
    ap.add_argument("--critic-ret", type=Path, default=None,
                    help="可选：用这个小文件里的 ret 覆盖 critic-real 的 ret（fit_critic_real "
                         "--critic-target gae 的产物），用于把 critic 目标从 MC 换成 TD(λ)")
    ap.add_argument("--critic-real", type=Path, default=ROOT / "runs/wm_real/real_returns.pt",
                    help="--critic-source real 用的真回报文件（scripts/build_real_returns.py 生成）")
    ap.add_argument("--posture-weight", type=float, default=0.0,
                    help="连续的姿态惩罚权重 -w*(relu(tilt-0.5)^2 + relu(0.6-height)^2)。"
                         "WM 的终止头几乎不触发，所以摔倒/歪斜在想象里没有梯度，这个项补上")
    ap.add_argument("--reward-mode", choices=("saturating", "linear", "huber", "huber_err"), default="saturating",
                    help="跟踪核：saturating=exp(-e/std^2)（大误差处梯度→0，实测 B 线中位 0.7m 全在死区）；"
                         "linear=-(e/std^2)（**任何误差都保持常梯度**，世界模型当裁判时必须用它）；"
                         "huber=根位置用 Huber(δ,cap)，其余线性（2026-11-02，cm 级主项）")
    ap.add_argument("--huber-delta", type=float, default=0.02, help="Huber δ（米），默认 2 cm")
    ap.add_argument("--huber-cap", type=float, default=0.15, help="Huber 远场截断（米），默认 15 cm")
    # ★ 2026-11-04 **奖励死区**：d > d_max 之后惩罚恒为常数、梯度 0（不再硬拉回参考）。
    #   依据：VLA 的 state46 里**没有 root 位置**（只有 29 关节 + 14 手 + 3 重力），
    #   所以大范围位置漂移它看不见；强行拉回等于拟合噪声。小漂移（姿态级）仍然拉。
    ap.add_argument("--executor", choices=("wbc", "sonic"), default="sonic",
                    help="token→动作 的执行器：sonic=官方 release decoder（默认，与采集/部署同口径）；"
                         "wbc=旧的 SuperMimic 策略")
    ap.add_argument("--sonic-decoder", type=Path, default=None,
                    help="--executor sonic 用的 ONNX（默认 release/model_decoder.onnx）")
    ap.add_argument("--reward-d-max", type=float, default=0.30,
                    help="huber_err 的死区（米/弧度）；0=关闭（旧行为，永远线性拉回）。推荐 0.30")
    ap.add_argument("--reward-delta-pos", type=float, default=0.05,
                    help="huber_err 位置 δ（米）；R1=0.05，R2=0.02，R3=0.005，R4=0.002，R5=0.001")
    ap.add_argument("--term-off", action="store_true",
                    help="关掉想象 rollout 里的终止/存活折扣（survival≡1）。"
                         "新 WM 的朝向通道弱 ⇒ soft_termination 误触发（survival 0.25–0.49）")
    ap.add_argument("--reward-delta-vel", type=float, default=0.05,
                    help="huber_err **速度项**的 Huber δ（m/s / rad/s）；配合 --reward-vel-scale 使用")
    ap.add_argument("--reward-delta-ori", type=float, default=0.05,
                    help="huber_err 朝向 δ（弧度）；与 delta-pos 同步退火")
    ap.add_argument("--reward-vel-scale", type=float, default=1.0,
                    help="奖励里速度两项的系数（0=关）。WM 的速度通道最不准（qd_corr~0.64）")
    # ★ 2026-10-29（照 mjlab 交接文档 §1 的 reward 口径）：
    #   他们 root 项 w=1.0 / std=0.2(pos)、0.25(ori)，并记录"root 项原来在 4 cm 处饱和"。
    #   我们 ckpt 里是 w_root=0.5 / root_pos_std=0.3 ⇒ root 权重只有一半、核更钝。
    ap.add_argument("--reward-root-weight", type=float, default=None,
                    help="覆盖 root_pos/root_ori 的奖励权重（我们 ckpt 是 0.5；mjlab 用 1.0）")
    ap.add_argument("--reward-root-pos-std", type=float, default=None,
                    help="覆盖 root_pos_std（我们 ckpt 是 0.3；mjlab 用 0.2）")
    ap.add_argument("--reward-root-ori-std", type=float, default=None,
                    help="覆盖 root_ori_std（我们 ckpt 是 0.4；mjlab 用 0.25）")
    ap.add_argument("--reward-body-pos-std", type=float, default=None,
                    help="覆盖 body_pos_std（ckpt 0.3；新线 cm 级用 0.05）")
    ap.add_argument("--reward-body-ori-std", type=float, default=None,
                    help="覆盖 body_ori_std（ckpt 0.4；新线用 0.10）")
    ap.add_argument("--wm-ckpt", type=Path, default=None,
                    help="可选：用 A' 重训的 WM 覆盖 --wbc-ckpt 里的 wm（pretrain_wm 的输出）")
    ap.add_argument("--allow-legacy-reward", action="store_true",
                    help="放行旧奖励架构（mono / tail=0 / posture>0）。默认禁用")
    ap.add_argument("--root-ori-mode", choices=("mono", "split", "split_huber"), default="split",
                    help="root 朝向奖励：mono=单块（旧行为）；split=拆 tilt/yaw/yaw_rate（1.0/0.2/0.3）。"
                         "SONIC token 按定义 yaw-相对，绝对 yaw 架构上拿不到，满权重只会制造不可学的常数项")
    ap.add_argument("--tail-cap-slope", type=float, default=0.0,
                    help="e2：hinge 尾部越过 cap 之后保留的斜率（0=旧行为，梯度在远场为 0；"
                         "0.5=线性延续，远场仍有梯度）。B 线 52%% 的时刻在 0.5 m 以外。")
    ap.add_argument("--tail-weight", type=float, default=2.0,
                    help="hinge 尾部权重 λ（0=关）。exp 核两端梯度都趋 0，纯 exp 会拿尾部换中位；"
                         "mjlab 实测 λ=2.0 把 pos p90 从 52.1 cm 压到 26.9 cm")
    ap.add_argument("--tail-pos-thr", type=float, default=None,
                    help="根位置 hinge 阈值 e_h（米）。空=reward_shape.ROOT_POS_THR(0.35)；"
                         "E=1.0 课程必须按 scripts/calibrate_hinge_5round.py 的标定值传。")
    ap.add_argument("--tail-pos-cap", type=float, default=None,
                    help="根位置 hinge 饱和上限（米）。空=reward_shape.ROOT_POS_CAP(1.0)。")
    ap.add_argument("--yaw-hinge-lambda", type=float, default=0.3,
                    help="AB6：yaw 远场 hinge 的 λ（0=关掉远场补偿，只剩 exp 近场）。"
                         "原来写死成 reward_shape.YAW_HINGE_LAMBDA=0.3 ⇒ 消融没法做。")
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
    ap.add_argument("--token-quant", default="none", choices=["none", "ste", "hard"],
                    help="token 空间的 FSQ 网格量化（详见 diff_vla_wbc/quant.py）："
                         "none=连续喂(旧) / ste=前向格点+反向恒等(推荐，部署须同步量化) "
                         "/ hard=硬量化、梯度归零(消融对照，应当训不动)")
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
    # ---- DDP（torchrun 多进程时生效；单进程行为完全不变）----
    _world = int(_os.environ.get("WORLD_SIZE", "1") or 1)
    _local = int(_os.environ.get("LOCAL_RANK", "0") or 0)
    _is_main = True
    _rank = 0
    _base_seed = int(args.seed)
    if _world > 1:
        import torch.distributed as _dist
        if not _dist.is_initialized():
            _dist.init_process_group(backend="nccl")
        _rank = _dist.get_rank()
        _is_main = (_rank == 0)
        # ★ 2026-11-01（8 卡）：**模型构造必须用同一个种子**。原来直接 +rank，
        #   于是 LoRA 的 kaiming 初值各 rank 不同；DDP 只同步梯度 ⇒ 8 个 rank 越走越远，
        #   rank0 存出来的 ckpt 并不是"全局 batch 那个"。数据侧靠 args.seed 再 +rank。
        args.seed = _base_seed + _rank              # 每卡不同种子 ⇒ 抽不同的 batch
        torch.manual_seed(_base_seed)               # 相同初值（LoRA/critic/state_extra）
        device = torch.device(f"cuda:{_local}")
        args.device = str(device)
        print(f"[ddp] rank={_rank}/{_world} local_rank={_local} device={device} "
              f"data_seed={args.seed} init_seed={_base_seed} "
              f"sync_gradients={bool(args.sync_gradients)}", flush=True)
    else:
        torch.manual_seed(_base_seed)
        device = torch.device(args.device)
    files = collect_npz(args.motion)

    fingerprint = {
        k: (str(v) if isinstance(v, Path) else (list(map(str, v)) if isinstance(v, list) else v))
        for k, v in vars(args).items()
    }
    fingerprint["_wbc_sha256_1m"] = _sha256_head(args.wbc_ckpt)
    fingerprint["_n_clips"] = len(files)
    if _is_main:
        (out / "args.json").write_text(json.dumps(fingerprint, indent=1, default=str))
    print(f"[audit] 参数指纹 -> {out / 'args.json'}", flush=True)

    # ---------------------------------------------------------------- bundle（冻结执行器 + WM）
    from diff_vla_wbc.isaac_joint import is_isaac_checkpoint, load_isaac_bundle

    if not is_isaac_checkpoint(args.wbc_ckpt):
        raise SystemExit(f"{args.wbc_ckpt}: 只接受 Isaac/trackerlab WBC ckpt（含 pi/wm）")
    # ★ 2026-10-29（照 mjlab 交接文档 §1 的 reward 口径）：root 项 w=1.0 / std=0.2(pos)、0.25(ori)，
    #   他们记录"root 项原来在 4 cm 处饱和"。我们 ckpt 里是 w_root=0.5 / root_pos_std=0.3。
    #   走 load_isaac_bundle 的 reward_cfg 通道（在构造 RewardCfg 时合并），不改运行中的对象。
    _reward_override = {}
    _rw = getattr(args, "reward_root_weight", None)
    if _rw is not None:
        _reward_override.update(w_root_pos=float(_rw), w_root_ori=float(_rw))
    if getattr(args, "reward_root_pos_std", None) is not None:
        _reward_override["root_pos_std"] = float(args.reward_root_pos_std)
    if getattr(args, "reward_root_ori_std", None) is not None:
        _reward_override["root_ori_std"] = float(args.reward_root_ori_std)
    if getattr(args, "reward_body_pos_std", None) is not None:
        _reward_override["body_pos_std"] = float(args.reward_body_pos_std)
    if getattr(args, "reward_body_ori_std", None) is not None:
        _reward_override["body_ori_std"] = float(args.reward_body_ori_std)
    bundle = load_isaac_bundle(
        args.wbc_ckpt, device=device, npz=files,
        packed_motion=args.motion_packed, obs_mode=None, hist=None, token_dim=64,
        reward_cfg=(_reward_override or None),
    )
    # ★★ 2026-11-04：**执行器口径**。默认 `wbc`（旧行为，SuperMimic 策略把 token 变动作），
    #   `sonic` = 用官方 SONIC release decoder（冻结、可微）替换 `bundle.act`。
    #   为什么要它：我们的采集与部署都是 SONIC decoder；如果微调还用 SuperMimic 把 token
    #   变动作，优化出来的 token 对部署没有意义（"训 A 部 B"）。
    if str(getattr(args, "executor", "wbc")) == "sonic":
        from diff_vla_wbc.sonic_isaac import SonicIsaacBundle  # noqa: PLC0415
        bundle = SonicIsaacBundle(
            bundle,
            onnx_path=getattr(args, "sonic_decoder", None) or None,
            token_quant=str(getattr(args, "token_quant", "none") or "none"),
            device=device,
        )
        print(f"[AUDIT][executor] sonic_release_decoder onnx={bundle.executor.onnx_path} "
              f"obs_dim={bundle.obs_dim}（动作返回 STANDARD 序）", flush=True)
    # ★ 2026-11-05：奖励口径改为调**唯一实现**（`diff_vla_wbc/reward_cfg.py`），
    #   补算奖励脚本走同一个函数 ⇒ 两侧不可能分叉。
    from diff_vla_wbc.reward_cfg import apply_reward_cfg, reward_cfg_summary  # noqa: PLC0415
    apply_reward_cfg(bundle, args)
    if getattr(args, "wm_ckpt", None):
        _wp = Path(args.wm_ckpt)
        if not _wp.exists():
            raise SystemExit(f"[fail] --wm-ckpt 不存在: {_wp}")
        _wd = torch.load(str(_wp), map_location="cpu", weights_only=False)
        _wsd = _wd.get("wm", _wd)
        bundle.wm.load_state_dict(_wsd, strict=True)
        print(f"[wm] swapped from {_wp} (arch={getattr(bundle, 'wm_arch', '?')})", flush=True)
    # ★ 2026-10-31：**旧奖励架构已被移出默认路径**（用户口径："把之前的架构直接删了"）。
    #   新架构 = root 朝向拆 tilt/yaw/yaw_rate + hinge 尾部。想跑旧的必须显式加 --allow-legacy-reward，
    #   否则 mono 或 tail=0 直接报错 —— 防止某轮"忘了传参"就又跑回旧行为（d2 就是这么中招的）。
    if not bool(getattr(args, "allow_legacy_reward", False)):
        _bad = []
        if str(getattr(args, "root_ori_mode", "split")) not in ("split", "split_huber"):
            _bad.append(f"root_ori_mode={args.root_ori_mode}（应为 split）")
        if float(getattr(args, "tail_weight", 2.0) or 0.0) <= 0.0:
            _bad.append(f"tail_weight={args.tail_weight}（应 >0）")
        if float(getattr(args, "posture_weight", 0.0) or 0.0) != 0.0:
            _bad.append(f"posture_weight={args.posture_weight}（该项已删除，应恒 0）")
        if _bad:
            raise SystemExit("[reward] 旧奖励架构已禁用：" + "; ".join(_bad)
                             + "。要跑旧行为请显式加 --allow-legacy-reward")
    # ★ 2026-11-01：根位置 hinge 的 e_h/cap 已并入 apply_reward_cfg（按轮覆盖）。
    # ★ 2026-10-31（AB6）：远场 hinge λ 同样并入。
    print(f"[AUDIT][reward-canon] {reward_cfg_summary(bundle)}", flush=True)
    print(f"[AUDIT][reward] root_ori_mode={bundle.root_ori_mode} "
          f"tail_weight={bundle.tail_weight} "
          f"posture_weight={getattr(bundle, 'posture_weight', 0.0)}（已删除，恒 0） "
          f"tail_cap_slope={bundle.tail_cap_slope:g} "
          f"tail_pos_thr={getattr(bundle, 'tail_pos_thr', 'default')} "
          f"tail_pos_cap={getattr(bundle, 'tail_pos_cap', 'default')} "
          f"yaw_hinge_lambda={bundle.yaw_hinge_lambda:g} "
          f"yawrate_weight={getattr(bundle, 'yawrate_weight', 'default')} "
          f"mode={bundle.reward_mode} vel_scale={bundle.reward_vel_scale} "
          f"root_w={getattr(bundle.mdp.cfg,'w_root_pos','?')} "
          f"root_pos_std={getattr(bundle.mdp.cfg,'root_pos_std','?')} "
          f"root_ori_std={getattr(bundle.mdp.cfg,'root_ori_std','?')}", flush=True)
    print(f"[AUDIT][reward] mode={bundle.reward_mode} vel_scale={bundle.reward_vel_scale} "
          f"(速度通道 qd_corr≈0.64 不可信；0=精确扣掉两项)", flush=True)
    # 2026-10-29：原来这里硬编码 0.0，等于把"歪/快倒"的连续惩罚关掉了
    # （world model 的终止头几乎不触发，所以摔倒在想象里没有梯度）。
    bundle.posture_weight = float(getattr(args, "posture_weight", 0.0) or 0.0)
    if _reward_override:
        _cfg = getattr(getattr(bundle, "mdp", None), "cfg", None)
        print(f"[AUDIT][reward] root 覆盖 {_reward_override} ⇒ "
              f"w_root={getattr(_cfg, 'w_root_pos', '?')}/{getattr(_cfg, 'w_root_ori', '?')} "
              f"std={getattr(_cfg, 'root_pos_std', '?')}/{getattr(_cfg, 'root_ori_std', '?')}",
              flush=True)
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
    # ---------------------------------------------------------------- 多轮闭环：π_k → π_{k+1}
    # 没有这一段，每一轮都从官方底模重新开始，"闭环"就断在第 1 轮。
    # 载入的是上一轮 `save_vla` 写的 `vla_*.pt`（= {step, fm_steps, fm:{...}}），
    # 只覆盖**可训**张量（LoRA + proj_out），冻结的底模逐位不变 ⇒ π_k = 底模 + 这份 LoRA。
    # 必须放在 apply_lora 之后：LoRA 层要先存在，否则整份状态字典都会是 unexpected。
    if str(getattr(args, "init_vla", "") or ""):
        _iv = Path(args.init_vla)
        if not _iv.exists():
            raise SystemExit(f"[fail] --init-vla 不存在: {_iv}")
        if int(args.lora_rank) <= 0:
            raise SystemExit("[fail] --init-vla 需要 --lora-rank > 0（round-trip 靠 LoRA 名对齐）")
        _d = torch.load(str(_iv), map_location="cpu", weights_only=False)
        _sd = _d.get("fm") if isinstance(_d, dict) and "fm" in _d else _d
        if not isinstance(_sd, dict):
            raise SystemExit(f"[fail] --init-vla {_iv} 不是状态字典（也没有 'fm'）")
        _missing, _unexpected = vlm.load_state_dict(_sd, strict=False)
        if _unexpected:
            raise SystemExit(
                f"[fail] --init-vla unexpected={len(_unexpected)}（例：{list(_unexpected)[:3]}）；"
                f"LoRA 配置必须与保存时一致（rank={args.lora_rank} targets={args.lora_targets} "
                f"also_train={'proj_out' if args.lora_train_proj_out else '-'}）")
        _lora_b = [v for k, v in _sd.items() if k.endswith("lora_B")]
        _absmean = (sum(float(v.float().abs().mean()) for v in _lora_b) / max(1, len(_lora_b)))
        if isinstance(_d, dict) and _d.get("state_extra_scale") is not None:
            args._state_extra_scale_ckpt = torch.as_tensor(_d["state_extra_scale"]).float().reshape(-1)
            print(f"[AUDIT][init-vla] ckpt 带 state_extra_scale="
                  f"{[round(float(x), 5) for x in args._state_extra_scale_ckpt]}")
        print(f"[AUDIT][init-vla] loaded {_iv}: fm={len(_sd)} "
              f"missing(frozen)={len(_missing)} unexpected=0 lora_B_absmean={_absmean:.6f} "
              f"saved_step={_d.get('step') if isinstance(_d, dict) else '?'} "
              f"(= 本轮 π_k 起点)", flush=True)
        if all(float(v.float().abs().max()) == 0.0 for v in _lora_b):
            raise SystemExit(f"[fail] --init-vla {_iv} 的 lora_B 全 0 ⇒ 等于没加载")

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
        # ★★ 2026-11-04：**A' 覆盖不全时也要能用**。原来的填充只补"孤立的 1 帧"
        #   （`_map[miss] = _map[miss-1]` 是向量化的，连续缺失 > 1 帧就填不上），
        #   leash 数据的覆盖率是 74%（每 clip ~97%，但窗口起点前移 warm=20 帧、
        #   以及 clip 末尾掩码，会在每个窗口边界留下 20 帧空洞）⇒ 直接 exit 1。
        #   正确做法：
        #     ① 用**前向/后向累计**做真正的 fill（保证整张表都有效）；
        #     ② 另外记一张 `_covered` 掩码，**起点抽样只在覆盖帧上取**（见 --start-max-drift），
        #        这样不会把"邻居顶替"的假起点喂给训练。
        _map = torch.full((_n_frames,), -1, dtype=torch.long)
        _map[_fr] = torch.arange(_fr.numel())
        _covered = (_map >= 0)
        _idx = torch.arange(_n_frames)
        _last = torch.where(_covered, _idx, torch.full_like(_idx, -1))
        _last = torch.cummax(_last, 0).values                       # 前一个有效帧
        _first = torch.where(_covered, _idx, torch.full_like(_idx, _n_frames))
        _first = torch.flip(torch.cummin(torch.flip(_first, [0]), 0).values, [0])
        _nearest = torch.where(_last >= 0, _last, _first)            # 最近的**已覆盖帧号**
        # ★ 注意：`_map` 存的是**行号**，所以要用 `_map[_nearest]` 取那一行的行号，
        #   不能直接塞帧号（那就是刚刚那个 IndexError：615331 超出 611822 行）。
        _map = torch.where(_covered, _map, _map[_nearest])
        if int((_map < 0).sum()):
            raise SystemExit(f"[fail] 连前后填充都救不了：{int((_map < 0).sum())} 帧")
        print(f"[v3] 帧覆盖 {float(_covered.float().mean())*100:.1f}%"
              f"（{int(_covered.sum())}/{_n_frames}）；未覆盖帧用最近邻顶上，"
              f"且**不会被抽作起点**", flush=True)
        _recorded_states = _real["states"][_map].to(device)
        print(
            f"[v3] recorded_states 按帧对齐: {tuple(_recorded_states.shape)} "
            f"（原始转移 {tuple(_real['states'].shape)}，补 {int((~_covered).sum())} 帧 = 未采到的帧）",
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
    if args.wbc_token_window == "hold" and _lib_kind not in ("command", "command_exec") and not args.allow_library_mismatch:
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

    # ---------------------------------------------------------------- B 线重规划起点（评审 §3）
    _b_train = _b_eval = None
    _self_tgt = None
    if getattr(args, "b_self_targets", None):
        _p_self = Path(args.b_self_targets)
        if not _p_self.exists():
            raise SystemExit(f"[fail] --b-self-targets 不存在: {_p_self}")
        _self_tgt = torch.load(str(_p_self), map_location="cpu", weights_only=False)
        _m = _self_tgt.pop("_meta", {}) or {}
        print(f"[b-self] 真值标签：{_m.get('n_dumps', '?')} 条 dump / {_m.get('n_rows', '?')} 个时刻；"
              f"realized_err p25={(_m.get('err_quantiles') or {}).get('p25', float('nan')):.3f} "
              f"p50={(_m.get('err_quantiles') or {}).get('p50', float('nan')):.3f} m；"
              f"筛选阈值={getattr(args, 'b_self_err_max', None)}", flush=True)
    if str(getattr(args, "start_from_b", "") or ""):
        from diff_vla_wbc.b_start import BStartSource  # noqa: PLC0415
        _h_req = max(int(args.horizon), int(getattr(args, "eval_horizon", 0) or 0))
        _b_train = BStartSource(
            args.start_from_b, bundle, lib,
            stride=int(args.start_from_b_stride), horizon=_h_req,
            order=str(args.start_from_b_order), split="train",
            strict_online=not bool(args.b_start_nonstrict), log=print,
            self_targets=_self_tgt, max_err=getattr(args, "b_self_err_max", None),
            max_e_inst=getattr(args, "b_start_max_err", None),
            fresh_glob=(args.b_fresh_glob or None), fresh_weight=float(args.b_fresh_weight),
        )
        try:
            _b_eval = BStartSource(
                args.start_from_b, bundle, lib,
                stride=int(args.start_from_b_stride), horizon=_h_req,
                order=str(args.start_from_b_order), split="eval",
                strict_online=not bool(args.b_start_nonstrict), log=print,
                self_targets=_self_tgt, max_err=getattr(args, "b_self_err_max", None),
                max_e_inst=getattr(args, "b_start_max_err", None),
                fresh_glob=(args.b_fresh_glob or None), fresh_weight=float(args.b_fresh_weight),
            )
        except SystemExit:
            _b_eval = None
        if _b_eval is None or _b_eval.n_rows == 0:
            print("[b-start] WARN: eval 切分为 0 行（数据太少），评测仍用录制状态", flush=True)
            _b_eval = None
        print(f"[b-start] 训练起点 {_b_train.n_rows} 行 / 评测起点 "
              f"{0 if _b_eval is None else _b_eval.n_rows} 行；"
              "burn-in 关闭、DR 不叠加", flush=True)

    # ---------------------------------------------------------------- 训练 / 评测 provider
    # 2026-11-02：参考起点加权（phase/task）
    _start_weights = None
    if str(args.start_sampler) != "uniform":
        if str(args.start_sampler) == "task":
            _start_weights = _build_task_weights(lib)
            _src = "<task-balanced>"
        else:
            _p = Path(args.start_weights) if args.start_weights else (
                Path(a.out).resolve().parents[1] / "runs/wm_real/exp_start_weights.pt")
            if not _p.exists():
                raise SystemExit(
                    f"[fail] --start-weights 不存在: {_p}；先跑 reward_from_objects.py --weights-out")
            _d = torch.load(str(_p), map_location="cpu", weights_only=False)
            _start_weights = (_d["weights"] if isinstance(_d, dict) else _d).float()
            _src = str(_p)
        if int(_start_weights.numel()) != int(lib.num_frames):
            raise SystemExit(
                f"[fail] start_weights {int(_start_weights.numel())} != 库帧数 {int(lib.num_frames)}")
        print(f"[start] sampler={args.start_sampler} src={_src} "
              f"mean={float(_start_weights.mean()):.3f} max={float(_start_weights.max()):.3f} "
              f"min={float(_start_weights.min()):.3f}", flush=True)

    # ★ 2026-11-04：按"起点漂移上限"建帧权重（只在 --start-source recorded 时有意义）
    if float(getattr(args, "start_max_drift", 0.0) or 0.0) > 0.0:
        if _real is None:
            raise SystemExit("[start][fail] --start-max-drift 需要 --real-transitions（A' 转移）")
        _delta = float(args.start_max_drift)
        _n_fr = int(lib.num_frames)
        _fr = _real["frames"].long()
        _st = _real["states"].float()
        _pk = torch.load(ROOT / "runs/isaac_motion/ha_cmd_exec/packed.pt",
                         map_location="cpu", weights_only=False)
        _ref_all = torch.as_tensor(_pk["ref"]).float()
        _off = (_st[:, 0:3] - _ref_all[_fr][:, 0:3]).norm(dim=-1)
        _w = torch.zeros(_n_fr, dtype=torch.float32)
        _ok = _off <= _delta
        # ★ 只在**真的采到过**的帧上取起点（`_covered` 来自上面的对齐表）
        try:
            _ok = _ok & _covered[_fr].to(_ok.device)      # _covered 按**帧号**索引
        except Exception:  # noqa: BLE001
            pass
        _w[_fr[_ok]] = 1.0
        # 没采到的帧（clip 末几帧）沿用前一帧的判定，避免出现孤立的 0 洞
        _zero = (_w == 0).nonzero(as_tuple=False).flatten()
        if _zero.numel():
            _w[_zero] = _w[(_zero - 1).clamp(min=0)]
        _start_weights = _w
        args.start_sampler = "weights"
        _kept = float((_off <= _delta).float().mean())
        print(f"[start] drift-cap δ={_delta:.3f} m -> 保留 {_kept*100:.1f}% 的转移行作起点 "
              f"（全量偏移 p50={float(_off.median())*100:.2f}cm "
              f"p90={float(_off.quantile(0.9))*100:.2f}cm p99={float(_off.quantile(0.99))*100:.2f}cm）",
              flush=True)
        if _kept < 0.5:
            print(f"[start][warn] δ={_delta} 只保留了 {_kept*100:.0f}% 的行，考虑放大 δ", flush=True)

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
        b_starts=_b_train,
        start_weights=_start_weights, start_audit=bool(args.start_audit),
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
        b_starts=_b_eval,
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
        token_quant=str(args.token_quant),
        head_eval_mode=not args.head_train_mode,
        grad_accum=args.grad_accum, grad_ratio_every=args.grad_ratio_every,
        eval_horizon=int(args.eval_horizon), amp=str(args.amp),
        train_wbc=bool(args.train_wbc), wbc_lr=float(args.wbc_lr),
        wbc_unfreeze=str(args.wbc_unfreeze),
        sync_gradients=bool(args.sync_gradients),
    )
    if args.v3:
        from diff_vla_wbc.v3_step import JointTrainerV3, V3Config  # noqa: PLC0415

        v3cfg = V3Config(
            lambda_r=args.lambda_r, lambda_hand=args.lambda_hand, use_critic=args.critic,
            critic_lr=args.critic_lr, gamma=args.gamma, lam=args.lam,
            push_std=args.push_std, push_ang=args.push_ang,
            anchor_weight=args.anchor_weight,
            token_bc_weight=float(args.token_bc_weight),
            token_bc_source=str(args.token_bc_source),
            token_bc_span=int(args.token_bc_span),
            trust_eps=args.trust_eps, trust_weight=args.trust_weight,
            act_eps=args.act_eps, act_weight=args.act_weight,
            act_penalty=str(args.act_penalty),
            reward_horizon=int(getattr(args, "reward_horizon", 0) or 0),
            action_target_weight=float(getattr(args, "action_target_weight", 0.0) or 0.0),
            action_target_eta=float(getattr(args, "action_target_eta", 0.15) or 0.15),
            action_target_steps=int(getattr(args, "action_target_steps", 20) or 20),
            vla_state_from_burnin=bool(args.vla_state_from_burnin),
            critic_grad_through_state=bool(args.critic_grad_through_state),
            trust_center=str(args.trust_center),
            hybrid_err=float(getattr(args, "hybrid_err", 0.5) or 0.5),
            critic_state_grad_scale=float(args.critic_state_grad_scale),
            critic_rel_clip=float(args.critic_rel_clip),
            act_hinge_cap=float(args.act_hinge_cap),
            critic_source=args.critic_source,
            critic_use_diff=bool(args.critic_use_diff),
            grad_log_every=args.grad_log_every,
            clip_head=args.clip_head,
            adv_baseline=str(getattr(args, "adv_baseline", "none") or "none"),
            adv_tau=float(getattr(args, "adv_tau", 0.0) or 0.0),
            state_extra=str(getattr(args, "state_extra", "none") or "none"),
            state_extra_scale_init=float(getattr(args, "_state_extra_scale_init", 1.0)),
            wbc_kl_anchor=float(getattr(args, "wbc_kl_anchor", 0.0) or 0.0),
            vla_kl_anchor=float(getattr(args, "vla_kl_anchor", 0.0) or 0.0),
        )
        # processor 传给 trainer 是为了 A1（P1-6）：覆写 state 时要复用 ckpt 自己的归一化
        trainer = JointTrainerV3(bundle, vlm, codec, cfg=cfg, v3=v3cfg, device=device,
                                 processor=processor)
        _se_ckpt = getattr(args, "_state_extra_scale_ckpt", None)
        if _se_ckpt is not None and getattr(trainer, "state_extra_scale", None) is not None:
            with torch.no_grad():
                trainer.state_extra_scale.copy_(
                    torch.as_tensor(_se_ckpt, dtype=torch.float32,
                                    device=trainer.state_extra_scale.device).reshape(-1))
            print(f"[AUDIT][state-extra] 从 --init-vla 恢复 scale="
                  f"{[round(float(x), 5) for x in trainer.state_extra_scale.detach().cpu()]}")
        elif getattr(trainer, "state_extra_scale", None) is not None:
            print("[AUDIT][state-extra] 无 ckpt 里的 scale ⇒ 用初值 "
                  f"{[round(float(x), 5) for x in trainer.state_extra_scale.detach().cpu()]}")
        if args.critic_source == "real":
            _rrp = Path(args.critic_real)
            if not _rrp.exists():
                raise SystemExit(f"[fail] 缺真回报文件 {_rrp}；先跑 scripts/build_real_returns.py")
            _rr = torch.load(str(_rrp), map_location="cpu", weights_only=False)
            # 2026-11-04：口径检查**必须在这里**（加载后立刻做）。
            #   原来这段被写在下面那个 `raise SystemExit` 之后 ⇒ 是**死代码**，
            #   于是 runs/wm_real/real_returns.pt（reward_mode='saturating', vel_scale=1.0）
            #   一直没被拦下——用它训 grounded critic 等于把 critic 训在**另一套奖励**上。
            _rr_meta = _rr.get("meta") or {}
            _want_meta = {
                "reward_mode": str(args.reward_mode),
                "reward_vel_scale": float(args.reward_vel_scale),
                "reward_delta_pos": float(getattr(args, "reward_delta_pos", 0.05)),
                "reward_delta_ori": float(getattr(args, "reward_delta_ori", 0.05)),
                "root_ori_mode": str(getattr(args, "root_ori_mode", "split")),
            }
            _bad_meta = [(k, _rr_meta[k], v) for k, v in _want_meta.items()
                         if k in _rr_meta and _rr_meta[k] != v]
            if _bad_meta:
                _detail = "；".join(f"{k}: 文件={got!r} 训练={want!r}"
                                    for k, got, want in _bad_meta)
                raise SystemExit(
                    f"[fail] grounded critic 的回报文件口径不符（{_rrp}）：{_detail}。\n"
                    f"      重建：python3 scripts/build_real_returns.py --reward-mode "
                    f"{_want_meta['reward_mode']} --reward-vel-scale {_want_meta['reward_vel_scale']} "
                    f"--reward-delta-pos {_want_meta['reward_delta_pos']} "
                    f"--reward-delta-ori {_want_meta['reward_delta_ori']} "
                    f"--root-ori-mode {_want_meta['root_ori_mode']} ...")
            _rr_ret = _rr["ret"]
            if getattr(args, "critic_ret", None) is not None:
                _ov = torch.load(str(args.critic_ret), map_location="cpu", weights_only=False)
                _ov_ret = _ov["ret"]
                if int(_ov_ret.shape[0]) != int(_rr_ret.shape[0]):
                    raise SystemExit(f"[fail] --critic-ret 长度 {int(_ov_ret.shape[0])} != "
                                     f"critic-real ret {int(_rr_ret.shape[0])}")
                _rr_ret = _ov_ret
                print(f"[critic] ret 被 --critic-ret 覆盖：kind={_ov.get('kind')} "
                      f"lam={_ov.get('lam')} n={int(_rr_ret.shape[0])}", flush=True)
            if "states" in _rr and "ref_rows" in _rr:
                # P1-2：MC 目标文件自带 states/ref_rows（按 episode 采的 on-policy 状态）
                print(f"[P1-2] critic 目标来自 on-policy MC：n={int(_rr_ret.shape[0])} "
                      f"（{(_rr.get('meta') or {}).get('episodes', '?')} 集）", flush=True)
                trainer.attach_real_returns(_rr_ret, _rr["states"].float(), _rr["ref_rows"].float())
            elif _recorded_states is None:
                raise SystemExit(
                    "[fail] --critic-source real 需要一个带 states/ref_rows 的文件（如 "
                    "runs/wm_real/real_transitions_v2b.pt 的 ret/states/ref_rows），"
                    "或 --start-source recorded")
            else:
                if int(_rr_ret.shape[0]) != int(_recorded_states.shape[0]):
                    raise SystemExit(
                        f"[fail] real returns 长度 {int(_rr_ret.shape[0])} != recorded_states "
                        f"{int(_recorded_states.shape[0])}；"
                        "用同一个 --reward-mode/--reward-vel-scale 重建")
                trainer.attach_real_returns(_rr_ret, _recorded_states,
                                            bundle.motion.ref.detach().float())
        if args.critic_ckpt:
            _cp = Path(args.critic_ckpt)
            if not _cp.exists():
                raise SystemExit(f"[fail] 缺 critic ckpt {_cp}；先跑 scripts/fit_critic_real.py")
            _c = torch.load(str(_cp), map_location="cpu", weights_only=False)
            _sd = _c.get("critic") or _c
            _miss, _unexp = trainer.critic.load_state_dict(_sd, strict=False)
            _meta = _c.get("meta") or {}
            if bool(_meta.get("use_diff", False)) != bool(args.critic_use_diff):
                raise SystemExit(
                    f"[fail] critic ckpt 的 use_diff={_meta.get('use_diff')} 与 "
                    f"--critic-use-diff={args.critic_use_diff} 不一致")
            print(f"[critic] prefit loaded {_cp} v_ev={_meta.get('v_ev_val')} "
                  f"missing={len(_miss)} unexpected={len(_unexp)}", flush=True)
        if _is_main:
            (out / "v3_config.json").write_text(json.dumps(v3cfg.__dict__, indent=1, default=str))
        # 口径的一行真相：token 监督 / 手部 / 锚定 / DR 分桶 / critic 自举 / WM 保鲜
        _pn, _pm = float(args.dr_p_nominal), float(args.dr_p_mid)
        print(
            f"[AUDIT][v3] token_BC=OFF hand_mask=64:78 lam_hand={args.lambda_hand} "
            f"lam_r={args.lambda_r} anchor={args.anchor_weight} "
            f"trust=({args.trust_eps},{args.trust_weight}) "
            f"dr={bool(args.dr)} buckets={_pn:.2f}/{_pm:.2f}/{max(0.0, 1-_pn-_pm):.2f} "
            f"dr_max(xy,yaw,dq,dqd)=({args.dr_xy_max},{args.dr_yaw_max},{args.dr_dq_max},{args.dr_dqd_max}) "
            f"start={args.start_source} b_start={_b_train.n_rows if _b_train else 0} "
            f"burnin={0 if args.no_burnin else args.burnin_steps} "
            f"push=({args.push_std},{args.push_ang}) "
            f"critic={args.critic}/{args.critic_source} lam={args.lam} gamma^H={args.gamma ** args.horizon:.4f} "
            f"vla_state_burnin={args.vla_state_from_burnin} train_wbc={bool(args.train_wbc)} "
            f"act=({args.act_eps},{args.act_weight},{args.act_penalty}) "
            f"rewH={int(getattr(args, 'reward_horizon', 0) or 0)} "
            f"reward=({args.reward_mode},{args.reward_vel_scale}) "
            f"wm_refresh_every={args.wm_refresh_every} clip_head={args.clip_head} "
            f"eval_H={int(args.eval_horizon) or int(args.horizon)} amp={args.amp} "
            f"H={args.horizon} K={args.fm_steps} batch={args.batch_size} accum={args.grad_accum}",
            flush=True,
        )
    else:
        trainer = JointTrainerV2(bundle, vlm, codec, cfg=cfg, device=device)
    # ★ 2026-11-01（8 卡 DDP）：起点对齐 + 训练期采样/噪声的 RNG 按 rank 分开。
    #   构造阶段所有 rank 用同一 init_seed（保证初值逐位相同），到这里再切成 data_seed，
    #   让 FM 采样噪声、任何 torch 全局随机都与数据一样逐 rank 不同。
    if _world > 1:
        _ddp_broadcast_trainables(trainer)
        torch.manual_seed(_base_seed + _rank)
    if _is_main:
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
        if not _is_main:
            return {}
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
        m["token_quant"] = str(args.token_quant)
        if _is_main:
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
    _prof = _os.environ.get("J1_STEP_PROFILE") == "1"
    batch = next_batch()
    step = 0
    for step in range(1, int(args.steps) + 1):
        _p0 = time.perf_counter()
        nxt = next_batch()
        _p1 = time.perf_counter()
        batch.model_inputs = prefetch.get(batch.frames, lambda: memo.get(batch.frames))
        _p2 = time.perf_counter()
        metrics = trainer.train_step(batch)
        _p3 = time.perf_counter()
        if _prof:
            print(f"[prof] step={step} provider.sample={_p1 - _p0:.2f}s "
                  f"prefetch.get={_p2 - _p1:.2f}s train_step={_p3 - _p2:.2f}s", flush=True)
        metrics["step"] = step
        metrics["s_per_step"] = (time.time() - _t0) / step
        # 每一步都落盘：grad_ratio 只在打印步产生，只写在打印分支里会丢
        if _is_main:
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
            # ★ DDP：只有 rank0 落盘（两个进程同时写同一个 .pt 会写坏）
            if _is_main:
                trainer.save_joint(out / f"joint_{step:06d}.pt")
            if not args.train_wbc:
                assert_wbc_frozen(bundle, args.wbc_ckpt)
            if args.save_vla_every and step % args.save_vla_every == 0 and _is_main:
                p = trainer.save_vla(out / f"vla_{step:06d}.pt")
                print(f"[saved] {p}", flush=True)
        batch = nxt
        memo.clear()
        if deadline is not None and time.time() >= deadline:
            print(f"[budget] {args.time_budget_minutes:g} min 到达（step {step}）", flush=True)
            break

    # ---------------------------------------------------------------- 收尾
    vla_final = None
    if _is_main:
        trainer.save_joint(out / "joint_final.pt")
    frozen_ok = True if args.train_wbc else assert_wbc_frozen(bundle, args.wbc_ckpt)
    if _is_main:
        vla_final = trainer.save_vla(out / "vla_final.pt")
        print(f"[saved] {vla_final}", flush=True)
    final_eval = run_eval(int(step)) if eval_batches and step else {}
    summary = {
        "steps": int(step),
        "out": str(out),
        "wbc_ckpt": str(args.wbc_ckpt),
        "wbc_sha256_1m": fingerprint["_wbc_sha256_1m"],
        "vlm_ckpt": str(args.vlm_ckpt),
        "freeze_gate": ("skipped(wbc_trainable)" if args.train_wbc else bool(frozen_ok)),
        "train_wbc": bool(args.train_wbc),
        "fm_steps": int(args.fm_steps),
        "token_quant": str(args.token_quant),
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
    # 2026-11-02：起点覆盖审计
    if _is_main and args.start_audit:
        _frames = provider.audit_frames()
        if _frames.numel():
            torch.save({"frames": _frames}, out / "start_coverage.pt")
            _starts = torch.tensor([int(c.start) for c in lib.clips], dtype=torch.long)
            _names = [str(c.name) for c in lib.clips]
            _tasks = [nm.split("__")[0] for nm in _names]
            _cid = (torch.searchsorted(_starts, _frames, right=True) - 1).clamp(0, len(_names) - 1)
            _uniq = torch.unique(_frames)
            _task_cnt: dict[str, int] = {}
            _used_clips = 0
            for i, nm in enumerate(_names):
                m = _cid == i
                c = int(m.sum())
                if c:
                    _used_clips += 1
                    t = str(nm).split("__")[0]
                    _task_cnt[t] = _task_cnt.get(t, 0) + c
            _cov = {
                "windows": int(_frames.numel()),
                "unique_frames": int(_uniq.numel()),
                "unique_frame_ratio": round(int(_uniq.numel()) / max(1, int(lib.num_frames)), 4),
                "unique_clips": _used_clips,
                "total_clips": len(_names),
                "per_task": _task_cnt,
                "sampler": str(args.start_sampler),
            }
            (out / "start_coverage.json").write_text(json.dumps(_cov, ensure_ascii=False, indent=1))
            print("[start][coverage] " + json.dumps(_cov, ensure_ascii=False), flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("[AUDIT][v2-summary] " + json.dumps(summary, default=str), flush=True)
    prefetch.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
