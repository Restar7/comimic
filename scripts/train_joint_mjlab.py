#!/usr/bin/env python3
"""联合微调（J1）· **mjlab / MuJoCo 后端**：VLA head ← 奖励/BC 反传，观测 **93**。

    VLA(GR00T head) ──token(64)──> SuperMimic WBC(mjlab) ──a(29)──> WM(mjlab) ──> reward
          ▲                                                                          │
          └────────────────────── ∂loss/∂head ◄── 想象 rollout (H 步) ◄────────────┘

与 `scripts/train_joint_v2.py`（Isaac 版）的关系：
* **算法完全同一份**：`diff_vla_wbc.train_v2.JointTrainerV2` / `v3_step.JointTrainerV3`、
  `fm_bc`、`token_critic`、`burnin`、`dr`、`v2_gates`、`fixed_eval` —— 一行不改；
* **只换后端**：执行器/世界模型/解析件换成 `diff_vla_wbc.mjlab_joint.MjlabJointBundle`（语义等价，
  但状态元组的角速度是**体系**，见 §1.3 of `docs/MUJOCO_MIGRATION_93OBS_20261028.md`）；
* **token 源 = 数据集录制标签**（`data/library/ft318_tokens.pt`），不重建 hold、不用侧车；
  这与 Isaac 锁定的 `--bc-token-space label` 是同一个空间（`docs/TOKEN_SPACE_ANCHOR_LOCKDOWN_20261028.md` F3）。

示例
----
```bash
# 1) 真值门禁（不训练，~2 分钟）
python3 scripts/train_joint_mjlab.py --wbc-ckpt <ckpt> --motion data/npz/g1_tasks \
    --tokens data/library/ft318_tokens.pt --vlm-ckpt <ckpt> --vlm-path <vlm> \
    --out runs/joint_mjlab_ft318 --gate-only

# 2) BC warmup（800 步，lambda_r=0）
python3 scripts/train_joint_mjlab.py --wbc-ckpt <ckpt> --motion data/npz/g1_tasks \
    --tokens data/library/ft318_tokens.pt --vlm-ckpt <ckpt> --vlm-path <vlm> \
    --out runs/joint_mjlab_ft318 --steps 800 --lambda-r 0 --lora-rank 32 --amp bf16

# 3) +奖励（v3，2000 步）
python3 scripts/train_joint_mjlab.py ... --v3 --steps 2800 --lambda-r 0.3 \
    --anchor-weight 1e-3 --trust-eps 0.2 --trust-weight 1.0
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "Isaac-GR00T"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_WBC = (
    ROOT / "third_party/SuperMimic-Reproduced/logs/wbc_ft318_mjlab/ckpt_latest.pt"
)
DEFAULT_VLM = ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights"
DEFAULT_TOKENS = ROOT / "data/library/ft318_tokens.pt"
DEFAULT_MOTION = ROOT / "data/npz/g1_tasks"
TOKEN_HORIZON = 40
TOKEN_DIM = 64


# --------------------------------------------------------------------------- helpers
def _sha256_head(path: Path, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def collect_npz(paths: list[Path]) -> list[Path]:
    """与 `build_library.py` / `TaskFrameIndex` 相同的顺序：排序后的 npz 列表。"""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.npz")))
        elif p.suffix == ".npz":
            files.append(p)
        else:
            raise SystemExit(f"[fail] --motion 只接受目录或 .npz：{p}")
    if not files:
        raise SystemExit("[fail] --motion 里没有 npz")
    return files


class TokenPack:
    """数据集录制 token（逐帧，(N,64)，与参考库的全局帧轴对齐）。"""

    def __init__(self, path: Path, device):
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        self.tokens = d["tokens"].float()
        self.names = d.get("names")
        self.space = str(d.get("space", "label"))
        self.device = device
        self.path = path
        self.n = int(self.tokens.shape[0])

    def chunk(self, frames: torch.Tensor) -> torch.Tensor:
        """(B,) -> (B,40,64)：一次查询给整段（部署口径）。"""
        idx = frames[:, None].to("cpu") + torch.arange(TOKEN_HORIZON)
        idx = idx.clamp(max=self.n - 1)
        return self.tokens[idx].to(self.device)

    def frame(self, frames: torch.Tensor) -> torch.Tensor:
        return self.tokens[frames.to("cpu")].to(self.device)

    def audit(self) -> str:
        return (
            f"[tokens] {self.path} | {self.n} frames x {self.tokens.shape[-1]} "
            f"| space={self.space} | clips={len(self.names) if self.names else '?'}"
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- 执行器 / 数据 ----
    ap.add_argument("--wbc-ckpt", type=Path, default=DEFAULT_WBC,
                    help="mjlab 训练 ckpt（B1: sm-train 产物 / B2: 联合微调产物）")
    ap.add_argument("--motion", nargs="+", type=Path, default=[DEFAULT_MOTION],
                    help="参考库 npz（full 库；参考帧/奖励要 body 状态）")
    ap.add_argument("--motion-packed", type=Path, default=None,
                    help="compact 库（可选；与 --motion 二选一，body 靠可微 FK 现算）")
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS,
                    help="数据集录制 token 包（与参考库同帧轴）")
    ap.add_argument("--xml-path", type=Path, default=None)
    ap.add_argument("--hand-dir", type=Path, default=None,
                    help="手部标签目录（318 npz 自带 g1_hand_joints_14）。默认关：mjlab policy 无手部输出")
    # ---- VLA ----
    ap.add_argument("--vlm-ckpt", type=Path, required=True)
    ap.add_argument("--vlm-path", type=Path, default=DEFAULT_VLM)
    ap.add_argument("--vlm-select-layer", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=64,
                    help="LoRA 秩（0 = 训全量 head 1.62B）。实测参数：rank8=8.4M / rank32=33.5M / "
                         "rank64=67M（4.1%%）/ rank128=134M（8.2%%）；显存 ≈ 3×参数量（fp32 权重+梯度+Adam）")
    ap.add_argument("--lora-alpha", type=float, default=None)
    ap.add_argument("--lora-targets", type=str, default="attn1,ff")
    ap.add_argument("--lora-train-proj-out", action="store_true", default=True)
    ap.add_argument("--amp", choices=("off", "bf16"), default="bf16")
    # ---- 训练 ----
    ap.add_argument("--out", type=Path, default=ROOT / "runs/joint_mjlab_ft318")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="实测(4090, H=16, LoRA32, bf16, λ_r=0.3): batch4=3.5s/step(25.6GB) "
                         "batch8=3.7s(26.4GB) batch16=3.9s(32.5GB) ⇒ 固定开销约 24GB，"
                         "batch 主要吃显存不吃饭时间。默认 64 是 H200(141GB) 的目标值；"
                         "4090(46GB) 请降到 16~32（batch16 = 32.5GB, batch32 ≈ 41GB）")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="batch 已经够大时保持 1；显存紧再靠 accum 换 batch（样本数= batch×accum）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--eval-horizon", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sync-gradients", dest="sync_gradients", action="store_true", default=None,
                    help="多卡梯度同步（torchrun 起 >1 进程时默认开）")
    ap.add_argument("--no-sync-gradients", dest="sync_gradients", action="store_false")
    # ---- 损失 ----
    ap.add_argument("--lambda-bc", type=float, default=0.0,
                    help="token 的监督（标签）权重。**Isaac v3 口径 = 0**（token 只由奖励监督）；"
                         ">0 只剩诊断用途（纯 BC 对照臂）")
    ap.add_argument("--lambda-r", type=float, default=0.0)
    ap.add_argument("--lambda-s", type=float, default=0.0,
                    help="state 项。318 库的根速度恒为 0（prepare_g1_tasks_npz.py:215），**别开**")
    ap.add_argument("--lambda-a", type=float, default=0.0)
    ap.add_argument("--lambda-hand", type=float, default=0.0,
                    help="手部辅助项（hand-only FM 掩码 mask[...,64:78]）；v3 默认 0.2")
    ap.add_argument("--state-vel-weight", type=float, default=0.05)
    ap.add_argument("--train-wbc", dest="train_wbc", action="store_true", default=False,
                    help="解冻执行器（对照臂）。主线必须**冻结**；解冻后 B_freeze 门会 FAIL，需跳过门禁")
    ap.add_argument("--wbc-lr", type=float, default=1e-6)
    ap.add_argument("--executor", default="supermimic", choices=("supermimic", "sonic"),
                    help="身体执行器：supermimic（旧）/ sonic（SONIC release decoder，冻结）")
    ap.add_argument("--decoder", type=Path, default=None,
                    help="SONIC decoder ONNX（默认 release 路径，见 diff_vla_wbc/sonic_mjlab.py）")
    ap.add_argument("--token-quant", default="none", choices=("none", "ste", "hard"),
                    help="FSQ 格点化；实测亚格外推 rel<1e-2 时默认 none")
    ap.add_argument("--fm-steps", type=int, default=4)
    ap.add_argument("--fm-lr", type=float, default=1e-5)
    ap.add_argument("--fm-clip", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--fm-grad-backend", default="pathwise")
    ap.add_argument("--fm-truncated-tail", type=int, default=1)
    ap.add_argument("--head-train-mode", action="store_true", default=False)
    ap.add_argument("--grad-ratio-every", type=int, default=50)
    ap.add_argument("--grad-log-every", type=int, default=0)
    ap.add_argument("--clip-head", type=float, default=10.0)
    # ---- v3 ----
    ap.add_argument("--v3", action="store_true")
    ap.add_argument("--critic", dest="critic", action="store_true", default=True)
    ap.add_argument("--no-critic", dest="critic", action="store_false")
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--critic-source", choices=("imagined", "real"), default="imagined")
    ap.add_argument("--critic-real", type=Path, default=None,
                    help="--critic-source real 的文件（含 ret/states/ref_rows）")
    ap.add_argument("--critic-ckpt", type=Path, default=None)
    ap.add_argument("--critic-ret", type=Path, default=None,
                    help="可选：用这个小文件 {ret,kind,lam} 覆盖 transitions 的 ret"
                         "（fit_critic_real.py --critic-target gae 的产物）")
    ap.add_argument("--anchor-weight", type=float, default=0.0)
    ap.add_argument("--trust-eps", type=float, default=0.0)
    ap.add_argument("--trust-weight", type=float, default=0.0)
    ap.add_argument("--trust-center", choices=("self", "label", "base", "prev"), default="self",
                    help="信任域中心：**self = 采集时策略自己真正下发的 token（推荐）**；"
                         "label = 录制标签（属于另一个执行器）；prev = π_k 快照；base = 底模输出")
    ap.add_argument("--act-eps", type=float, default=0.0)
    ap.add_argument("--act-weight", type=float, default=0.0)
    ap.add_argument("--act-hinge-cap", type=float, default=2.0)
    ap.add_argument("--push-std", type=float, default=0.0)
    ap.add_argument("--push-ang", type=float, default=0.0)
    ap.add_argument("--vla-state-from-burnin", action="store_true", default=False)
    ap.add_argument("--critic-grad-through-state", action="store_true", default=False)
    ap.add_argument("--critic-state-grad-scale", type=float, default=0.1)
    ap.add_argument("--critic-rel-clip", type=float, default=5.0)
    # ---- DR / burn-in ----
    ap.add_argument("--dr", dest="dr", action="store_true", default=True)
    ap.add_argument("--no-dr", dest="dr", action="store_false")
    ap.add_argument("--dr-xy-max", type=float, default=0.20)
    ap.add_argument("--dr-yaw-max", type=float, default=0.25)
    ap.add_argument("--dr-dq-max", type=float, default=0.05)
    ap.add_argument("--dr-dqd-max", type=float, default=0.30)
    ap.add_argument("--dr-p-nominal", type=float, default=0.25)
    ap.add_argument("--dr-p-mid", type=float, default=0.45)
    ap.add_argument("--burnin-steps", type=int, default=10)
    ap.add_argument("--no-burnin", action="store_true", default=False)
    ap.add_argument("--no-perturb", action="store_true", default=False)
    ap.add_argument("--perturb-xy", type=float, default=0.05)
    ap.add_argument("--perturb-yaw", type=float, default=0.10)
    ap.add_argument("--perturb-dq", type=float, default=0.05)
    ap.add_argument("--perturb-dqd", type=float, default=0.30)
    ap.add_argument("--obs-noise", type=float, default=0.0)
    # ---- 起点 / 真数据 ----
    ap.add_argument("--start-source", choices=("reference", "recorded"), default="reference")
    ap.add_argument("--start-from-b", type=str, default="",
                    help="B 线 dump 的 glob（MuJoCo collect 产物）；顺序用 --b-start-order")
    ap.add_argument("--b-start-order", default="standard", choices=("standard", "ha_v31"))
    ap.add_argument("--b-start-stride", type=int, default=20)
    ap.add_argument("--start-from-b-stride", type=int, default=None,
                    help="兼容别名（等同 --b-start-stride）")
    ap.add_argument("--b-start-nonstrict", action="store_true", default=False)
    ap.add_argument("--b-self-targets", type=Path, default=None,
                    help="build_b_self_targets.py 的产物（trust_center=self 必需）")
    ap.add_argument("--b-max-err", type=float, default=None,
                    help="只保留接下来 H 步实际漂移 <= 该值的重规划时刻（None=不筛）")
    ap.add_argument("--real-transitions", type=Path, default=None,
                    help="MuJoCo collect 的 transitions（WM 保鲜 / critic 真回报）")
    ap.add_argument("--wm-pretrain-steps", type=int, default=0,
                    help="训练**之前**先用 real transitions 把 WM 训稳，训完冻结（reward 固定靶）")
    ap.add_argument("--wm-refresh-every", type=int, default=0)
    ap.add_argument("--wm-refresh-steps", type=int, default=200)
    ap.add_argument("--wm-refresh-lr", type=float, default=1e-4)
    ap.add_argument("--wm-kl-anchor", type=float, default=1e-3)
    ap.add_argument("--reward-mode", choices=("saturating", "linear"), default="saturating")
    ap.add_argument("--reward-vel-scale", type=float, default=1.0)
    # ---- 保存 / 评测 ----
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=0)
    ap.add_argument("--eval-out", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--save-vla-every", type=int, default=200)
    ap.add_argument("--prefetch-vla", dest="prefetch_vla", action="store_true", default=True)
    ap.add_argument("--no-prefetch-vla", dest="prefetch_vla", action="store_false")
    ap.add_argument("--time-budget-minutes", type=float, default=0.0)
    # ---- 门禁 ----
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--gate-out", type=Path, default=None)
    ap.add_argument("--gate-batches", type=int, default=4)
    ap.add_argument("--gate-subset", type=str, default="all",
                    help="只跑这几道门（逗号分隔：A_codec,B_freeze,C_burnin,C2_token_contrast,D_fm_bc）；"
                         "all = 全跑。launcher 用「一道门一个进程」避免显存累积")
    return ap


# --------------------------------------------------------------------------- main
def main() -> int:
    args = build_parser().parse_args()
    if args.start_from_b_stride is not None:
        args.b_start_stride = int(args.start_from_b_stride)
    # 手部标签只喂 **VLA 头的辅助损失**（mjlab policy 只输出 29 维，不吃手部）。
    # 318 npz 自带 `g1_hand_joints_14`，所以默认就用 motion 目录；`bc_target` 需要它。
    if args.hand_dir is None and args.motion and Path(args.motion[0]).is_dir():
        args.hand_dir = Path(args.motion[0])
    print(f"[info] hand labels: {args.hand_dir or '(none)'} "
          f"{'→ 只用于 VLA 侧 BC 目标/手部项' if args.hand_dir else ''}", flush=True)

    # ---------------- 前置检查 ----------------
    from diff_vla_wbc.mjlab_joint import OBS_DIM, STATE_DIM, is_mjlab_checkpoint, load_mjlab_bundle

    for p, what in ((args.wbc_ckpt, "--wbc-ckpt"), (args.tokens, "--tokens"),
                    (args.vlm_ckpt, "--vlm-ckpt"), (args.vlm_path, "--vlm-path")):
        if not Path(p).exists():
            raise SystemExit(f"[fail] 缺 {what}: {p}")
    if not is_mjlab_checkpoint(args.wbc_ckpt):
        raise SystemExit(
            f"[fail] {args.wbc_ckpt} 不是 mjlab 训练 ckpt（需要 policy+wm、没有 pi）。"
            "Isaac 的 {pi,wm} 布局请用 scripts/train_joint_v2.py"
        )

    # ---------------- 分布式（torchrun 起多进程时自动开梯度同步） ----------------
    import os

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        import torch.distributed as dist

        if str(args.device).startswith("cuda"):
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(args.device)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        print(f"[dist] rank {rank}/{world} local={local_rank} device={device}", flush=True)
    else:
        device = torch.device(args.device)
    sync_grads = bool(world > 1) if args.sync_gradients is None else bool(args.sync_gradients)
    if sync_grads and world == 1:
        print("[warn] --sync-gradients 但没有 dist（world=1），已忽略", flush=True)

    out = args.out
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    # 模型初始化必须**各 rank 相同**（LoRA 的 A 是随机的！按 rank 换种子会让各卡起点不同，
    # 梯度平均就失效了）；只有**数据**才按 rank 换种子（见 provider 的 seed 偏移）。
    torch.manual_seed(int(args.seed))
    if world > 1:
        import torch.distributed as dist
        dist.barrier()

    files = collect_npz(list(args.motion)) if args.motion else []
    fingerprint = {k: (str(v) if isinstance(v, Path) else (list(map(str, v)) if isinstance(v, list) else v))
                   for k, v in vars(args).items()}
    fingerprint["_wbc_sha256_1m"] = _sha256_head(Path(args.wbc_ckpt))
    fingerprint["_n_clips"] = len(files)
    fingerprint["_backend"] = "mjlab"
    fingerprint["_obs_dim"] = OBS_DIM
    if rank == 0:
        (out / "args.json").write_text(json.dumps(fingerprint, indent=1, default=str))
        print(f"[audit] 参数指纹 -> {out / 'args.json'}", flush=True)

    # ---------------- bundle（冻结执行器 + WM） ----------------
    bundle = load_mjlab_bundle(
        args.wbc_ckpt, device=device,
        npz=files or None, packed=args.motion_packed,
        token_dim=TOKEN_DIM, xml_path=args.xml_path, freeze=True,
    )
    if args.executor == "sonic":
        from diff_vla_wbc.sonic_mjlab import SonicMjlabBundle  # noqa: PLC0415

        bundle = SonicMjlabBundle(bundle, onnx_path=args.decoder,
                                  token_quant=args.token_quant, device=device)
        if rank == 0:
            print(f"[AUDIT][executor] sonic_release_decoder onnx={bundle.executor.onnx_path} "
                  f"token_quant={args.token_quant} (冻结；动作空间与 mjlab 同底)", flush=True)
    lib = bundle.library
    if lib is None:
        raise SystemExit("[fail] 需要参考库：给 --motion 或 --motion-packed")
    print(f"[AUDIT][backend] mjlab | obs={bundle.obs_width} term_dims={bundle.term_dims} "
          f"token_dim={bundle.token_dim} max_step_reward={bundle.max_step_reward:g} "
          f"integration={getattr(bundle.world_model_cfg.state_spec, 'explicit_integration', None)}",
          flush=True)
    print(f"[info] library = {lib.summary() if hasattr(lib, 'summary') else '?'}", flush=True)

    # ---------------- token 源（数据集录制标签） ----------------
    pack = TokenPack(args.tokens, device)
    print(pack.audit(), flush=True)
    if pack.n != int(lib.num_frames):
        raise SystemExit(f"[fail] token 包 {pack.n} 帧 != 参考库 {int(lib.num_frames)} 帧；"
                         "两者必须同源同序（build_library.py + build_mjlab_tokens.py）")

    # ---------------- VLA 观测批 ----------------
    from diff_vla_wbc.g1_tasks import (
        TaskFrameIndex, TaskObservationBuilder, load_gr00t_processor, processed_batch,
    )
    from diff_vla_wbc.action_codec import ActionCodec

    processor = load_gr00t_processor(args.vlm_ckpt, vlm_path=args.vlm_path)
    codec = ActionCodec(processor, device=device)
    print(codec.audit(), flush=True)
    frame_index = TaskFrameIndex(files)
    if args.motion and frame_index.num_frames != int(lib.num_frames):
        raise SystemExit(f"[fail] npz 索引 {frame_index.num_frames} 帧 != 参考库 {int(lib.num_frames)} 帧")
    builder = TaskObservationBuilder(frame_index)

    from diff_vla_wbc.prefetch import BatchPrefetcher, StepMemo

    def _processed(frames):  # noqa: ANN001
        return processed_batch(processor, builder, frames)

    memo = StepMemo(_processed)
    prefetch = BatchPrefetcher(_processed, enabled=bool(args.prefetch_vla))
    print(f"[prefetch] enabled={prefetch.enabled}", flush=True)

    # ---------------- VLA（backbone 冻结 + head/LoRA 可训） ----------------
    from diff_vla_wbc.gr00t_vlm import Gr00tVLM

    vlm = Gr00tVLM.from_pretrained(
        args.vlm_path, checkpoint=args.vlm_ckpt, select_layer=args.vlm_select_layer,
        device=device, action_horizon=40, max_state_dim=132,
        freeze_vlm=True, attn_implementation="sdpa",
    )
    if int(args.lora_rank) > 0:
        from diff_vla_wbc.lora import apply_lora, lora_summary

        n = apply_lora(
            vlm.action_head, rank=int(args.lora_rank), alpha=args.lora_alpha,
            targets=tuple(s for s in args.lora_targets.split(",") if s),
            also_train=(("proj_out",) if args.lora_train_proj_out else ()),
        )
        sm = lora_summary(vlm.action_head)
        print(f"[AUDIT][lora] rank={args.lora_rank} layers={n} "
              f"trainable={sm['trainable']:,}/{sm['total']:,} "
              f"({100 * sm['trainable_frac']:.3f}%) targets={args.lora_targets}", flush=True)
    if int(vlm.action_head.action_horizon) < int(args.horizon):
        raise SystemExit(f"[fail] action_horizon {vlm.action_head.action_horizon} < H {args.horizon}")

    # ---------------- provider（burn-in / DR / B 线起点） ----------------
    from diff_vla_wbc.train_v2 import BurnInBatchProvider, JointTrainerV2, JointV2Config
    from diff_vla_wbc.dr import DRConfig

    dr_cfg = DRConfig(
        enabled=bool(args.dr), p_nominal=args.dr_p_nominal, p_mid=args.dr_p_mid,
        xy_max=args.dr_xy_max, yaw_max=args.dr_yaw_max,
        dq_max=args.dr_dq_max, dqd_max=args.dr_dqd_max,
        push_std=args.push_std, push_ang=args.push_ang,
    )
    print(f"[AUDIT][dr] enabled={dr_cfg.enabled} p={dr_cfg.p_nominal}/{dr_cfg.p_mid}/rest "
          f"max(xy,yaw,dq,dqd)=({dr_cfg.xy_max},{dr_cfg.yaw_max},{dr_cfg.dq_max},{dr_cfg.dqd_max})",
          flush=True)

    b_train = b_eval = None
    if str(args.start_from_b or ""):
        from diff_vla_wbc.b_start import BStartSource

        h_req = max(int(args.horizon), int(args.eval_horizon or 0))
        _st = None
        if args.b_self_targets is not None:
            _st = torch.load(str(args.b_self_targets), map_location="cpu", weights_only=False)
            _st = _st.get("targets", _st) if isinstance(_st, dict) else _st
            print(f"[b-start] self_targets: {len(_st)} 条 dump 带真值", flush=True)
            if args.b_max_err is not None:
                print(f"[b-start] max_err={args.b_max_err}: 只保留真值漂移不超过它的重规划时刻", flush=True)
        common = dict(stride=int(args.b_start_stride), horizon=h_req,
                      order=str(args.b_start_order),
                      strict_online=not bool(args.b_start_nonstrict), log=print,
                      self_targets=_st, max_err=args.b_max_err)
        b_train = BStartSource(args.start_from_b, bundle, lib, split="train", **common)
        try:
            b_eval = BStartSource(args.start_from_b, bundle, lib, split="eval", **common)
        except SystemExit:
            b_eval = None
        print(f"[b-start] 训练 {b_train.n_rows} 行 / 评测 {0 if b_eval is None else b_eval.n_rows} 行；"
              "burn-in 关闭、DR 不叠加", flush=True)

    recorded_states = None
    real = None
    if args.real_transitions is not None:
        if not Path(args.real_transitions).exists():
            raise SystemExit(f"[fail] 缺真转移文件 {args.real_transitions}")
        real = torch.load(str(args.real_transitions), map_location="cpu", weights_only=False)
        print(f"[real] transitions: states {tuple(real['states'].shape)}, "
              f"keys={sorted(real.keys())}", flush=True)
        if args.start_source == "recorded":
            _map = real.get("frame_map")
            if _map is None:
                raise SystemExit("[fail] --start-source recorded 需要 transitions 里的 frame_map")
            recorded_states = real["states"][_map].to(device)
            print(f"[real] recorded_states -> {tuple(recorded_states.shape)}", flush=True)
    elif args.start_source == "recorded" or args.wm_refresh_every:
        raise SystemExit("[fail] --start-source recorded / --wm-refresh-every 需要 --real-transitions"
                         "（先跑 scripts/build_real_transitions_mjlab.py）")

    def _mk_provider(horizon: int, burnin: bool, seed: int, b_starts):
        return BurnInBatchProvider(
            bundle, lib,
            port_source=pack.chunk, port_frame_source=pack.frame,
            wbc_chunk_source=pack.chunk, online_tokens=None,
            recorded_states=recorded_states, b_starts=b_starts, dr=dr_cfg,
            batch_size=args.batch_size, horizon=horizon, token_horizon=TOKEN_HORIZON,
            seed=seed, hand_dir=args.hand_dir,
            burnin=burnin and not args.no_burnin, burnin_steps=int(args.burnin_steps),
            perturb=not args.no_perturb,
            perturb_xy=args.perturb_xy, perturb_yaw=args.perturb_yaw,
            perturb_dq=args.perturb_dq, perturb_dqd=args.perturb_dqd,
        )

    provider = _mk_provider(int(args.horizon), True, int(args.seed) + 1000 * rank, b_train)
    h_eval = max(int(args.horizon), int(args.eval_horizon or 0))
    eval_provider = _mk_provider(h_eval, True, 20261028, b_eval)

    # ---------------- trainer ----------------
    cfg = JointV2Config(
        horizon=args.horizon, gamma=args.gamma,
        lambda_bc=args.lambda_bc, lambda_r=args.lambda_r, lambda_s=args.lambda_s,
        lambda_a=args.lambda_a, state_vel=args.state_vel_weight,
        eval_horizon=int(args.eval_horizon), amp=str(args.amp),
        fm_steps=args.fm_steps, fm_lr=args.fm_lr, fm_clip=args.fm_clip,
        weight_decay=args.weight_decay, grad_backend=args.fm_grad_backend,
        fm_truncated_tail=args.fm_truncated_tail,
        head_eval_mode=not args.head_train_mode,
        grad_accum=args.grad_accum, grad_ratio_every=args.grad_ratio_every,
        sync_gradients=sync_grads,
        train_wbc=bool(args.train_wbc), wbc_lr=float(args.wbc_lr),
    )
    trainer = JointTrainerV2(bundle, vlm, codec, cfg=cfg, device=device)
    if args.v3:
        from diff_vla_wbc.v3_step import JointTrainerV3, V3Config

        v3cfg = V3Config(
            lambda_r=args.lambda_r, lambda_hand=args.lambda_hand, use_critic=args.critic,
            critic_lr=args.critic_lr, gamma=args.gamma, lam=args.lam,
            push_std=args.push_std, push_ang=args.push_ang,
            anchor_weight=args.anchor_weight, trust_eps=args.trust_eps,
            trust_weight=args.trust_weight, act_eps=args.act_eps, act_weight=args.act_weight,
            vla_state_from_burnin=bool(args.vla_state_from_burnin),
            critic_grad_through_state=bool(args.critic_grad_through_state),
            trust_center=str(args.trust_center),
            critic_state_grad_scale=float(args.critic_state_grad_scale),
            critic_rel_clip=float(args.critic_rel_clip), act_hinge_cap=float(args.act_hinge_cap),
            critic_source=args.critic_source, grad_log_every=args.grad_log_every,
            clip_head=args.clip_head,
        )
        trainer = JointTrainerV3(bundle, vlm, codec, cfg=cfg, v3=v3cfg, device=device,
                                 processor=processor)
        (out / "v3_config.json").write_text(json.dumps(v3cfg.__dict__, indent=1, default=str))
        if args.critic_source == "real":
            if real is None or "ret" not in real:
                raise SystemExit("[fail] --critic-source real 需要带 ret/states/ref_rows 的 transitions")
            _ret_use = real["ret"]
            if args.critic_ret is not None:
                _ov = torch.load(str(args.critic_ret), map_location="cpu", weights_only=False)
                if int(_ov["ret"].shape[0]) != int(_ret_use.shape[0]):
                    raise SystemExit(f"[fail] --critic-ret 长度 {int(_ov['ret'].shape[0])} != "
                                     f"transitions ret {int(_ret_use.shape[0])}")
                _ret_use = _ov["ret"]
                print(f"[critic] ret 被 --critic-ret 覆盖：kind={_ov.get('kind')} "
                      f"lam={_ov.get('lam')} n={int(_ret_use.shape[0])}", flush=True)
            trainer.attach_real_returns(_ret_use, real["states"].float(),
                                        real.get("ref_rows", _ref_rows_of(real, lib)))
        if args.critic_ckpt is not None:
            c = torch.load(str(args.critic_ckpt), map_location="cpu", weights_only=False)
            sd = c.get("critic") or c
            miss, unexp = trainer.critic.load_state_dict(sd, strict=False)
            print(f"[critic] prefit {args.critic_ckpt} missing={len(miss)} unexpected={len(unexp)}",
                  flush=True)
    (out / "v2_config.json").write_text(json.dumps(cfg.__dict__, indent=1))

    # ---------------- 门禁（不训练） ----------------
    if args.gate_only:
        from diff_vla_wbc.v2_gates import run_gates

        ref_provider = _mk_provider(int(args.horizon), False, int(args.seed), b_train)
        _only = None
        if str(args.gate_subset).lower() not in ("all", "*", ""):
            _only = tuple(x.strip() for x in str(args.gate_subset).split(",") if x.strip())
        gates = run_gates(
            args, bundle, vlm, codec, trainer, provider, ref_provider, processor,
            args.gate_out or (out / "gates.json"), n_batches=int(args.gate_batches),
            only=_only,
        )
        prefetch.close()
        ok = all(bool(v.get("pass", True)) for k, v in gates.items() if not k.startswith("_"))
        print(f"[GATE] overall {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1

    # ---------------- 固定评测窗口 ----------------
    from diff_vla_wbc.train_v2 import fixed_eval

    eval_batches = []
    if args.eval_every and args.eval_batches and rank == 0:
        t0 = time.time()
        eval_batches = [eval_provider.sample() for _ in range(args.eval_batches)]
        print(f"[eval] fixed {len(eval_batches)} x batch {args.batch_size} "
              f"= {len(eval_batches) * args.batch_size} 窗口（{time.time() - t0:.1f}s）", flush=True)
    eval_out = args.eval_out or (out / "eval_curve.jsonl")

    import copy as _copy

    eval_wm = _copy.deepcopy(bundle.wm)

    def run_eval(step: int) -> dict:
        for b in eval_batches:
            b.model_inputs = memo.get(b.frames)
        live = bundle.wm
        try:
            bundle.wm = eval_wm
            m = fixed_eval(trainer, eval_batches)
        finally:
            bundle.wm = live
        m["step"] = int(step)
        m["h"] = int(args.horizon)
        m["lambda_r"] = float(args.lambda_r)
        with open(eval_out, "a") as f:
            f.write(json.dumps(m) + "\n")
        print(f"[eval {step}] " + json.dumps({k: round(v, 6) if isinstance(v, float) else v
                                              for k, v in m.items()}), flush=True)
        return m

    if eval_batches and rank == 0:
        run_eval(0)

    # ---------------- 训练循环 ----------------
    def next_batch():
        b = provider.sample()
        if prefetch.enabled:
            prefetch.submit(b.frames)
        return b

    deadline = time.time() + args.time_budget_minutes * 60.0 if args.time_budget_minutes > 0 else None
    metrics_path = out / "train_metrics.jsonl"
    t0 = time.time()
    batch = next_batch()
    step = 0
    if int(args.wm_pretrain_steps) > 0 and real is not None and rank == 0:
        print(f"[wm-pretrain] 训练前先训 WM {args.wm_pretrain_steps} 步（之后冻结，reward 固定靶）", flush=True)
    if int(args.wm_pretrain_steps) > 0 and real is not None:
        rm = trainer.refresh_wm(real, steps=int(args.wm_pretrain_steps), lr=args.wm_refresh_lr,
                                kl_anchor=args.wm_kl_anchor, log=print if rank == 0 else (lambda *a, **k: None))
        if rank == 0:
            print("[wm-pretrain] " + json.dumps(
                {k: v for k, v in rm.items() if k in ("steps", "loss_first", "loss_last")}), flush=True)

    for step in range(1, int(args.steps) + 1):
        nxt = next_batch()
        batch.model_inputs = prefetch.get(batch.frames, lambda: memo.get(batch.frames))
        metrics = trainer.train_step(batch)
        if os.environ.get("J1_DEBUG_SYNC") == "1":
            # 多卡同步自检：同一参数在各 rank 的校验和必须相等（浮点逐位一致）
            _p = trainer.fm_parameters[0]
            _sum = float(_p.detach().float().sum())
            _sq = float((_p.detach().float() ** 2).sum())
            print(f"[sync-check] rank={rank} step={step} sum={_sum:.8f} sq={_sq:.8f}", flush=True)
        metrics["step"] = step
        metrics["s_per_step"] = (time.time() - t0) / step
        if rank == 0:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        if rank == 0 and (step % max(1, args.log_every) == 0 or step == 1):
            print(f"[j1 {step}/{args.steps}] "
                  + json.dumps({k: round(v, 6) for k, v in metrics.items()}), flush=True)
        if args.v3 and args.wm_refresh_every and step % args.wm_refresh_every == 0 and real:
            try:
                rm = trainer.refresh_wm(real, steps=args.wm_refresh_steps, lr=args.wm_refresh_lr,
                                        kl_anchor=args.wm_kl_anchor, log=print)
                print(f"[wm-refresh {step}] " + json.dumps(
                    {k: v for k, v in rm.items() if k in ("steps", "loss_first", "loss_last")}),
                    flush=True)
            except Exception as exc:  # 保鲜失败不能挡住训练
                print(f"[wm-refresh][warn] {exc}", flush=True)
        if rank == 0 and eval_batches and args.eval_every and step % args.eval_every == 0:
            run_eval(step)
        if rank == 0 and args.save_every and step % args.save_every == 0:
            trainer.save_joint(out / f"joint_{step:06d}.pt")
            if args.save_vla_every and step % args.save_vla_every == 0:
                print(f"[saved] {trainer.save_vla(out / f'vla_{step:06d}.pt')}", flush=True)
        batch = nxt
        memo.clear()
        if deadline is not None and time.time() >= deadline:
            print(f"[budget] {args.time_budget_minutes:g} min 到达（step {step}）", flush=True)
            break

    # ---------------- 收尾 ----------------
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
    if rank == 0:
        trainer.save_joint(out / "joint_final.pt")
        vla_final = trainer.save_vla(out / "vla_final.pt")
    else:
        vla_final = out / "vla_final.pt"
    final_eval = run_eval(int(step)) if (eval_batches and step and rank == 0) else {}
    summary = {
        "backend": "mjlab",
        "steps": int(step),
        "out": str(out),
        "wbc_ckpt": str(args.wbc_ckpt),
        "wbc_sha256_1m": fingerprint["_wbc_sha256_1m"],
        "vlm_ckpt": str(args.vlm_ckpt),
        "obs_dim": OBS_DIM,
        "state_dim": STATE_DIM,
        "token_space": pack.space,
        "tokens": str(args.tokens),
        "horizon": int(args.horizon),
        "lambda_bc": float(args.lambda_bc),
        "lambda_r": float(args.lambda_r),
        "v3": bool(args.v3),
        "lora_rank": int(args.lora_rank),
        "amp": str(args.amp),
        "final_eval": final_eval,
        "vla_final": str(vla_final),
        "wall_s": time.time() - t0,
    }
    if rank == 0:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print("[AUDIT][j1-summary] " + json.dumps(summary, default=str), flush=True)
    prefetch.close()
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _ref_rows_of(real: dict, lib) -> torch.Tensor:
    """critic 真回报的参考行：优先用 transitions 里的 frames，否则用帧序。"""
    fr = real.get("frames")
    if fr is None:
        raise SystemExit("[fail] transitions 里既没有 ref_rows 也没有 frames")
    return lib.ref71(fr.long()).float()


if __name__ == "__main__":
    raise SystemExit(main())
