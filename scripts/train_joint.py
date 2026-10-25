#!/usr/bin/env python3
"""Joint training / gradient smoke CLI for the differentiable framework.

Three token sources, most synthetic first:

* default: random FM tokens (fast CPU smoke of the whole graph);
* ``--real-gr00t-head``: the real ``Gr00tN1d7ActionHead`` with a tiny DiT and
  random backbone features;
* ``--vlm``: the **real top-level VLM** -- Cosmos-Reason2-2B (local weights) ->
  GR00T N1.7 flow-matching head -> ``[B, 40, 64]`` chunk, with gradients reaching
  the head (and the VLM with ``--train-vlm``).  There is no paired RGB data in
  this repository, so the observation is a deterministic synthetic image unless
  ``--vlm-image`` is given.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter  # noqa: E402
from diff_vla_wbc.data import MotionBatchProvider, sample_reference_windows  # noqa: E402
from diff_vla_wbc.gr00t_fm import sample_model_with_grad  # noqa: E402
from diff_vla_wbc.train import JointTrainer, JointTrainerConfig  # noqa: E402

# ---------------------------------------------------------------------------------------------
# 写死的口径（A4 及以后所有正式实验都只允许这些）：
#   * WBC / 世界模型：Isaac Lab / PhysX 版 SuperMimic（trackerlab-supermimic）;
#   * 数据：纯 HA（data/npz/ha_tokens，700 条）。mixed / ft318 **永久禁止**（下面会硬性拦截）;
#   * mjlab(third_party/SuperMimic-Reproduced) 的权重、解析头、MuJoCo 栈**永不参与**。
# ---------------------------------------------------------------------------------------------
ISAAC_SM_CKPT = ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500.pt"
HA_MOTION_DIR = ROOT / "data/npz/ha_tokens"
PACKED_MOTION = ROOT / "runs/isaac_motion/ha_tokens/packed.pt"
FORBIDDEN_DATA_TOKENS = ("mixed", "ft318")


def _forbid_legacy_dataset(paths, flag: str) -> None:
    """Hard stop on any dataset that is not pure HA (mixed / ft318 are permanently banned)."""
    for p in paths:
        text = str(p).lower()
        bad = [tok for tok in FORBIDDEN_DATA_TOKENS if tok in text]
        if bad:
            raise SystemExit(
                f"{flag} {p}: 含禁用数据集标记 {bad}。本工程永久只走纯 HA 数据 "
                f"({HA_MOTION_DIR.name})，mixed / ft318 数据一律不得使用。"
            )


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


def load_vlm_image(path: Path | None, size: int = 256):
    """Observation image for the VLM path (deterministic synthetic if unset)."""
    from PIL import Image

    if path is not None:
        return Image.open(path).convert("RGB")
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, size), torch.linspace(0.0, 1.0, size), indexing="ij"
    )
    rgb = torch.stack((x, y, (x + y) / 2.0), dim=-1).numpy()
    return Image.fromarray((rgb * 255.0).round().astype("uint8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", nargs="+", type=Path, default=[HA_MOTION_DIR],
                    help=f"纯 HA npz 目录（默认 {HA_MOTION_DIR}）；mixed/ft318 会被硬性拒绝")
    ap.add_argument("--out", type=Path, default=Path("runs/joint_smoke"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--small", action="store_true", help="tiny SuperMimic for CPU smoke runs")
    ap.add_argument("--supermimic-ckpt", type=Path, default=ISAAC_SM_CKPT,
                    help=f"Isaac/trackerlab SuperMimic ckpt（默认 {ISAAC_SM_CKPT}）")
    ap.add_argument("--motion-packed", type=Path, default=None,
                    help="Isaac-side packed motion library ({'ref','names','starts','lengths'}, "
                         "canonical layout). Needed with an Isaac ckpt; if omitted it is built "
                         "from --motion into --motion-packed-dir and cached")
    ap.add_argument("--motion-packed-dir", type=Path, default=None,
                    help="directory for the cached packed library (default runs/isaac_motion/<set>)")
    ap.add_argument("--obs-mode", default=None,
                    help="override the policy observation mode; default = whatever the checkpoint "
                         "was trained with (Isaac HA run = relative_ori99, 99 dims/frame)")
    ap.add_argument("--hist", type=int, default=10, help="observation history length")
    ap.add_argument("--adapter-lr", type=float, default=1e-4)
    ap.add_argument("--policy-lr", type=float, default=1e-5)
    ap.add_argument("--fm-lr", type=float, default=1e-5)
    ap.add_argument("--pretrain-wm", type=int, default=0)
    ap.add_argument("--pretrain-wm-batch", type=int, default=8)
    ap.add_argument("--pretrain-wm-window", type=int, default=8)
    # ---- 世界模型消融：联合训练过程中要不要继续用真实转移数据更新 F_phi ----
    # 默认 0 = 冻结 WM（SuperMimic ckpt 原样），这是当前所有主线实验的口径。
    # --wm-update-every N：每 N 步用 sample_reference_windows 采真实转移窗口，做一次
    # WM 有监督拟合（unfreeze → loss → step → 立刻 freeze），reward 回路本身不变。
    ap.add_argument("--wm-update-every", type=int, default=0,
                    help="ablation: every N steps refit the world model on real transition windows "
                         "(0 = keep it frozen, the default)")
    ap.add_argument("--wm-update-batch", type=int, default=2048,
                    help="windows per refit; SuperMimic does 10x2048=20480 windows per update "
                         "(wm_trainer.py: batch_size=2048, batches_per_update=10)")
    ap.add_argument("--wm-update-window", type=int, default=8)
    ap.add_argument("--wm-weight-decay", type=float, default=0.0,
                    help="AdamW weight decay for the world model; SuperMimic uses 0.0")
    ap.add_argument("--wm-update-warmup", type=int, default=500,
                    help="do not refit the world model during the first N steps "
                         "(SuperMimic: warmup_iterations=40 env steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--save-every", type=int, default=0, help="every N steps save joint_<step>.pt (0=off)")
    ap.add_argument("--save-vla-every", type=int, default=0,
                    help="every N steps save the fine-tuned VLA head (vla_<step>.pt, ~6.5 GB); "
                         "trainer.save() alone does NOT include it")
    ap.add_argument("--eval-every", type=int, default=0, help="every N steps run a fixed-batch eval (0=off)")
    ap.add_argument("--eval-batches", type=int, default=2, help="how many fixed eval batches")
    ap.add_argument("--eval-out", type=Path, default=None, help="JSONL file for eval curves")
    ap.add_argument("--hand-loss-weight", type=float, default=0.0,
                    help="手部辅助监督的权重 λ（0=关闭）。手部不经过 SuperMimic，纯 label 回归")
    ap.add_argument("--hand-grasp-boost", type=float, default=5.0,
                    help="抓握帧的损失放大倍数（抓握帧只占 ~7%%，不加权会塌到恒定张开）")
    ap.add_argument("--hand-mask-threshold", type=float, default=0.05,
                    help="frames whose reference hand is below this are excluded from the hand "
                         "loss (HA has 4/7 tasks with all-zero hands)")
    ap.add_argument("--token-anchor-weight", type=float, default=0.0,
                    help="token 锚定损失权重 λ：λ*MSE(VLA token, 端口 token)。reward 一饱和就没有"
                         "东西钉住 token，模型会在平坦区里漂移（实测 2000→3000 步 token_mse "
                         "0.077→0.093）。建议 0.5–2.0；0 = 关闭（旧行为）")
    ap.add_argument("--prefetch-vla", dest="prefetch_vla", action="store_true", default=False,
                    help="后台线程预取下一步的 VLA 预处理（CPU 侧；不碰 CUDA）。默认开，"
                         "用 --no-prefetch-vla 关闭。数值与现算逐位相同（见 scripts/check_prefetch_parity.py）")
    ap.add_argument("--no-prefetch-vla", dest="prefetch_vla", action="store_false",
                    help="关闭预取（排查用）")
    ap.add_argument("--resume-joint", type=Path, default=None, metavar="CKPT",
                    help="从 joint_<step>.pt 续训：恢复 policy/wm/adapter + 步数"
                         "（Isaac 的 {pi,wm,adapter} 与 mjlab 的 {policy,wm,adapter} 都支持）")
    ap.add_argument("--resume-vla", type=Path, default=None, metavar="CKPT",
                    help="从 vla_<step>.pt 续训：恢复 VLM 的可训练参数（action head）")
    ap.add_argument("--start-step", type=int, default=0,
                    help="从该步之后继续（默认取 --resume-joint 里记录的步数）")
    ap.add_argument("--state-vel-weight", type=float, default=0.05,
                    help="速度通道（v/w/qd）在 state 项里的额外权重。Isaac WM 想象出的速度与参考"
                         "速度相关仅 0.077（位置/关节 1e-4/0.03），不降权会让 state 项 96%% 的"
                         "能量来自不可学的速度噪声（详见 rollout.LossWeights.state_vel）")
    ap.add_argument("--recorded-tokens", type=Path, default=None, metavar="DIR",
                    help="use the tokens recorded in DIR/*.npz (g1_motion_token_64, real ft318) "
                         "instead of a VLM/FM sampler — the 'real data' token source")
    ap.add_argument("--real-gr00t-head", action="store_true",
                    help="use the real GR00T action head with a tiny DiT (gradient smoke)")
    vlm = ap.add_argument_group("top-level VLM (Cosmos-Reason2-2B)")
    vlm.add_argument("--vlm", action="store_true",
                     help="use the real Cosmos-Reason2-2B VLM + GR00T head as the token source")
    vlm.add_argument("--vlm-path", type=Path, default=None,
                     help="local VLM directory (default: auto-resolve, see diff_vla_wbc.gr00t_vlm)")
    vlm.add_argument("--vlm-ckpt", type=Path, default=None,
                     help="optional full GR00T checkpoint (backbone + action head)")
    vlm.add_argument("--vlm-select-layer", type=int, default=16,
                     help="LLM layers kept by the backbone (GR00T N1.7 truncation)")
    vlm.add_argument("--vlm-steps", type=int, default=2,
                     help="Euler steps of the differentiable flow-matching sampler")
    vlm.add_argument("--vlm-image", type=Path, default=None,
                     help="observation image (default: deterministic synthetic RGB)")
    vlm.add_argument("--vlm-instruction", default="Pick up the carton and hand it over.")
    vlm.add_argument("--vlm-state-dim", type=int, default=43, help="raw robot state width")
    vlm.add_argument("--vlm-embodiment-id", type=int, default=0,
                     help="embodiment id fed to the action head; a GR00T checkpoint needs the "
                          "id it was trained with (e.g. 11 = unitree_g1_sonic)")
    vlm.add_argument("--vlm-dataset", action="store_true",
                     help="feed per-sample REAL observations: each step reads the ego_view frame, "
                          "46-dim state and instruction belonging to the sampled motion window, "
                          "preprocessed by the checkpoint's own Gr00tN1d7Processor. Requires "
                          "--vlm-ckpt and a --motion directory built by prepare_g1_tasks_npz.py")
    vlm.add_argument("--train-vlm", action="store_true",
                     help="also unfreeze the VLM backbone (otherwise only the action head trains)")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--fm-grad-backend", default="pathwise",
                    choices=["pathwise", "checkpointed", "truncated", "adjoint"],
                    help="how the gradient crosses the FM Euler ODE (see diff_vla_wbc/fm_grad.py). "
                         "pathwise = exact BPTT (default), checkpointed = exact + O(1) activations, "
                         "truncated = biased but cheap, adjoint = exact discrete adjoint")
    ap.add_argument("--fm-truncated-tail", type=int, default=1,
                    help="for --fm-grad-backend truncated: how many trailing Euler steps keep grad")
    ap.add_argument("--time-budget-minutes", type=float, default=0.0,
                    help="stop after N wall-clock minutes (0 = off); used by the fixed-budget "
                         "BPTT-vs-PPO ablation so both arms get identical GPU time")
    args = ap.parse_args()

    # ---------------- 2 卡数据并行（torchrun 启动时自动生效）----------------
    import os as _os

    _world = int(_os.environ.get("WORLD_SIZE", "1"))
    _rank = int(_os.environ.get("RANK", "0"))
    if _world > 1:
        import torch.distributed as _dist

        _local = int(_os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(_local)
        _dist.init_process_group("nccl")
        print(f"[dist] rank {_rank}/{_world} local_rank={_local} backend=nccl", flush=True)
    _is_main = _rank == 0

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{_os.environ.get('LOCAL_RANK', '0')}") if _world > 1 else torch.device(args.device)
    _forbid_legacy_dataset(args.motion, "--motion")
    if args.recorded_tokens is not None:
        _forbid_legacy_dataset([args.recorded_tokens], "--recorded-tokens")
    files = collect_npz(args.motion)

    # ------------------------------------------------------------------ 运行指纹（可审计）
    # 消融的合法性靠"除了被消融的那一个变量，其余命令行参数逐字相同"来保证，所以把**实际生效**
    # 的全部参数落盘一份（rank0 写 args.json），事后可以直接 diff 两个格子证明只差那个变量。
    if _is_main:
        try:
            import json as _json

            args.out.mkdir(parents=True, exist_ok=True)
            fingerprint = {
                k: (str(v) if isinstance(v, Path) else (list(map(str, v)) if isinstance(v, tuple) else v))
                for k, v in vars(args).items()
            }
            fingerprint["_world_size"] = _world
            fingerprint["_nproc_per_node"] = _world
            fingerprint["_data_clips"] = len(files)
            fingerprint["_forbidden_data_tokens"] = list(FORBIDDEN_DATA_TOKENS)
            (args.out / "args.json").write_text(_json.dumps(fingerprint, indent=1, default=str))
            print(f"[audit] 参数指纹 -> {args.out / 'args.json'}", flush=True)
        except Exception as _exc:  # 审计绝不能挡住训练
            print(f"[audit][warn] 写 args.json 失败: {_exc}", flush=True)

    # ------------------------------------------------------------------ bundle + library
    # 只有一条线路：Isaac Lab / PhysX（trackerlab-supermimic）。
    #   权重（policy+WM）、99 维观测、奖励、软终止、可微 FK（14 tracked bodies）、参考运动库
    #   全部来自 trackerlab；mjlab(third_party/SuperMimic-Reproduced) 的权重/解析头**永不参与**。
    #   非 Isaac ckpt 直接报错退出（不做任何回退）。
    from diff_vla_wbc.isaac_joint import is_isaac_checkpoint as _is_isaac
    from diff_vla_wbc.isaac_joint import load_isaac_bundle

    if not _is_isaac(args.supermimic_ckpt):
        raise SystemExit(
            f"{args.supermimic_ckpt}: 只接受 Isaac/trackerlab SuperMimic ckpt（含 'pi'/'wm'）。"
            "本工程永久停用 mjlab/MuJoCo 链路。官方底模见 runs/OFFICIAL_SUPERMIMIC.json。"
        )
    bundle = load_isaac_bundle(
        args.supermimic_ckpt, device=device, npz=files,
        packed_motion=args.motion_packed, packed_dir=args.motion_packed_dir,
        obs_mode=args.obs_mode, hist=args.hist, token_dim=64,
    )
    library = bundle.library
    print(f"[info] {library.summary() if hasattr(library, 'summary') else library.provenance}")

    adapter = MotionGoalAdapter(AdapterConfig(
        token_dim=64,
        token_horizon=12 if args.real_gr00t_head else 40,
        hidden=128 if args.small else 256,
        transformer_layers=1 if args.small else 2,
        transformer_heads=4,
    )).to(device)

    fm_sampler = None
    fm_params = None
    if args.recorded_tokens is not None:
        from diff_vla_wbc.data import RecordedTokenSource

        fm_sampler = RecordedTokenSource(
            args.recorded_tokens,
            library,
            token_horizon=adapter.cfg.token_horizon,
            token_dim=adapter.cfg.token_dim,
            device=device,
        )
        fm_params = []
        print(
            f"[info] token source: recorded g1_motion_token_64 from {args.recorded_tokens} "
            f"(real ft318 tokens, horizon={adapter.cfg.token_horizon})"
        )

    if args.real_gr00t_head:
        from diff_vla_wbc.gr00t_tiny import TinyGr00tModel, make_tiny_action_head, tiny_model_inputs

        head = make_tiny_action_head()
        model = TinyGr00tModel(head).to(device)
        inputs = tiny_model_inputs(args.batch_size, device=device)
        fixed_noise = torch.randn(args.batch_size, 12, 64, device=device)

        def fm_sampler(meta):  # noqa: ANN001
            return sample_model_with_grad(
                model, inputs, initial_noise=fixed_noise, num_steps=3,
                grad_backend=args.fm_grad_backend, truncated_tail=args.fm_truncated_tail,
            )

        fm_params = [p for p in head.parameters() if p.requires_grad]
        print(f"[info] real GR00T action head params: {sum(p.numel() for p in head.parameters()):,}")

    vlm_model = None  # 循环里的 VLA 头存档要用；未走 --vlm 时保持 None
    if args.vlm:
        if args.real_gr00t_head:
            raise SystemExit("--vlm and --real-gr00t-head are mutually exclusive")
        from diff_vla_wbc.gr00t_vlm import Gr00tVLM, describe_vlm, resolve_vlm_source, vlm_available

        if not vlm_available(args.vlm_path) and args.vlm_path is None and args.vlm_ckpt is None:
            raise SystemExit(
                "no local Cosmos-Reason2-2B weights found; pass --vlm-path, set "
                "$GR00T_VLM_PATH, or run scripts/link_vlm_weights.py"
            )
        print(f"[info] top-level VLM source: {resolve_vlm_source(args.vlm_path)}")
        vlm_model = Gr00tVLM.from_pretrained(
            args.vlm_path,
            checkpoint=args.vlm_ckpt,
            select_layer=args.vlm_select_layer,
            device=device,
            action_horizon=adapter.cfg.token_horizon,
            max_state_dim=args.vlm_state_dim,
            freeze_vlm=not args.train_vlm,
            # 反传微调里 VLM 输入是 fp32（SuperMimic 侧张量），FlashAttention 只吃
            # fp16/bf16 会直接抛 "FlashAttention only support fp16 and bf16"。
            # 强制走 sdpa，数值等价且 dtype 通吃。
            attn_implementation="sdpa",
        )
        print("[info] top-level VLM: " + json.dumps(describe_vlm(vlm_model), default=str))

        # The head emits its own action width (``Gr00tN1d7Config.max_action_dim``:
        # 64 for the standalone default, 132 for a GR00T checkpoint).  GR00T packs
        # an embodiment's action vector into that width following modality-key
        # order, so for ``unitree_g1_sonic`` columns 0:64 are the motion token and
        # the rest is unsupervised padding.  Sample at the head's width, then slice.
        head_action_dim = int(vlm_model.action_head.action_dim)

        def _sample_full(inputs) -> torch.Tensor:
            noise = torch.randn(
                int(inputs["input_ids"].shape[0]),
                adapter.cfg.token_horizon,
                head_action_dim,
                device=device,
                dtype=vlm_model.dtype,
            )
            return sample_model_with_grad(
                vlm_model, inputs, initial_noise=noise, num_steps=args.vlm_steps,
                grad_backend=args.fm_grad_backend, truncated_tail=args.fm_truncated_tail,
            )

        def _sample(inputs) -> torch.Tensor:
            return _sample_full(inputs)[..., : adapter.cfg.token_dim]

        if args.vlm_dataset:
            if not args.vlm_ckpt:
                raise SystemExit(
                    "--vlm-dataset needs --vlm-ckpt: the 46-dim state layout, percentile "
                    "normalisation and image transform all come from the checkpoint's processor"
                )
            from diff_vla_wbc.g1_tasks import (
                TaskFrameIndex,
                TaskObservationBuilder,
                load_gr00t_processor,
                processed_batch,
            )

            frame_index = TaskFrameIndex(files)
            if frame_index.num_frames != library.num_frames:
                raise SystemExit(
                    f"npz index has {frame_index.num_frames} frames but the motion library has "
                    f"{library.num_frames}; rebuild it with scripts/prepare_g1_tasks_npz.py"
                )
            builder = TaskObservationBuilder(frame_index)
            processor = load_gr00t_processor(args.vlm_ckpt, vlm_path=args.vlm_path)
            print(
                f"[info] real observations: {frame_index.num_episodes} episodes / "
                f"{frame_index.num_frames} frames, processor={type(processor).__name__}"
            )

            # ---- CPU 热路径：步骤内缓存 + 跨步预取 -------------------------------
            # 每步的两路 VLA 采样（token / 手部）用的是**同一组 frames**，原来各算一遍
            # processed_batch；再叠加"主线程 backward 时后台线程预先处理下一步"，
            # 可以把 GPU 空转的 ~1.4 s/step 大部分吃掉。
            from diff_vla_wbc.prefetch import BatchPrefetcher, StepMemo  # noqa: PLC0415

            def _processed(frames):  # noqa: ANN001
                return processed_batch(processor, builder, frames)

            _vla_memo = StepMemo(_processed)
            _vla_prefetch = BatchPrefetcher(
                _processed, enabled=bool(args.prefetch_vla and args.vlm and args.vlm_dataset)
            )
            print(
                f"[prefetch] enabled={_vla_prefetch.enabled} (CPU 侧；--no-prefetch-vla 可关)",
                flush=True,
            )

            def _vla_inputs(meta_or_batch):  # noqa: ANN001
                """meta dict（含 frames）→ 处理好的 batch；已经是 batch 就原样返回。"""
                if isinstance(meta_or_batch, dict) and "frames" in meta_or_batch:
                    frames = meta_or_batch["frames"]
                    return _vla_prefetch.get(frames, lambda: _vla_memo.get(frames))
                return meta_or_batch

            def fm_sampler(meta):  # noqa: ANN001
                return _sample(_vla_inputs(meta))

        else:
            observations = [load_vlm_image(args.vlm_image)] * args.batch_size
            instructions = [args.vlm_instruction] * args.batch_size
            vlm_state = torch.zeros(args.batch_size, args.vlm_state_dim)
            vlm_batch = vlm_model.build_batch(
                observations, instructions, vlm_state, embodiment_id=args.vlm_embodiment_id
            )
            print(
                "[warn] --vlm without --vlm-dataset reuses ONE synthetic image every step; "
                "fine-tuning the head this way collapses its visual conditioning. Use "
                "--vlm-dataset for anything beyond a gradient smoke test."
            )

            def fm_sampler(meta):  # noqa: ANN001
                return _sample(vlm_batch)

        fm_params = [p for p in vlm_model.parameters() if p.requires_grad]
        print(f"[info] trainable VLM/head params: {sum(p.numel() for p in fm_params):,}")

    # 手部辅助损失：标签目录（npz 里带 g1_hand_joints_14）
    hand_dir = None
    if args.hand_loss_weight > 0:
        hand_dir = args.motion[0] if args.motion[0].is_dir() else args.motion[0].parent

    # 预取模式：``sample()`` 只做轻活（帧索引/参考/观测），VLA 采样挪到主线程的
    # ``trainer._tokens``（走 ``batch.model_inputs``），这样后台线程可以先把下一步的
    # CPU 预处理做掉。
    _use_prefetch = bool(args.prefetch_vla and args.vlm and args.vlm_dataset and vlm_model is not None)
    provider = MotionBatchProvider(
        bundle,
        library,
        batch_size=args.batch_size,
        horizon=args.horizon,
        token_source=None if _use_prefetch else fm_sampler,
        token_dim=64,
        token_horizon=adapter.cfg.token_horizon,
        seed=args.seed,
        hand_dir=hand_dir,
    )
    cfg = JointTrainerConfig(
        horizon=args.horizon,
        adapter_lr=args.adapter_lr,
        policy_lr=args.policy_lr,
        fm_lr=args.fm_lr,
        # 世界模型消融：SuperMimic 用 weight_decay=0.0 / grad_clip=1.0（wm_trainer.py）
        wm_weight_decay=args.wm_weight_decay,
    )
    # 速度通道降权（2026-10-25）：Isaac WM 想象出来的速度与参考速度相关仅 0.077（位置/关节
    # 精度 1e-4/0.03），速度通道占 state 项 96% 且不可学习 —— 默认 0.05，详见 rollout.LossWeights。
    cfg.weights.state_vel = float(args.state_vel_weight)
    # 手部辅助损失需要 full chunk（token64 + hands14）；这里包一层 meta → 模型输入
    fm_full_from_meta = None
    if args.vlm and args.hand_loss_weight > 0:
        if args.vlm_dataset:
            def fm_full_from_meta(meta):  # noqa: ANN001
                return _sample_full(_vla_inputs(meta))
        else:
            def fm_full_from_meta(meta):  # noqa: ANN001
                return _sample_full(vlm_batch)

    token_anchor = None
    if args.token_anchor_weight > 0:
        from diff_vla_wbc.data import RecordedTokenSource  # noqa: PLC0415

        _tdir = args.motion[0] if args.motion[0].is_dir() else args.motion[0].parent
        _anchor_src = RecordedTokenSource(
            _tdir, library, token_horizon=adapter.cfg.token_horizon,
            token_dim=adapter.cfg.token_dim, device=device,
        )
        token_anchor = lambda b: _anchor_src({"frames": b.frames})  # noqa: E731
        print(
            f"[anchor] ON: λ={args.token_anchor_weight} src={_tdir} port_dir={_anchor_src.port_dir}",
            flush=True,
        )

    trainer = JointTrainer(
        bundle, adapter, fm_sampler=fm_sampler, fm_parameters=fm_params, cfg=cfg, device=device,
        token_anchor=token_anchor, token_anchor_weight=args.token_anchor_weight,
        hand_weight=args.hand_loss_weight,
        hand_grasp_boost=args.hand_grasp_boost,
        hand_mask_threshold=args.hand_mask_threshold,
        fm_full_sampler=fm_full_from_meta,
        sync_gradients=(_world > 1),
    )
    if args.hand_loss_weight > 0:
        print(f"[hand] 辅助监督开启: λ={args.hand_loss_weight} boost={args.hand_grasp_boost} "
              f"labels={hand_dir}", flush=True)

    if _world > 1:
        import torch.distributed as _dist

        for mod in (bundle.policy, bundle.wm, adapter, *( [vlm_model] if vlm_model is not None else [] )):
            for p_ in mod.parameters():
                _dist.broadcast(p_.data, src=0)
            for b_ in mod.buffers():
                _dist.broadcast(b_.data, src=0)
        print("[dist] 初始参数已从 rank0 广播", flush=True)

    if args.pretrain_wm:
        for i in range(args.pretrain_wm):
            windows = sample_reference_windows(
                bundle, library, batch_size=args.pretrain_wm_batch,
                window=args.pretrain_wm_window, seed=args.seed + i,
            )
            m = trainer.train_world_model(windows)
            print(f"[wm {i + 1}/{args.pretrain_wm}] " + json.dumps({k: round(v, 6) for k, v in m.items()}))

    # ---------------------------------------------------------------- 固定评测批次
    # 评测窗口必须与 --horizon 无关：``MotionBatchProvider.sample`` 用
    # ``randint(HISTORY, num_frames - horizon - 2)`` 抽起点，上界一旦随 H 变化，**同一份
    # 固定种子抽到的帧就变了**，跨格比较就不是同一批样本。所以评测提供器固定用最长 horizon
    # (40，= token chunk 长度)，rollout 用各格自己的 --horizon。
    EVAL_HORIZON = 40
    # 前缀指标：reward / state 的"H 步均值"随 H 变化，跨 H 不可比；统一再报前 8 步（=最小 H）
    # 的同一窗口口径，H 扫描才有可比读数。
    PREFIX_STEPS = 8
    eval_batches = None
    recorded_src = None
    if args.eval_every and args.eval_batches:
        # 只导入这里需要的名字：在 main() 里重复 import MotionBatchProvider 会让它变成
        # 函数局部变量，后面 line 266 的 MotionBatchProvider(...) 就会 UnboundLocalError。
        from diff_vla_wbc.data import RecordedTokenSource
        from diff_vla_wbc.rollout import compute_joint_loss, imagine_with_tokens

        token_dir = args.motion[0] if args.motion[0].is_dir() else args.motion[0].parent
        recorded_src = RecordedTokenSource(
            token_dir, library, token_horizon=adapter.cfg.token_horizon,
            token_dim=adapter.cfg.token_dim, device=device,
        )
        eval_provider = MotionBatchProvider(
            bundle, library, batch_size=args.batch_size,
            horizon=max(EVAL_HORIZON, args.horizon),
            token_source=None, token_dim=adapter.cfg.token_dim,
            token_horizon=adapter.cfg.token_horizon, seed=12345,
            hand_dir=hand_dir,
        )
        eval_batches = [eval_provider.sample() for _ in range(args.eval_batches)]
        print(
            f"[eval] fixed eval batches: {len(eval_batches)} x batch {args.batch_size} "
            f"(windows drawn with horizon={max(EVAL_HORIZON, args.horizon)}, rollout={args.horizon}) "
            f"every {args.eval_every} steps",
            flush=True,
        )

    @torch.no_grad()
    def run_eval(step: int) -> dict:
        """同一批帧上测：VLA token 与录制 token 的 MSE + 想象 rollout 的状态/奖励项。

        报两种口径：
        * 满 horizon（`state_term` / `rollout_reward`）= 该格自己的 H 步均值；
        * 固定前缀（`state_term@8` / `rollout_reward@8` / `survival@8`）= 只取前 8 步。
          H 步均值随 H 变化（窗口更长 → WM 漂移更多、连乘存活更少），跨 H 不可比；
          前缀口径让 H 扫描的四个格子落在**同一窗口长度**上。
        """
        import torch.nn.functional as F

        tok_mse, st, rw, hm = [], [], [], []
        st8, rw8, surv, surv8, rsum = [], [], [], [], []
        pos_e, ori_e, q_e, vel_e = [], [], [], []   # state 分项（pos/ori/q/vel）
        # 采样噪声是随机初始化：评测必须固定种子，否则曲线抖动会被误读成训练噪声。
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        for i, b in enumerate(eval_batches):
            torch.manual_seed(1234 + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1234 + i)
            try:
                # VLA 采样要求 grad 打开（flow-matching 走 sample_model_with_grad），
                # 评测只要数值，所以借用 enable_grad 后立刻 detach，不建反传图。
                with torch.enable_grad():
                    vla = fm_sampler(b.as_dict()) if fm_sampler is not None else b.fm_tokens
            finally:
                pass
            vla = vla.detach()
            rec = recorded_src({"frames": b.frames})
            n = min(vla.shape[1], rec.shape[1])
            tok_mse.append(float(F.mse_loss(vla[:, :n], rec[:, :n])))
            if (
                args.hand_loss_weight > 0
                and fm_full_from_meta is not None
                and getattr(b, "hand_ref", None) is not None
            ):
                with torch.enable_grad():   # VLA 采样要求 grad 打开
                    full = fm_full_from_meta(b.as_dict())
                full = full.detach()
                hs = full[..., 64:78]
                hr = b.hand_ref.to(hs.dtype)
                nn = min(hs.shape[1], hr.shape[1])
                hm.append(float(((hs[:, :nn] - hr[:, :nn]) ** 2).mean()))
            ro = imagine_with_tokens(
                bundle, tokens=vla, state=b.start_state, obs_hist=b.obs_hist, history=b.history,
                xi=b.xi, references=b.references, prev_action=b.prev_action,
                horizon=args.horizon, obs_noise=0.0,
            )
            _, terms = compute_joint_loss(
                ro, teacher_states=b.teacher_states, gamma=trainer.cfg.gamma,
                weights=trainer.cfg.weights,
            )
            st.append(float(terms.get("state", torch.zeros(()))))
            rw.append(float(ro.rewards.mean()))
            # ---- 固定前缀口径（跨 H 可比）----
            pref = min(PREFIX_STEPS, ro.rewards.shape[0])
            _, terms8 = compute_joint_loss(
                ro, teacher_states=b.teacher_states[:, :pref], gamma=trainer.cfg.gamma,
                weights=trainer.cfg.weights,
            )
            from diff_vla_wbc.compat import quat_error_magnitude as _qem  # noqa: PLC0415
            _n = min(ro.states.shape[0] - 1, b.teacher_states.shape[1])
            _pred, _tgt = ro.states[1:_n + 1], b.teacher_states[:, :_n].permute(1, 0, 2)
            pos_e.append(float(F.mse_loss(_pred[..., 0:3], _tgt[..., 0:3])))
            ori_e.append(float(_qem(_pred[..., 3:7], _tgt[..., 3:7]).square().mean()))
            q_e.append(float(F.mse_loss(_pred[..., 7:36], _tgt[..., 7:36])))
            vel_e.append(float(F.mse_loss(_pred[..., 36:71], _tgt[..., 36:71])))
            st8.append(float(terms8.get("state", torch.zeros(()))))
            rw8.append(float(ro.rewards[:pref].mean()))
            surv.append(float(ro.survival[-1].mean()))
            surv8.append(float(ro.survival[pref].mean()))
            rsum.append(float(ro.rewards.sum(0).mean()))
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        m = {
            "step": step,
            "elapsed_s": _time.time() - _T0 if "_T0" in globals() else 0.0,
            "h": int(args.horizon),
            "token_mse": sum(tok_mse) / len(tok_mse),
            "state_term": sum(st) / len(st),
            "rollout_reward": sum(rw) / len(rw),
            "reward_sum": sum(rsum) / len(rsum),
            "state_term@8": sum(st8) / len(st8),
            "state_pos": sum(pos_e) / len(pos_e),
            "state_ori": sum(ori_e) / len(ori_e),
            "state_q": sum(q_e) / len(q_e),
            "state_vel": sum(vel_e) / len(vel_e),
            "rollout_reward@8": sum(rw8) / len(rw8),
            "survival": sum(surv) / len(surv),
            "survival@8": sum(surv8) / len(surv8),
        }
        if hm:
            m["hand_mse"] = sum(hm) / len(hm)
        print(f"[eval {step}] " + json.dumps({k: round(v, 6) for k, v in m.items()}), flush=True)
        if args.eval_out is not None:
            args.eval_out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.eval_out, "a") as f:
                f.write(json.dumps(m) + "\n")
        return m

    # ---------------------------------------------------------------- 续训
    _start_step = int(args.start_step or 0)
    if args.resume_joint is not None:
        _ck = trainer.load(args.resume_joint)
        _start_step = max(_start_step, int(_ck.get("steps", _ck.get("iteration", 0)) or 0))
        if _is_main:
            print(f"[resume] joint <- {args.resume_joint}  (step={_start_step})", flush=True)
    if args.resume_vla is not None:
        if vlm_model is None:
            raise SystemExit("--resume-vla needs --vlm")
        _sd = torch.load(args.resume_vla, map_location="cpu", weights_only=False)
        _fm = _sd.get("fm") or _sd
        _missing, _unexpected = vlm_model.load_state_dict(_fm, strict=False)
        if _is_main:
            print(
                f"[resume] vla <- {args.resume_vla}: tensors={len(_fm)} "
                f"missing={len(_missing)} unexpected={len(_unexpected)}",
                flush=True,
            )
        _start_step = max(_start_step, int(_sd.get("step", 0) or 0))

    global _T0
    _T0 = _time.time()          # 墙钟起点：eval 行里要带 elapsed_s（A3 是同墙钟比较）
    if eval_batches and _is_main:
        run_eval(_start_step)   # 起点基线（续训时记录的是"续训起点"，不是 0）

    if args.fm_grad_backend != "pathwise":
        from diff_vla_wbc.fm_grad import BACKEND_DOC  # noqa: PLC0415

        print(f"[fm-grad] backend={args.fm_grad_backend}: {BACKEND_DOC[args.fm_grad_backend]}", flush=True)

    _deadline = (
        _time.time() + args.time_budget_minutes * 60.0 if args.time_budget_minutes > 0 else float("inf")
    )
    _t0 = _time.time()

    def _next_batch():
        b = provider.sample()
        if _use_prefetch:
            _vla_prefetch.submit(b.frames)     # 后台线程先处理，主线程去算上一步
        return b

    batch = _next_batch()
    step = _start_step          # 续训时 steps<=start 会让循环体不执行，收尾保存仍要能读到 step
    for step in range(_start_step + 1, args.steps + 1):
        nxt = _next_batch()
        if _use_prefetch:
            batch.model_inputs = _vla_prefetch.get(
                batch.frames, lambda: _vla_memo.get(batch.frames)
            )
        metrics = trainer.train_step(batch)
        if (
            args.wm_update_every
            and step > args.wm_update_warmup
            and step % args.wm_update_every == 0
        ):
            # 消融臂 B：用真实转移窗口更新世界模型（有监督，不经策略梯度），随后立刻冻结。
            wm_windows = sample_reference_windows(
                bundle, library,
                batch_size=args.wm_update_batch,
                window=args.wm_update_window,
                seed=args.seed + 100000 + step,
            )
            metrics.update(trainer.train_world_model(wm_windows, sync=True))
        if _is_main and (step % args.log_every == 0 or step == 1):
            print(f"[step {step}/{args.steps}] " + json.dumps({k: round(v, 6) for k, v in metrics.items()}), flush=True)
        if _is_main and args.eval_every and eval_batches and step % args.eval_every == 0:
            run_eval(step)
        if _is_main and args.save_every and step % args.save_every == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            trainer.save(args.out / f"joint_{step:06d}.pt")
            print(f"[saved] {args.out / f'joint_{step:06d}.pt'}", flush=True)
        if _is_main and args.save_vla_every and vlm_model is not None and step % args.save_vla_every == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            vpath = args.out / f"vla_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "fm": {
                        n: p.detach().to("cpu", dtype=torch.float32)
                        for n, p in vlm_model.named_parameters()
                        if p.requires_grad
                    },
                },
                vpath,
            )
            print(f"[saved] {vpath}", flush=True)
        # ---------------------------------------------------------------- 墙钟停止（协同退出）
        # 必须由所有 rank **一致**决定何时退出：若 rank0（带 eval，较慢）先到墙钟就 break，
        # rank1 会继续下一步并卡在 sync_grads 的 all_reduce 上 —— 40 min 的格子会静默挂死。
        # 所以每个 rank 各自算 locally-expired，取 MAX 协商，再统一退出。
        if _world > 1:
            import torch.distributed as _d

            if not isinstance(_T0, float):
                _T0 = float(_T0)
            stop_flag = torch.tensor(
                [1.0 if _time.time() >= _deadline else 0.0], device=device
            )
            try:
                _d.all_reduce(stop_flag, op=_d.ReduceOp.MAX)
            except Exception:  # 单卡回退 / 后端异常时不阻塞训练
                pass
            stop = bool(stop_flag.item() > 0)
            del stop_flag
        else:
            stop = _time.time() >= _deadline
        batch = nxt
        if _use_prefetch:
            _vla_memo.clear()
        if stop:
            if _is_main:
                print(
                    f"[budget] {args.time_budget_minutes:g} min reached after {step} steps "
                    f"({_time.time() - _t0:.0f}s)",
                    flush=True,
                )
            break

    args.out.mkdir(parents=True, exist_ok=True)
    if _is_main and vlm_model is not None:
        # 与 train_ppo.py 对齐：两臂都在结束时落一份 vla_final.pt，后续可以直接接 A1 的成功率流程。
        vpath = args.out / "vla_final.pt"
        torch.save(
            {
                "step": step,
                "fm": {
                    n: p.detach().to("cpu", dtype=torch.float32)
                    for n, p in vlm_model.named_parameters()
                    if p.requires_grad
                },
            },
            vpath,
        )
        print(f"[saved] {vpath}", flush=True)
    if args.save:
        path = args.out / "joint_latest.pt"
        trainer.save(path)
        print(f"[saved] {path}")
    print("[done] differentiable FM -> adapter -> WBC -> WM -> task loss step completed")


if __name__ == "__main__":
    main()
