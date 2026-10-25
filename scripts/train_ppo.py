#!/usr/bin/env python3
"""PPO arm of the BPTT-vs-PPO fixed-time-budget ablation.

Everything except the gradient estimator is shared with ``scripts/train_joint.py``:
same motion npz, same frozen SuperMimic WBC, same frozen world model, same
reward, same VLA checkpoint, same hand auxiliary labels.  See
``diff_vla_wbc/ppo.py`` for why the objective reduces to a one-step clipped
policy gradient on the token chunk.

Reference commands (both arms must get the same ``--time-budget-minutes``):

    # BPTT (exact gradient through FM + WBC + WM)
    bash scripts/launch_joint_ha.sh                     # or train_joint.py directly

    # PPO (score-function gradient of the same reward)
    python3 -m torch.distributed.run --nproc_per_node 2 --master_port 29644 \
      scripts/train_ppo.py \
        --motion data/npz/g1_tasks --supermimic-ckpt <sm.pt> \
        --vlm --vlm-path <cosmos> --vlm-ckpt <gr00t ckpt> --vlm-dataset \
        --hand-loss-weight 0.2 --time-budget-minutes 120 \
        --out runs/ablation_ppo_2h --eval-out runs/ablation_ppo_2h/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.compat import ensure_supermimic_importable  # noqa: E402

ensure_supermimic_importable()

from diff_vla_wbc.data import MotionBatchProvider  # noqa: E402
from diff_vla_wbc.gr00t_fm import sample_model_with_grad  # noqa: E402
from diff_vla_wbc.ppo import PpoConfig, PpoTrainer  # noqa: E402
from diff_vla_wbc.supermimic_backend import (  # noqa: E402
    SuperMimicBundle,
    build_g1_assets,
    load_motion_library,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/ppo_smoke"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=10**9, help="upper bound; the time budget usually stops first")
    ap.add_argument("--time-budget-minutes", type=float, default=0.0, help="wall-clock budget (0 = off)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--supermimic-ckpt", type=Path, required=True)
    ap.add_argument("--motion-packed", type=Path, default=None,
                    help="Isaac (trackerlab) packed motion library; default = build/cache one "
                         "from --motion under --motion-packed-dir")
    ap.add_argument("--motion-packed-dir", type=Path, default=None,
                    help="cache dir for the packed motion library (must be the HA-only one)")
    ap.add_argument("--obs-mode", default=None, help="override the checkpoint's observation mode")
    ap.add_argument("--hist", type=int, default=None, help="override the checkpoint's history length")
    ap.add_argument("--fm-lr", type=float, default=1e-5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--sigma-init", type=float, default=0.25)
    ap.add_argument("--value-lr", type=float, default=1e-3)
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--hand-loss-weight", type=float, default=0.0,
                    help="supervised hand auxiliary term, added exactly like in train_joint.py")
    ap.add_argument("--hand-grasp-boost", type=float, default=5.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--prefetch-vla", dest="prefetch_vla", action="store_true", default=False,
                    help="后台线程预取下一步 VLA 预处理（CPU 侧，默认开；--no-prefetch-vla 关闭）")
    ap.add_argument("--no-prefetch-vla", dest="prefetch_vla", action="store_false")
    ap.add_argument("--eval-batches", type=int, default=2)
    ap.add_argument("--eval-out", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--save-vla-every", type=int, default=0)
    vlm = ap.add_argument_group("top-level VLM (shared with train_joint.py)")
    vlm.add_argument("--vlm", action="store_true")
    vlm.add_argument("--vlm-path", type=Path, default=None)
    vlm.add_argument("--vlm-ckpt", type=Path, required=True)
    vlm.add_argument("--vlm-select-layer", type=int, default=16)
    vlm.add_argument("--vlm-steps", type=int, default=2)
    vlm.add_argument("--vlm-state-dim", type=int, default=43)
    vlm.add_argument("--vlm-embodiment-id", type=int, default=0)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    if world > 1:
        import torch.distributed as dist

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[dist] rank {rank}/{world} local_rank={local_rank} backend={backend}", flush=True)
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed + rank)
    files = collect_npz(args.motion)

    # 与 train_joint.py 完全同一条线路：官方 ckpt 是 Isaac/trackerlab 的 {pi,wm}，
    # policy/WM/观测/奖励/软终止/FK/参考运动库全部来自 trackerlab-supermimic（Isaac Lab 实现），
    # 不碰 third_party/SuperMimic-Reproduced 的 mjlab（MuJoCo）实现。
    from diff_vla_wbc.isaac_joint import is_isaac_checkpoint as _is_isaac  # noqa: PLC0415

    if not _is_isaac(args.supermimic_ckpt):
        raise SystemExit(
            f"{args.supermimic_ckpt} is not an Isaac/trackerlab ckpt: the BPTT-vs-PPO ablation "
            "must use the Isaac Lab chain (ckpt_010500.pt), not the mjlab/MuJoCo replica")
    from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: PLC0415

    bundle = load_isaac_bundle(
        args.supermimic_ckpt, device=device, npz=files,
        packed_motion=args.motion_packed, packed_dir=args.motion_packed_dir,
        obs_mode=args.obs_mode, hist=args.hist, token_dim=64,
    )
    library = bundle.library
    print(f"[info] Isaac joint bundle ready: {library}", flush=True)

    hand_dir = args.motion[0] if args.motion[0].is_dir() else args.motion[0].parent
    provider = MotionBatchProvider(
        bundle, library, batch_size=args.batch_size, horizon=args.horizon,
        token_source=None, token_dim=64, token_horizon=40, seed=args.seed, hand_dir=hand_dir,
    )

    if not args.vlm:
        raise SystemExit("--vlm is required: PPO must train the same action head as the BPTT arm")

    from diff_vla_wbc.g1_tasks import (  # noqa: PLC0415
        TaskFrameIndex,
        TaskObservationBuilder,
        load_gr00t_processor,
        processed_batch,
    )
    from diff_vla_wbc.gr00t_vlm import Gr00tVLM, describe_vlm, resolve_vlm_source  # noqa: PLC0415

    print(f"[info] top-level VLM source: {resolve_vlm_source(args.vlm_path)}", flush=True)
    vlm_model = Gr00tVLM.from_pretrained(
        args.vlm_path,
        checkpoint=args.vlm_ckpt,
        select_layer=args.vlm_select_layer,
        device=device,
        action_horizon=40,
        max_state_dim=args.vlm_state_dim,
        freeze_vlm=True,
        attn_implementation="sdpa",
    )
    print("[info] top-level VLM: " + json.dumps(describe_vlm(vlm_model), default=str), flush=True)

    frame_index = TaskFrameIndex(files)
    if frame_index.num_frames != library.num_frames:
        raise SystemExit(
            f"npz index has {frame_index.num_frames} frames but the motion library has "
            f"{library.num_frames}; rebuild it with scripts/prepare_g1_tasks_npz.py"
        )
    builder = TaskObservationBuilder(frame_index)
    processor = load_gr00t_processor(args.vlm_ckpt, vlm_path=args.vlm_path)
    hand_action_dim = int(vlm_model.action_head.action_dim)

    def fm_sampler(inputs, *, initial_noise=None):  # noqa: ANN001
        # sample_model_with_grad 要求 grad 打开（它要对 Euler 链建图）；PPO 的采集循环常在
        # torch.no_grad() 里，直接调用会 RuntimeError。这里局部打开 grad，保持原调用语义。
        with torch.enable_grad():
            return sample_model_with_grad(
                vlm_model, inputs, initial_noise=initial_noise, num_steps=args.vlm_steps
            )

    # 采样器要求的噪声宽度 = max_action_dim（本 ckpt = 132，见 checkpoint-8000/config.json），
    # 与 action_head.action_dim（64）不是同一个量；用 action_head.action_dim 造噪声会报 shape 错。
    _noise_w = int(getattr(getattr(vlm_model, "config", None), "max_action_dim", 0) or 0) \
        or int(getattr(vlm_model.action_head, "max_action_dim", 0) or 132)

    def fm_full_from_meta(meta):  # noqa: ANN001
        inputs = processed_batch(processor, builder, meta["frames"])
        # 宽度必须与 batch 里 action 的最后一维一致（GR00T processor 会 pad 到 max_action_dim=132）；
        # 用 model config 的 max_action_dim 不够稳（PPO 里可能被建成 64）。
        w = _noise_w
        if isinstance(inputs, dict):
            for key in ("action", "actions"):
                v = inputs.get(key)
                if hasattr(v, "shape") and v.ndim >= 1:
                    w = int(v.shape[-1])
                    break
        noise = torch.randn(
            int(inputs["input_ids"].shape[0]), 40, w,
            device=device, dtype=vlm_model.dtype,
        )
        return fm_sampler(inputs, initial_noise=noise)

    fm_params = [p for p in vlm_model.action_head.parameters() if p.requires_grad]
    print(f"[info] trainable action-head params: {sum(p.numel() for p in fm_params):,}", flush=True)
    optimizer = torch.optim.AdamW(fm_params, lr=args.fm_lr)

    if world > 1:
        import torch.distributed as dist

        for mod in (bundle.policy, bundle.wm, vlm_model):
            for p in mod.parameters():
                dist.broadcast(p.data, src=0)
            for b in mod.buffers():
                dist.broadcast(b.data, src=0)
        print("[dist] initial parameters broadcast from rank0", flush=True)

    cfg = PpoConfig(
        clip_eps=args.clip_eps,
        epochs=args.ppo_epochs,
        sigma_init=args.sigma_init,
        value_lr=args.value_lr,
        entropy_coef=args.entropy_coef,
        aux_hand_weight=args.hand_loss_weight,
        aux_grasp_boost=args.hand_grasp_boost,
    )
    trainer = PpoTrainer(
        bundle=bundle,
        fm_sampler=fm_sampler,
        fm_parameters=fm_params,
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        horizon=args.horizon,
        token_dim=64,
        noise_dim=_noise_w,
        hand_sampler=fm_full_from_meta if args.hand_loss_weight > 0 else None,
        sync_gradients=(world > 1),
    )
    if args.hand_loss_weight > 0:
        print(f"[hand] auxiliary supervision on: lambda={args.hand_loss_weight} boost={args.hand_grasp_boost}",
              flush=True)

    eval_batches = None
    if args.eval_every:
        eval_batches = [provider.sample() for _ in range(args.eval_batches)]
        # BPTT 臂的 eval 会算 token_mse（VLA token vs 录制 token）；PPO 臂必须用同一个口径，
        # 否则两张曲线表里一条 token_mse 是 0（"没算"）看起来像"完美"。
        from diff_vla_wbc.data import RecordedTokenSource  # noqa: PLC0415

        recorded_src = RecordedTokenSource(
            hand_dir, library, token_horizon=40, token_dim=64, device=device
        )
        for b in eval_batches:
            # evaluate() 走 fm_sampler(batch.model_inputs)：训练循环里才填 model_inputs，
            # 评测批次必须自己填一次，否则会拿着 meta dict 去调 VLM（KeyError: input_ids）。
            b.model_inputs = processed_batch(processor, builder, b.frames)
            b.recorded_tokens = recorded_src({"frames": b.frames}).detach()
        print(f"[eval] {len(eval_batches)} fixed batches every {args.eval_every} steps "
              f"(token_mse vs recorded g1_motion_token_64)", flush=True)

    _T0 = time.time()           # 墙钟起点（eval 行带 elapsed_s，A3 按同墙钟比较）

    def run_eval(step: int) -> dict:
        metrics = trainer.evaluate(eval_batches)
        metrics = {"step": step, "elapsed_s": round(time.time() - _T0, 1),
                   **{k: round(v, 6) for k, v in metrics.items()}}
        print(f"[eval {step}] " + json.dumps(metrics), flush=True)
        if is_main and args.eval_out is not None:
            args.eval_out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.eval_out, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        return metrics

    if is_main and eval_batches:
        run_eval(0)

    deadline = (
        time.time() + args.time_budget_minutes * 60.0 if args.time_budget_minutes > 0 else float("inf")
    )
    t_start = time.time()
    # ---- CPU 热路径：跨步预取（与 train_joint.py 同一套；默认开）----------------
    from diff_vla_wbc.prefetch import BatchPrefetcher  # noqa: PLC0415

    _prefetch = BatchPrefetcher(
        lambda frames: processed_batch(processor, builder, frames),
        enabled=bool(getattr(args, "prefetch_vla", True)),
    )
    print(f"[prefetch] enabled={_prefetch.enabled}", flush=True)

    def _next_batch():
        b = provider.sample()
        _prefetch.submit(b.frames)
        return b

    step = 0
    batch = _next_batch()
    while step < args.steps:
        step += 1
        nxt = _next_batch()
        batch.model_inputs = _prefetch.get(
            batch.frames, lambda: processed_batch(processor, builder, batch.frames)
        )
        metrics = trainer.train_step(batch)
        if is_main and (step % args.log_every == 0 or step == 1):
            print(f"[step {step}] " + json.dumps({k: round(v, 6) for k, v in metrics.items()}), flush=True)
        if is_main and args.eval_every and eval_batches and step % args.eval_every == 0:
            run_eval(step)
        if is_main and args.save_every and step % args.save_every == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "opt": optimizer.state_dict()}, args.out / f"ppo_state_{step:06d}.pt")
        if (
            is_main
            and args.save_vla_every
            and step % args.save_vla_every == 0
        ):
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
        batch = nxt
        if time.time() >= deadline:
            print(f"[budget] {args.time_budget_minutes:g} min reached after {step} steps "
                  f"({time.time() - t_start:.0f}s)", flush=True)
            break

    args.out.mkdir(parents=True, exist_ok=True)
    vpath = args.out / "vla_final.pt"
    if is_main:
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
        if eval_batches:
            run_eval(step)
    print("[done] PPO arm finished", flush=True)


if __name__ == "__main__":
    main()
