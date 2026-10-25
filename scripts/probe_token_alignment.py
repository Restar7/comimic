#!/usr/bin/env python3
"""Is the *recorded* token aligned with the reference motion the reward uses?

Context: ``scripts/probe_reward_landscape.py`` shows that on the Isaac bundle the recorded
token (``g1_motion_token_64``) scores ``rollout_reward`` **0.0026** / ``state_term`` **22.5**,
while the reward-optimised VLA token (token_mse 0.53 away from the recording) scores
**0.0742** / **0.99**.  A correct token route must score the *recorded* token near the top,
so either

  (a) ``RecordedTokenSource`` returns tokens for different frames than the batch's
      ``start_state`` / ``references`` (frame-index misalignment), or
  (b) the token convention (scale / normalisation) differs from what the Isaac WBC expects.

This script separates the two:

  1. compare ``RecordedTokenSource(frames)`` against ``batch.teacher_goals`` (the sequence the
     batch itself carries) -> catches (a);
  2. per-step tracking error of the recorded-token rollout (t = 0..H) and the reward-term
     breakdown at t = 0 vs t = H -> tells whether the very first step already fails;
  3. a per-dimension re-standardisation control: renormalise the recorded token with the
     dataset's own per-dim mean/std (in both directions) -> catches a pure (b) scale bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter  # noqa: E402
from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource  # noqa: E402
from diff_vla_wbc.isaac_joint import load_isaac_bundle  # noqa: E402
from diff_vla_wbc.rollout import _state_error, imagine_with_tokens  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", type=Path, default=ROOT / "data/npz/ha_tokens")
    ap.add_argument("--packed", type=Path, default=ROOT / "runs/isaac_motion/ha_tokens/packed.pt")
    ap.add_argument("--supermimic-ckpt", type=Path,
                    default=ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--hist", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    files = sorted(args.motion.glob("*.npz"))
    bundle = load_isaac_bundle(
        args.supermimic_ckpt, device=device, npz=files, packed_motion=args.packed,
        packed_dir=None, obs_mode=None, hist=args.hist, token_dim=64,
    )
    library = bundle.library
    adapter = MotionGoalAdapter(AdapterConfig(
        token_dim=64, token_horizon=40, hidden=256, transformer_layers=2, transformer_heads=4,
    )).to(device)
    recorded = RecordedTokenSource(
        args.motion, library, token_horizon=adapter.cfg.token_horizon,
        token_dim=adapter.cfg.token_dim, device=device,
    )
    provider = MotionBatchProvider(
        bundle, library, batch_size=args.batch_size, horizon=40,
        token_source=None, token_dim=adapter.cfg.token_dim,
        token_horizon=adapter.cfg.token_horizon, seed=args.seed, hand_dir=None,
    )
    b = provider.sample()
    rec = recorded({"frames": b.frames})

    print("\n[1] 对齐检查：RecordedTokenSource vs batch.teacher_goals")
    tg = b.teacher_goals
    print(f"    recorded      {tuple(rec.shape)}  std={rec.std().item():.4f} mean={rec.mean().item():+.4f}")
    if tg is not None:
        print(f"    teacher_goals {tuple(tg.shape)}  std={tg.std().item():.4f} mean={tg.mean().item():+.4f}")
        if tg.shape[-1] == rec.shape[-1]:
            n = min(tg.shape[1], rec.shape[1])
            a, c = tg[:, :n], rec[:, :n]
            print(f"    MSE(teacher_goals, recorded) = {torch.nn.functional.mse_loss(a, c).item():.6f}")
            print(f"    cos 逐元素 = {(a * c).mean().item() / (a.std().item() * c.std().item() + 1e-9):+.4f}")
        else:
            print("    维度不同 -> teacher_goals 不是 64 维 token（是 goal 编码，跳过比对）")
    print(f"\n    batch.frames = {b.frames.tolist()}  (clip 起点帧)")

    def rollout(tokens):
        with torch.no_grad():
            return imagine_with_tokens(
                bundle, tokens=tokens, state=b.start_state, obs_hist=b.obs_hist,
                history=b.history, xi=b.xi, references=b.references,
                prev_action=b.prev_action, horizon=args.horizon, obs_noise=0.0,
            )

    print("\n[2] 逐步跟踪误差（录制 token）：t=0 就已经错 = 口径问题；随 t 涨 = WM 漂移")
    ro = rollout(rec)
    ts = b.teacher_states
    print(f"    {'t':>2} {'state_err':>10} {'pos_mse':>9} {'quat_err':>9} {'joint_mse':>10} {'reward':>9}")
    for t in range(min(args.horizon, ro.states.shape[0] - 1)):
        pred, tgt = ro.states[t + 1], ts[:, t]
        per_t = _state_error(pred, tgt).item()
        pos = torch.nn.functional.mse_loss(pred[..., 0:3], tgt[..., 0:3]).item()
        q = torch.nn.functional.mse_loss(pred[..., 7:36], tgt[..., 7:36]).item()
        print(f"    {t:>2} {per_t:>10.4f} {pos:>9.4f} {'-':>9} {q:>10.4f} {ro.rewards[t].mean().item():>9.4f}")

    print("\n[3] 奖励分项（录制 token）：t=0 vs t=H-1")
    for name, val in ro.reward_terms.items():
        v0 = val[0].mean().item()
        v1 = val[min(args.horizon, val.shape[0]) - 1].mean().item()
        print(f"    {name:<32} t0={v0:+.5f}  t_end={v1:+.5f}")

    print("\n[4] 每维标准化控制（把录制 token 按数据集自身 per-dim mean/std 变换）")
    # 用整库的 token 统计做 per-dim 标准化/反标准化，看是不是"少了一步 normalize"
    tok_all = []
    for i, clip in enumerate(library.clips[: min(40, len(library.clips))]):
        idx = {"frames": torch.tensor([clip.start], device=device)}
        tok_all.append(recorded(idx)[0])
    big = torch.stack(tok_all)                     # (N, 40, 64)
    mu, sd = big.mean((0, 1)), big.std((0, 1)) + 1e-6
    print(f"    数据集 token per-dim: mu[-1:1]={mu.min().item():+.3f}..{mu.max().item():+.3f} "
          f"sd={sd.min().item():.4f}..{sd.max().item():.4f}（全局 sd={big.std().item():.4f}）")
    for label, tok in [
        ("recorded", rec),
        ("(rec-mu)/sd", (rec - mu) / sd),
        ("rec*sd", rec * sd),
        ("rec/sd", rec / sd),
    ]:
        r = rollout(tok)
        err = _state_error(r.states[1], ts[:, 0]).item()
        print(f"    {label:<14} reward={r.rewards.mean().item():+.5f}  state_err(t=1)={err:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
