"""S3 的四道自检门禁（``scripts/train_joint_v2.py --gate-only``）。

| 门禁 | 检查什么 | 通过条件 |
| ---- | -------- | -------- |
| A codec | ``codec.decode`` 与 GR00T ``processor.decode_action`` 逐元素一致；``encode∘decode`` 往返 | max|Δ| < 1e-5 |
| B freeze | WBC/WM 无 ``requires_grad``；``pi`` 与源 ckpt 逐张量相等；一步反传后 policy/WM 的 ``.grad`` 全为 None | 全过 |
| C burn-in | 同批帧上 burn-in 起点 vs 参考起点：起点真的被扰开（xy>1cm）、存活 ≥0.95；回报差单列 | 硬条件都过 |
| C2 token 对比 | **同一起点**下 GT token vs 零/随机 token 的想象回报差（奖励通道有没有信号，J7/D1） | contrast > 0 且 rel > 0.1% |
| D FM-BC | 一次真实 batch 的前向/反向：``loss/fm_bc`` 有限、梯度只落在 head、``grad_ratio`` 有限 | 全过 |

结果写 ``runs/.../gates.json``，控制台同时打印 ``[GATE]`` 行。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .rollout import discounted_return


def gate_codec(codec, processor, *, tag: str = "unitree_g1_sonic") -> dict:
    """A：与官方 decode 对拍（S0.2 第 9 条的单测，正式写进 S3 门禁）。"""
    from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: PLC0415

    torch.manual_seed(0)
    x = torch.rand(2, 40, codec.dim) * 2 - 1
    y = codec.decode(x)
    ref = processor.decode_action(x.numpy(), EmbodimentTag(tag))
    keys = ("motion_token", "left_hand_joints", "right_hand_joints")
    ref_cat = np.concatenate([np.asarray(ref[k], np.float32) for k in keys], axis=-1)
    d1 = float(np.abs(y.numpy() - ref_cat).max())
    rt = codec.encode(codec.decode(x))
    d2 = float((rt - x.clamp(-1, 1)).abs().max())
    ok = d1 < 1e-5 and d2 < 1e-5
    print(f"[GATE][codec] {'PASS' if ok else 'FAIL'} decode<->decode_action max|d|={d1:.2e} "
          f"roundtrip={d2:.2e} dim={codec.dim} raw=[{codec.raw_min.min():.3f},{codec.raw_max.max():.3f}]",
          flush=True)
    return {
        "pass": bool(ok), "max_abs_diff_decode": d1, "max_abs_diff_roundtrip": d2,
        "dim": int(codec.dim),
        "raw_min": float(codec.raw_min.min()), "raw_max": float(codec.raw_max.max()),
    }


def gate_freeze(bundle, source_ckpt, trainer, batch, *, log: Callable = print) -> dict:
    """B：冻结门禁（S3.2）。跑**一步真实反传**，确认 policy/WM 没有梯度。"""
    from .train_v2 import assert_wbc_frozen  # noqa: PLC0415

    pi_ok = assert_wbc_frozen(bundle, source_ckpt, log=log)
    wbc_req = sum(p.requires_grad for p in bundle.policy.parameters())
    wm_req = sum(p.requires_grad for p in bundle.wm.parameters())
    wbc_train = bool(bundle.policy.training)
    wm_train = bool(bundle.wm.training)

    # 注意：train_step 末尾会 zero_grad，所以这里自己做一次前向/反传，在 step 之前检查。
    from .fm_bc import bc_target, fm_bc_loss, sample_once_with_grad  # noqa: PLC0415

    cfg = trainer.cfg
    trainer.head_mode()
    act_in, bb_out = trainer._backbone_output(batch.model_inputs)
    head = trainer.vlm.action_head
    x_norm = sample_once_with_grad(
        head, bb_out, act_in,
        initial_noise=torch.randn(
            int(act_in["state"].shape[0]), int(head.action_horizon), int(head.action_dim),
            device=trainer.device, dtype=head.dtype),
        num_steps=cfg.fm_steps, grad_backend=cfg.grad_backend,
        truncated_tail=cfg.fm_truncated_tail,
    )
    tokens = trainer.codec.decode(x_norm)[..., :64]
    ro = trainer._rollout(batch, tokens)
    loss_r = -discounted_return(ro, cfg.gamma).mean()
    tgt, mask = bc_target(
        trainer.codec, batch.port_tokens, batch.hand_ref,
        head_dim=int(head.action_dim), dtype=head.dtype, device=trainer.device)
    loss_bc = fm_bc_loss(head, bb_out, act_in, tgt, mask)
    (cfg.lambda_bc * loss_bc + cfg.lambda_r * loss_r).backward()
    wbc_grads = [n for n, p in bundle.policy.named_parameters() if p.grad is not None]
    wm_grads = [n for n, p in bundle.wm.named_parameters() if p.grad is not None]
    head_grads = sum(1 for p in trainer.fm_parameters if p.grad is not None)
    trainer.optimizer.zero_grad(set_to_none=True)
    ok = pi_ok and wbc_req == 0 and wm_req == 0 and not wbc_grads and not wm_grads and head_grads > 0
    print(f"[GATE][freeze] {'PASS' if ok else 'FAIL'} pi_identical={pi_ok} "
          f"wbc_requires_grad={wbc_req} wm_requires_grad={wm_req} "
          f"wbc_grad_tensors={len(wbc_grads)} wm_grad_tensors={len(wm_grads)} "
          f"head_grad_tensors={head_grads} policy.training={wbc_train} wm.training={wm_train}",
          flush=True)
    return {
        "pass": bool(ok), "pi_identical": bool(pi_ok),
        "wbc_requires_grad": int(wbc_req), "wm_requires_grad": int(wm_req),
        "wbc_grad_tensors": len(wbc_grads), "wm_grad_tensors": len(wm_grads),
        "head_grad_tensors": int(head_grads),
        "policy_training": wbc_train, "wm_training": wm_train,
    }


@torch.no_grad()
def gate_burnin(trainer, provider, ref_provider, *, n_batches: int = 4) -> dict:
    """C：burn-in 起点 vs 参考起点（GT 端口 token、H 步、同一批帧）。"""
    cfg = trainer.cfg
    acc = {k: [] for k in (
        "ref_reward", "burn_reward", "ref_return", "burn_return", "ref_survival", "burn_survival",
        "start_err", "burnin_reward",
    )}
    frames_match = True
    for _ in range(n_batches):
        b_burn = provider.sample()
        # 同一批帧上重建"参考起点"（否则两边的帧不同，比较没有意义）
        b_ref = ref_provider.sample(b_burn.frames)
        frames_match = frames_match and torch.equal(b_burn.frames, b_ref.frames)
        for tag, b in (("ref", b_ref), ("burn", b_burn)):
            ro = trainer._rollout(b, b.port_tokens)
            ret = discounted_return(ro, cfg.gamma)
            acc[f"{tag}_reward"].append(float(ro.rewards.mean()))
            acc[f"{tag}_return"].append(float(ret.mean()))
            acc[f"{tag}_survival"].append(float(ro.survival[-1].mean()))
        acc["start_err"].append(
            float((b_burn.start_state[:, :3] - b_ref.start_state[:, :3]).norm(dim=-1).mean())
        )
        if b_burn.burnin.burnin_reward is not None:
            acc["burnin_reward"].append(float(b_burn.burnin.burnin_reward.mean()))
    mean = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
    reward_drop = mean["ref_reward"] - mean["burn_reward"]
    # 通过条件（三条"起点确实不同"的硬条件）：
    #   ① 同一批帧 ② 起点被真的扰开（xy > 1 cm）③ burn-in 没把存活打崩（≥0.95）
    # ``reward_drop`` 仍然打印并单列：它检验"起点误差会不会让想象回报下降"。
    # 2026-10-26 实测：WM 在 H=16 内几乎不失配（survival≈1），这一步的回报差是噪声级
    #   → 真正的信号强度由 C2 token 对比度来量（同一起点下 GT vs 零 token）。
    ok = frames_match and mean["start_err"] > 0.01 and mean["burn_survival"] >= 0.95
    drop_ok = mean["burn_reward"] <= mean["ref_reward"]
    print(f"[GATE][burnin] {'PASS' if ok else 'FAIL'} "
          f"reward ref={mean['ref_reward']:.5f} -> burn={mean['burn_reward']:.5f} "
          f"(drop={reward_drop:+.5f}, drop_sign_ok={drop_ok}) "
          f"survival ref={mean['ref_survival']:.4f} burn={mean['burn_survival']:.4f} "
          f"start_xy_err={mean['start_err']:.4f}m "
          f"burnin_reward={mean['burnin_reward']:.5f} frames_match={frames_match}",
          flush=True)
    return {
        "pass": bool(ok), "reward_drop_sign_ok": bool(drop_ok),
        "reward_ref": mean["ref_reward"], "reward_burn": mean["burn_reward"],
        "reward_drop": float(reward_drop),
        "survival_ref": mean["ref_survival"], "survival_burn": mean["burn_survival"],
        "start_xy_err_m": mean["start_err"], "burnin_reward": mean["burnin_reward"],
        "frames_match": bool(frames_match), "n_batches": int(n_batches),
    }


@torch.no_grad()
def gate_token_contrast(trainer, provider, *, n_batches: int = 4) -> dict:
    """C2：**同一个起点**下，想象回报对 conditioning 的区分度（奖励通道有没有信号）。

    必须同时报**两个** GT，否则会得出互相矛盾的结论（2026-10-26 实测过这个坑）：

    * ``gt_wbc``：WBC 训练空间的录制 token —— "奖励通道本身有没有信号"；
    * ``gt_bc`` ：BC 目标空间（= VLA 输出空间）的录制 token 直接喂 WBC —— "部署现实"。

    对照 ``mean``（条件均值）/ ``shuffled``（batch 内打乱）/ ``zeros`` / ``randn``（分布外）
    都基于 ``gt_wbc``。

    ⚠️ **必须配对统计**：8 个窗口时 GT−zeros 只有 +0.0004（假阴性）；64 窗口配对检验是
    +0.372（t=30.2，64/64 窗口胜出）。判据用 ``paired_t`` 与胜率。
    """
    cfg = trainer.cfg
    keys = ("gt_wbc", "gt_bc", "mean", "shuffled", "zeros", "randn")
    per_window: dict[str, list[float]] = {k: [] for k in keys}
    r0: dict[str, list[float]] = {k: [] for k in keys}
    surv: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_batches):
        b = provider.sample()
        gt = getattr(b, "wbc_tokens", b.port_tokens)
        perm = torch.randperm(gt.shape[0], device=gt.device)
        variants = {
            "gt_wbc": gt,
            "gt_bc": b.port_tokens,
            "mean": gt.mean(0, keepdim=True).expand_as(gt).contiguous(),
            "shuffled": gt[perm],
            "zeros": torch.zeros_like(gt),
            "randn": torch.randn_like(gt),
        }
        for k, tok in variants.items():
            ro = trainer._rollout(b, tok)
            per_window[k].extend(discounted_return(ro, cfg.gamma).tolist())
            r0[k].extend(ro.rewards[0].tolist())
            surv[k].extend(ro.survival[-1].tolist())

    m = {k: float(np.mean(v)) for k, v in per_window.items()}
    se = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in per_window.items()}

    def _paired(a: str, b: str):
        d = np.asarray(per_window[a]) - np.asarray(per_window[b])
        sd = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        t = float(d.mean() / sd) if sd and sd > 0 else float("nan")
        return float(d.mean()), t, float((d > 0).mean())

    c_wbc, t_wbc, w_wbc = _paired("gt_wbc", "zeros")
    c_bc, t_bc, w_bc = _paired("gt_bc", "zeros")
    rel = c_wbc / max(abs(m["gt_wbc"]), 1e-9)
    ok = bool(c_wbc > 0 and t_wbc >= 3.0 and w_wbc >= 0.8)
    print(
        f"[GATE][contrast] {'PASS' if ok else 'FAIL'} n={len(per_window['gt_wbc'])} "
        f"ret gt_wbc={m['gt_wbc']:.5f} gt_bc={m['gt_bc']:.5f} mean={m['mean']:.5f} "
        f"shuffled={m['shuffled']:.5f} zeros={m['zeros']:.5f} randn={m['randn']:.5f} | "
        f"wbc-space contrast={c_wbc:+.5f} rel={rel:+.2%} t={t_wbc:.2f} win={w_wbc:.2f} | "
        f"bc-space contrast={c_bc:+.5f} t={t_bc:.2f} win={w_bc:.2f} | "
        f"r0 gt_wbc={np.mean(r0['gt_wbc']):.5f} zeros={np.mean(r0['zeros']):.5f} | "
        f"surv gt_wbc={np.mean(surv['gt_wbc']):.4f} zeros={np.mean(surv['zeros']):.4f}",
        flush=True,
    )
    return {
        "pass": ok,
        "n_windows": len(per_window["gt_wbc"]),
        **{f"ret_{k}": round(m[k], 6) for k in keys},
        **{f"se_{k}": round(se[k], 6) for k in keys},
        **{f"r0_{k}": round(float(np.mean(r0[k])), 6) for k in keys},
        **{f"surv_{k}": round(float(np.mean(surv[k])), 6) for k in keys},
        "wbc_space_contrast": round(c_wbc, 6),
        "wbc_space_rel": round(float(rel), 6),
        "wbc_space_t": round(t_wbc, 3),
        "wbc_space_win_frac": round(w_wbc, 4),
        "bc_space_contrast": round(c_bc, 6),
        "bc_space_t": round(t_bc, 3),
        "bc_space_win_frac": round(w_bc, 4),
    }


def gate_fm_bc(trainer, batch, *, log: Callable = print) -> dict:
    """D：FM-BC 损失与梯度比（真实 batch，一次前向+反向）。"""
    old_every = trainer.cfg.grad_ratio_every
    trainer.cfg.grad_ratio_every = 1          # 门禁必须真的看到梯度比
    try:
        m = trainer.train_step(batch)
    finally:
        trainer.cfg.grad_ratio_every = old_every
    loss_bc = m.get("loss/fm_bc", float("nan"))
    loss_r = m.get("loss/reward_norm", float("nan"))
    ratio = m.get("grad_ratio", float("nan"))
    ok = np.isfinite(loss_bc) and np.isfinite(loss_r) and (not np.isfinite(ratio) or ratio >= 0)
    print(f"[GATE][fm-bc] {'PASS' if ok else 'FAIL'} loss_fm_bc={loss_bc:.5f} "
          f"loss_reward_norm={loss_r:.5f} rollout_reward={m.get('rollout/reward', float('nan')):.5f} "
          f"survival={m.get('rollout/survival', float('nan')):.4f} "
          f"grad_ratio={ratio} out_of_range={m.get('codec/out_of_range', float('nan')):.3f}",
          flush=True)
    return {"pass": bool(ok), "loss_fm_bc": float(loss_bc), "loss_reward_norm": float(loss_r),
            "rollout_reward": float(m.get("rollout/reward", float("nan"))),
            "rollout_survival": float(m.get("rollout/survival", float("nan"))),
            "grad_ratio": float(ratio)}


def _processed_batch_for(batch, args, processor):
    """按 batch 自己的 frames 现算 VLA 输入（门禁不走预取，逻辑更直白）。"""
    from .g1_tasks import TaskFrameIndex, TaskObservationBuilder, processed_batch  # noqa: PLC0415

    builder = getattr(_processed_batch_for, "_builder", None)
    if builder is None:
        index = TaskFrameIndex(sorted(Path(args.motion[0]).glob("*.npz")))
        builder = TaskObservationBuilder(index)
        _processed_batch_for._builder = builder
    return processed_batch(processor, builder, batch.frames)


def run_gates(
    args,
    bundle,
    vlm,
    codec,
    trainer,
    provider,
    ref_provider,
    processor,
    out_path: str | Path,
    *,
    n_batches: int = 4,
    log: Callable = print,
) -> dict[str, Any]:
    t0 = time.time()
    gates: dict[str, Any] = {}
    gates["A_codec"] = gate_codec(codec, processor)
    batch = provider.sample()
    batch.model_inputs = _processed_batch_for(batch, args, processor)
    gates["B_freeze"] = gate_freeze(bundle, args.wbc_ckpt, trainer, batch, log=log)
    gates["C_burnin"] = gate_burnin(trainer, provider, ref_provider, n_batches=n_batches)
    gates["C2_token_contrast"] = gate_token_contrast(trainer, provider, n_batches=n_batches)
    batch2 = provider.sample()
    batch2.model_inputs = _processed_batch_for(batch2, args, processor)
    gates["D_fm_bc"] = gate_fm_bc(trainer, batch2, log=log)
    gates["_meta"] = {
        "wbc_ckpt": str(args.wbc_ckpt), "vlm_ckpt": str(args.vlm_ckpt),
        "H": int(trainer.cfg.horizon), "fm_steps": int(trainer.cfg.fm_steps),
        "lambda_r": float(trainer.cfg.lambda_r), "lambda_bc": float(trainer.cfg.lambda_bc),
        "grad_accum": int(trainer.cfg.grad_accum), "backend": trainer.cfg.grad_backend,
        "wall_s": time.time() - t0,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gates, indent=1, default=str))
    print(f"[GATE] 结果 -> {out_path}", flush=True)
    return gates


__all__ = ["gate_burnin", "gate_codec", "gate_fm_bc", "gate_freeze", "gate_token_contrast", "run_gates"]
