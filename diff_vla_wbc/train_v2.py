"""S3 阶段 C：联合微调 **v2**（小幅精修）。

与 v1（``scripts/train_joint.py`` + ``diff_vla_wbc.train.JointTrainer``）的差别，逐条对应
``docs/JOINT_FINETUNE_REVIEW_REDESIGN_V2_20261026.md`` §2.1 的 J* 问题：

======  ==========================================================================
J3/J4   每步**只采样一次**（K = 服务端 ``fm_steps``），主监督换成 GR00T 原生
        flow-matching 损失（``L_bc``）；删掉 ``token_anchor`` 与 hand-MSE 分支。
J2      采样结果先 ``codec.decode`` 回原始 token/手部空间再喂 WBC；BC 目标用
        ``codec.encode`` 回到归一化空间（与 head 训练口径一致）。
J5      WBC / WM **完全冻结**（``requires_grad_(False)`` + eval），优化器只放 head；
        保存时逐张量与源 ckpt 对比（``torch.equal``）。
J7/J1'' 起点不再是参考帧：``burnin.sample_burnin`` 用冻结 WBC 在 WM 里走 K 步 + 扰动。
J8      H=16（覆盖闭环 12 步重取间隔）。
J9      固定 64 窗口评测（不是 4 个），闭环选 ckpt 交给 ``scripts/mini_closedloop_v2.sh``。
J10     目标函数仍只优化"可执行性"：不产生新任务能力。门禁只看闭环（S3.8）。
======  ==========================================================================

计算图（§3.5）::

    obs → GR00T(backbone 冻结) → head → x_norm → codec.decode → token64 → WBC(冻结)
        → WM(冻结) → reward
    L = λ_bc·L_FM + λ_r·(−J/(H·MAX_STEP_REWARD)) [+ λ_s·L_state] [+ λ_a·蒸馏]

注意：``imagine_with_tokens`` 在 H 步里对 ``tokens[:, t]`` 取条件，而 head 只输出
``action_horizon=40`` 帧，所以 ``H ≤ 40``；H=16 时用到 chunk 的前 16 帧。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from .burnin import BurnInResult, sample_burnin
from .data import MotionBatch, MotionBatchProvider
from .fm_bc import (
    bc_target,
    copy_backbone_output,
    fm_bc_loss,
    grad_norm_ratio,
    grad_ratio_parameters,
    sample_once_with_grad,
)
from .rollout import (
    _state_error,
    discounted_return,
    freeze_world_model,
    imagine_with_tokens,
)

#: 与 trackerlab MDP 的 ``MAX_STEP_REWARD`` 一致（每步奖励上界，归一化分母）
try:  # pragma: no cover - only importable with the trackerlab tree
    from .isaac_joint import MAX_STEP_REWARD as _MAX_STEP_REWARD  # noqa: PLC0415
except Exception:  # pragma: no cover
    _MAX_STEP_REWARD = 0.1


@dataclass
class JointV2Config:
    # --- rollout ---
    horizon: int = 16
    gamma: float = 0.99
    obs_noise: float = 0.0
    # --- 损失权重 ---
    lambda_bc: float = 1.0
    lambda_r: float = 0.3
    lambda_s: float = 0.0
    lambda_a: float = 0.0
    state_vel: float = 0.05
    # --- 生成式 head ---
    fm_steps: int = 4
    fm_lr: float = 1e-5
    fm_clip: float = 0.5
    weight_decay: float = 0.0
    grad_backend: str = "pathwise"
    fm_truncated_tail: int = 1
    head_eval_mode: bool = True
    # --- 优化 ---
    grad_accum: int = 4
    # --- 诊断 ---
    grad_ratio_every: int = 50


class JointTrainerV2:
    """冻结执行器的联合微调 v2（只更新 GR00T action head）。"""

    def __init__(
        self,
        bundle,
        vlm,
        codec,
        *,
        cfg: JointV2Config | None = None,
        device: str | torch.device | None = None,
        adapter=None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.bundle = bundle
        self.vlm = vlm
        self.codec = codec
        self.cfg = cfg or JointV2Config()
        self.device = torch.device(device or bundle.device)
        self.log = log
        self.steps = 0
        self.adapter = adapter

        # ---------------- J5 / S3.2：执行器与动力学完全冻结 -----------------
        self._freeze_executor()
        self.fm_parameters = [p for p in vlm.action_head.parameters() if p.requires_grad]
        if not self.fm_parameters:
            raise SystemExit("[v2] action head 没有可训练参数（tune_* 全 False？）")
        self.optimizer = torch.optim.AdamW(
            [{"params": self.fm_parameters, "lr": self.cfg.fm_lr}],
            weight_decay=self.cfg.weight_decay,
        )
        self._ratio_params = grad_ratio_parameters(vlm.action_head)
        self._accum_count = 0
        self.last_metrics: dict[str, float] = {}
        log(
            f"[AUDIT][v2] trainable=head_only params={sum(p.numel() for p in self.fm_parameters):,} "
            f"wbc_trainable={sum(p.numel() for p in bundle.policy.parameters() if p.requires_grad)} "
            f"wm_trainable={sum(p.numel() for p in bundle.wm.parameters() if p.requires_grad)} "
            f"H={self.cfg.horizon} K={self.cfg.fm_steps} lam_bc={self.cfg.lambda_bc} "
            f"lam_r={self.cfg.lambda_r} lam_s={self.cfg.lambda_s} lam_a={self.cfg.lambda_a} "
            f"grad_accum={self.cfg.grad_accum} backend={self.cfg.grad_backend} "
            f"head_eval={self.cfg.head_eval_mode}",
            flush=True,
        )

    # ------------------------------------------------------------------ freeze
    def _freeze_executor(self) -> None:
        for p in self.bundle.policy.parameters():
            p.requires_grad_(False)
        self.bundle.policy.eval()
        freeze_world_model(self.bundle.wm)          # 同时 .eval()
        if self.adapter is not None:
            for p in self.adapter.parameters():
                p.requires_grad_(False)
            self.adapter.eval()

    def head_mode(self) -> None:
        """head 的 train/eval 由配置决定（默认 eval，与部署一致、且关掉 state dropout）。"""
        if self.cfg.head_eval_mode:
            self.vlm.action_head.eval()
        else:
            self.vlm.action_head.train()

    # ------------------------------------------------------------------ 前向
    def _backbone_output(self, model_inputs: dict):
        """冻结 backbone 前向（no_grad），返回 ``(act_in, bb_out_副本)``。"""
        _bb_in, act_in = self.vlm.prepare_input(dict(model_inputs))
        with torch.no_grad():
            bb_out = self.vlm.backbone(_bb_in)
        # 浅拷贝：下游 ``process_backbone_output`` 会原地改写字典
        return act_in, copy_backbone_output(bb_out)

    def _sample_tokens(self, act_in, bb_out, *, generator: torch.Generator | None = None):
        """一次有梯度的 FM 采样 → ``(x_norm, x_raw, tokens)``。"""
        head = self.vlm.action_head
        batch = int(act_in["state"].shape[0])
        horizon = int(head.action_horizon)
        dim = int(head.action_dim)
        noise = torch.randn(
            batch, horizon, dim, device=self.device, dtype=head.dtype,
            generator=generator,
        )
        x_norm = sample_once_with_grad(
            head, bb_out, act_in,
            initial_noise=noise,
            num_steps=self.cfg.fm_steps,
            grad_backend=self.cfg.grad_backend,
            truncated_tail=self.cfg.fm_truncated_tail,
        )
        x_raw = self.codec.decode(x_norm)
        tokens = x_raw[..., :64]
        return x_norm, x_raw, tokens

    def _rollout(self, batch, tokens):
        return imagine_with_tokens(
            self.bundle,
            tokens=tokens,
            state=batch.start_state,
            obs_hist=batch.obs_hist,
            history=batch.history,
            xi=batch.xi,
            references=batch.references,
            prev_action=batch.prev_action,
            horizon=self.cfg.horizon,
            obs_noise=self.cfg.obs_noise,
        )

    # ------------------------------------------------------------------ 训练步
    def train_step(self, batch) -> dict[str, float]:
        cfg = self.cfg
        self.bundle.policy.eval()
        self.bundle.wm.eval()
        self.head_mode()

        model_inputs = batch.model_inputs
        if model_inputs is None:
            raise SystemExit("[v2] batch 没有 model_inputs：v2 只走真实观测（dataset）路径")
        act_in, bb_out = self._backbone_output(model_inputs)
        x_norm, x_raw, tokens = self._sample_tokens(act_in, bb_out)
        out_of_range = self.codec.out_of_range_frac(x_norm)

        # ---------- (1) 奖励项（§3.5 归一化回报）----------
        # λ_r = λ_s = λ_a = 0 时（纯 BC 对照臂）不需要想象 rollout：省一半时间，
        # 也避免把"两套 token 空间"的回报当读数（S0.4 之后 port≠label）。
        _needs_rollout = (cfg.lambda_r > 0 or cfg.lambda_s > 0 or cfg.lambda_a > 0)
        if not _needs_rollout:
            loss_bc_only = fm_bc_loss(
                self.vlm.action_head, bb_out, act_in,
                *bc_target(
                    self.codec, batch.port_tokens, batch.hand_ref,
                    head_dim=int(self.vlm.action_head.action_dim),
                    dtype=self.vlm.action_head.dtype, device=self.device,
                ),
            )
            accum0 = max(1, int(cfg.grad_accum))
            self._accum_count += 1
            if self._accum_count == 1:
                self.optimizer.zero_grad(set_to_none=True)
            (cfg.lambda_bc * loss_bc_only / accum0).backward()
            if self._accum_count >= accum0:
                torch.nn.utils.clip_grad_norm_(self.fm_parameters, cfg.fm_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_count = 0
            self.steps += 1
            return {
                "loss": float((cfg.lambda_bc * loss_bc_only).detach()),
                "loss/fm_bc": float(loss_bc_only.detach()),
                "codec/out_of_range": float(out_of_range),
                "rollout/skipped": 1.0,
            }
        rollout = self._rollout(batch, tokens)
        ret = discounted_return(rollout, cfg.gamma)
        ret_norm = ret / (cfg.horizon * float(_MAX_STEP_REWARD))
        loss_r = -ret_norm.mean()
        metrics: dict[str, float] = {
            "loss/reward_raw": float(ret.mean().detach()),
            "loss/reward_norm": float(ret_norm.mean().detach()),
            "rollout/reward": float(rollout.rewards.mean().detach()),
            "rollout/survival": float(rollout.survival[-1].mean().detach()),
            "rollout/action_abs": float(rollout.actions.abs().mean().detach()),
        }

        # ---------- (2) FM-BC（J4：GR00T 原生损失，单一主项）----------
        target, mask = bc_target(
            self.codec, batch.port_tokens, batch.hand_ref,
            head_dim=int(self.vlm.action_head.action_dim),
            dtype=self.vlm.action_head.dtype, device=self.device,
        )
        loss_bc = fm_bc_loss(self.vlm.action_head, bb_out, act_in, target, mask)

        loss = cfg.lambda_bc * loss_bc + cfg.lambda_r * loss_r

        # ---------- (3) 可选：state 项（速度通道降权，见 rollout.LossWeights）----------
        if cfg.lambda_s > 0 and batch.teacher_states is not None:
            _h = rollout.states.shape[0] - 1
            loss_s = _state_error(
                rollout.states[1:],
                batch.teacher_states[:, :_h].permute(1, 0, 2),
                vel_weight=cfg.state_vel,
            )
            loss = loss + cfg.lambda_s * loss_s
            metrics["loss/state"] = float(loss_s.detach())

        # ---------- (4) 可选：动作蒸馏（行为空间度量，WBC 冻结）----------
        if cfg.lambda_a > 0:
            oh = batch.obs_hist
            a_vla = self.bundle.act(oh, tokens[:, 0])
            with torch.no_grad():
                a_ref = self.bundle.act(oh, batch.port_tokens[:, 0].to(tokens.dtype))
            loss_a = torch.nn.functional.mse_loss(a_vla.float(), a_ref.float())
            loss = loss + cfg.lambda_a * loss_a
            metrics["loss/action_distill"] = float(loss_a.detach())

        # ---------- 梯度比诊断（每 N 步；只打印，不写 .grad）----------
        if cfg.grad_ratio_every > 0 and (self.steps + 1) % cfg.grad_ratio_every == 0:
            try:
                metrics["grad_ratio"] = grad_norm_ratio(
                    cfg.lambda_r * loss_r, cfg.lambda_bc * loss_bc, self._ratio_params
                )
            except Exception as exc:  # 诊断绝不能挡住训练
                metrics["grad_ratio"] = float("nan")
                self.log(f"[v2][warn] grad_ratio 失败: {exc}")

        # ---------- 累积 + 反传 + 步进 ----------
        accum = max(1, int(cfg.grad_accum))
        self._accum_count += 1
        if self._accum_count == 1:
            self.optimizer.zero_grad(set_to_none=True)
        (loss / accum).backward()
        if self._accum_count >= accum:
            torch.nn.utils.clip_grad_norm_(self.fm_parameters, cfg.fm_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_count = 0
        self.steps += 1

        metrics.update({
            "loss": float(loss.detach()),
            "loss/fm_bc": float(loss_bc.detach()),
            "codec/out_of_range": float(out_of_range),
        })
        self.last_metrics = metrics
        return metrics

    # ------------------------------------------------------------------ 保存
    @torch.no_grad()
    def save_vla(self, path: str | Path, *, extra: dict | None = None) -> Path:
        """写 ``vla_*.pt``：``{step, fm_steps, fm:{name:tensor}}``（与 export_vla_ckpt 兼容）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(self.steps),
            "fm_steps": int(self.cfg.fm_steps),
            "fm": {
                n: p.detach().to("cpu", dtype=torch.float32)
                for n, p in self.vlm.named_parameters()
                if p.requires_grad
            },
            "v2": asdict(self.cfg),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    @torch.no_grad()
    def save_joint(self, path: str | Path) -> Path:
        """写 ``joint_*.pt``（trackerlab 布局；``pi`` 应与源 ckpt 逐张量相等）。"""
        path = Path(path)
        extra: dict = {"joint_v2": asdict(self.cfg), "steps": int(self.steps)}
        try:  # 执行器 spec 必须跟着权重走（W1/S0.1）
            from trackerTask.supermimic import robot as _rm  # noqa: PLC0415

            extra["robot_spec"] = _rm.robot_spec_snapshot(self.bundle.spec)
        except Exception as exc:  # pragma: no cover
            self.log(f"[v2][warn] robot_spec 快照失败（ckpt 将缺 robot_spec）: {exc}")
        if self.adapter is not None:
            extra["adapter"] = {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()}
        self.bundle.save_checkpoint(path, **extra)
        return path


# --------------------------------------------------------------------------- 冻结门禁


def assert_wbc_frozen(bundle, source_ckpt: str | Path, log: Callable[[str], None] = print) -> bool:
    """S3.2 门禁：``bundle.policy`` 与源 WBC ckpt 的 ``pi`` 逐张量相等（``torch.equal``）。"""
    ck = torch.load(str(source_ckpt), map_location="cpu", weights_only=False)
    src = ck.get("pi")
    if src is None:
        log(f"[GATE][freeze] SKIP: {source_ckpt} 没有 'pi'")
        return False
    cur = {k: v.detach().cpu() for k, v in bundle.policy.state_dict().items()}
    bad = [
        k for k in src
        if k not in cur or cur[k].shape != src[k].shape or not torch.equal(cur[k], src[k])
    ]
    extra = [k for k in cur if k not in src]
    ok = not bad and not extra
    log(
        f"[GATE][freeze] {'PASS' if ok else 'FAIL'} tensors={len(src)} "
        f"mismatch={bad[:5]}{'...' if len(bad) > 5 else ''} extra={extra[:5]}",
    )
    return ok


# --------------------------------------------------------------------------- 批数据


class BurnInBatchProvider:
    """在 ``MotionBatchProvider`` 之上把起点换成 **burn-in 起点**（S3.4）。"""

    def __init__(
        self,
        bundle,
        library,
        *,
        port_source: Callable[[torch.Tensor], torch.Tensor],
        port_frame_source: Callable[[torch.Tensor], torch.Tensor] | None = None,
        wbc_chunk_source: Callable[[torch.Tensor], torch.Tensor] | None = None,
        online_tokens=None,
        recorded_states: torch.Tensor | None = None,
        batch_size: int = 4,
        horizon: int = 16,
        token_horizon: int = 40,
        seed: int | None = None,
        hand_dir: str | Path | None = None,
        burnin_steps: int = 10,
        burnin: bool = True,
        perturb: bool = True,
        perturb_xy: float = 0.05,
        perturb_yaw: float = 0.10,
        perturb_dq: float = 0.05,
        perturb_dqd: float = 0.30,
    ) -> None:
        self.bundle = bundle
        self.library = library
        self.port_source = port_source
        self.port_frame_source = port_frame_source or (lambda f: port_source(f)[:, 0])
        # **WBC 空间**的整段 GT token（门禁要区分"奖励通道在 WBC 空间里的信号"和
        # "部署时 VLA 空间的 token 喂给 WBC"两件事）；缺省时与 BC 源相同。
        self.wbc_chunk_source = wbc_chunk_source or port_source
        # 环内 token（hold builder + 仿真 anchor）：非 None 时 burn-in 与 GT 臂都用它，
        # 侧车源只作回退（交接文档 §5.3）。
        self.online_tokens = online_tokens
        # v3：真录制机器人状态（起点用它，而不是参考帧）
        self.recorded_states = recorded_states
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self.token_horizon = int(token_horizon)
        self.hand_dir = hand_dir
        self.burnin_steps = int(burnin_steps)
        self.burnin = bool(burnin)
        self.perturb = bool(perturb)
        self.p = dict(xy=perturb_xy, yaw=perturb_yaw, dq=perturb_dq, dqd=perturb_dqd)
        self.rng = torch.Generator(device="cpu")
        if seed is not None:
            self.rng.manual_seed(int(seed))
        self.inner = MotionBatchProvider(
            bundle, library,
            batch_size=self.batch_size, horizon=self.horizon,
            token_source=None, token_dim=64, token_horizon=self.token_horizon,
            seed=seed, hand_dir=hand_dir,
        )

    def port_tokens_at(self, frames: torch.Tensor) -> torch.Tensor:
        """帧级端口 GT token ``(B,64)``（burn-in 的 conditioning）。"""
        return self.port_frame_source(frames)

    def sample(self, frames: torch.Tensor | None = None) -> MotionBatch:
        """采一个 batch；``frames`` 给定时在**同一批帧**上重建起点（门禁对拍用）。"""
        base = self.inner.sample(frames)
        frames = base.frames
        port_chunk = self.port_source(frames)
        if self.burnin and self.burnin_steps > 0:
            bi = sample_burnin(
                self.bundle, self.library, frames,
                port_tokens=self.port_frame_source,
                token_fn=(self.online_tokens if self.online_tokens is not None else None),
                recorded_states=self.recorded_states,
                horizon=self.horizon, k=self.burnin_steps,
                generator=self.rng,
                perturb=self.perturb, **self.p,
            )
        else:
            bi = self._reference_start(frames)
        base.start_state = bi.state
        base.history = bi.history
        base.obs_hist = bi.obs_hist
        base.prev_action = bi.prev_action
        base.references = bi.references
        base.xi = bi.xi
        base.teacher_states = bi.teacher_states
        base.port_tokens = port_chunk
        if self.online_tokens is not None:
            # GT 臂：整段用 burn-in 结束时的机器人 root 算 token（= 部署时一次查询出整段）
            # 夹到 clip 末尾：frames 上界是 num_frames-horizon-2，+39 会越界（实测触发
            # CUDA device-side assert）。与 teacher_references 同口径。
            _last = self.library.clip_end[self.library.clip_id[frames]].to(frames.device)
            _idx = torch.minimum(
                frames[:, None] + torch.arange(self.token_horizon, device=frames.device)[None, :],
                _last[:, None],
            )
            base.wbc_tokens = self.online_tokens.chunk(_idx, robot_quat=bi.state[:, 3:7])
        else:
            base.wbc_tokens = self.wbc_chunk_source(frames)
        base.burnin = bi
        return base

    def _reference_start(self, frames: torch.Tensor) -> BurnInResult:
        """``burnin=False`` 时的对照起点（= 旧口径：起点就是参考帧 f）。"""
        start = self.bundle.reference_start(self.library, frames)
        return BurnInResult(
            frames=frames,
            f0=frames,
            state=start["state"],
            history=start["observations"],
            obs_hist=start["obs_hist"],
            prev_action=start["prev_action"],
            xi=start["xi"],
            references=self.bundle.teacher_references(self.library, frames, self.horizon),
            teacher_states=self.bundle.teacher_states(self.library, frames, self.horizon),
        )


# --------------------------------------------------------------------------- 固定评测


@torch.no_grad()
def fixed_eval(
    trainer: JointTrainerV2,
    eval_batches: list,
    *,
    log: Callable[[str], None] = print,
    decode_diag: bool = True,
) -> dict[str, float]:
    """S3.7 固定评测：``L_fm(val)``、burn-in 起点上的想象回报、decode 后的 token/手部 MSE。"""
    import torch.nn.functional as F  # noqa: PLC0415

    cfg = trainer.cfg
    head = trainer.vlm.action_head
    dev = trainer.device
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(20261026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261026)

    acc: dict[str, list[float]] = {}

    def add(k: str, v) -> None:
        acc.setdefault(k, []).append(float(v))

    try:
        for b in eval_batches:
            act_in, bb_out = trainer._backbone_output(b.model_inputs)
            with torch.enable_grad():
                x_norm = sample_once_with_grad(
                    head, bb_out, act_in,
                    initial_noise=torch.randn(
                        int(act_in["state"].shape[0]), int(head.action_horizon), int(head.action_dim),
                        device=dev, dtype=head.dtype,
                    ),
                    num_steps=cfg.fm_steps,
                    grad_backend=cfg.grad_backend,
                    truncated_tail=cfg.fm_truncated_tail,
                )
            x_norm = x_norm.detach()
            tgt, mask = bc_target(
                trainer.codec, b.port_tokens, b.hand_ref,
                head_dim=int(head.action_dim), dtype=head.dtype, device=dev,
            )
            loss_bc = fm_bc_loss(head, bb_out, act_in, tgt, mask)
            add("L_fm_val", loss_bc.detach())

            x_raw = trainer.codec.decode(x_norm)
            tokens_vla = x_raw[..., :64]
            if decode_diag:
                n = min(tokens_vla.shape[1], b.port_tokens.shape[1])
                add("token_mse_decoded", F.mse_loss(tokens_vla[:, :n], b.port_tokens[:, :n].float()))
                hands = x_raw[..., 64:78]
                m = min(hands.shape[1], b.hand_ref.shape[1])
                add("hand_mse_decoded", F.mse_loss(hands[:, :m], b.hand_ref[:, :m].float()))
                add("hand_absmean_pred", hands[:, :m].abs().mean())
                add("hand_absmean_label", b.hand_ref[:, :m].abs().mean())
                add("hand_gt170_pred", (hands[:, :m].abs() > 1.70).float().mean())
                add("hand_gt170_label", (b.hand_ref[:, :m].abs() > 1.70).float().mean())
                add("token_shift", (tokens_vla - b.port_tokens[:, :n].float()).abs().mean())

            for name, tok in (("vla", tokens_vla), ("gt", b.port_tokens)):
                ro = trainer._rollout(b, tok)
                ret = discounted_return(ro, cfg.gamma)
                add(f"imag_return_{name}", ret.mean())
                add(f"imag_return_norm_{name}", (ret / (cfg.horizon * float(_MAX_STEP_REWARD))).mean())
                add(f"imag_reward_{name}", ro.rewards.mean())
                add(f"imag_survival_{name}", ro.survival[-1].mean())
                if b.teacher_states is not None:
                    _h = ro.states.shape[0] - 1
                    add(f"imag_state_{name}", _state_error(
                        ro.states[1:], b.teacher_states[:, :_h].permute(1, 0, 2),
                        vel_weight=cfg.state_vel,
                    ))
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    out = {k: sum(v) / len(v) for k, v in acc.items()}
    out["step"] = int(trainer.steps)
    out["n_windows"] = (
        int(len(eval_batches) * int(eval_batches[0].start_state.shape[0])) if eval_batches else 0
    )
    log(f"[eval-v2 {trainer.steps}] " + json.dumps({k: round(v, 6) for k, v in out.items()}))
    return out


__all__ = [
    "BurnInBatchProvider",
    "JointTrainerV2",
    "JointV2Config",
    "assert_wbc_frozen",
    "fixed_eval",
]
