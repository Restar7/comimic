"""S3 v3：**端到端联合微调（token 不做任何标签监督）**。

与 `train_v2.JointTrainerV2` 的差别（见 `docs/S3_END_TO_END_V3_PLAN_20261026.md`）：

| 部件 | v2 | v3 |
|---|---|---|
| token 监督 | FM-BC（标签） | **无**（`lambda_bc_token = 0`） |
| 手部 | 混在 78 维 BC | **hand-only FM 掩码**（`mask[..., 64:78]=1`） |
| 信用分配 | H 步 pathwise | **+ token-critic bootstrap（GAE λ）** |
| 起点 | 参考帧 / WM burn-in | **真录制机器人状态**（provider 传 `recorded_states`） |
| 场景 | 温吞 | **大扰动 + rollout 内推扰** |
| WM | 冻结 | 冻结（策略回路）+ **定期用真转移保鲜**（`refresh_wm`） |

设计原则：模型/执行器在**策略回路里**保持冻结（只让梯度穿过），
WM 的更新只来自**真实转移**（HA 录制），并且与策略图解耦（stop-gradient）——
这正是 SuperMimic「真数据保鲜 + 想象反传」的结构，只是把策略换成了 VLA。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .fm_bc import bc_target, copy_backbone_output, fm_bc_loss, sample_once_with_grad
from .isaac_joint import canonical_to_flat
from .rollout import discounted_return, imagine_with_tokens
from .token_critic import TokenCritic, critic_update, gae_targets
from .train_v2 import JointTrainerV2, JointV2Config, _MAX_STEP_REWARD


@dataclass
class V3Config:
    lambda_r: float = 0.3
    lambda_hand: float = 0.2
    use_critic: bool = True
    critic_hidden: int = 256
    critic_lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    critic_clip: float = 1.0          # SuperMimic: clip_grad_norm_(critic, 1.0)
    critic_value_clip: float = 0.1    # SuperMimic: clip_grad_value_(critic, 0.1)
    push_std: float = 0.0             # rollout 内过程扰动：root **线速度**（m/s）
    push_ang: float = 0.0             # 同上，root 角速度（rad/s）；太大会直接触发软终止
    # ---- 模型可信度（纯想象里的"约束"）----
    lambda_u: float = 0.0             # 不确定度惩罚：r_t ← r_t·exp(−λ_u·u_t)
    ensemble: bool = False            # 用已有的多个 WM 做集成分歧
    ensemble_ckpts: tuple = ()        # [(path, ...)]
    grad_log_every: int = 0           # 打印分项梯度范数
    clip_head: float = 0.5            # VLA head 的梯度范数裁剪（先测再定，见文档 §2.6）


class JointTrainerV3(JointTrainerV2):
    """端到端 v3：token 只由奖励（+critic bootstrap）监督，手部单独监督。"""

    def __init__(
        self,
        bundle,
        vlm,
        codec,
        *,
        cfg: JointV2Config | None = None,
        v3: V3Config | None = None,
        device=None,
        adapter=None,
        log: Callable[[str], None] = print,
    ) -> None:
        super().__init__(bundle, vlm, codec, cfg=cfg, device=device, adapter=adapter, log=log)
        self.v3 = v3 or V3Config()
        self.ensemble = None
        if self.v3.ensemble and self.v3.ensemble_ckpts and self.v3.lambda_u > 0:
            from .wm_ensemble import WMEnsemble  # noqa: PLC0415

            self.ensemble = WMEnsemble(self.v3.ensemble_ckpts, device=self.device)
            log(self.ensemble.audit())
        self.critic: TokenCritic | None = None
        self.critic_opt: torch.optim.Optimizer | None = None
        if self.v3.use_critic:
            self.critic = TokenCritic(hidden=self.v3.critic_hidden).to(self.device)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.v3.critic_lr)
        log(
            f"[AUDIT][v3] lambda_r={self.v3.lambda_r} lambda_hand={self.v3.lambda_hand} "
            f"token_bc=OFF critic={self.v3.use_critic} push_std={self.v3.push_std} "
            f"critic_clip={self.v3.critic_clip}/{self.v3.critic_value_clip} "
            f"head_clip={self.v3.clip_head} H={self.cfg.horizon} K={self.cfg.fm_steps}"
        )

    # ------------------------------------------------------------------ 参考状态
    def _ref_states(self, batch) -> torch.Tensor:
        """每个想象步的参考状态 ``(H+1,B,71)``（critic 输入 / 诊断用）。"""
        motion = self.bundle.motion
        rows = []
        for r in batch.references:
            f = r.f.to(motion.ref.device)
            rows.append(canonical_to_flat(motion.ref[f].float()))
        rows.append(rows[-1])                     # s_H 用最后一步的参考（同代理）
        return torch.stack(rows, dim=0)           # (H+1,B,71)

    # ------------------------------------------------------------------ 训练步
    def train_step(self, batch) -> dict[str, float]:  # noqa: C901
        cfg, v3 = self.cfg, self.v3
        self.bundle.policy.eval()
        self.bundle.wm.eval()
        self.head_mode()

        model_inputs = batch.model_inputs
        if model_inputs is None:
            raise SystemExit("[v3] batch 没有 model_inputs")
        act_in, bb_out = self._backbone_output(model_inputs)
        x_norm, x_raw, tokens = self._sample_tokens(act_in, bb_out)
        out_of_range = self.codec.out_of_range_frac(x_norm)

        rollout = imagine_with_tokens(
            self.bundle, tokens=tokens, state=batch.start_state, obs_hist=batch.obs_hist,
            history=batch.history, xi=batch.xi, references=batch.references,
            prev_action=batch.prev_action, horizon=cfg.horizon,
            obs_noise=cfg.obs_noise, push_std=v3.push_std, push_ang=v3.push_ang,
        )
        H = cfg.horizon
        # ---- 不确定度门控（纯想象里的"模型可信约束"）----
        unc_mean = 0.0
        if self.ensemble is not None and v3.lambda_u > 0:
            with torch.no_grad():
                u_steps = torch.stack(
                    [self.ensemble.disagreement(rollout.states[t], rollout.actions[t], batch.xi)
                     for t in range(H)], dim=0)                       # (H,B)
                w_steps = torch.exp(-v3.lambda_u * u_steps)
            rew = rollout.rewards * w_steps                              # 低可信度处不采信奖励
            disc = v3.gamma ** torch.arange(H, device=rew.device, dtype=rew.dtype)[:, None]
            ret = (disc * rollout.survival[:H] * rew).sum(0)
            unc_mean = float(u_steps.mean())
        else:
            ret = discounted_return(rollout, v3.gamma)               # (B,)
        ref_states = self._ref_states(batch)                     # (H+1,B,71)
        xi = batch.xi

        # ---------- 奖励项（+ critic bootstrap）----------
        boot = torch.zeros_like(ret)
        if self.critic is not None:
            with torch.no_grad():
                vH = self.critic(rollout.states[-1].detach(), ref_states[-1].detach(), xi.detach())
            boot = (v3.gamma ** H) * rollout.survival[-1] * vH
        loss_r = -(ret + boot).mean()

        # ---------- 手部（hand-only FM：token 那 64 维完全不监督）----------
        loss_hand = None
        if v3.lambda_hand > 0 and getattr(batch, "hand_ref", None) is not None:
            tgt, mask_full = bc_target(
                self.codec, batch.port_tokens, batch.hand_ref,
                head_dim=int(self.vlm.action_head.action_dim),
                dtype=self.vlm.action_head.dtype, device=self.device,
            )
            hand_mask = torch.zeros_like(mask_full)
            hand_mask[..., 64:78] = 1.0
            loss_hand = fm_bc_loss(self.vlm.action_head, bb_out, act_in, tgt, hand_mask)

        loss = v3.lambda_r * loss_r
        if loss_hand is not None:
            loss = loss + v3.lambda_hand * loss_hand

        metrics: dict[str, float] = {
            "loss/reward_norm": float(loss_r.detach()),
            "loss/reward_raw": float(ret.mean().detach()),
            "loss/boot": float(boot.mean().detach()),
            "rollout/reward": float(rollout.rewards.mean().detach()),
            "rollout/survival": float(rollout.survival[-1].mean().detach()),
            "codec/out_of_range": float(out_of_range),
            "loss": float(loss.detach()),
        }
        if loss_hand is not None:
            metrics["loss/hand_fm"] = float(loss_hand.detach())

        if unc_mean:
            metrics["unc/mean"] = unc_mean
        bucket = getattr(batch, "bucket", None)
        if bucket is not None:
            with torch.no_grad():
                rew = rollout.rewards.mean(0)                            # (B,)
                for bi, name in enumerate(("nominal", "mid", "hard")):
                    m = bucket == bi
                    metrics[f"bucket/{name}_n"] = float(m.sum())
                    if bool(m.any()):
                        metrics[f"bucket/{name}_rew"] = float(rew[m].mean())

        # ---------- 分项梯度范数（诊断；只打印，不写 .grad）----------
        if v3.grad_log_every > 0 and (self.steps + 1) % v3.grad_log_every == 0:
            from .fm_bc import grad_norm_ratio  # noqa: PLC0415

            if loss_hand is not None:
                metrics["grad/r_over_hand"] = grad_norm_ratio(
                    v3.lambda_r * loss_r, v3.lambda_hand * loss_hand, self._ratio_params)

        # ---------- 反传 / 累积 / 裁剪 ----------
        accum = max(1, int(cfg.grad_accum))
        self._accum_count += 1
        if self._accum_count == 1:
            self.optimizer.zero_grad(set_to_none=True)
        (loss / accum).backward()
        if self._accum_count >= accum:
            total = torch.nn.utils.clip_grad_norm_(self.fm_parameters, v3.clip_head)
            metrics["grad/head_preclip"] = float(total)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_count = 0
        self.steps += 1

        # ---------- critic 更新（GAE over 想象转移）----------
        if self.critic is not None and self.critic_opt is not None:
            dones = (rollout.termination_probs[1:].detach() > 0.5).float()
            cm = critic_update(
                self.critic, self.critic_opt,
                rollout.states.detach(), ref_states.detach(), xi.detach(),
                rollout.rewards.detach(), dones,
                gamma=v3.gamma, lam=v3.lam,
                epochs=v3.critic_epochs, minibatches=v3.critic_minibatches,
                clip_norm=v3.critic_clip, clip_value=v3.critic_value_clip,
            )
            metrics["critic/v_loss"] = cm["v_loss"]
            metrics["critic/v_ev"] = cm["v_ev"]
        self.last_metrics = metrics
        return metrics

    # ------------------------------------------------------------------ WM 保鲜
    def refresh_wm(self, data: dict, *, steps: int = 200, batch: int = 2048,
                   lr: float = 1e-4, vel_weight: float = 5.0, kl_anchor: float = 1e-3,
                   log=None) -> dict:
        from .wm_refresh import refresh_wm  # noqa: PLC0415

        out = refresh_wm(self.bundle.wm, data, steps=steps, batch=batch, lr=lr,
                         vel_weight=vel_weight, kl_anchor=kl_anchor, device=self.device,
                         log=self.log if log is None else log)
        from .rollout import freeze_world_model  # noqa: PLC0415

        freeze_world_model(self.bundle.wm)
        return out


__all__ = ["JointTrainerV3", "V3Config"]
