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
    # ---- 先验锚定（A 方案）：λ_a·‖token − 底模先验‖² ----
    # 说明：底模（checkpoint-8000）的输出与录制标签数值上几乎一致（token MSE 0.004），
    # 所以这里用 batch 里已有的标签 chunk 当**先验代理**，零额外成本；
    # 语义上是"别离开底模自己会说的地方"，不是"学标签"（λ_a=0 时完全不参与）。
    anchor_weight: float = 0.0
    # ---- token trust region（尺度无关的硬约束，B 方案）----
    # 为什么需要：anchor 是 MSE·λ_a，数值上比 λ_r·|loss_r| 小 2–3 个数量级，
    # 挡不住奖励梯度（A1 在 step100 把 tok_mse 顶到 ×3.75 就是这么来的）。
    # 这里改成无量纲 hinge：只在 RMSE 超出 eps 时才给力，且力度与奖励项同量级。
    trust_eps: float = 0.0            # 0 = 关；容差 = token 相对先验的 RMSE（归一化 token 空间）
    trust_weight: float = 0.0         # hinge 权重（无量纲；1.0 ≈ 越界即与 λ_r·|J| 同量级）
    # ---- 动作空间 trust region（比 token 空间更本质：动作才是 WBC/WM 的输入）----
    act_eps: float = 0.0              # 0 = 关；容差 = (a − a_ref) 的逐关节 RMS（归一化动作空间）
    act_weight: float = 0.0           # hinge 权重（无量纲）
    # ---- critic 的监督来源 ----
    # "imagined"（旧）= 想象转移上的 GAE；"real"（新）= 真录制轨迹的 return-to-go
    critic_source: str = "imagined"
    critic_batch: int = 256
    # 按 clip 去时间趋势：return-to-go 天然随"剩余步数"下降，直接当 bootstrap 会诱导策略
    # "让状态看起来像更早的帧"（滞后）。减去同 clip 均值后，V 变成"相对质量"，自举才有意义。
    critic_center: bool = True
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

    # ------------------------------------------------------------------ 真数据 critic
    def attach_real_returns(self, rets: torch.Tensor, states: torch.Tensor,
                            ref_rows: torch.Tensor) -> None:
        """挂上 grounded 的价值目标：``rets[f]`` = 真录制轨迹从帧 f 起的折扣回报。

        ``ref_rows`` = motion library 的 canonical 参考行 ``(N,71)``（与 ``_ref_states`` 同口径）。
        """
        from .isaac_joint import canonical_to_flat, default_xi  # noqa: PLC0415

        n = int(rets.shape[0])
        if int(states.shape[0]) != n or int(ref_rows.shape[0]) != n:
            raise SystemExit(
                f"[v3] real returns 长度不一致：ret={n} states={int(states.shape[0])} ref={int(ref_rows.shape[0])}")
        self._real_ret = rets.to(self.device).float()
        if bool(getattr(self.v3, "critic_center", True)):
            motion = self.bundle.motion
            starts = motion.clip_start.detach().cpu().long()
            lens = motion.clip_len.detach().cpu().long()
            cen = self._real_ret.clone()
            for ci in range(int(lens.numel())):
                s, L = int(starts[ci]), int(lens[ci])
                if L > 0:
                    seg = self._real_ret[s:s + L]
                    cen[s:s + L] = seg - seg.mean()
            self._real_ret_raw = self._real_ret
            self._real_ret = cen
        self._real_states = states.to(self.device).float()
        self._real_ref_states = canonical_to_flat(ref_rows.float()).to(self.device)
        self._real_xi = default_xi(n, self.device)
        self.log(f"[AUDIT][v3][critic] source={self.v3.critic_source} n={n} "
                 f"ret_mean={float(self._real_ret.mean()):+.4f} std={float(self._real_ret.std()):.4f} "
                 f"ref_shape={tuple(self._real_ref_states.shape)}")

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
        # ---- 先验锚定（A 方案）：把 token 按在一根弹簧上，堵掉"跑出流形骗 WM"的路 ----
        loss_anchor = None
        if v3.anchor_weight > 0:
            with torch.no_grad():
                prior = batch.port_tokens.to(tokens.dtype)            # 底模先验代理（≈标签）
            n_anchor = min(int(tokens.shape[1]), int(prior.shape[1]))
            loss_anchor = torch.nn.functional.mse_loss(
                tokens[:, :n_anchor].float(), prior[:, :n_anchor].float())
            loss = loss + v3.anchor_weight * loss_anchor
        # ---- token trust region：relu(RMSE/eps − 1)^2 ----
        loss_trust = None
        rmse_tok = None
        if v3.trust_eps > 0 and v3.trust_weight > 0:
            with torch.no_grad():
                prior_t = batch.port_tokens.to(tokens.dtype)          # 底模先验代理（≈标签）
            n_t = min(int(tokens.shape[1]), int(prior_t.shape[1]))
            diff = tokens[:, :n_t].float() - prior_t[:, :n_t].float()  # (B,T,64) 带梯度
            rmse_tok = diff.pow(2).mean(dim=tuple(range(1, diff.dim()))).sqrt()   # (B,)
            excess = torch.relu(rmse_tok / float(v3.trust_eps) - 1.0)
            loss_trust = float(v3.trust_weight) * excess.pow(2).mean()
            loss = loss + loss_trust
        # ---- 动作空间 trust region：relu(RMS(a − a_ref)/eps − 1)^2 ----
        # 参照 = **用标签 token 跑同一段 rollout** 得到的动作序列（无梯度）。
        # 为什么不用库里的命令 q_cmd：实测 WBC 的动作与 (q_cmd−dq)/scale 有 ~1.1 RMS 的
        # 系统性偏差（反馈项/口径差），当中心没有意义。用"标签 token 的动作"作中心，
        # 度量的是**token 变化导致的行为变化**——这正是 WBC/WM 训练域的定义方式。
        loss_act = None
        act_rms = None
        if v3.act_eps > 0 and len(getattr(batch, "references", []) or []) > 0:
            with torch.no_grad():
                roll_ref = imagine_with_tokens(
                    self.bundle, tokens=batch.port_tokens.detach().to(tokens.dtype),
                    state=batch.start_state, obs_hist=batch.obs_hist,
                    history=batch.history, xi=batch.xi, references=batch.references,
                    prev_action=batch.prev_action, horizon=cfg.horizon,
                )
            diff_a = rollout.actions.float() - roll_ref.actions.float()
            act_rms = diff_a.pow(2).mean(-1).sqrt()                              # (H,B)
            excess_a = torch.relu(act_rms / float(v3.act_eps) - 1.0)
            loss_act = float(v3.act_weight) * excess_a.pow(2).mean()
            if float(v3.act_weight) > 0:
                loss = loss + loss_act
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

        if loss_anchor is not None:
            metrics["loss/anchor"] = float(loss_anchor.detach())
        if loss_act is not None:
            metrics["loss/act"] = float(loss_act.detach())
            metrics["act/rmse"] = float(act_rms.mean().detach())
            metrics["act/frac_out"] = float((act_rms > float(v3.act_eps)).float().mean().detach())
        if loss_trust is not None:
            metrics["loss/trust"] = float(loss_trust.detach())
            metrics["trust/rmse"] = float(rmse_tok.mean().detach())
            metrics["trust/frac_out"] = float(
                (rmse_tok > float(v3.trust_eps)).float().mean().detach())
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

        # ---------- critic 更新 ----------
        if (self.critic is not None and self.critic_opt is not None
                and getattr(v3, "critic_source", "imagined") == "real"
                and getattr(self, "_real_ret", None) is not None):
            from .token_critic import critic_fit_supervised  # noqa: PLC0415
            from .isaac_joint import FrameRef, canonical_to_flat, default_xi  # noqa: PLC0415

            cm = critic_fit_supervised(
                self.critic, self.critic_opt,
                self._real_states, self._real_ref_states, self._real_xi, self._real_ret,
                epochs=v3.critic_epochs, minibatches=v3.critic_minibatches,
                batch=v3.critic_batch, clip_norm=v3.critic_clip, clip_value=v3.critic_value_clip,
            )
            metrics["critic/v_loss"] = cm["v_loss"]
            metrics["critic/v_ev"] = cm["v_ev"]
        elif self.critic is not None and self.critic_opt is not None:
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
