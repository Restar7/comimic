"""Joint VLA / WBC / world-model training loop.

The world model is frozen for policy improvement, but its forward pass is
differentiable w.r.t. the action (and hence w.r.t. the WBC, adapter and FM
head).  A separate optimizer updates the world model from real transition
windows, which are *not* differentiated into the policy.

For the real experiment use:

* ``fm_sampler=Gr00tTokenSource(gr00t)`` with real preprocessed batches;
* ``bundle = SuperMimicBundle.from_checkpoint("runs/.../ckpt_012000.pt", assets)``;
* a ``MotionBatchProvider`` whose token source returns the paired GR00T tokens
  for the same motion window.

For the smoke run in this repository the token source is random and the WBC/WM
are randomly initialized, which still exercises every differentiable edge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn

from .adapter import MotionGoalAdapter
from .rollout import (
    LossWeights,
    compute_joint_loss,
    freeze_world_model,
    imagine_with_goals,
    imagine_with_tokens,
    unfreeze_world_model,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only: the Isaac path never imports the mjlab stack
    from .supermimic_backend import SuperMimicBundle  # noqa: F401


@dataclass
class JointTrainerConfig:
    horizon: int = 8
    gamma: float = 0.99
    adapter_lr: float = 1e-4
    policy_lr: float = 1e-5
    fm_lr: float = 1e-5
    wm_lr: float = 2e-4
    # SuperMimic 的 WMTrainerCfg 用 weight_decay=0.0 / grad_clip=1.0（wm_trainer.py）
    wm_weight_decay: float | None = None
    wm_grad_clip: float = 1.0
    weight_decay: float = 1e-4
    adapter_clip: float = 1.0
    policy_clip: float = 1.0
    fm_clip: float = 0.5
    wm_clip: float = 1.0
    obs_noise: float = 0.0
    pretrain_wm_steps: int = 0
    pretrain_wm_batch: int = 8
    pretrain_wm_window: int = 8
    weights: LossWeights = field(default_factory=LossWeights)


class JointTrainer:
    def __init__(
        self,
        bundle: SuperMimicBundle,
        adapter: MotionGoalAdapter,
        *,
        fm_sampler: Callable | None = None,
        fm_parameters: Iterable[torch.nn.Parameter] | None = None,
        cfg: JointTrainerConfig | None = None,
        device: str | torch.device | None = None,
        hand_weight: float = 0.0,
        hand_grasp_boost: float = 5.0,
        # 只在“有手部信号”的帧上算 hand loss：HA 的 7 个任务里 4 个手全程为 0
        # （实测 pp_box/open_door/sit_sofa/vision_navi），全帧 MSE 会把 14 维输出拉向常零。
        hand_mask_threshold: float = 0.05,
        fm_full_sampler: Callable | None = None,
        sync_gradients: bool = False,
        # token 锚定：把 VLA 输出的 64 维 token 拉向"执行器能吃的"端口 token。
        # 不加这一项时，reward 一饱和就没有任何东西钉住 token，模型会在 reward 平坦区里
        # 随机漂移（实测 token_mse 2000 步 0.077 → 3000 步 0.093/0.108）。
        token_anchor: Callable | None = None,
        token_anchor_weight: float = 0.0,
        # 行为级监督：π(VLA_token) 对齐 π(录制端口 token)，WBC 冻结（默认关闭）
        action_distill_weight: float = 0.0,
        # 梯度累积（等效 batch = batch_size * grad_accum）
        grad_accum: int = 1,
    ):
        self.bundle = bundle
        self.adapter = adapter
        self.fm_sampler = fm_sampler
        # 手部辅助监督：不经过 SuperMimic，直接用录制 label 回归 VLA 的 14 维手部输出。
        self.hand_weight = float(hand_weight)
        self.hand_grasp_boost = float(hand_grasp_boost)
        self.hand_mask_threshold = float(hand_mask_threshold)
        self.fm_full_sampler = fm_full_sampler
        # 2 卡数据并行：梯度在 optimizer.step() 之前手动 all-reduce（仓库既有做法，
        # 不用 DDP 包 module，避免和 torch.compile 的自定义前向打架）。
        self.sync_gradients = bool(sync_gradients)
        self.token_anchor = token_anchor
        self.token_anchor_weight = float(token_anchor_weight)
        self.action_distill_weight = float(action_distill_weight)
        self.grad_accum = max(1, int(grad_accum or 1))
        self._accum_count = 0
        self.cfg = cfg or JointTrainerConfig()
        self.device = torch.device(device or bundle.device)
        self.bundle.to(self.device)
        self.adapter.to(self.device)
        # Goal-width compatibility is only meaningful on the **paper route** (680-d motion goal
        # -> adapter -> policy encoder).  On the **token route** the conditioning *is* the
        # VLA's 64-d SONIC token and the adapter is never called (``imagine_with_tokens``), so a
        # token-conditioned checkpoint legitimately reports ``goal_dim = token_dim``
        # (trackerlab/Isaac HA ckpt: 64).  Raising here would block every token-route run.
        token_route = int(getattr(bundle.policy, "token_dim", 0) or 0) > 0
        if not token_route and self.bundle.goal_dim != self.adapter.goal_dim:
            raise ValueError(
                f"adapter goal_dim {self.adapter.goal_dim} != policy goal_dim {self.bundle.goal_dim}"
            )

        params = list(self.adapter.parameters()) + list(self.bundle.policy.parameters())
        if fm_parameters is None and isinstance(fm_sampler, nn.Module):
            fm_parameters = fm_sampler.parameters()
        self.fm_parameters = [p for p in (fm_parameters or []) if p.requires_grad]
        if self.fm_parameters:
            params += self.fm_parameters
        self.optimizer = torch.optim.AdamW(
            [
                {"params": list(self.adapter.parameters()), "lr": self.cfg.adapter_lr},
                {"params": list(self.bundle.policy.parameters()), "lr": self.cfg.policy_lr},
                *(
                    [{"params": self.fm_parameters, "lr": self.cfg.fm_lr}]
                    if self.fm_parameters
                    else []
                ),
            ],
            weight_decay=self.cfg.weight_decay,
        )
        wd = self.cfg.weight_decay if self.cfg.wm_weight_decay is None else self.cfg.wm_weight_decay
        self.wm_optimizer = torch.optim.AdamW(
            self.bundle.wm.parameters(), lr=self.cfg.wm_lr, weight_decay=wd
        )
        # Freeze by default: policy improvement must not update the dynamics.
        freeze_world_model(self.bundle.wm)
        self.steps = 0

    # ------------------------------------------------------------------ distributed

    @staticmethod
    def _sync_param_grads(group, max_chunk_bytes: int = 256 << 20) -> None:
        """Average the gradients of one parameter group across ranks (no-op if single).

        The action head alone is 1.62 B fp32 parameters (6.5 GB of gradients), so concatenating
        the whole group into one flat buffer costs another 6.5 GB of *transient* device memory —
        on a 48 GB card that is the difference between fitting and OOM.  Gradients are therefore
        averaged in bounded chunks; the arithmetic is unchanged (plain average, same order of
        parameters), only the peak allocation is.
        """
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        grads = [p.grad for p in group if p.grad is not None]
        if not grads:
            return
        dtype = grads[0].dtype
        chunk, used = [], 0

        def flush(buf: list[torch.Tensor]) -> None:
            flat = torch.cat([g.reshape(-1) for g in buf])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            off = 0
            for g in buf:
                n = g.numel()
                g.copy_(flat[off : off + n].view_as(g))
                off += n
            del flat

        for g in grads:
            if g.dtype != dtype:      # mixed dtypes cannot share one communication buffer
                continue
            nbytes = g.numel() * g.element_size()
            if chunk and used + nbytes > max_chunk_bytes:
                flush(chunk)
                chunk, used = [], 0
            chunk.append(g)
            used += nbytes
        if chunk:
            flush(chunk)

    def sync_grads(self) -> None:
        """把 adapter / policy / VLA(head) 的梯度在 rank 间取平均。"""
        for group in (
            list(self.adapter.parameters()),
            list(self.bundle.policy.parameters()),
            list(self.fm_parameters),
        ):
            self._sync_param_grads(group)

    # ------------------------------------------------------------------ policy / FM

    def _tokens(self, batch) -> torch.Tensor:
        if self.fm_sampler is None:
            return batch.fm_tokens
        if getattr(batch, "model_inputs", None) is None:
            # The sampler may only need the metadata dict; let it decide.
            return self.fm_sampler(batch.as_dict())
        return self.fm_sampler(batch.model_inputs)

    def train_step(self, batch) -> dict[str, float]:
        self.bundle.policy.train()
        self.adapter.train()
        self.bundle.wm.eval()
        freeze_world_model(self.bundle.wm)

        tokens = self._tokens(batch)
        # Token 路线：SuperMimic ckpt 是 token 条件化（PolicyCfg.token_dim > 0），conditioning 就是
        # VLA 自己输出的 64 维 motion token，不再经 adapter 生成 680 维 goal（FINAL_HANDOVER §0）。
        token_route = int(getattr(self.bundle.policy, "token_dim", 0)) > 0
        if token_route:
            rollout = imagine_with_tokens(
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
            # 没有 680 维 goal → goal 对齐项由 compute_joint_loss 自动跳过
            loss, terms = compute_joint_loss(
                rollout,
                teacher_states=batch.teacher_states,
                gamma=self.cfg.gamma,
                weights=self.cfg.weights,
            )
        else:
            rollout = imagine_with_goals(
                self.bundle,
                fm_tokens=tokens,
                adapter=self.adapter,
                state=batch.start_state,
                obs_hist=batch.obs_hist,
                history=batch.history,
                xi=batch.xi,
                references=batch.references,
                prev_action=batch.prev_action,
                horizon=self.cfg.horizon,
                obs_noise=self.cfg.obs_noise,
            )
            loss, terms = compute_joint_loss(
                rollout,
                teacher_goals=batch.teacher_goals,
                teacher_goal_sequence=batch.teacher_goal_sequence,
                teacher_states=batch.teacher_states,
                gamma=self.cfg.gamma,
                weights=self.cfg.weights,
            )
        # ---- 手部辅助损失（支路，绝不经过 SuperMimic / 世界模型 / 奖励）----
        if (
            self.hand_weight > 0
            and self.fm_full_sampler is not None
            and getattr(batch, "hand_ref", None) is not None
        ):
            import torch.nn.functional as _F

            with torch.enable_grad():
                full = self.fm_full_sampler(batch.as_dict())   # (B, T, >=78)
            if full.shape[-1] < 78:
                raise RuntimeError(f"VLA chunk has {full.shape[-1]} dims; need >=78 (token64+hands14)")
            hands = full[..., 64:78]
            ref = batch.hand_ref.to(hands.dtype)
            n = min(hands.shape[1], ref.shape[1])
            grasp = (ref[:, :n].abs().mean(-1) > 0.10).to(hands.dtype)      # 抓握帧
            # 无手样本（参考手全 0）不给梯度，否则 60% 的样本在教“手别动”
            active = (ref[:, :n].abs().amax(-1) > self.hand_mask_threshold).to(hands.dtype)
            w = (1.0 + self.hand_grasp_boost * grasp) * active
            per = ((hands[:, :n] - ref[:, :n]) ** 2).mean(-1)               # (B, n)
            hand_loss = (per * w).sum() / w.sum().clamp(min=1.0)
            loss = loss + self.hand_weight * hand_loss
            terms = dict(terms)
            terms["hand_mse"] = hand_loss.detach()

        # token 锚定（可选）：直接监督 action head 输出到端口 token，
        # 防止 reward 饱和后 token 在平坦区里漂移。
        if self.token_anchor is not None and self.token_anchor_weight > 0:
            target = self.token_anchor(batch)
            if target is not None:
                n = min(tokens.shape[1], target.shape[1])
                anchor_loss = torch.nn.functional.mse_loss(
                    tokens[:, :n].float(), target[:, :n].to(tokens.device, tokens.dtype).float()
                )
                loss = loss + self.token_anchor_weight * anchor_loss
                terms = dict(terms)
                terms["token_anchor"] = anchor_loss.detach()

        # 动作级蒸馏（可选）：让"用 VLA token 的动作"贴近"用录制端口 token 的动作"。
        # 完全在可微链路里（只多一次 WBC 前向），不经过仿真器，也不替代世界模型。
        if (
            self.action_distill_weight > 0
            and self.token_anchor is not None
            and hasattr(self.bundle, "act")
        ):
            ref_tokens = self.token_anchor(batch)
            if ref_tokens is not None:
                oh = batch.obs_hist
                if oh.ndim == 3:
                    oh = oh.reshape(oh.shape[0], -1)
                a_vla = self.bundle.act(oh, tokens[:, 0])
                with torch.no_grad():
                    a_ref = self.bundle.act(oh, ref_tokens[:, 0].to(tokens.device, tokens.dtype))
                distill = torch.nn.functional.mse_loss(a_vla.float(), a_ref.float())
                loss = loss + self.action_distill_weight * distill
                terms = dict(terms)
                terms["action_distill"] = distill.detach()

        # 梯度累积：accum=1 时行为与旧代码完全一致
        _accum = self.grad_accum
        self._accum_count += 1
        if self._accum_count == 1:
            self.optimizer.zero_grad(set_to_none=True)
        (loss / _accum).backward()
        if self.sync_gradients:
            self.sync_grads()
        if self.fm_parameters:
            torch.nn.utils.clip_grad_norm_(self.fm_parameters, self.cfg.fm_clip)
        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.cfg.adapter_clip)
        torch.nn.utils.clip_grad_norm_(self.bundle.policy.parameters(), self.cfg.policy_clip)
        if self._accum_count >= _accum:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_count = 0
        self.steps += 1

        metrics = {"loss": float(loss.detach())}
        metrics.update({f"loss/{k}": float(v.detach()) for k, v in terms.items()})
        metrics["rollout/reward"] = float(rollout.rewards.mean().detach())
        metrics["rollout/survival"] = float(rollout.survival[-1].mean().detach())
        metrics["rollout/action_abs"] = float(rollout.actions.abs().mean().detach())
        return metrics

    # ------------------------------------------------------------------ world model

    def train_world_model(self, windows: dict, *, sync: bool = False) -> dict[str, float]:
        """Fit ``F_phi`` on real transition windows (no policy gradient).

        ``sync=True`` averages the world-model gradients across ranks before the
        optimiser step; needed when this is called inside the distributed training loop
        (the ``--wm-update-every`` ablation), otherwise the two ranks diverge.
        """
        self.bundle.wm.train()
        unfreeze_world_model(self.bundle.wm)
        loss, metrics = self.bundle.wm.window_loss(
            windows["states"],
            windows["actions"],
            windows["xi"],
            windows["done"],
            windows["refs"],
        )
        self.wm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if sync:
            self._sync_param_grads(list(self.bundle.wm.parameters()))
        torch.nn.utils.clip_grad_norm_(self.bundle.wm.parameters(), self.cfg.wm_grad_clip)
        self.wm_optimizer.step()
        freeze_world_model(self.bundle.wm)
        out = {f"wm/{k}": float(v.detach()) for k, v in metrics.items()}
        out["wm/loss"] = float(loss.detach())
        return out

    # ------------------------------------------------------------------ checkpoints

    @torch.no_grad()
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.bundle, "save_checkpoint"):   # Isaac/trackerlab bridge: keep its layout
            self.bundle.save_checkpoint(path, adapter=self.adapter.state_dict(), steps=self.steps)
            return
        torch.save(
            {
                "adapter": self.adapter.state_dict(),
                "adapter_config": asdict(self.adapter.cfg),
                "policy": self.bundle.policy.state_dict(),
                "wm": self.bundle.wm.state_dict(),
                "policy_config": asdict(self.bundle.policy_cfg),
                "world_model_config": asdict(self.bundle.world_model_cfg),
                "config": asdict(self.cfg),
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str | Path) -> dict:
        """Resume: restore adapter / policy / WM (+ step counter). Returns the raw ckpt.

        Two key layouts exist: mjlab-style ``{policy, wm, adapter}`` and the Isaac bridge's
        ``{pi, wm, adapter, wm_integration}`` (``IsaacBundle.save_checkpoint``). Both work;
        the integration flag is honoured so a checkpoint trained with explicit integration
        does not silently load into the residual structure.
        """
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if "adapter" in ck:
            self.adapter.load_state_dict(ck["adapter"])
        policy = ck.get("policy", ck.get("pi"))
        if policy is not None:
            self.bundle.policy.load_state_dict(policy)
        if "wm" in ck:
            self.bundle.wm.load_state_dict(ck["wm"])
        integ = str(ck.get("wm_integration") or "").strip().lower()
        if integ.startswith("explicit") and hasattr(self.bundle.wm, "explicit_integration"):
            self.bundle.wm.explicit_integration = True
            if "dt" in integ:
                try:
                    self.bundle.wm.dt = float(integ.split("dt", 1)[1])
                except ValueError:
                    pass
        self.steps = int(ck.get("steps", ck.get("iteration", 0)) or 0)
        return ck

    @staticmethod
    def load_bundle_from_checkpoint(path: str | Path, assets):
        """Rebuild a bundle with the exact architecture saved by ``JointTrainer.save``."""
        from .supermimic_backend import SuperMimicBundle, _coerce_cfg
        from supermimic.policy.nets import PolicyCfg
        from supermimic.world_model.world_model import WorldModelCfg

        ck = torch.load(path, map_location=assets.device, weights_only=False)
        if "policy_config" not in ck or "world_model_config" not in ck:
            raise ValueError("checkpoint lacks policy/world-model config; pass --supermimic-ckpt instead")
        policy_cfg = _coerce_cfg(PolicyCfg, ck["policy_config"])
        wm_cfg = _coerce_cfg(WorldModelCfg, ck["world_model_config"])
        bundle = SuperMimicBundle.from_configs(assets, policy_cfg, wm_cfg)
        bundle.policy.load_state_dict(ck["policy"])
        bundle.wm.load_state_dict(ck["wm"])
        return bundle

    @staticmethod
    def load_adapter_from_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
        from .adapter import AdapterConfig, MotionGoalAdapter

        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = AdapterConfig(**ck["adapter_config"])
        adapter = MotionGoalAdapter(cfg).to(device)
        adapter.load_state_dict(ck["adapter"])
        adapter.eval()
        return adapter


def pretrain_world_model(
    trainer: JointTrainer,
    provider,
    *,
    steps: int,
    batch_size: int = 8,
    window: int = 8,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Warm-start the world model from reference windows (no simulator)."""
    from .data import sample_reference_windows

    logs = []
    for i in range(steps):
        windows = sample_reference_windows(
            trainer.bundle,
            provider.library,
            batch_size=batch_size,
            window=window,
            seed=seed + i,
        )
        logs.append(trainer.train_world_model(windows))
    return logs


__all__ = ["JointTrainer", "JointTrainerConfig", "pretrain_world_model"]
