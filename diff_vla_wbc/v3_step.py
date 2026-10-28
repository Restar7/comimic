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
    # P1-6（A1）：把 **burn-in 末状态**的本体感觉写回 VLA 的 state 输入。
    # 默认关；打开后 VLA 才能"看到"DR 扰动/实际偏离"，梯度才可能学成"状态相关的修正"
    # 而不是对扰动取平均的常数偏置（见 docs/S3_FIX_PLAN_AFTER_REVIEW_20261027.md §1.1）。
    vla_state_from_burnin: bool = False
    # P1-4：信任域（动作 hinge）的**中心**用谁
    #   "label" = 标签 token 的 rollout（旧行为，稳定）
    #   "base"  = 底模自己的输出（无标签，更正确，但当前冒烟会 NaN —— 待查）
    trust_center: str = "label"
    # P1-1 的梯度部分：V 对 s_H 保留梯度（SHAC 本意）。默认 **关**——冒烟显示它会引入 NaN，
    # 先把相对形式（V(s_H)−V(ref_H)）稳定下来，再单独开这个开关做 A/B。
    critic_grad_through_state: bool = False
    # P1-1 稳定性三件套（不动数值语义，只控梯度量级）
    critic_state_grad_scale: float = 0.1   # 只缩放"经 s_H 的梯度"，数值不变（straight-through）
    critic_rel_clip: float = 5.0           # V 相对值 clamp（超界梯度为 0）
    # P1-4 稳定性：hinge 上界（min(excess, cap)^2），防止两条 rollout 分叉后一次爆掉
    act_hinge_cap: float = 2.0
    act_eps: float = 0.0              # 0 = 关；容差 = (a − a_ref) 的逐关节 RMS（归一化动作空间）
    act_weight: float = 0.0           # hinge 权重（无量纲）
    # ---- 参考执行动作误差（2026-10-28 用户口径，**主目标**）------------------------
    # 目标：`a_exec(θ) − a_ref`，其中
    #   a_exec = **冻结执行器在想象里执行出来的 29 维动作**（rollout.actions，可微）；
    #   a_ref  = 录制时**真正下发的动作**（`decoder_target_action`，asset→canonical，
    #            见 scripts/build_ref_action_lib.py，覆盖 634016 帧 / 699 clip 100%）。
    # 与 BC 的区别：BC 监督 token 标签（模仿"参考动作的编码"），这里监督**执行结果**，
    # 允许 token 学出"预补偿执行器偏差"的解；路径 θ→token→执行器→动作，**不经过奖励项**。
    lambda_aref: float = 0.0          # 0 = 关；建议做主目标时 1.0 且 lambda_bc_token=0
    aref_hinge_eps: float = 0.0       # >0 = hinge(relu(RMS/eps−1)^2)，0 = 纯 MSE
    aref_rel: bool = True             # True: 按参考动作尺度归一化（逐关节 std），更稳
    #: 参考动作库路径；空 = ``runs/wm_real/ref_action_lib.pt``。
    #: 缺这个字段会让 ``train_joint_v2.py`` 里的 ``V3Config(ref_action_lib=...)`` 直接 TypeError。
    ref_action_lib: str = ""
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
        processor=None,
        embodiment_tag: str = "unitree_g1_sonic",
        log: Callable[[str], None] = print,
    ) -> None:
        super().__init__(bundle, vlm, codec, cfg=cfg, device=device, adapter=adapter, log=log)
        self.v3 = v3 or V3Config()
        # P1-6 / A1：覆写 VLA 的 state 输入要复用 ckpt 自己的归一化，所以需要 processor
        self.processor = processor
        self.embodiment_tag = str(embodiment_tag)
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

    # ------------------------------------------------------------------ P1-6 / A1
    #: 46 维状态里由 burn-in 末状态决定的分组（**手部保持录制值** ⇒ 不在此列）
    _BURNIN_STATE_GROUPS = ("left_leg", "right_leg", "waist", "left_arm",
                            "right_arm", "projected_gravity")
    #: canonical ``q``(29) 在各分组里的切分（与 ``g1_state_46`` 逐位对拍过：29/29 恒等）
    _JOINT_SEGMENTS = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15),
                       "left_arm": (15, 22), "right_arm": (22, 29)}

    def prepare_model_inputs(self, batch) -> dict:
        """``batch.model_inputs``（+ 可选的 A1 状态覆写）。训练与评测**共用这一条路**。"""
        model_inputs = batch.model_inputs
        if model_inputs is None:
            raise SystemExit("[v3] batch 没有 model_inputs")
        if bool(getattr(self.v3, "vla_state_from_burnin", False)):
            model_inputs = self._patch_state_from_burnin(model_inputs, batch)
        return model_inputs

    def _state_norm_params(self, key: str) -> dict:
        sap = self.processor.state_action_processor
        return sap.norm_params[self.embodiment_tag]["state"][key]

    def _patch_state_from_burnin(self, model_inputs: dict, batch) -> dict:
        """P1-6 / A1：用 burn-in 末状态覆写 VLA 的 ``state`` 输入（**图像仍是录制帧**）。

        为什么不能覆写 ``state.<group>`` 原始键：``model_inputs`` 是 processor 的输出，
        7 个 ``state.<group>`` 键在 ``process_observation`` 里已经被归一化并拼成单个
        ``state`` 张量（``(B,1,max_state_dim)``，本 ckpt 46 → 132 补零），原始键不再存在
        —— 对它们赋值是**空操作**，模型读到的仍是录制状态。

        正确做法：用**同一套** ``state_action_processor.apply_state`` 重新归一化，
        只替换「五组关节 + 重力投影」对应的切片；手部与补零区保留 processor 原值。
        归一化口径随 ckpt 走（本 ckpt：``use_percentiles`` + ``clip_outliers``，
        无 sin/cos 编码 ⇒ 宽度仍是 46）。

        数值前提（已对拍 ``data/npz/ha_tokens/*`` 的 ``g1_state_46``，2080 帧）：
          * 关节序：canonical ``q29`` 与 46 维前 29 维**逐位恒等**（29/29）；
          * 重力：``state[43:46] == -R[:,2]``，平均绝对误差 0.0。
        """
        import numpy as np  # noqa: PLC0415
        from .trackerlab_adapter import quat_to_mat  # noqa: PLC0415

        if self.processor is None:
            raise SystemExit(
                "[v3] --vla-state-from-burnin 需要 processor：构造 JointTrainerV3 时传 "
                "processor=processor（见 scripts/train_joint_v2.py）")
        sap = self.processor.state_action_processor
        tag = self.embodiment_tag
        keys = list(self.processor.modality_configs[tag]["state"].modality_keys)

        s = batch.start_state.detach().float().cpu()
        q = s[:, 7:36].numpy()                        # (B,29) canonical
        R = quat_to_mat(s[:, 3:7]).cpu().numpy()      # (B,3,3) body->world
        grav = -R[:, 2, :]                            # body 系重力投影 = -R^T e_z

        raw: dict[str, "np.ndarray"] = {}
        for key in keys:
            if key in self._JOINT_SEGMENTS:
                a, b = self._JOINT_SEGMENTS[key]
                raw[key] = np.ascontiguousarray(q[:, a:b], dtype=np.float32)
            elif key == "projected_gravity":
                raw[key] = np.ascontiguousarray(grav, dtype=np.float32)
            else:
                # 手部：只为凑齐 apply_state 的入参，切片不会被写回
                width = int(np.asarray(self._state_norm_params(key)["min"]).reshape(-1).shape[0])
                raw[key] = np.zeros((q.shape[0], width), dtype=np.float32)
        normed = sap.apply_state(state=raw, embodiment_tag=tag)

        out = dict(model_inputs)
        state = out["state"].clone()                  # (B,1,max_state_dim)
        flat = state.reshape(state.shape[0], -1)
        touched: list[tuple[int, int]] = []
        off = 0
        for key in keys:
            arr = np.asarray(normed[key], dtype=np.float32)
            width = int(arr.shape[-1])
            if width and key in self._BURNIN_STATE_GROUPS:
                flat[:, off:off + width] = torch.from_numpy(
                    np.ascontiguousarray(arr)).to(dtype=flat.dtype)
                touched.append((off, off + width))
            off += width
        out["state"] = state

        if not getattr(self, "_a1_logged", False):
            self._a1_logged = True
            delta = float((state - model_inputs["state"]).abs().max()) if touched else 0.0
            self.log(f"[A1] state ← burn-in 末状态：改了 {len(touched)}/{len(keys)} 组，"
                     f"切片 {touched}，|Δstate|max={delta:.4f}"
                     f"（归一化后；手部仍为录制值）。图像仍是录制帧。")
        return out

    # ------------------------------------------------------------------ 参考状态
    # ---- 参考执行动作（decoder_target_action）----------------------------------
    def _aref_lib(self):
        """懒加载参考动作库；返回 (actions(N,29), clip_end(N,), std(29,))。"""
        from pathlib import Path as _P  # noqa: PLC0415
        lib = getattr(self, "_aref_cache", None)
        if lib is not None:
            return lib
        path = _P(getattr(self.v3, "ref_action_lib", "") or
                  (_P(__file__).resolve().parents[1] / "runs/wm_real/ref_action_lib.pt"))
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        acts = d["actions"].float().to(self.device)
        starts, lengths = d["starts"].long(), d["lengths"].long()
        clip_id = torch.zeros(acts.shape[0], dtype=torch.long)
        for i, (st, ln) in enumerate(zip(starts.tolist(), lengths.tolist())):
            clip_id[st:st + ln] = i
        clip_end = (starts + lengths - 1)[clip_id].to(self.device)
        _v = d["valid"].bool() if "valid" in d else torch.ones(acts.shape[0], dtype=torch.bool)
        std = acts[_v.to(acts.device)].std(0).clamp_min(1e-3).to(self.device)
        self._aref_cache = (acts, clip_end, std)
        print(f"[aref] 参考执行动作库 {path.name}: rows={acts.shape[0]} "
              f"逐关节 std 中位={float(std.median()):.3f}", flush=True)
        return self._aref_cache

    def _ref_actions(self, frames, H: int) -> torch.Tensor:
        """``frames (B,)`` -> ``a_ref (H,B,29)``（canonical 序；越过 clip 尾就夹住）。"""
        acts, clip_end, _ = self._aref_lib()
        f = torch.as_tensor(frames, dtype=torch.long, device=self.device).reshape(-1)
        offs = torch.arange(H, device=self.device).unsqueeze(0)
        rows = torch.minimum(f.unsqueeze(1) + offs, clip_end[f].unsqueeze(1))
        return acts[rows].permute(1, 0, 2)

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

        # A1：观测侧覆写走 prepare_model_inputs（训练/评测同一条路，避免评测看不到改动）
        model_inputs = self.prepare_model_inputs(batch)
        act_in, bb_out = self._backbone_output(model_inputs)
        # 固定噪声：自适应头与"底模参照"用**同一个噪声**采样，两者只差 LoRA 增量
        head = self.vlm.action_head
        _noise = torch.randn(int(act_in["state"].shape[0]), int(head.action_horizon),
                             int(head.action_dim), device=self.device, dtype=head.dtype)
        x_norm, x_raw, tokens = self._sample_tokens(act_in, bb_out, noise=_noise)
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
            # P1-1（对齐 SuperMimic 的 SHAC 本意）：V 对 s_H **保留梯度**，只冻结 critic 的参数
            #   （参数在自己的监督损失里更新，避免在策略损失里双计）。
            # B2（消除符号隐患）：用 V(s_H) − V(ref_H) 的**相对形式**——ref_H 是同一帧的参考状态，
            #   于是 boot 的语义是"偏离参考带来的额外价值"，不会出现负值乘 survival 把策略往摔里推。
            _sg = float(getattr(v3, "critic_state_grad_scale", 0.1))
            _clip = float(getattr(v3, "critic_rel_clip", 5.0))
            if bool(getattr(v3, "critic_grad_through_state", False)):
                _g = [q.requires_grad for q in self.critic.parameters()]
                for q in self.critic.parameters():
                    q.requires_grad_(False)
                try:
                    # critic 这条前向强制 fp32：s_H 的梯度不再被 bf16 的动态范围搞爆
                    with torch.autocast("cuda", enabled=False):
                        vH = self.critic(rollout.states[-1].float(), ref_states[-1].float(), xi.float())
                        v_ref = self.critic(ref_states[-1].float(), ref_states[-1].float(), xi.float())
                finally:
                    for q, g in zip(self.critic.parameters(), _g):
                        q.requires_grad_(g)
            else:
                with torch.no_grad():
                    vH = self.critic(rollout.states[-1].detach(), ref_states[-1].detach(), xi.detach())
                    v_ref = self.critic(ref_states[-1].detach(), ref_states[-1].detach(), xi.detach())
            rel = (vH - v_ref.detach())
            if _sg != 1.0:
                # straight-through：loss 里仍是 rel 的数值，但反传时只有 _sg 倍
                rel = rel.detach() + _sg * (rel - rel.detach())
            if _clip > 0:
                rel = torch.clamp(rel, -_clip, _clip)
            boot = (v3.gamma ** H) * rollout.survival[-1] * rel
            metrics_v = float(rel.mean().detach())
        else:
            metrics_v = 0.0
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
            with self._amp_ctx():
                loss_hand = fm_bc_loss(self.vlm.action_head, bb_out, act_in, tgt, hand_mask)
            loss_hand = loss_hand.float()

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
            # P1-4：`trust_center == "base"` 时中心 = **底模自己的输出**（LoRA 增量关掉、同一噪声）；
            # 默认 "label" 保持旧行为（稳定）。base 模式当前冒烟会 NaN（见 FIX_PLAN P1-4 备注）。
            if str(getattr(v3, "trust_center", "label")) == "base":
                from .lora import lora_disabled  # noqa: PLC0415
                with lora_disabled(self.vlm.action_head):
                    _, _, _ref_tok = self._sample_tokens(
                        act_in, bb_out, noise=_noise, backend="truncated")
                _ref_tok = _ref_tok.detach()
            else:
                _ref_tok = batch.port_tokens.detach()
            with torch.no_grad():
                roll_ref = imagine_with_tokens(
                    self.bundle, tokens=_ref_tok.to(tokens.dtype),
                    state=batch.start_state, obs_hist=batch.obs_hist,
                    history=batch.history, xi=batch.xi, references=batch.references,
                    prev_action=batch.prev_action, horizon=cfg.horizon,
                )
            diff_a = rollout.actions.float() - roll_ref.actions.float()
            # ★ sqrt 在 0 处的导数是 inf：`diff_a == 0`（LoRA 的 B 零初始化时，
            #   `trust_center=base` 的两条 rollout 逐位相同）会让反向得到 0×inf = NaN，
            #   哪怕 loss_act 本身是 0（relu 当时为 0）。加 eps 让梯度有限。
            act_rms = (diff_a.pow(2).mean(-1) + 1e-12).sqrt()                    # (H,B)
            excess_a = torch.relu(act_rms / float(v3.act_eps) - 1.0)
            _cap = float(getattr(v3, "act_hinge_cap", 0.0))
            if _cap > 0:
                excess_a = torch.clamp(excess_a, max=_cap)      # 上界：力封顶，不会一次爆
            loss_act = float(v3.act_weight) * excess_a.pow(2).mean()
            if float(v3.act_weight) > 0:
                loss = loss + loss_act
        # ---- 参考执行动作误差（主目标，2026-10-28）：a_exec(θ) vs 真正下发的动作 ----
        loss_aref = None
        aref_rms = None
        if float(getattr(v3, "lambda_aref", 0.0)) > 0.0:
            _H = int(rollout.actions.shape[0])
            a_ref = self._ref_actions(batch.frames, _H).to(rollout.actions.dtype)
            diff = rollout.actions.float() - a_ref
            if bool(getattr(v3, "aref_rel", True)):
                diff = diff / self._aref_lib()[2].to(diff.device)
            aref_rms = diff.pow(2).mean(-1).sqrt()
            _eps = float(getattr(v3, "aref_hinge_eps", 0.0))
            if _eps > 0:
                loss_aref = float(v3.lambda_aref) * torch.relu(aref_rms / _eps - 1.0).pow(2).mean()
            else:
                loss_aref = float(v3.lambda_aref) * diff.pow(2).mean()
            loss = loss + loss_aref

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
        if loss_aref is not None:
            metrics["loss/aref"] = float(loss_aref.detach())
            metrics["aref/rmse"] = float(aref_rms.mean().detach())
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
        metrics["critic/v_adv_H"] = float(metrics_v)
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
