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

import contextlib
import copy
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
    critic_use_diff: bool = False     # critic 显式吃 s−ref（价值主要由偏差决定）
    critic_lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    critic_clip: float = 1.0          # SuperMimic: clip_grad_norm_(critic, 1.0)
    critic_value_clip: float = 0.1    # SuperMimic: clip_grad_value_(critic, 0.1)
    # 2026-10-29：bootstrap 只许**惩罚**偏离参考，不许**奖励**偏离。
    # 依据：`rel = V(s_H) − V(ref_H)`，而 `V(ref_H) = critic(ref_H, ref_H, xi)` 是"状态差为 0"
    #   那个点，跟踪奖励在它最大 ⇒ 正确的 critic 应给 rel ≤ 0（R1 实测一路走到 −0.45）。
    #   R2 实测 rel 恒正（80% 步为正、max +0.76，第一桶就 +0.15），语义上说不通，而且
    #   `v_ev=0.995` 说明 critic 拟合很好 ⇒ 问题在**目标**（想象 rollout 的 GAE）而不在 critic。
    #   分布外 qd_ratio 1.33–1.42（样本内只有 1.12–1.26）正是可乘之机。
    # 打开后 rel := min(rel, 0)，只保留"别飘"的护栏，不再为漂移付钱。原始值仍记到
    #   `critic/v_adv_H_raw`，诊断用（否则等于把自己刚发现的信号蒙上）。
    boot_one_sided: bool = False   # 2026-10-29 回滚为 False：R1 的原始配方没有它。
                                  # 与 R1 保持单变量对比（本轮只测奖励尾部铰链）。
    push_std: float = 0.0             # rollout 内过程扰动：root **线速度**（m/s）
    push_ang: float = 0.0             # 同上，root 角速度（rad/s）；太大会直接触发软终止
    # ---- 先验锚定（A 方案）：λ_a·‖token − 底模先验‖² ----
    # 说明：底模（checkpoint-8000）的输出与录制标签数值上几乎一致（token MSE 0.004），
    # 所以这里用 batch 里已有的标签 chunk 当**先验代理**，零额外成本；
    # 语义上是"别离开底模自己会说的地方"，不是"学标签"（λ_a=0 时完全不参与）。
    anchor_weight: float = 0.0
    # token BC（2026-10-28）：用 B dump 的命令流 token 当标签做 FM 回归。
    # v3 原来**完全没有** token 级监督（`token_bc=OFF`），所以奖励+hinge 是仅有的两个信号，
    # 而它们都不含"正确 token 是什么"。>0 时每步额外算一次 GR00T 原生 FM loss（前 64 维）。
    token_bc_weight: float = 0.0
    # 2026-10-29：token BC 的**目标从哪来**
    #   "ref"  = batch.port_tokens（参考命令流 token）—— 实测会把补偿抹掉，闭环变差
    #   "self" = batch.self_tokens（采集时策略**自己下发**的 token，配合结果筛选）
    #            这才是"用仿真真值做监督"，见 scripts/build_b_self_targets.py
    token_bc_source: str = "ref"
    # 2026-10-29：dump 里的 `tokens` 是**单步** token（每 20 步重规划一次），
    # 所以 self 目标只有前 `token_bc_span` 步（=HA_VLA_EXEC_STEPS=20）是"那次观测真发生的"，
    # 再往后属于下一次观测、不能当标签。>0 时把 mask 的后半段置 0。0=不限制（ref 目标用）。
    token_bc_span: int = 0
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
    # hybrid: 低漂移用 label、高漂移用 self；阈值=realized_err（米）
    hybrid_err: float = 0.5
    # P1-1 的梯度部分：V 对 s_H 保留梯度（SHAC 本意）。默认 **关**——冒烟显示它会引入 NaN，
    # 先把相对形式（V(s_H)−V(ref_H)）稳定下来，再单独开这个开关做 A/B。
    critic_grad_through_state: bool = False
    # P1-1 稳定性三件套（不动数值语义，只控梯度量级）
    critic_state_grad_scale: float = 0.1   # 只缩放"经 s_H 的梯度"，数值不变（straight-through）
    critic_rel_clip: float = 5.0           # V 相对值 clamp（超界梯度为 0）
    # P1-4 稳定性：hinge 上界（min(excess, cap)^2），防止两条 rollout 分叉后一次爆掉
    act_hinge_cap: float = 2.0
    # ★ 2026-11-06：**反常量项**。直接最大化"跨窗口残差/均值"（= 状态相关度）。
    #   为什么需要：`action_target` 关掉后就没有任何东西惩罚"固定方向"了（v10 就是这样
    #   学到 0.398 的常向量）。常量解会让 ratio→0、log→-∞ ⇒ 无法被占。
    var_weight: float = 0.0
    # ★ 2026-11-06（与上面三条**并行**，默认关）：**命令层残差** token' = token + g(token, 偏差)。
    #   为什么另开一条：LoRA/state_extra/反常量项 都是"给信息指望它自己学"，四轮实测
    #   残差/均值只从 0.098 → 0.164（阈值 0.5），而残差/噪声 3.79→4.00 没变（学的是噪声）。
    #   这里把状态相关写进**结构**：g 的输入就是偏差特征，末层零初始化 ⇒ 第 0 步与底模逐位相同。
    #   部署口径零客户端改动（server 用 state64[:6] 与 state64[35:64]）。
    delta_token: bool = False
    delta_token_hidden: int = 1024     # 99→1024→1024→64 = 1,217,600 参数（LoRA 的 3.6%）
    delta_token_depth: int = 2
    #: 末层初始化 std。**不是严格 0**：`dL/dW_前层 ∝ W_out`，严格 0 会让前两层第 0 步没梯度。
    #: 4e-4 ⇒ 初始 ‖Δtoken‖ ≈ 0.009（底模 ‖token‖ ≈ 7.9 的 0.12%）⇒ 等价于从底模出发。
    #: 设 0 可退回严格零初始化（A/B 用）。
    delta_token_init_std: float = 4e-4
    #: delta-token 的 lr。**不能跟 fm_lr(1e-5) 走** —— 那是给 2B 主干上的 LoRA 用的；
    #: 独立的 1.2M 小 MLP 用 1e-5 等于几乎不学（实测 250 步 win 只 0.342→0.382，
    #: 按此速率追平参考要 ~740 步）。小网络常规区间 3e-4 ~ 1e-3。
    delta_token_lr: float = 5e-4
    #: >0 时对 ‖Δtoken‖ 做**软**上限（tanh 收缩，保留梯度）。防止把 token 推出 codec 流形。
    delta_token_cap: float = 0.0
    # ★ 2026-11-06（第 2 刀）：特征从 35 维(rot6d+dof_vel) 扩成**部署 state64 全 64 维**
    #   （多了 dof_pos 29 维 —— 之前根本没用上）。依据：cos 0.24 → 0.39~0.47。
    delta_token_fullstate: bool = False
    # ★ 2026-11-06（第 1+3 刀）：**稠密 ĝ 监督** —— 直接把"理想修正方向"当标签回归，
    #   不再只靠 20 步报酬反传（那条路跨窗口求和只剩均值 ⇒ 常向量）。
    #   目标 = eta·unit(ĝ_token)，ĝ_token = ∂J/∂token（一次 backward）。
    #   eta=0.08 = 0.15/1.8（真机实测 WM 高估幅度 1.8×：预测 +0.128 vs 实测 +0.059）。
    grad_target_weight: float = 0.0
    grad_target_n: int = 32        # 子批大小（每步只在这 32 个窗口上求 ĝ，控成本）
    grad_target_eta: float = 0.08  # 目标幅度（token RMS）；★ 已按 1.8× 高估修正
    grad_target_every: int = 1     # 每几步算一次（>1 时复用上一次的 ĝ）
    # ★ 2026-11-06（第 5 刀，用户指出的那一刀）：**短视界标签**。
    #   20 步的 ĝ 是 20 次 Jacobian 连乘，可学性 cos 0.22~0.35；单步 ĝ₁ 无连乘，cos 0.47~0.50。
    #   部署时每 chunk（16 步）重算一次 ⇒ 短视界标签 = 滚动重规划，与用法一致。
    #   0 = 沿用 cfg.horizon（老行为）；建议 1~5（真机 H=2 已验证 96/96）。
    grad_target_horizon: int = 0
    act_eps: float = 0.0              # 0 = 关；容差 = (a − a_ref) 的逐关节 RMS（归一化动作空间）
    act_weight: float = 0.0           # hinge 权重（无量纲）
    # ★ 2026-11-05：惩罚形状。`hinge`（历史默认）= relu(r/eps-1)²，**eps 以内梯度恰为 0**；
    #   `l2` = (r/eps)²，**从 0 起就有梯度**。
    #   为什么需要它：round-5 实测动作漂移一路滑到 0.80（≈20°/关节）才被 hinge 接住 ——
    #   前 ~90 次更新在动作方向上**零阻力**（奖励 λ_r=5 独自推）。收紧刹车不只是调小 eps，
    #   还要把死区去掉，否则 eps 以内仍然没人管。
    act_penalty: str = "hinge"        # "hinge" | "l2"
    # ★ 2026-11-05：奖励只用想象轨迹的**前 k 步**（0=旧行为，用满 H）。
    #   理由：WM 单步准、多步靠积分；把奖励压在它最准的时间尺度上。
    reward_horizon: int = 0
    # ★ 2026-11-05（P1）：**把更新放到动作空间**。
    #   为什么：实测微调"实际走的方向"在真机上只有 35% 同号（比抛硬币还差），而
    #   WM 梯度方向是 69% 同号、漂移 −1.23 cm ⇒ 方向在 token→4 步采样→动作 这条链上被压歪了。
    #   做法：把 rollout 的动作当自由变量，只滚 WM+奖励拿到 ∂J/∂A，
    #   目标 A* = A + η·ĝ，损失改成 **动作空间回归** `w·‖actions − A*‖²`。
    #   它仍然通过 token→decoder 可微，但损失是良态的回归，不再是奖励穿采样。
    action_target_weight: float = 0.0   # 0 = 关（旧行为）
    action_target_eta: float = 0.15     # 动作空间的步长（SONIC 单位，与真机实验的 δ 同口径）
    # 回归多少步：目标序列是"base 动作 + η·梯度"这条**完整计划**，
    # 前几步 base 与当前策略状态重合、越往后分叉越大。默认 10（用户口径 8–12 区间）。
    action_target_steps: int = 20   # 默认 = 全视界（用户口径：8/20，不是 5）
    # ---- Path B：联合训练的双 KL 锚（2026-11-01）----
    # WBC action KL：起点 WBC 在同一状态/token 下的动作 a0，λ*MSE(a, a0)。
    # VLA token KL：起点 action head 在同一 obs/noise 下的 token，λ*MSE(t, t0)。
    # 默认 0 = 完全不影响 Path A / 历史 joint 的行为。
    wbc_kl_anchor: float = 0.0
    vla_kl_anchor: float = 0.0
    # ---- critic 的监督来源 ----
    # "imagined"（旧）= 想象转移上的 GAE；"real"（新）= 真录制轨迹的 return-to-go
    critic_source: str = "imagined"
    critic_batch: int = 256
    # 按 clip 去时间趋势：return-to-go 天然随"剩余步数"下降，直接当 bootstrap 会诱导策略
    # "让状态看起来像更早的帧"（滞后）。减去同 clip 均值后，V 变成"相对质量"，自举才有意义。
    critic_center: bool = True
    grad_log_every: int = 0           # 打印分项梯度范数
    clip_head: float = 0.5            # VLA head 的梯度范数裁剪（先测再定，见文档 §2.6）
    # ---- e1（2026-10-31）：把"参考相对姿态误差"送进 VLA 的 state（46 → 52 维）----
    # 动机：state 里只有 gravity ⇒ 偏航/位置对策略**不可观测**（`gravity_from_rot6d` 把
    # rot6d 的朝向信息扔了，重力对绕世界 z 的旋转不变）。token 又是"以当前朝向左乘"的相对帧，
    # 于是朝向一偏命令整段转歪、正反馈放大。
    #   "none" = 现状 46 维
    #   "ori6" = 在 132 宽 state 的 **46:52** 写 rot6d(R_robotᵀ R_ref) × 6 个可训练缩放
    # 缩放初值必须 0：state 投影层在**冻结主干**里（ckpt 只存 432 个 fm 张量），新列从未被训练，
    # 直接灌 O(1) 的值等于踹预训练模型一脚；缩放=0 ⇒ 第 0 步与 d2b **逐位相同**，是真续接。
    state_extra: str = "none"
    # ★ 2026-10-31（P3，用户口径"至少得保证正确"）：初值从 0 改成 **1.0**。
    #   0 的初衷是"第 0 步与上轮逐位相同"（不出事），但代价是**这一轮里 yaw 输入等于没装**，
    #   只能指望 6 个缩放在 800 步里自己长出来。用户判断：正确性优先于"不踹一脚"。
    #   1.0 = 从第一步起就把 rot6d(R) 按原值喂进去（量级 O(1)，与已归一的其它 state 维同量级）。
    state_extra_scale_init: float = 1.0
    # ---- e2（2026-10-31）：配对基线（控制变量）----
    # "self" = 用这一时刻**真机实际执行**的 token 再跑同一条想象，得到 ret_exec，
    #          adv = ret − ret_exec。两者共享"真机里发生的那一段"，窗口难度被减掉。
    # ⚠️ 关键数学事实：`-(ret).mean()` 是 **pathwise** 梯度，减一个 detach 的常数**不改变梯度**；
    #    而且 Adam 对全局标量缩放不敏感（会被二阶矩归一掉）。所以单单"减基线"是个**安慰剂**。
    #    必须配上 `adv_tau`：`-tanh(adv/τ)` 把每样本权重压成有界的，窗口难度大的样本不再
    #    独占梯度；τ 取"逐窗配对的典型量级"（实测 imag_delta_std≈0.019）。τ=0 ⇒ 退回均值形式
    #    （梯度与现在逐位相同，仅作诊断对照）。
    adv_baseline: str = "none"
    # ★ 2026-11-07（BPTT 那条路的"求和偏置"解药）：**每窗口梯度归一化**。
    #   现状：loss_r = -mean(ret_w)，梯度 = Σ_w ∂ret_w/∂θ / B ⇒ 大回报窗口主导，
    #         而"跨窗口求和只剩均值"正是常向量的机制。
    #   开启后：每个窗口先按自身量级归一化再加权 ⇒ 每窗口影响力相同
    #         （梯度里带 1/|ret_w| 权重，等价于逐窗口缩放后的求和）。
    #   默认 False ⇒ 老行为逐位不变。
    window_norm: bool = False
    window_norm_floor: float = 0.2   # 归一化下限（×|ret| 的中位数），防止接近 0 的窗口爆掉
    adv_tau: float = 0.02


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
            self.critic = TokenCritic(hidden=self.v3.critic_hidden,
                                      use_diff=bool(getattr(self.v3, "critic_use_diff", False))).to(self.device)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.v3.critic_lr)
        # π_k 快照：训练启动时（= 采集本轮 B 数据用的那个策略）拷一份 LoRA。
        # 之后每次取信任域参照都临时换回来，所以中心**永远**是 π_k，不会随更新漂走。
        # π_k 的 **执行器** 快照（train_wbc 时用）：参考 rollout 必须由"上一轮那个 WBC"给出，
        # 否则动作 hinge 变成"自己跟自己比"，约束恒为 0。WBC 只有 ~4.8M，深拷约 19 MB。
        self._wbc_ref = None
        if bool(getattr(self.cfg, "train_wbc", False)):
            self._wbc_ref = copy.deepcopy(self.bundle.policy).eval()
            for _p in self._wbc_ref.parameters():
                _p.requires_grad_(False)
            log(f"[WBC][train] 解冻执行器：params={sum(p.numel() for p in self.wbc_parameters):,} "
                f"lr={self.cfg.wbc_lr:g}；参考 rollout 用 π_k 的 WBC 快照（动作 hinge 中心）")
        # ★ 2026-11-01（Path B）：VLA token KL 的起点快照。默认 0 = 不建、不影响 Path A。
        self._vla_ref = None
        if float(getattr(self.v3, "vla_kl_anchor", 0.0) or 0.0) > 0:
            self._vla_ref = copy.deepcopy(self.vlm.action_head).eval()
            for _p in self._vla_ref.parameters():
                _p.requires_grad_(False)
            log(f"[VLA][kl] token 锚：起点 action head 快照 "
                f"params={sum(p.numel() for p in self._vla_ref.parameters()):,}")
        if float(getattr(self.v3, "wbc_kl_anchor", 0.0) or 0.0) > 0 and self._wbc_ref is None:
            raise SystemExit("[v3] wbc_kl_anchor>0 但 train_wbc=False（没有起点 WBC 快照）")
        self._prev_lora = None
        if str(getattr(self.v3, "trust_center", "label")) == "prev":
            from .lora import LoRASnapshot  # noqa: PLC0415

            self._prev_lora = LoRASnapshot(self.vlm.action_head)
            self.v3.trust_eps = 0.0        # 中心换成 π_k 后 token 空间不再有"先验"
            log(f"[TRUST][prev] 冻结 π_k：LoRA {self._prev_lora.summary()}；"
                f"中心=π_k 自身采样（同一噪声），不含标签")
        log(
            f"[AUDIT][v3] lambda_r={self.v3.lambda_r} lambda_hand={self.v3.lambda_hand} "
            f"token_bc={self.v3.token_bc_weight:g} critic={self.v3.use_critic} push_std={self.v3.push_std} "
            f"critic_clip={self.v3.critic_clip}/{self.v3.critic_value_clip} "
            f"head_clip={self.v3.clip_head} trust_center={self.v3.trust_center} "
            f"H={self.cfg.horizon} K={self.cfg.fm_steps} boot_1side={self.v3.boot_one_sided}"
        )
        # v3 的 loss 里**没有** cfg.lambda_bc 这一项（v2 的 FM-BC 已被 train_step 覆写掉），
        # 但 --lambda-bc 仍会被写进 args.json / v2_config.json / summary.json。传非 0 值时
        # 这里显式告警，避免交接时被"配置说开了 BC、代码其实没开"误导（2026-10-29）。
        if float(getattr(self.cfg, "lambda_bc", 0.0)) != 0.0:
            log(f"[AUDIT][v3][warn] lambda_bc={self.cfg.lambda_bc:g} 在 v3 路径**不生效**"
                f"（等效 BC 权重 0）——配置里那个数字只是记录，别当成开了 BC")

        # ---- e1：参考相对姿态误差的 6 个可训练缩放 ----
        self.state_extra_scale = None
        _extra = str(getattr(self.v3, "state_extra", "none") or "none").lower()
        if _extra not in ("none", "rotlocal6", "abs6", "vel6", "ori6"):
            raise SystemExit(f"[state-extra] 未知 mode={_extra!r}（只支持 none|rotlocal6|abs6|vel6|ori6）")
        if _extra in ("rotlocal6", "abs6", "vel6", "ori6"):
            from .state_extra import STATE_EXTRA_DIM  # noqa: PLC0415

            _init = float(getattr(self.v3, "state_extra_scale_init", 0.0))
            self.state_extra_scale = torch.nn.Parameter(
                torch.full((STATE_EXTRA_DIM,), _init, dtype=torch.float32, device=self.device))
            # 进优化器（否则永远是 0 = 白加 6 维）。lr 跟 action head 一致。
            # 用 _ensure_optimizer：`--freeze-vla` 时基类建不出优化器（空参数列表）。
            self._ensure_optimizer([self.state_extra_scale], float(self.cfg.fm_lr))
            self.v3.state_extra = _extra
            log(f"[AUDIT][state-extra] mode={_extra} slots=46:52 × scale(init={_init:g}) "
                f"params={self.state_extra_scale.numel()} "
                f"⇒ state 有效宽度 46→52（投影层在冻结主干里，靠缩放学习启用）")
            log(f"[AUDIT][state-extra] 初值={_init:g}："
                + ("从第 0 步起 yaw 输入就真的进入前向（P3 口径：正确性优先）"
                   if _init != 0.0 else
                   "第 0 步与该轮起点逐位相同，但这一轮里新维度等于没装（仅适用于纯 A/B）"))
        else:
            self.v3.state_extra = "none"

        # ---- 命令层残差 g(state)：token' = token + g(token, 偏差) ----
        # 与上面三条路**并行**，默认关（DELTA_TOKEN=0）。开启后 LoRA 可以照样开着（做 A/B），
        # 但推荐同时冻掉（否则两个可训练通道混在一起，归因不清）。
        self.delta_token_net = None
        self._dt_norm = None
        self._dt_state_sens = None
        if bool(getattr(self.v3, "delta_token", False)):
            from .delta_token import FEAT_DIM, STATE64_DIM, DeltaTokenNet  # noqa: PLC0415
            _fdim = (STATE64_DIM if bool(getattr(self.v3, "delta_token_fullstate", False))
                     else FEAT_DIM)

            self.delta_token_net = DeltaTokenNet(
                feat_dim=int(_fdim),
                hidden=int(getattr(self.v3, "delta_token_hidden", 1024) or 1024),
                depth=int(getattr(self.v3, "delta_token_depth", 2) or 2),
                init_std=float(getattr(self.v3, "delta_token_init_std", 4e-4) or 0.0),
            ).to(self.device)
            _dt_lr = float(getattr(self.v3, "delta_token_lr", 5e-4) or 5e-4)
            self._ensure_optimizer(list(self.delta_token_net.parameters()), _dt_lr)
            log(f"[AUDIT][delta-token] 启用：feat={FEAT_DIM}(rot6d+dof_vel) + token64 → Δtoken64；"
                f"hidden={getattr(self.v3, 'delta_token_hidden', 256)} "
                f"params={self.delta_token_net.n_params}"
                f" lr={float(getattr(self.v3, 'delta_token_lr', 5e-4) or 5e-4):g}"
                f" cap={float(getattr(self.v3, 'delta_token_cap', 0.0) or 0.0):g}")

        # 兜底：如果既没有 LoRA、没有 WBC、也没有任何延迟添加的参数组 ⇒ 明确失败，
        # 而不是等到 train_step 里 `self.optimizer.step()` 抛 NoneType。
        if getattr(self, "optimizer", "missing") is None:
            raise SystemExit(
                "[v3] 没有任何可训练参数（既无 LoRA/无 WBC，也未开 --delta-token / --state-extra）。"
                " 用 --freeze-vla 时必须同时开 --delta-token。")

    # ---------------------------------------------------------------- delta token
    def _delta_token_features(self, batch) -> torch.Tensor:
        """``(B, 35)`` = rot6d(6) + dof_vel(29)，与部署 ``state64`` 同口径可复算。

        rot6d 用 **rotlocal6**（heading-canonical，回合起始为 identity）—— 部署端
        ``state64[:6]`` 就是这个量，所以训练/部署必须走同一条 ``extra6_for``。
        """
        from .isaac_joint import flat_state_tuple  # noqa: PLC0415
        from .state_extra import extra6_for  # noqa: PLC0415

        start = batch.start_state.to(self.device).float()
        _p, _R, _v, _w, _q, _qd = flat_state_tuple(start)
        q0 = getattr(batch, "start_quat0", None)
        if q0 is None:
            # 兜底：没有回合起始姿态时退化为绝对朝向（与部署口径会有偏，但不会崩）
            rot6 = extra6_for("abs6", quat=start[:, 3:7])
        else:
            rot6 = extra6_for("rotlocal6", quat=start[:, 3:7],
                              q_init=q0.to(self.device).float().reshape(-1, 4))
        if bool(getattr(self.v3, "delta_token_fullstate", False)):
            # ★ 第 2 刀：与部署 state64 逐位同序 —— [ rot6d(6) | dof_pos(29) | dof_vel(29) ]
            #   dof_pos 是之前完全没喂的那 29 维（cos 0.24 → 0.39~0.47 的主要来源）。
            return torch.cat([rot6.float(), _q.float(), _qd.float()], dim=-1)
        return torch.cat([rot6.float(), _qd.float()], dim=-1)

    def _delta_token_delta(self, batch, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, 1, 64)`` 的广播用 Δtoken（用窗口起点的命令 token[0] + 起点偏差）。"""
        feat = self._delta_token_features(batch)
        base_tok = tokens[:, 0].detach()
        # ★★ 2026-11-07 修复：**标签必须算在这个 token 上**。
        #   之前 `_grad_target_loss` 用的是 `batch.port_tokens`（**录制数据里的 token**），
        #   而 Δ 实际加在 VLA 本次前向输出的 token 上 ⇒ 两者不是同一个点，
        #   ĝ 的方向对不上标的 token ⇒ 网络只能学到"长度"（ratio≈1）而学不到方向（cos≈0.05）。
        self._dt_base_tok = base_tok
        d = self.delta_token_net(base_tok, feat)
        # ★ **状态敏感度**（2026-11-06）：同一批 token，把状态打乱再算一次。
        #   `‖g(tok,s) − g(tok,s_perm)‖ / ‖g(tok,s)‖` 才是这条路的"残差/均值"等价物：
        #   原来的 `action_var/ratio` 是跨窗口比较，在 token 残差路径下会被"token 本来就不同"
        #   混淆（token 里已含相位）；这个量把 token 固定住，只量状态贡献。
        #   ≈0 ⇒ g 根本没看状态（这条路当场判定无效）；随训练上升 ⇒ 真的在按状态改命令。
        with torch.no_grad():
            _perm = torch.randperm(feat.shape[0], device=feat.device)
            _d_perm = self.delta_token_net(base_tok, feat[_perm])
            _n = d.norm(dim=-1).mean()
            self._dt_state_sens = float(((d - _d_perm).norm(dim=-1).mean()
                                         / _n.clamp_min(1e-9)).detach())
        # 软上限：**与部署端同一个函数**（口径漂移的唯一防线）
        from .delta_token import apply_norm_cap  # noqa: PLC0415

        d = apply_norm_cap(d, float(getattr(self.v3, "delta_token_cap", 0.0) or 0.0))
        self._dt_norm = float(d.norm(dim=-1).mean().detach())
        self._dt_delta_last = d          # ★ 第 3 刀：稠密监督要用（带梯度）
        return d.unsqueeze(1)

    # ------------------------------------------------------------------ 第 3 刀
    def _grad_target_loss(self, batch):
        """稠密监督：让 Δ 直接对齐 ``eta·unit(∂J/∂token)``。

        为什么这是"最重要的一刀"
        ------------------------
        20 步报酬反传时，参数梯度是**跨窗口求和**的 ``Σ_w ∂J_w/∂θ``，各窗口独有的那部分
        互相抵消，只剩均值 ⇒ 收敛到常向量（四种参数化实测 0.08~0.17 都是同一个病）。
        而真机 ±δ 实验证明**单窗口的 ĝ 方向本身是好的**（H=2 96/96、H=20 63/64）。
        所以把 ĝ 当**标签**用：逐窗口回归，不做求和，方向信息就保住了。

        成本：子批 ``grad_target_n`` 个窗口各一次 20 步反传（默认 16 个 ⇒ ~3 s/step）。
        """
        w = float(getattr(self.v3, "grad_target_weight", 0.0) or 0.0)
        delta = getattr(self, "_dt_delta_last", None)
        if w <= 0 or delta is None:
            return None, {}
        try:
            from .isaac_joint import FrameRef  # noqa: PLC0415
            from .rollout import imagine_with_tokens  # noqa: PLC0415

            cfg = self.cfg
            # ★ 优先用"Δ 实际依附的那个 token"（VLA 输出），退路才是录制 token
            _bt = getattr(self, "_dt_base_tok", None)
            tok = _bt[:, None, :] if _bt is not None else getattr(batch, "port_tokens", None)
            if tok is None:
                return None, {}
            if tok.ndim == 2:
                tok = tok[:, None, :]
            n = max(1, min(int(getattr(self.v3, "grad_target_n", 16) or 16), int(tok.shape[0])))
            # ★ 第 5 刀：短视界标签（0 ⇒ 用 cfg.horizon）
            _H_gt = int(getattr(self.v3, "grad_target_horizon", 0) or 0) or int(cfg.horizon)
            _H_gt = max(1, min(_H_gt, len(batch.references)))
            sl = slice(0, n)
            _tok = tok[sl].detach().float().clone().requires_grad_(True)
            _refs = []
            for r in batch.references:
                _f = getattr(r, "f", None)
                _refs.append(FrameRef(_f[sl].to(self.device)) if _f is not None else r)
            # history 两种布局都可能（(B,T,w) / (T,B,w)）；有 history 时 obs_hist 会被忽略
            _hist = getattr(batch, "history", None)
            _oh = getattr(batch, "obs_hist", None)
            if _hist is not None:
                _hist_arg = (_hist[sl] if _hist.shape[0] == tok.shape[0]
                             else (_hist[:, sl] if _hist.shape[1] == tok.shape[0] else _hist))
                _oh_arg = None
            else:
                _hist_arg = None
                _oh_arg = (_oh[sl] if (_oh is not None and _oh.shape[0] == tok.shape[0]) else None)
            with torch.enable_grad():
                _ro = imagine_with_tokens(
                    self.bundle, tokens=_tok, state=batch.start_state[sl],
                    obs_hist=_oh_arg,
                    history=_hist_arg,
                    xi=batch.xi[sl], references=_refs,
                    prev_action=(batch.prev_action[sl] if getattr(batch, "prev_action", None) is not None else None),
                    horizon=_H_gt,
                )
                _J = _ro.rewards.sum()
                _g = torch.autograd.grad(_J, _tok)[0].mean(1).detach()      # (n,64)
            _gn = _g.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            eta = float(getattr(self.v3, "grad_target_eta", 0.08) or 0.08)
            _tgt = (eta * _g / _gn).to(delta.dtype)                        # (n,64)
            _d = delta.reshape(-1, delta.shape[-1])[:n]
            loss = ((_d - _tgt) ** 2).sum(-1).mean()
            with torch.no_grad():
                _cos = (_d * _g.to(_d.dtype)).sum(-1) / (_d.norm(dim=-1).clamp_min(1e-9) * _gn.squeeze(-1).to(_d.dtype))
                meta = {
                    "delta_token/grad_cos": float(_cos.mean()),
                    "delta_token/grad_norm_ratio": float(_d.norm(dim=-1).mean() / max(eta, 1e-9)),
                    "delta_token/grad_J": float(_J.detach()),
                    "delta_token/grad_n": float(n),
                    "delta_token/grad_H": float(_H_gt),
                }
            return loss, meta
        except Exception as _e:  # noqa: BLE001 - 监督项绝不能挡住训练
            if not getattr(self, "_gt_warned", False):
                self._gt_warned = True
                print(f"[grad-target][warn] {_e}", flush=True)
            return None, {}

    def save_vla(self, path, *, extra: dict | None = None):
        """带上 ``state_extra_scale``（否则下一轮续接时这 6 个数会丢，新维度白学）。"""
        _extra = dict(extra or {})
        if getattr(self, "state_extra_scale", None) is not None:
            _extra["state_extra_scale"] = self.state_extra_scale.detach().to("cpu", dtype=torch.float32)
            _extra["state_extra_mode"] = str(getattr(self.v3, "state_extra", "none"))
            if not _extra.get("state_extra_mode"):
                raise SystemExit("[state-extra] 保存时 mode 为空")
        if getattr(self, "delta_token_net", None) is not None:
            _extra["delta_token_state"] = {
                k: v.detach().to("cpu", dtype=torch.float32)
                for k, v in self.delta_token_net.state_dict().items()
            }
            _extra["delta_token_meta"] = {
                "hidden": int(getattr(self.v3, "delta_token_hidden", 1024) or 1024),
                "init_std": float(getattr(self.v3, "delta_token_init_std", 4e-4) or 0.0),
                "depth": int(getattr(self.v3, "delta_token_depth", 2) or 2),
                "cap": float(getattr(self.v3, "delta_token_cap", 0.0) or 0.0),
            }
        return super().save_vla(path, extra=_extra)

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
        if getattr(self, "state_extra_scale", None) is not None:
            model_inputs = self._patch_state_extra(model_inputs, batch)
        return model_inputs

    def _patch_state_extra(self, model_inputs: dict, batch) -> dict:
        """e1：把 ``extra6 × scale`` 写进 state 的 46:52。

        * ``vel6``  = 体坐标系 ``[Rᵀv, Rᵀw]``（**闭环也能算**，推荐）
          v/w 从 ``start_state`` 按 ``flat_state_tuple`` 的布局取（世界系，见该函数），
          旋转用 ``start_state[3:7]`` 的四元数——不硬编码 36:42 之外的偏移。
        * ``ori6``  = ``rot6d(R_robotᵀ R_ref)``，参考帧取 ``batch.frames``。
          ⚠️ 闭环采集器没有参考帧（``FrameRef(zeros)`` 占位），所以 ori6 **不能**用于评测，
          只能做离线诊断——真要用必须先把评测改成跟参考帧。
        """
        from .state_extra import STATE_EXTRA_SLICE, extra6_for, write_state_extra  # noqa: PLC0415

        if "state" not in model_inputs:
            raise SystemExit("[state-extra] model_inputs 里没有 state")
        mode = str(getattr(self.v3, "state_extra", "none"))
        start = getattr(batch, "start_state", None)
        if start is None:
            raise SystemExit("[state-extra] batch 没有 start_state")
        start = start.to(self.device).float()
        if mode == "vel6":
            from .isaac_joint import flat_state_tuple  # noqa: PLC0415

            _p, _R, _v, _w, _q, _qd = flat_state_tuple(start)
            extra6 = extra6_for("vel6", quat=start[:, 3:7], v_world=_v, w_world=_w)
        elif mode == "rotlocal6":
            # **线上协议本来就在发的 6 维**：rot6d(conj(heading(q0)) * q_now)。
            # q0 = 本回合起始姿态（b_start 从 dump 第 0 帧取的）；与第三方实现已逐位对拍。
            q0 = getattr(batch, "start_quat0", None)
            if q0 is None:
                raise SystemExit("[state-extra] rotlocal6 需要 batch.start_quat0（回合起始姿态）")
            extra6 = extra6_for("rotlocal6", quat=start[:, 3:7],
                                q_init=q0.to(self.device).float())
        elif mode == "abs6":
            # 零改动可落地版：线上 state64[0:6] 本来就是 rot6d(quat)，训练侧用同一个编码
            extra6 = extra6_for("abs6", quat=start[:, 3:7])
        elif mode == "ori6":
            frames = getattr(batch, "frames", None)
            if frames is None:
                raise SystemExit("[state-extra] ori6 需要 batch.frames（参考帧）")
            ref_flat = self.bundle.ref_state_at(frames.to(self.device))
            extra6 = extra6_for("ori6", quat=start[:, 3:7],
                                quat_ref=ref_flat.to(self.device).float()[:, 3:7])
        else:
            raise SystemExit(f"[state-extra] mode={mode!r} 不该走到这里")
        if not torch.isfinite(extra6).all():
            raise SystemExit("[state-extra] extra6 出现非有限值（四元数退化？）")
        out = dict(model_inputs)
        out["state"] = write_state_extra(out["state"], extra6 * self.state_extra_scale)
        return out

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
    def _ref_states(self, batch) -> torch.Tensor:
        """每个想象步的参考状态 ``(H+1,B,71)``（critic 输入 / 诊断用）。

        **后端无关版**：优先用 ``bundle.ref_state_at(frames)``；
        老路径（``r.f`` + ``motion.ref`` 张量）只对 trackerlab 成立 —— mjlab 的
        ``ReferenceFrame`` 没有 ``.f``、``library.ref`` 是方法（2026-10-28 实测崩过）。
        """
        if hasattr(self.bundle, "ref_state_at"):
            H = len(batch.references)
            base = batch.frames.to(self.bundle.device).long()
            offs = torch.arange(1, H + 1, device=base.device)
            lib = self.bundle.motion
            # 2026-10-28 20:30：`clip_last`/`frame_clip` 只有 trackerlab 的 MotionLib 有；
            # 换机后这里是 `CompactMotionLib`（只有 `clip_start`/`clip_len`/`bin_clip`/`frame_bin`），
            # 直接取 `lib.clip_last` 会在第 1 步就 AttributeError。
            # 两种 lib 都给一条等价路径：frame → clip → 该 clip 的最后一帧。
            if hasattr(lib, "clip_last") and hasattr(lib, "frame_clip"):
                _cl = lib.frame_clip[base]
                last = lib.clip_last[_cl].to(base.device)
            else:
                _cl = lib.bin_clip[lib.frame_bin[base]]
                last = (lib.clip_start[_cl] + lib.clip_len[_cl] - 1).to(base.device)
            idx = torch.minimum(base[:, None] + offs[None, :], last[:, None])   # (B,H)
            rows = self.bundle.ref_state_at(idx.reshape(-1)).reshape(
                batch.frames.shape[0], H, -1).permute(1, 0, 2).contiguous()   # (H,B,71)
            return torch.cat((rows, rows[-1:]), dim=0)                         # (H+1,B,71)
        motion = self.bundle.motion
        rows = []
        for r in batch.references:
            f = r.f.to(motion.ref.device)
            rows.append(canonical_to_flat(motion.ref[f].float()))
        rows.append(rows[-1])
        return torch.stack(rows, dim=0)

    @contextlib.contextmanager
    def _wbc_ref_ctx(self):
        """临时把执行器换成 π_k 快照（只用于参考 rollout）；退出时换回训练中的 WBC。"""
        if self._wbc_ref is None:
            yield
            return
        live = self.bundle.policy
        self.bundle.policy = self._wbc_ref
        try:
            yield
        finally:
            self.bundle.policy = live

    # ------------------------------------------------------------------ 训练步
    def train_step(self, batch) -> dict[str, float]:  # noqa: C901
        cfg, v3 = self.cfg, self.v3
        import os as _os_p  # noqa: PLC0415
        import time as _time_p  # noqa: PLC0415
        _prof = _os_p.environ.get("J1_STEP_PROFILE") == "1"
        _marks: list[tuple[str, float]] = []

        def _mark(_n: str) -> None:
            if _prof:
                _marks.append((_n, _time_p.perf_counter()))

        _mark("start")
        self.bundle.policy.eval()
        self.bundle.wm.eval()
        self.head_mode()

        # A1：观测侧覆写走 prepare_model_inputs（训练/评测同一条路，避免评测看不到改动）
        model_inputs = self.prepare_model_inputs(batch)
        # ★ 2026-11-06：**VLA 全冻时把主干+FM 采样放进 no_grad**。
        #   否则 `tokens` 自带整条 2B 主干的计算图，`loss.backward()` 会把 2B 的激活全部
        #   反传一遍 —— 而里面**一个可训参数都没有**（纯浪费）。实测后果：DDP 下 backward
        #   直接 OOM（47.4 GiB 只剩 200 MiB）。
        #   冻 VLA 时唯一需要有梯度的是 `g`（delta-token），把它接在 detach 后的 tokens 上即可。
        # ⚠️ 不能用 torch.no_grad() 包：`sample_fm_from_features` 在 no_grad 下会主动
        #   raise（"the joint training graph would be cut"）。VLA 全冻时本来就没有可训参数、
        #   输入也不带梯度 ⇒ 采样天然不建图，不需要 no_grad。
        _vla_frozen = not self.fm_parameters
        act_in, bb_out = self._backbone_output(model_inputs)
        _mark("backbone")
        # 固定噪声：自适应头与"底模参照"用**同一个噪声**采样，两者只差 LoRA 增量
        head = self.vlm.action_head
        _noise = torch.randn(int(act_in["state"].shape[0]), int(head.action_horizon),
                             int(head.action_dim), device=self.device, dtype=head.dtype)
        x_norm, x_raw, tokens = self._sample_tokens(act_in, bb_out, noise=_noise)
        # 冻结主采样点的量化诊断：下面 trust-center（prev/base）还会再采样，会覆盖 _quant_stats
        self._quant_main = self.quant_metrics()
        _mark("fm_sample")
        out_of_range = self.codec.out_of_range_frac(x_norm)

        if _vla_frozen:
            # 显式切断：之后唯一的梯度来源是 `g`
            x_norm = x_norm.detach(); x_raw = x_raw.detach(); tokens = tokens.detach()

        # 未加 Δ 的底模 token（trust_center=base 的参照；下面可能直接复用，省一次 VLA 前向）
        tokens_base = tokens

        # ★ 命令层残差：token' = token + g(token_0, 偏差)，**在进 WBC 之前**加。
        #   只到 g 的梯度（VLA 若冻结则不回传），act/惩罚/diverge 全都不需要改口径。
        if self.delta_token_net is not None:
            tokens = tokens + self._delta_token_delta(batch, tokens).to(tokens.dtype)
            _mark("delta_token")

        rollout = imagine_with_tokens(
            self.bundle, tokens=tokens, state=batch.start_state, obs_hist=batch.obs_hist,
            history=batch.history, xi=batch.xi, references=batch.references,
            prev_action=batch.prev_action, horizon=cfg.horizon,
            obs_noise=cfg.obs_noise, push_std=v3.push_std, push_ang=v3.push_ang,
            anchor_policy=(
                self._wbc_ref
                if float(getattr(v3, "wbc_kl_anchor", 0.0) or 0.0) > 0
                else None
            ),
        )
        _mark("rollout")
        H = cfg.horizon
        # 奖励视界：0=满 H（旧行为）；k>0 只积前 k 步（见 v3.reward_horizon 注释）
        _rh = int(getattr(v3, "reward_horizon", 0) or 0)
        ret = discounted_return(rollout, v3.gamma, steps=_rh)    # (B,)
        ref_states = self._ref_states(batch)                     # (H+1,B,71)
        xi = batch.xi
        _mark("ref")

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
            metrics_v_raw = float(rel.mean().detach())          # 未裁剪的原始符号（诊断）
            if bool(getattr(v3, "boot_one_sided", True)):
                # 只惩罚"比参考差"，不为"比参考好"付钱：见 V3Config.boot_one_sided 的注释
                rel = torch.minimum(rel, torch.zeros_like(rel))
            if _sg != 1.0:
                # straight-through：loss 里仍是 rel 的数值，但反传时只有 _sg 倍
                rel = rel.detach() + _sg * (rel - rel.detach())
            if _clip > 0:
                rel = torch.clamp(rel, -_clip, _clip)
            boot = (v3.gamma ** H) * rollout.survival[-1] * rel
            metrics_v = float(rel.mean().detach())
        else:
            metrics_v = 0.0
            metrics_v_raw = 0.0
        _mark("boot")
        # ★ 每窗口归一化归约（window_norm=1 才生效；默认等价于 .mean()）
        _win = bool(getattr(v3, "window_norm", False))

        def _red(v: torch.Tensor) -> torch.Tensor:
            if not _win:
                return v.mean()
            _a = v.detach().abs()
            _floor = float(getattr(v3, "window_norm_floor", 0.2) or 0.0) * float(_a.median())
            _den = _a.clamp_min(max(_floor, 1e-6))
            return (v / _den).mean()
        # ---------- e2：配对基线（用"执行 token"当控制变量）----------
        _adv_stats: dict[str, float] = {}
        _bl = str(getattr(v3, "adv_baseline", "none"))
        if _bl in ("self", "label"):
            # 2026-11-04：新增 `label` 基线——用**标签 token**跑同一段想象当配对基线。
            #   为什么需要它：`self` 要 B 线 dump（R1 没有 B），于是 R1 只能 `none`；
            #   而没有基线时 5-6 个 clip × 不同 phase 的公共方差全进梯度。
            #   `label` 不需要任何额外数据，且语义正好是"我这个 token 比参考 token 好在哪"。
            if _bl == "self":
                _st = getattr(batch, "self_tokens", None)
                if _st is None:
                    raise SystemExit(
                        "[adv-baseline=self] batch 里没有 self_tokens（B 线真值 token）；"
                        "R1 请用 ADV_BASELINE=label")
            else:
                _st = batch.port_tokens
            if _st is not None:
                if _st.ndim == 2:
                    _st = _st[:, None, :]
                with torch.no_grad():
                    _roll_exec = imagine_with_tokens(
                        self.bundle, tokens=_st.detach().to(self.device), state=batch.start_state,
                        obs_hist=batch.obs_hist, history=batch.history, xi=batch.xi,
                        references=batch.references, prev_action=batch.prev_action,
                        horizon=cfg.horizon,
                    )
                    ret_exec = discounted_return(_roll_exec, v3.gamma).detach()
                adv = ret - ret_exec
                _tau = float(getattr(v3, "adv_tau", 0.0) or 0.0)
                _r_std = float(ret.detach().std())
                _a_std = float(adv.detach().std())
                _adv_stats = {
                    "adv/baseline": 1.0 if _bl == "label" else 0.0,
                    "adv/ret_std": _r_std,
                    "adv/std": _a_std,
                    "adv/std_ratio": (_a_std / _r_std) if _r_std > 1e-9 else 1.0,
                    "adv/absmean": float(adv.detach().abs().mean()),
                    "adv/ret_exec_mean": float(ret_exec.mean()),
                }
                if _tau > 0.0:
                    _adv_stats["adv/frac_inside_tau"] = float(
                        (adv.detach().abs() < _tau).float().mean())
                    loss_r = -_red(torch.tanh((adv + boot) / _tau))
                else:
                    loss_r = -_red(adv + boot)
            else:
                loss_r = -_red(ret + boot)
        else:
            loss_r = -_red(ret + boot)
        _mark("self_adv")

        # ---------- (B) state 项：把**误差直接反传**（2026-10-30 新增到 v3）----------
        # 为什么：奖励是 `exp(-e²/σ²)` 的均值，e>2σ 处梯度≈0（实测 >0.4m 的时刻占 21% 却只给
        # 0.22% 位置梯度）⇒ 均值目标主动"拿尾部换中位"。MSE 在远处梯度 ~2e **线性增长**，
        # 天然修尾部。`lambda_s` 原本只在 v2 的 train_step 里被读，v3 覆写后**整项空转**。
        loss_s = None
        if float(getattr(self.cfg, "lambda_s", 0.0)) > 0 and getattr(batch, "teacher_states", None) is not None:
            from .rollout import _state_error  # noqa: PLC0415
            _h = rollout.states.shape[0] - 1
            loss_s = _state_error(
                rollout.states[1:],
                batch.teacher_states[:, :_h].permute(1, 0, 2),
                vel_weight=float(getattr(self.cfg, "state_vel", 0.05)),
                root_vel_weight=float(getattr(self.cfg, "state_root_vel", 0.0)),
            )
            loss = loss + float(self.cfg.lambda_s) * loss_s

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

        # ---------- token BC（2026-10-28 新增：v3 唯一的"该输出什么"监督）----------
        # 为什么必须加：v3 的 token 只有两个信号来源——奖励项（经 WM 想象）和 trust hinge。
        # 两者都只回答"往哪偏一点点"，谁也不回答"这个 drifted 状态下正确 token 是什么"。
        # 实测后果：6 轮训练全在 π₀ 附近震荡（合并 n=24 反而更差 0.101 m）。
        # 这里用 B dump 自己的 `port_tokens`（该帧的命令流 token）当标签做 GR00T 原生
        # flow-matching 回归 —— 即 **on-policy 自监督**：状态是真实漂移状态，标签是参考 token。
        # 只监督前 64 维（token），手部仍由上面的 lambda_hand 分支负责。
        loss_token_bc = None
        _bc_src = str(getattr(v3, "token_bc_source", "ref"))
        _bc_tok = batch.port_tokens
        if _bc_src == "self":
            _bc_tok = getattr(batch, "self_tokens", None)
            if _bc_tok is None:
                raise SystemExit(
                    "[fail] token_bc_source=self 但 batch 里没有 self_tokens；"
                    "需要 --b-self-targets 指向 build_b_self_targets.py 的产物")
        if float(getattr(v3, "token_bc_weight", 0.0)) > 0 and getattr(batch, "hand_ref", None) is not None:
            tgt_bc, mask_bc_full = bc_target(
                self.codec, _bc_tok, batch.hand_ref,
                head_dim=int(self.vlm.action_head.action_dim),
                dtype=self.vlm.action_head.dtype, device=self.device,
            )
            tok_mask = torch.zeros_like(mask_bc_full)
            tok_mask[..., :64] = 1.0
            _span = int(getattr(v3, "token_bc_span", 0) or 0)
            if _span > 0:
                # 只监督前 _span 步（(B,T,dim) 的第 1 维是时间）
                tok_mask[:, _span:, :] = 0.0
            with self._amp_ctx():
                loss_token_bc = fm_bc_loss(self.vlm.action_head, bb_out, act_in, tgt_bc, tok_mask)
            loss_token_bc = loss_token_bc.float()

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
                _tc = str(getattr(v3, "trust_center", "label"))
                if (_tc == "self" and getattr(batch, "self_tokens", None) is None
                        and not getattr(self, "_self_warned", False)):
                    self.log("[TRUST][self][warn] batch 没有 self_tokens（缺 build_b_self_targets "
                             "或该 dump 不在表里）⇒ token 中心降级为 port_tokens(label)")
                    self._self_warned = True
                if _tc == "self" and getattr(batch, "self_tokens", None) is not None:
                    # 中心 = **采集那一刻策略自己真正下发的 token**（B 线真值），
                    # 既不是录制标签（属于另一个执行器），也不是上一轮 π_k（那个在本状态下未必更好）
                    prior_t = batch.self_tokens.to(tokens.dtype)
                else:
                    prior_t = batch.port_tokens.to(tokens.dtype)      # 底模先验代理（≈标签）
            # ★ 2026-10-30：形状防御。self_tokens 现在恒为 B 行（缺标签用 0 占位，见 b_start.py），
            #   万一还有别的来源行数不一致，这里退回 port_tokens 而不是崩。
            n_t = min(int(tokens.shape[1]), int(prior_t.shape[1]))
            if int(prior_t.shape[0]) != int(tokens.shape[0]):
                self.log(f"[TRUST][self][warn] prior_t 行数 {int(prior_t.shape[0])} != B "
                         f"{int(tokens.shape[0])} ⇒ 本步退回 port_tokens 中心")
                prior_t = batch.port_tokens.to(tokens.dtype)
                n_t = min(int(tokens.shape[1]), int(prior_t.shape[1]))
            diff = tokens[:, :n_t].float() - prior_t[:, :n_t].float()  # (B,T,64) 带梯度
            rmse_tok = diff.pow(2).mean(dim=tuple(range(1, diff.dim()))).sqrt()   # (B,)
            excess = torch.relu(rmse_tok / float(v3.trust_eps) - 1.0)
            # 有真标签的位置才计入（占位行的 0 token 不算违规）
            _has = getattr(batch, "self_tokens_has", None)
            if _has is not None and int(_has.numel()) == int(excess.numel()):
                _w = _has.to(excess.dtype)
                loss_trust = float(v3.trust_weight) * (excess.pow(2) * _w).sum() / _w.sum().clamp_min(1.0)
            else:
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
            _center = str(getattr(v3, "trust_center", "label"))
            if _center == "prev":
                # 中心 = π_k（本轮起点，冻结快照）：**同一输入、同一噪声**再采一次。
                # 不含标签、不含底模先验；球内只剩执行误差的梯度（见 docs/REVIEW §2）。
                if self._prev_lora is None:
                    raise SystemExit("[TRUST][prev] 没有 π_k 快照；构造 trainer 时 trust_center 不是 prev")
                # 不能包 no_grad：joint sampler 会显式拒绝无梯度上下文（图会被切断）。
                # 和 base 分支同口径：正常梯度上下文采样，然后立刻 detach（不参与反传）。
                with self._prev_lora.applied(self.vlm.action_head):
                    _, _, _ref_tok = self._sample_tokens(
                        act_in, bb_out, noise=_noise, backend="truncated")
                _ref_tok = _ref_tok.detach()
            elif _center == "base":
                # ★ 2026-11-06：VLA 全冻（--freeze-vla）时 `lora_disabled()` 是**空操作**，
                #   再采一次得到的 ref token 与 tokens_base **逐位相同** ⇒ 直接复用，
                #   省掉一次 1.09B 的前向（实测每步 ~4 s，这一次约占 1/3）。
                if _vla_frozen:
                    _ref_tok = tokens_base.detach()
                else:
                    from .lora import lora_disabled  # noqa: PLC0415
                    with lora_disabled(self.vlm.action_head):
                        _, _, _ref_tok = self._sample_tokens(
                            act_in, bb_out, noise=_noise, backend="truncated")
                    _ref_tok = _ref_tok.detach()
            elif _center == "self" and getattr(batch, "self_tokens", None) is not None:
                # 中心 = 采集时真正下发的 token（B 线真值）：用它跑同一段 rollout 得 a_ref
                _ref_tok = batch.self_tokens.detach()
            elif _center == "hybrid" and getattr(batch, "self_tokens", None) is not None:
                # 低漂移用 label（能教运动），高漂移用 self（参考 token 在漂移状态未必纠偏）。
                _err = getattr(batch, "realized_err", None)
                if _err is None:
                    _ref_tok = batch.port_tokens.detach()
                else:
                    _thr = float(getattr(v3, "hybrid_err", 0.5))
                    _use_self = (~torch.isfinite(_err)) | (_err > _thr)
                    _has = getattr(batch, "self_tokens_has", None)
                    if _has is not None:
                        _use_self = _use_self | (_has < 0.5)
                    _st = batch.self_tokens
                    _pt = batch.port_tokens
                    if _st.ndim == 2:
                        _st = _st[:, None, :]
                    if _pt.ndim == 2:
                        _pt = _pt[:, None, :]
                    _tt = min(int(_st.shape[1]), int(_pt.shape[1]))
                    _mask = _use_self.to(_pt.dtype)[:, None, None].expand(-1, _tt, -1).bool()
                    _ref_tok = torch.where(_mask, _st[:, :_tt], _pt[:, :_tt]).detach()
            else:
                _ref_tok = batch.port_tokens.detach()
            with torch.no_grad(), self._wbc_ref_ctx():
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
            _shape = str(getattr(v3, "act_penalty", "hinge") or "hinge").lower()
            if _shape == "l2":
                # 无死区：r=0 处梯度也为 0（平方项），但**离开 0 就有斜率 2r/eps²**，
                # 所以刹车从第一步就参与平衡。平衡点 ≈ eps·sqrt(奖励斜率/(2·weight/eps))。
                excess_a = act_rms / float(v3.act_eps)
                _pen = excess_a.pow(2)
            else:
                excess_a = torch.relu(act_rms / float(v3.act_eps) - 1.0)
                _cap = float(getattr(v3, "act_hinge_cap", 0.0))
                if _cap > 0:
                    # ``clamp`` alone makes the penalty CONSTANT once excess_a > cap, so the worst
                    # violations get ZERO gradient.  Keep a constant slope outside the cap with a
                    # linear continuation: quad^2 + 2*cap*(excess - quad).
                    _quad = torch.clamp(excess_a, max=_cap)
                    _pen = _quad.pow(2) + 2.0 * _cap * (excess_a - _quad)
                else:
                    _pen = excess_a.pow(2)
            loss_act = float(v3.act_weight) * _pen.mean()
            if float(v3.act_weight) > 0:
                loss = loss + loss_act

            # ★ 反常量项：把"每个窗口的 20 帧平均动作改变"拆成 均值 + 残差，
            #   最大化 rms(残差)/rms(均值)。常向量 ⇒ 分子 0 ⇒ 梯度把策略推离常量。
            _vw = float(getattr(v3, "var_weight", 0.0) or 0.0)
            if _vw > 0.0:
                _dmean = diff_a.mean(dim=0)                      # (B,29) 每窗口平均改变
                _mu = _dmean.mean(dim=0)                         # (29,)
                _res = _dmean - _mu                              # (B,29)
                _num = (_res.pow(2).mean() + 1e-12).sqrt()
                _den_raw = (_mu.pow(2).mean() + 1e-12).sqrt()
                # ★ 2026-11-06 11:20：分母加下限，堵掉"把均值压小"这条捷径。
                #   实测证据（VAR_WEIGHT=0.1、无下限、step50 的落地探针，24 请求 × 3 噪声）：
                #     残差/均值 0.097 → 0.391（看着变好），但 ‖Δa‖ 0.132 → 0.0069（0.05×η，
                #     要求 0.10–0.20），效应/噪声 1.49 → 0.33 ⇒ 闭环里根本测不到。
                #   原因：-log(rms(残)/rms(均值)) 里"压小均值"和"做大残差"等价，而压小更省力。
                #   钳住分母后，均值低于下限时该项对 _mu 的梯度为 0 ⇒ 只能靠"做大残差"降 loss，
                #   均值回到"奖励推力 vs act 刹车"的自然平衡（v13 实测 rms(mean)=0.0245）。
                #   P1B_MU_FLOOR=0.02 ⇒ ‖Δa‖_L2 ≈ sqrt(29)·0.02 ≈ 0.108，正好是 G5 区间下界。
                _mu_floor = float(__import__("os").environ.get("P1B_MU_FLOOR", "0.02") or 0.02)
                _den = _den_raw.clamp_min(max(_mu_floor, 1e-3))
                _ratio = _num / _den
                loss_var = -_vw * torch.log(_ratio + 1e-3)
                loss = loss + loss_var
                self._var_ratio = float(_ratio.detach())
                self._var_mu_rms = float(_den_raw.detach())

        # ★ 2026-11-06（一次性判读探针，J1_GRAD_PROBE=1 才跑）：量"**理想修正方向**"本身
        #   有没有状态依赖。做法与 action_target 完全同口径：在 base 动作处对 20 步想象回报
        #   求梯度 ĝ = ∂J/∂A（(H,B,29)），再算**跨窗口**的 残差/均值。
        #   为什么需要：四种参数化（LoRA/反常量项/state_extra/delta-token）全都收敛到
        #   常向量，唯一还没量过的就是"最优点本身是不是常数"。
        #     残差/均值 ≈0.1 ⇒ 最优点就是常数 ⇒ 这条路从目标上就死了，别再训。
        #     残差/均值 >0.5 ⇒ 最优点状态相关 ⇒ 是学习信号/幅度的问题。
        # ★★ 2026-11-06（**可学性**判读，J1_GRAD_DUMP=<dir> 才跑，每步都跑）：
        #   前面所有 R² 都是**线性**（ridge）的，全 ≤0 只证明"线性不可解"，不证明"不可学"。
        #   ∂r/∂a 里含 FK 雅可比与 huber 分段符号 ⇒ 天然强非线性，线性模型本来就测不出来。
        #   这里把**单步**梯度 ĝ1 = ∂r_0/∂a_0（无 Jacobian 连乘、最干净的对象）连同可观测输入
        #   一起 dump，离线用**和 delta-token 同量级的 MLP** 做 8 折 CV：
        #     R²_mlp 高 ⇒ 有可学结构，训练失败是数据/优化/幅度问题（继续投入）
        #     R²_mlp ≈0 ⇒ ĝ 对可观测输入是高维噪声，整条"梯度蒸馏"路线从目标上就死
        _dumpdir = __import__("os").environ.get("J1_GRAD_DUMP", "")
        if _dumpdir:
            try:
                _rr = locals().get("roll_ref", None)
                if _rr is None:
                    raise RuntimeError("没有 roll_ref")
                with torch.enable_grad():
                    _A1 = _rr.actions.detach().float().clone().requires_grad_(True)
                    _s1 = batch.start_state.to(self.device).float()
                    _n1 = self.bundle.step(_s1, _A1[0], batch.xi, step_idx=1)
                    _r1 = self.bundle.reward(_n1, _A1[0], _A1[0], batch.references[0])
                    _J1 = _r1["total"].sum()
                    _g1 = torch.autograd.grad(_J1, _A1)[0][0]              # (B,29)
                _X = {
                    "state": batch.start_state.detach().float().cpu().numpy(),
                    "act0": _rr.actions[0].detach().float().cpu().numpy(),
                    "feats": self._delta_token_features(batch).detach().float().cpu().numpy(),
                }
                _pt2 = getattr(batch, "port_tokens", None)
                if _pt2 is not None:
                    _X["token"] = _pt2.detach().float().mean(1).cpu().numpy()
                _fr2 = getattr(batch, "frames", None)
                if _fr2 is not None:
                    _X["ref0"] = self.bundle.ref_state_at(
                        _fr2.to(self.device).long()).detach().float().cpu().numpy()
                _X["g1"] = _g1.detach().float().cpu().numpy()
                _X["r1"] = _r1["total"].detach().float().cpu().numpy()
                # ★★ `J1_GRAD_DUMP_TOK=1`：**token 空间**的理想修正 ∂J/∂token。
                #   这不是 ∂J/∂A 换皮：Δtoken 该学的目标就是它（含解码器雅可比 J_dec^T ĝ）。
                #   一次 backward 即得，比 20 次 Jacobian 连乘干净。用它回答"原路线（Δ 加在
                #   token 上）到底有没有可学信息"，而不是靠 A 空间的结果反推。
                if __import__("os").environ.get("J1_GRAD_DUMP_TOK", "0") == "1":
                    from diff_vla_wbc.rollout import imagine_with_tokens as _iwt  # noqa: PLC0415
                    _tp = getattr(batch, "port_tokens", None)
                    if _tp is None:
                        raise RuntimeError("没有 port_tokens")
                    if _tp.ndim == 2:
                        _tp = _tp[:, None, :]
                    _tg = _tp.detach().float().clone().requires_grad_(True)
                    with torch.enable_grad(), self._wbc_ref_ctx():
                        _ro = _iwt(
                            self.bundle, tokens=_tg, state=batch.start_state,
                            obs_hist=batch.obs_hist, history=batch.history, xi=batch.xi,
                            references=batch.references, prev_action=batch.prev_action,
                            horizon=int(cfg.horizon),
                        )
                        _g1t = torch.autograd.grad(_ro.rewards[0].sum(), _tg, retain_graph=True)[0]
                        _gHt = torch.autograd.grad(_ro.rewards.sum(), _tg)[0]
                    _X["g_tok1"] = _g1t.detach().float().mean(1).cpu().numpy()   # (B,64)
                    _X["g_tokH"] = _gHt.detach().float().mean(1).cpu().numpy()   # (B,64)
                    _X["actH"] = _ro.actions.detach().float().transpose(0, 1).reshape(
                        _ro.actions.shape[1], -1).cpu().numpy()                 # (B,H*29)
                import os as _os  # noqa: PLC0415
                _os.makedirs(_dumpdir, exist_ok=True)
                _n = int(getattr(self, "_gdump_n", 0))
                import numpy as _np2  # noqa: PLC0415
                _np2.savez(f"{_dumpdir}/dump_{_n:03d}.npz", **_X)
                self._gdump_n = _n + 1
                print(f"[gradump] {_dumpdir}/dump_{_n:03d}.npz n={_X['g1'].shape[0]} "
                      f"|g1|={float(_np2.abs(_X['g1']).mean()):.5f}", flush=True)
            except Exception as _e3:  # noqa: BLE001
                print(f"[gradump] FAILED: {_e3}", flush=True)
                self._gdump_n = int(getattr(self, "_gdump_n", 0)) + 1

        if __import__("os").environ.get("J1_GRAD_PROBE", "0") == "1" and not getattr(self, "_gp_done", False):
            try:
                _ref_roll = locals().get("roll_ref", None)
                if _ref_roll is None:
                    raise RuntimeError("没有 roll_ref（需要 trust_center=base 且 act_eps>0）")
                with torch.enable_grad():
                    _A = _ref_roll.actions.detach().float().clone().requires_grad_(True)
                    _s = batch.start_state.to(self.device).float()
                    _J = torch.zeros((), device=self.device)
                    for _t in range(int(cfg.horizon)):
                        _nxt = self.bundle.step(_s, _A[_t], batch.xi, step_idx=_t + 1)
                        _r = self.bundle.reward(_nxt, _A[_t], _A[max(_t - 1, 0)],
                                                batch.references[_t])
                        _J = _J + _r["total"].sum()
                        _s = _nxt
                    _g = torch.autograd.grad(_J, _A)[0]            # (H,B,29)
                _gm = _g.mean(0)                                   # (B,29) 每窗口平均方向
                _mu = _gm.mean(0)
                _res = _gm - _mu
                _num = float(_res.pow(2).mean().sqrt())
                _den_rms = float(_mu.pow(2).mean().sqrt())
                _den_l2 = float(_mu.norm())
                # ★ 附加：**特征能解释多少 ĝ 的状态依赖**（8 折 CV 的 R²，避免 in-sample 虚高）
                def _cv_r2(X, Y, folds=8, lam=1e-3):
                    import numpy as _np
                    X = _np.asarray(X, _np.float64); Y = _np.asarray(Y, _np.float64)
                    if X.ndim == 1:
                        X = X[:, None]
                    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
                    X = _np.concatenate([X, _np.ones((len(X), 1))], 1)
                    n = len(X)
                    idx = _np.arange(n); _np.random.default_rng(0).shuffle(idx)
                    ss_res = ss_tot = 0.0
                    for f in range(min(folds, n)):
                        te = idx[f::folds]; tr = _np.setdiff1d(idx, te)
                        W = _np.linalg.solve(X[tr].T @ X[tr] + lam * _np.eye(X.shape[1]),
                                             X[tr].T @ Y[tr])
                        ss_res += float(((Y[te] - X[te] @ W) ** 2).sum())
                        ss_tot += float(((Y[te] - Y[tr].mean(0)) ** 2).sum())
                    return 1.0 - ss_res / max(ss_tot, 1e-12)

                import numpy as _np  # noqa: PLC0415

                _Y = _gm.detach().float().cpu().numpy()               # (B,29) 每窗口 ĝ
                _fdev = self._delta_token_features(batch).detach().float().cpu().numpy()  # (B,35)
                _pt = getattr(batch, "port_tokens", None)
                _ftok = (None if _pt is None
                         else _pt.detach().float().mean(1).cpu().numpy())  # (B,64)
                _r2 = {"gradprobe/r2_dev": _cv_r2(_fdev, _Y)}
                if _ftok is not None and _ftok.shape[0] == _Y.shape[0]:
                    _r2["gradprobe/r2_token"] = _cv_r2(_ftok, _Y)
                    _r2["gradprobe/r2_all"] = _cv_r2(_np.concatenate([_ftok, _fdev], 1), _Y)
                # ★ 关键对照：`xi` 是 rollout 的**潜变量**（训练时给、部署时没有）。
                #   如果 ĝ 的状态依赖其实是 xi 驱动的，那它不是"状态相关"，而是
                #   "训练期潜变量相关" ⇒ 部署端根本没有这个信息，谁也学不出来。
                _xi = getattr(batch, "xi", None)
                if _xi is not None:
                    _fx = _xi.detach().float().reshape(_Y.shape[0], -1).cpu().numpy()
                    _r2["gradprobe/r2_xi"] = _cv_r2(_fx, _Y)
                    _r2["gradprobe/r2_all_xi"] = _cv_r2(
                        _np.concatenate([_ftok if _ftok is not None else _fdev, _fdev, _fx], 1), _Y)
                    _r2["gradprobe/xi_dim"] = float(_fx.shape[1])
                # ★ 决定性对照：把**完整 71 维状态**当特征。
                #   r2_state 高 ⇒ 是我们喂的特征太少（修法明确）；≈0 ⇒ 跨窗口的 ĝ 变化主要是噪声。
                _fs = batch.start_state.detach().float().cpu().numpy()
                _r2["gradprobe/r2_state"] = _cv_r2(_fs, _Y)
                _r2["gradprobe/r2_state_dev"] = _cv_r2(
                    _np.concatenate([_fs, _fdev], 1), _Y)
                # ★ 历史：`obs_hist`(930) = WBC 的 10 帧观测历史。p>>n 必须强正则。
                _fh = getattr(batch, "obs_hist", None)
                if _fh is not None:
                    _fh = _fh.detach().float().cpu().numpy()
                    _r2["gradprobe/hist_dim"] = float(_fh.shape[1])
                    _r2["gradprobe/r2_hist_lam100"] = _cv_r2(_fh, _Y, lam=100.0)
                    _r2["gradprobe/r2_hist_lam1000"] = _cv_r2(_fh, _Y, lam=1000.0)
                    # 最近 2 帧（186 维，p<n 一点）
                    if _fh.shape[1] >= 186:
                        _r2["gradprobe/r2_hist_last2"] = _cv_r2(_fh[:, -186:], _Y, lam=20.0)
                    # 用 PCA 压到 24 维（在全体上拟合，轻微乐观，仅作趋势）
                    _Xc = _fh - _fh.mean(0)
                    _U, _S, _Vt = _np.linalg.svd(_Xc, full_matrices=False)
                    _P = _Vt[:24].T
                    _r2["gradprobe/r2_hist_pca24"] = _cv_r2(_Xc @ _P, _Y, lam=1.0)
                    _r2["gradprobe/r2_hist_pca24_dev"] = _cv_r2(
                        _np.concatenate([_Xc @ _P, _fdev], 1), _Y, lam=1.0)
                # ★ 2026-11-06（B 假设）：**完整 20 步参考**当特征 —— token 只压了 64 维均值，
                #   丢掉的是"未来要做什么"的细节。它才是 ĝ 最可能的真正条件。
                # `FrameRef` 只有帧号 ⇒ 用参考库现取未来 H 步的参考状态
                _frames = getattr(batch, "frames", None)
                if _frames is not None:
                    _dev_ = self.bundle.device if hasattr(self.bundle, "device") else self.device
                    _base_f = _frames.to(_dev_).long()
                    _rows = []
                    for _t in range(int(cfg.horizon)):
                        _rf = self.bundle.ref_state_at((_base_f + _t).clamp(max=10**9))
                        _rows.append(_rf.detach().float().cpu().numpy())
                    _fr = _np.concatenate(_rows, 1)                     # (B, H*D)
                    _r2["gradprobe/ref_dim"] = float(_fr.shape[1])
                    _r2["gradprobe/r2_ref_lam1000"] = _cv_r2(_fr, _Y, lam=1000.0)
                    _Xr = _fr - _fr.mean(0)
                    _Ur, _Sr, _Vtr = _np.linalg.svd(_Xr, full_matrices=False)
                    _Pr = _Vtr[:24].T
                    _r2["gradprobe/r2_ref_pca24"] = _cv_r2(_Xr @ _Pr, _Y, lam=1.0)
                    _r2["gradprobe/r2_ref_pca24_dev"] = _cv_r2(
                        _np.concatenate([_Xr @ _Pr, _fdev], 1), _Y, lam=1.0)
                    _r2["gradprobe/r2_ref_var24"] = float(
                        (_Sr[:24] ** 2).sum() / max((_Sr ** 2).sum(), 1e-12))
                self._gp = _r2
                self._gp.update({
                    "gradprobe/resid_ratio_l2": _num / max(_den_l2, 1e-9),
                    "gradprobe/resid_ratio_rms": _num / max(_den_rms, 1e-9),
                    "gradprobe/mean_norm_l2": _den_l2,
                    "gradprobe/resid_norm": _num,
                    "gradprobe/n_win": float(_gm.shape[0]),
                })
                self._gp_done = True
                print("[gradprobe] " + " ".join(f"{k}={v:.4f}" for k, v in self._gp.items()),
                      flush=True)

                # ★★ k-sweep（J1_GRAD_PROBE_K="2,5,10,20"）：**视界扫掠**
                #   动机：20 步的 ĝ 是 20 次 Jacobian 连乘，混沌系统里长视界梯度基本是噪声，
                #   所以"R²≈0"可能只是"20 步太远"，而不是"没有可学信号"。
                #   对每个 k 单独求 ĝ_k = ∂J_k/∂A[0:k]，报 残差/均值(L2) 与 dev/state/all 的 CV R²。
                #   判据：k 小（2~3）时 R² 明显 >0 ⇒ 短视界可学 ⇒ 改训练目标/缩短 horizon 有意义；
                #        全 k 都 ≈0 ⇒ 梯度噪声是本质的，别再在这个方向上投入。
                _kenv = __import__("os").environ.get("J1_GRAD_PROBE_K", "")
                if _kenv and not getattr(self, "_gpk_done", False):
                  try:
                    _ksweep = {}
                    _H = int(cfg.horizon)
                    for _k in [int(v) for v in _kenv.split(",") if v.strip()]:
                        if _k < 1 or _k > _H:
                            continue
                        with torch.enable_grad():
                            _Ak = _ref_roll.actions.detach().float().clone().requires_grad_(True)
                            _sk = batch.start_state.to(self.device).float()
                            _Jk = torch.zeros((), device=self.device)
                            for _t in range(_k):
                                _nxk = self.bundle.step(_sk, _Ak[_t], batch.xi, step_idx=_t + 1)
                                _rk = self.bundle.reward(_nxk, _Ak[_t], _Ak[max(_t - 1, 0)],
                                                         batch.references[_t])
                                _Jk = _Jk + _rk["total"].sum()
                                _sk = _nxk
                            _gk = torch.autograd.grad(_Jk, _Ak)[0]          # (H,B,29)
                        _gmk = _gk[:max(_k // 2, 1)].mean(0)                # 只用前 k/2 步，避免混入无关步
                        _muk = _gmk.mean(0)
                        _resk = _gmk - _muk
                        _numk = float(_resk.pow(2).mean().sqrt())
                        _denk = float(_muk.norm())
                        _Yk = _gmk.detach().float().cpu().numpy()
                        _ksweep[f"gradsweep/k{_k}_resid_l2"] = _numk / max(_denk, 1e-9)
                        _ksweep[f"gradsweep/k{_k}_mean_l2"] = _denk
                        _ksweep[f"gradsweep/k{_k}_r2_dev"] = _cv_r2(_fdev, _Yk)
                        _ksweep[f"gradsweep/k{_k}_r2_state"] = _cv_r2(_fs, _Yk)
                        if _ftok is not None and _ftok.shape[0] == _Yk.shape[0]:
                            _ksweep[f"gradsweep/k{_k}_r2_all"] = _cv_r2(
                                _np.concatenate([_ftok, _fdev], 1), _Yk)
                        # ★★ 决定性对照：**动作自己**能不能解释 ĝ 的跨窗口变化？
                        #   前面 state/token/hist/ref 全都解释不了；但 ĝ 是在**不同动作序列**处求的
                        #   （每个窗口的 A0 不同），而 ∂r/∂a 天然强烈依赖 a 本身（奖励是 huber 距离）。
                        #   ⇒ 若 r2_act 高：最优修正 = f(当前动作)，而"当前动作"部署端**有**
                        #      ⇒ 修法明确：delta-token 的输入加上动作（现在是 state64）。
                        #   ⇒ 若 r2_act 也 ≈0：跨窗口的 ĝ 差异是**WM 梯度噪声**，任何输入都学不出来。
                        _Anp = _ref_roll.actions.detach().float().cpu().numpy()   # (H,B,29)
                        _Bk = _Anp.shape[1]
                        _act1 = _Anp[0]                                          # (B,29)
                        _ksweep[f"gradsweep/k{_k}_r2_act1"] = _cv_r2(_act1, _Yk)
                        _ksweep[f"gradsweep/k{_k}_r2_act1_state"] = _cv_r2(
                            _np.concatenate([_act1, _fs], 1), _Yk, lam=10.0)
                        if _k >= 2:
                            _act2 = _Anp[:2].transpose(1, 0, 2).reshape(_Bk, -1)  # (B,58)
                            _ksweep[f"gradsweep/k{_k}_r2_act2"] = _cv_r2(_act2, _Yk, lam=1.0)
                        if _k >= 5:
                            _actH = _Anp.transpose(1, 0, 2).reshape(_Bk, -1)      # (B,H*29)
                            _ksweep[f"gradsweep/k{_k}_r2_actH"] = _cv_r2(_actH, _Yk, lam=100.0)
                    if _ksweep:
                        self._gp.update(_ksweep)
                        print("[gradsweep] " + " ".join(
                            f"{k}={v:.4f}" for k, v in _ksweep.items()), flush=True)
                    self._gpk_done = True
                  except Exception as _e2:  # noqa: BLE001
                    self._gpk_done = True
                    print(f"[gradsweep] FAILED: {_e2}", flush=True)
            except Exception as _e:  # noqa: BLE001
                self._gp = {"gradprobe/err": 1.0}
                self._gp_done = True
                print(f"[gradprobe] FAILED: {_e}", flush=True)

        if loss_hand is not None:
            loss = loss + v3.lambda_hand * loss_hand
        if loss_token_bc is not None:
            loss = loss + float(v3.token_bc_weight) * loss_token_bc
        # ★ 第 3 刀：稠密 ĝ 监督（grad_target_weight，默认 0 = 老行为逐位不变）
        _gt_loss, _gt_meta = self._grad_target_loss(batch)
        if _gt_loss is not None:
            loss = loss + float(v3.grad_target_weight) * _gt_loss
            self._grad_target_meta = dict(_gt_meta)

        # ---------- P1：动作空间目标（2026-11-05，v2 修正）----------
        #  ⚠️ v1 的 bug：锚点用了**当前动作** A0 = rollout.actions，于是
        #     A_tgt = A0 + η·ĝ，损失 ‖a − A_tgt‖² 对参数**恒等于 η²‖ĝ‖²**（梯度为 0）
        #     —— 实测 `loss/action_target` 全程钉在 0.0225 = 0.15²，就是它空转的证据。
        #  修正：锚点换成**base 策略的动作**（固定、不自指），目标 = base 动作 + η·(WM 在 base 动作处的梯度)。
        #     这样 "回归目标" 才是策略该朝的那个"改进后的动作"，且不随参数一起走。
        loss_at = None
        _atw = float(getattr(v3, "action_target_weight", 0.0) or 0.0)
        if _atw > 0.0:
            _eta = float(getattr(v3, "action_target_eta", 0.15) or 0.15)
            _ref_roll = locals().get("roll_ref", None)
            if _ref_roll is None:
                raise SystemExit(
                    "[P1] action_target 需要 trust_center=base/prev 先算出的 base rollout（roll_ref）；"
                    " 请保持 --act-eps>0 且 --trust-center base")
            A0 = _ref_roll.actions.detach()          # base 的动作（固定锚点）
            A_free = A0.clone().requires_grad_(True)
            _s = batch.start_state
            _J = torch.zeros((), device=A0.device)
            for _t in range(int(cfg.horizon)):
                _nxt = self.bundle.step(_s, A_free[_t], batch.xi, step_idx=_t + 1)
                _r = self.bundle.reward(_nxt, A_free[_t],
                                        A_free[max(_t - 1, 0)], batch.references[_t])
                _J = _J + _r["total"].sum()
                _s = _nxt
            _g = torch.autograd.grad(_J, A_free)[0]
            _rms = _g.pow(2).mean().sqrt().clamp_min(1e-9)
            A_tgt = (A0 + _eta * _g / _rms).detach()
            # 只用**前 k 步**：base 与当前策略的想象轨迹从第 6 步起会分叉，
            # 那时"A_tgt[t] 在哪个状态上"与"rollout.actions[t] 在哪个状态上"不再是同一个状态。
            # 前几步状态重合，回归目标才严格有意义。
            _k = int(getattr(v3, "action_target_steps", 20) or 20)
            _k = max(1, min(_k, int(rollout.actions.shape[0])))
            # ★ 2026-11-05 深夜（bias-正交）：probe 实测 FT 学到的 Δa 是一个**近似恒定偏置**
            #   （逐请求 cos(Δa_i, mean)=0.98），而训练目标里含有逐状态的 ĝ。
            #   原因很可能是：回归损失里"学一个 batch 常量偏置"就能吃掉大部分误差。
            #   `HA_AT_CENTER=1` 时把**预测与目标同时减去 batch 均值**（沿 batch 维 B）：
            #   常量偏置在损失里不再有任何收益，只剩"逐状态形状"可学。
            #   指标 `*/resid_ratio` 用来检查目标本身是否有状态依赖（太小 ⇒ centering 无意义）。
            _D_pred = rollout.actions[:_k] - A0[:_k]
            _D_tgt = A_tgt[:_k] - A0[:_k]
            _pred_mu = _D_pred.mean(dim=1, keepdim=True)
            _tgt_mu = _D_tgt.mean(dim=1, keepdim=True)
            _center = __import__("os").environ.get("HA_AT_CENTER", "0") == "1"
            if _center:
                loss_at = _atw * ((_D_pred - _pred_mu) - (_D_tgt - _tgt_mu)).pow(2).mean()
            else:
                loss_at = _atw * (_D_pred - _D_tgt).pow(2).mean()
            loss = loss + loss_at
            _pred_res = (_D_pred - _pred_mu).pow(2).mean().sqrt()
            _tgt_res = (_D_tgt - _tgt_mu).pow(2).mean().sqrt()
            metrics_at = {
                "loss/action_target": float(loss_at.detach()),
                "action_target/rms": float(A_tgt.sub(A0).pow(2).mean().sqrt()),
                "action_target/grad_rms": float(_rms),
                "action_target/center_on": float(_center),
                "action_target/pred_mean_norm": float(_pred_mu.pow(2).mean().sqrt()),
                "action_target/pred_resid_norm": float(_pred_res),
                "action_target/tgt_mean_norm": float(_tgt_mu.pow(2).mean().sqrt()),
                "action_target/tgt_resid_norm": float(_tgt_res),
                "action_target/pred_resid_ratio": float(_pred_res / _pred_mu.pow(2).mean().sqrt().clamp_min(1e-9)),
                "action_target/tgt_resid_ratio": float(_tgt_res / _tgt_mu.pow(2).mean().sqrt().clamp_min(1e-9)),
            }
        else:
            metrics_at = {}

        metrics: dict[str, float] = {
            "loss/reward_norm": float(loss_r.detach()),
            "loss/reward_raw": float(ret.mean().detach()),
            "loss/boot": float(boot.mean().detach()),
            "rollout/reward": float(rollout.rewards.mean().detach()),
            "action_var/ratio": float(getattr(self, "_var_ratio", float("nan"))),
            # 命令层残差：‖Δtoken‖（部署端同口径，>0 才说明这条路在动）
            "delta_token/norm": float(getattr(self, "_dt_norm", float("nan")) if self._dt_norm is not None else float("nan")),
            **getattr(self, "_gp", {}),
            **getattr(self, "_grad_target_meta", {}),
            # 状态敏感度 = 打乱状态后 Δ 的变化 / Δ 本身。≈0 ⇒ g 没在用状态（这条路无效）
            "delta_token/state_sens": float(getattr(self, "_dt_state_sens", float("nan"))),
            "rollout/survival": float(rollout.survival[-1].mean().detach()),
            "codec/out_of_range": float(out_of_range),
            "loss": float(loss.detach()),
            **getattr(self, "_quant_main", {}),
            **metrics_at,
        }
        # ★ 2026-11-06：量"策略轨迹 vs base 轨迹"的分叉，用来**决定 action_target_steps**
        #   现在这个 k=6 是抄代码注释的，没实测。这里把 t=5/10/20 的位置差与关节差打出来，
        #   下一轮就能按"位置差 <1cm 的最大 t"来定 k，而不是猜。
        #   ⚠️ 2026-11-06 10:25 修复：这段原本写在 `metrics = {...}` **之前**，一旦
        #   roll_ref 存在（trust_center=base 且 act_eps>0，本轮的配置就是）就
        #   UnboundLocalError: cannot access local variable 'metrics' —— P1b 首跑即崩。
        #   现在挪到 metrics 建好之后。
        _rr = locals().get("roll_ref", None)
        if _rr is not None:
            with torch.no_grad():
                _st = rollout.states[1:]                 # (H, B, 71)
                _sr = _rr.states[1:]
                _dp = (_st[..., 0:3] - _sr[..., 0:3]).norm(dim=-1)          # (H,B) 米
                _dq = (_st[..., 7:36] - _sr[..., 7:36]).abs().mean(-1)      # (H,B) 弧度
                for _t in (4, 9, 19):
                    if _t < _dp.shape[0]:
                        metrics[f"diverge/pos_t{_t+1}"] = float(_dp[_t].mean())
                        metrics[f"diverge/joint_t{_t+1}"] = float(_dq[_t].mean())

        if loss_s is not None:
            metrics["loss/state"] = float(loss_s.detach())
        if loss_hand is not None:
            metrics["loss/hand_fm"] = float(loss_hand.detach())
        if loss_token_bc is not None:
            metrics["loss/token_bc"] = float(loss_token_bc.detach())
            metrics["token_bc/weight"] = float(v3.token_bc_weight)

        if loss_anchor is not None:
            metrics["loss/anchor"] = float(loss_anchor.detach())
        metrics.update(_adv_stats)
        if getattr(self, "state_extra_scale", None) is not None:
            _sc = self.state_extra_scale.detach()
            metrics["state_extra/scale_absmax"] = float(_sc.abs().max())
            metrics["state_extra/scale_mean"] = float(_sc.mean())
            metrics["state_extra/scale_dim_max"] = int(_sc.abs().argmax())
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

        # ---------- Path B：双 KL 锚（默认 0 ⇒ 与 Path A / 历史 joint 完全一致）----------
        _wkl = float(getattr(v3, "wbc_kl_anchor", 0.0) or 0.0)
        if _wkl > 0 and rollout.anchor_actions is not None:
            # 动作空间 KL 代理：同状态、同 token 下，当前 WBC vs 起点 WBC。
            loss_wbc_kl = _wkl * (rollout.actions.float()
                                  - rollout.anchor_actions.float()).pow(2).mean()
            loss = loss + loss_wbc_kl
            metrics["loss/wbc_kl"] = float(loss_wbc_kl.detach())
            metrics["wbc_kl/weight"] = _wkl

        _vkl = float(getattr(v3, "vla_kl_anchor", 0.0) or 0.0)
        if _vkl > 0 and self._vla_ref is not None:
            # token 空间 KL 代理：同一 obs、同一 noise 下，当前 action head vs 起点 action head。
            # ⚠️ 不能在 torch.no_grad() 里调 `_sample_tokens`：`sample_once_with_grad` 会显式
            #   拒绝 no_grad（否则 joint 图会被切断）。起点 head 的参数已 requires_grad=False，
            #   所以正常 grad 上下文里采样、随后 detach 即可，不会给起点 head 累积梯度。
            _live_head = self.vlm.action_head
            self.vlm.action_head = self._vla_ref
            try:
                _, _, tokens0 = self._sample_tokens(act_in, bb_out, noise=_noise)
            finally:
                self.vlm.action_head = _live_head
            loss_vla_kl = _vkl * (tokens.float() - tokens0.detach().float()).pow(2).mean()
            loss = loss + loss_vla_kl
            metrics["loss/vla_kl"] = float(loss_vla_kl.detach())
            metrics["vla_kl/weight"] = _vkl

        # ---------- 反传 / 累积 / 裁剪 ----------
        accum = max(1, int(cfg.grad_accum))
        self._accum_count += 1
        if self._accum_count == 1:
            self.optimizer.zero_grad(set_to_none=True)
        (loss / accum).backward()
        if self._accum_count >= accum:
            # ★ DDP：只在累积边界做一次梯度 all_reduce（等价于全局 batch = 各卡之和）。
            #   `_sync_gradients` 原来**从未被调用**过（方法写了但没接线），这里接上。
            if self.sync_gradients:
                self._sync_param_grads()
            # ★ 2026-11-01：state_extra_scale 是单独 param group，裁剪也要覆盖它
            _clip_params = list(self.fm_parameters)
            if getattr(self, "state_extra_scale", None) is not None:
                _clip_params.append(self.state_extra_scale)
            # ★ 2026-11-06：delta-token 也必须在裁剪列表里（lr=5e-4，比 LoRA 大 50×）
            if getattr(self, "delta_token_net", None) is not None:
                _clip_params += list(self.delta_token_net.parameters())
            total = torch.nn.utils.clip_grad_norm_(_clip_params, v3.clip_head)
            metrics["grad/head_preclip"] = float(total)
            # **熔断**（2026-10-30 事故）：一旦梯度出现 NaN/inf 立刻停，不要继续烧下去。
            # 那次是第 535 步炸的，但因为没熔断，白烧了 266 步（~15 min）还写出一个 NaN ckpt。
            if not torch.isfinite(torch.as_tensor(total)):
                raise RuntimeError(
                    f"[NaN-guard] step={self.steps} grad/head_preclip={float(total)} 非有限 ⇒ 立即中止。"
                    " 检查奖励/损失里是否有 log-map / acos / sqrt(0) 之类的奇点。")
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_count = 0
        self.steps += 1
        _mark("bwd_opt")

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
        metrics["critic/v_adv_H_raw"] = float(metrics_v_raw)
        self.last_metrics = metrics
        if _prof:
            _parts, _prev = [], _marks[0][1]
            for _n, _ts in _marks[1:]:
                _parts.append(f"{_n}={_ts - _prev:.2f}s")
                _prev = _ts
            print("[prof-step] " + " ".join(_parts), flush=True)
        return metrics

    # ------------------------------------------------------------------ WM 保鲜
    def refresh_wm(self, data: dict, *, steps: int = 200, batch: int = 2048,
                   lr: float = 1e-4, vel_weight: float = 5.0, kl_anchor: float = 1e-3,
                   log=None) -> dict:
        from .wm_refresh import refresh_wm  # noqa: PLC0415

        out = refresh_wm(self.bundle.wm, data, steps=steps, batch=batch, lr=lr,
                         vel_weight=vel_weight, kl_anchor=kl_anchor, device=self.device,
                         log=self.log if log is None else log,
                         # 后端无关：mjlab 的 WM 直接吃/吐 flat；trackerlab 的由 bundle.step 适配
                         step_fn=self.bundle.step)
        from .rollout import freeze_world_model  # noqa: PLC0415

        freeze_world_model(self.bundle.wm)
        return out


__all__ = ["JointTrainerV3", "V3Config"]
