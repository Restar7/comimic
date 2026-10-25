# Patch：把 mjlab 的 `explicit_integration` 移植回 trackerlab 的 WM

> 目标文件：`trackerlab-supermimic/source/trackerLab/trackerLab/algorithms/supermimic/networks.py`
> 背景与证据：`docs/HANDOVER_SUPERMIMIC_FINETUNE.md` §2.2.0
> **必须成对修改 `step` 和 `targets`**：`step` 是前向积分，`targets` 是它的逆；只改一个会让
> 网络输出与积分互相矛盾（训练直接崩）。

## 0. 前置：WM 结构与 mjlab 的逐项对比（开工前先确认状态）

| # | 机制 | mjlab (`SuperMimic-Reproduced`) | trackerlab/Isaac | 结论 |
|---|---|---|---|---|
| 1 | **显式积分** `dt×速度` 进位置/朝向/关节 | ✅ `explicit_integration=True`（`state_spec.py:55,103`） | ❌ 六路输出独立 | **缺失 ← 本 patch 要修的根因** |
| 2 | 输入特征尺度 | 固定 `in_scale=(1,1,1,2,1,5)`（`state_spec.py:54`） | 学到 `RunningNorm(in_norm)` | 设计差异，可接受 |
| 3 | 误差尺度 | 固定 `err_scale=(0.05,0.1,0.5,1.0,0.1,2.0)`（`state_spec.py:50`） | 学到 `out_norm.std()` + `ERR_FLOOR=(3e-3,5e-3,2e-2,5e-2,3e-3,5e-2)` | 比值接近（速度/位置 ≈10× vs 6.7×），可接受 |
| 4 | `xi` 归一化 | `xi_offset=(0.75, 0×29, 0×3)`、`xi_scale=(0.45, 0.01×29, 0.05×3)` | `normalize_xi`：`(xi₀−0.75)/0.45`、`xi₁₋₂₉/0.01`、`xi₋₃/0.05` | **完全一致** ✅ |
| 5 | WM 的动作输入 | ReplayBuffer 里的 `a_exec`（实际执行动作） | 训练侧同为 `out["a"]`；**离线 bundle 里由参考关节反算** | 训练侧一致；离线侧是我们自己的口径（已用 `COMIMIC_ACTION_SCALE` 解耦） |
| 6 | 状态布局 | `[p, q, qj, v, w, qd]` | canonical `[p,quat,v,w,q,qd]` ↔ flat `[p,quat,q,v,w,qd]` | 有显式转换函数，单位置测试通过 |
| 7 | `dt` 来源 | `cfg.dt`（task cfg） | 硬编码缺失 | 本 patch 用 `EXPLICIT_DT=0.02`（`control_dt`） |

**结论**：WM 里唯一"结构性丢失"是 #1；#2/#3 是"固定常数 vs 学习统计量"的等价设计；#4 逐位一致。
真正需要复核的是**离线侧**的两件事（不在 WM 里）：token 条件空间（WBC↔VLA）与
部署侧观测末 6 维是否置零（`docs/HANDOVER_ISAAC_CLOSEDLOOP.md` 记录的已知不对齐）。

## 0. 常量（放在 `NUM_J = ...` 附近）
```python
#: 控制周期。Isaac task 的 control_dt = 0.02（env.py: `meta_control_dt`）。
#: 只在 explicit integration 下使用，可用环境变量覆盖（便于 A/B）。
EXPLICIT_DT = float(__import__("os").environ.get("SUPERMIMIC_EXPLICIT_DT", "0.02"))
```

## 1. `WorldModel.step` —— 前（现状）
```python
    def step(self, S, a, xi_n):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n)))
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd)
```

## 2. `WorldModel.step` —— 后（半隐式积分，与 mjlab `state_spec.integrate` 一致）
```python
    def step(self, S, a, xi_n):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n)))
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        # explicit integration: 位置/朝向/关节 = 当前值 + dt×新速度 + 学习残差。
        # 速度错 → 位置立即错 → 速度通道被位置误差隐式监督（mjlab 原版行为）。
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        qd1 = qd + dqd
        dt = EXPLICIT_DT
        p1 = p + dt * v1 + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dt * w1 + dth)
        return (p1, R1, v1, w1, q + dt * qd1 + dq, qd1)
```

## 3. `WorldModel.targets` —— 前（现状）
```python
    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        from trackerLab.kinematics.supermimic.rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )
```

## 4. `WorldModel.targets` —— 后（step 的逆：先扣掉 dt×速度）
```python
    @staticmethod
    def targets(S, S1):
        """Frame-local *residuals* (inverse of ``step`` with explicit integration)."""
        from trackerLab.kinematics.supermimic.rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        dt = EXPLICIT_DT
        return torch.cat(
            (
                # p1 = p + dt*v1 + H @ dp  ->  dp = H^T (p1 - p - dt*v1)
                (H.transpose(-1, -2) @ (p1 - p - dt * v1).unsqueeze(-1)).squeeze(-1),
                # R1 = R @ so3_exp(dt*w1 + dth)  ->  dth = log(R^T R1) - dt*w1
                so3_log(Rt @ R1) - dt * w1,
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                # q1 = q + dt*qd1 + dq  ->  dq = q1 - q - dt*qd1
                q1 - q - dt * qd1,
                qd1 - qd,
            ),
            -1,
        )
```

## 5. 切换后必须做的事
1. **重跑/续训**：`dq` 的含义从"整步增量"变成"残差"，旧 `wm` 权重不能直接用（`pi` 可以不动）；
2. 建议先做 **200–500 步 WM 有监督预热**（ReplayBuffer 里的 `(s, a, s1)` 即可，不用仿真），
   确认 `wm_loss` 不跳变、`wm_err8` 回到 ~0.07 量级再开 RL；
3. `out_norm` 的统计量会自动跟着新的 `targets` 更新（`runner.py:446` 的 `wm.out_norm.update(...)`），
   不需要手工改；
4. 可选：把 `runner.py` 里 `ERR_FLOOR` 的 `dv/dw/dqd` 从 `2e-2/5e-2/5e-2` 调小到 `1e-2` 量级
   （它们比位置大 6–17 倍，本身也在压速度梯度）。

## 6. 验收
```bash
# 1) 速度通道（用 policy 自己的动作，或参考动作 + 新积分）
python3 scripts/finetune_wm_velocity.py --steps 0          # 看 [before] 的 qd_corr / qd_ratio
# 目标：qd_corr > 0.9，qd_ratio ∈ [0.9, 1.1]（改结构前是 0.155 / 2.55）
# 2) 端到端：用 policy 动作跑 8 步想象 rollout
python3 scripts/probe_port_tokens.py
# 目标：state_term 明显低于 0.18，rollout_reward ≥ 0.07
# 3) RL 侧：train_log.csv 的 sim_rew ≥ 0.84，term_frac ~0.002
```

## 7. 回退
不想动结构就回退到 `docs/HANDOVER_SUPERMIMIC_FINETUNE.md` §2.2.1（速度维加权 + 参考速度辅助项），
或直接不修、用 `train_joint.py --state-vel-weight 0.05`（本容器现状）。
