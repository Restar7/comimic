# 论文要点（FM 梯度反传消融）

> 抓取时间：2026-10-24（arXiv 摘要页）。投稿前请再核对卷号/版本/作者。

## 1. Q-VGM（本目录 `qvgm_lite` 的来源）

- **标题**：Q-VGM: Q-Guided Value-Gradient Matching for Offline-to-Online RL of Flow-Matching VLA Policies
- **作者**：Ziqian Wang, Rui Zhang, Yitian Liu, Xingjian Mao, Minqian Wang, Yao Mu
- **arXiv**：2606.08015（v5, 2026-09-17）
- **URL**：https://arxiv.org/abs/2606.08015

### 摘要要点

- 直接对 flow-matching 策略做 critic 梯度需要穿过多步 denoising（BPTT），在 VLA 规模上**昂贵且不稳定**；
- 从**最优控制**视角出发，推导出"干净动作的值梯度"与"晚步局部 velocity 修正"之间的**局部近似**；
- 用这个关系构造 critic-guided target，通过 **velocity matching** 微调 action expert：
  **梯度只穿过局部 velocity 预测，不穿过整条 denoising 轨迹**；
- critic 是 **action-sensitive chunk critic**，离线用 **IQL**、在线用 **TD** 训练；
- 结果：LIBERO few-shot SFT 四套平均 84.6% → 90.7%（离线，每套 150 rollouts）→ 98.4%（在线）；
  相对 on-policy RL 样本效率 5.3×；真实双臂任务 66.7% → 98.3%。

### 我们的实现（`backends.py::QVGMLiteBackend`）

```text
k ~ late step (默认后 50%)
a_k = Euler(a_0, k)                       # no_grad
v   = v_theta(a_k, k)                     # 唯一保留图的局部 velocity
a_hat = a_k + (1 - t_k) * v               # 一阶干净动作估计
loss = -Q(a_hat)                          # 升值；梯度只回传 v 的这次调用
```

- 这是论文 **local approximation** 的最简复现，**不是官方代码**（官方未公开）；
- 官方 critic（IQL/TD、action-sensitive chunk）未实现；本目录提供：
  - `OracleCritic`：`Q(a) = -task_loss(a)`，用于验证"值梯度方向"是否正确；
  - `Critic`：MLP 回归 `-task_loss(a_K)`，用于展示"critic 不完美时会发生什么"。
- 实测（oracle，K=4/8）：与 pathwise 的 cosine ≈ 0.9998，rel err ≈ 0.5；tiny/lr=1e-2 下会过冲——
  说明局部近似**方向对、幅值敏感**，与论文强调 critic 训练/校准一致，不能据此否定 Q-VGM。

## 2. VITA（pathwise 的同类机制）

- **标题**：VITA: Vision-to-Action Flow Matching Policy
- **作者**：Dechen Gao, Boqi Zhao, Andrew Lee, Ian Chuang, Hanchu Zhou, Hang Wang, Zhe Zhao, Junshan Zhang, Iman Soltani
- **arXiv**：2507.13231（v4, 2026-03-03）
- **URL**：https://arxiv.org/abs/2507.13231

### 摘要要点

- flow 从**视觉表示**直接流到**动作 latent**（noise-free、conditioning-free），省去生成过程中的视觉条件注入；
- 动作自编码器把原始动作映射到与视觉 latent 同维的结构化空间，与 flow matching 联合训练；
- 为防止 latent 塌缩，提出 **flow latent decoding**：把动作重建 loss **反向穿过 flow matching ODE solver**；
- 9 个仿真 + 5 个真实任务，推理比常规条件式方法快 1.5–2×。

### 与我们的关系

`flow latent decoding` 的梯度机制就是 **unrolled Euler + autograd（pathwise）**，与
`diff_vla_wbc.gr00t_fm.sample_fm_from_features` 一致；差异在架构（视觉→动作 latent）与 loss（重建），
因此不单独做后端，而是在论文里作为"pathwise 梯度可行"的旁证。

## 3. Flow Policy Gradients / Potential-Guided Flow Matching（相关工作，未实现）

- **Flow Policy Gradients**（Yi et al., 2026）：用 reward-based 估计避免对 flow 求 action likelihood；
  本目录未实现，论文里只能作为相关工作引用，不能声称对比。
- **Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement**（arXiv:2606.04968）：
  用 potential 引导 flow 改进；同样未实现。

## 4. 离散伴随 / 梯度检查点（`adjoint` / `checkpointed` 的依据）

- **Neural ODE**：Chen, Rubanova, Bettencourt, Duvenaud, NeurIPS 2018 —— 连续伴随法；
  我们的 `adjoint` 是**离散伴随**（对 Euler 步做 reverse accumulation），不是连续时间 adjoint，
  与 BPTT 在数学上等价（浮点误差内），实现上不保存整条图、每步做一次 vjp。
- **Gradient checkpointing**：Griewank (1992) 的 recomputation 思想；PyTorch 实现为
  `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` —— 前向不存中间激活，反向重算。
- **Q-VGM 的动机**正是 BPTT 在 VLA 规模上的成本/不稳定性；`adjoint`/`checkpointed` 是"精确但换计算/显存"的两条路，
  与 Q-VGM 的"有偏但便宜"形成对照。

## 5. 引用注意

- 以上摘要来自 arXiv 页面，若论文正文引用，请补齐 venue/版本（例如 Q-VGM v5、VITA v4）；
- Q-VGM 的 "5.3× sample efficiency"、"98.3%" 等数字是**论文报告值**，不是我们在 G1/CoMimic 上的结果；
- 我们的 `qvgm_lite` 与官方实现的差异必须在论文中写明（reimplementation, oracle/regression critic）。
