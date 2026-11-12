# 实验 C（LoRA16-FT8000）问题陈述：为什么「加 LoRA」几乎不改善，以及现在能确认/不能确认什么

> 写于 2026-11-08。目的：把**已经跑过的事实、已排除的解释、仍然开着的假设**摊开，
> 交给外部 reviewer（GPT）给出下一步实验设计。**所有数字都来自真实日志**，出处写在每节末尾。
> 状态口径：`PASS / FAIL / RUNNING / STOPPED / NOT_RUN`，没跑过的不写跑过。

---

## 0. 一句话结论

在「冻结 FT8000 + rank16 LoRA（attn1/attn2/ff）+ 全秩 proj_out，4 步可微 flow-matching，
H=20，目标 = 冻结世界模型的想象回报 J20」这套设置下：

* **整包可训参数（LoRA + proj_out）确实能提高 WM 回报**：val J 从 1.49194 → 1.52938（**+2.51%**）；
* **但这 +2.51% 里 ~87% 来自 proj_out（全秩 6.3 M），LoRA（16.8 M）只贡献 +0.0051（+0.34%）**；
* **把 LoRA 的学习率提高 10×（1e-5 → 1e-4）、batch 提 4×（16 → 64）并没有让 LoRA 的效果变好**，
  val ΔJ 反而从「单调爬到 +0.0055」变成「在 0.0002 ~ 0.0037 之间震荡」；
* 因此当前**不能**说「LoRA 这个机制没用」，只能说：**在这套目标函数（冻结 WM 的期望回报）
  与这个预算（400~800 步、batch 16~64、lr 1e-5~1e-4）下，LoRA 子空间里的梯度信噪比不足以
  支撑稳定的方向性改进**。

---

## 1. 系统结构（三条路线共用）

```
冻结 FT8000 VLA (Gr00tN1d7, 1,637,293,184 参数, fp32)
  ├─ backbone: Qwen3-VL 2B（冻结）        → 观测特征（每步 no_grad 缓存）
  └─ action_head: AlternateVLDiT 1.0917 B（冻结）
        ├─ 216 个 attn1/attn2/ff 的 nn.Linear → 替换为 LoRALinear(rank16, alpha32, scale=2.0)
        ├─ 输出块 proj_out_1 [3072,1536] + proj_out_2 [1024,1536]（**全秩可训**）
        └─ 4 步 Euler flow-matching → x_norm (B,40,132)
              ↓ codec.decode（可微 min-max 反归一化，**不 detach**）
        token = x_raw[..., :64]   (B,40,64)
              ↓
        SONIC decoder / WBC（冻结）→ 关节目标 q_raw = q_default + G1_ACTION_SCALE ⊙ a
              ↓
        A′ 世界模型（冻结，wm_gain2.pt, sha fc5aeb30…）→ H=20 步想象 → J20（折扣回报）
```

可训参数拆分（从 ckpt 精确数出）：
`LoRA 16,777,216 + proj_out 6,295,552 = 23,072,768`（占全模型 **1.409%**，216 层，rank16/alpha32）。
**zero-LoRA 恒等**：新注入时 `max|Δ| = 0.0`（逐位一致，已验证）。

**训练目标**：

```
L = -J20 + λ_lim·L_lim(q_raw) + λ_peak·L_peak(e_same) + λ_same·L_same(e_same) (+ λ_rate·L_rate=0)
```

* `L_lim = mean_b[0.5·max_{t,j} v² + 0.5·mean_t max_j v²]`，v = 不对称限位违反（按关节归一化）；
* `e_same = q_raw − q_base`，其中 `q_base` 来自**同一批帧、同一 flow 噪声、关闭 LoRA** 的底模 token；
* `L_peak` 是单关节预算 `max_{t,j} relu(|e|/b − 1)²`，`b = 3·|e|.mean`（clamp 0.01）；
* λ（在 200 步 Adapter 上按梯度预算标定）：`λ_lim = 0.0369`、`λ_peak = 2.06e-4`、
  `λ_same = 0.01`（未激活）、`λ_rate = 0`（BLOCKED：RobotSpec 无 velocity 字段）。

---

## 2. 评测口径（决定了"能不能看见效应"）

* **成对同噪声**：同一批帧用**同一 flow-matching 初噪**采两次 token（`lora_scale=1` 与 `=0`），
  `ΔJ = J_lora − J_base`。**必须这样做**，否则采样噪声（token rms ~1e-3）与效应量同阶；
* 固定评测集：`runs/temporal_ac_bptt_v5/eval_set`（448 窗口：256 train / 96 val / 96 test，
  375 train clips，**训练只允许 train clips，已 fail-closed 校验**）；
* 底模参照（pipeline 第一跑，fresh-noise，96 窗口）：
  `val J_base = 1.49194`（cached-token 1.50594）、`test J_base = 1.57558`（1.57581）、
  限位违反率 `2.77% / 2.59%`、`q_raw_absmax = 4.603 / 3.580`；
* **一个重要的口径副作用**：因为 proj_out 也被训，`lora_scale=0` 的"底模"其实包含**已训练的
  proj_out**，所以 `J_base` 会随训练漂移（实测 1.50409 → 1.52432，+1.3%）。因此：
  * "LoRA 边际" = 与"已训 proj_out 的底模"比（数值小，但干净）；
  * "整包效应" = 与 pristine FT8000（1.49194）比（数值大，含 proj_out）。

---

## 3. 跑过的实验（4 个）

| # | 配置 | 状态 | 结果摘要 |
|---|---|---|---|
| E1 | **pilot 200**：lr1e-5、bs16、λ 用 A 臂（λ_lim 0.0529 / λ_peak 8.4e-4） | PASS | val ΔJ +0.00164（200 步）、test +0.00198、cond PASS |
| E2 | **主跑 lr1e-5**：bs16、checkpointed FM、λ 重标（见上）、预算 800 | **STOPPED at 700/800**（人工停） | val ΔJ 峰值 **+0.00548**（650 步）；test@200 +0.00198；vio 2.63%→2.45%；dtok 0.0011→0.0042 |
| E3 | **DDP×2 主跑**：eff bs64、**lr1e-4**、400 步 | **STOPPED at ~340**（人工停，效果不佳） | val ΔJ **震荡**：50:+0.00023 / 100:+0.00239 / 150:+0.00036 / 200:+0.00144 / 250:+0.00372 / 300:+0.00285 |
| E4 | **归因分解**（16 帧、同 seed、it=600 Adapter） | PASS | proj_out rms **3.053e-2**、LoRA 边际 **3.212e-3**、整包 **3.275e-2** |

### 3.1 E2（lr=1e-5）val ΔJ 曲线

| it | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ΔJ | +0.00204 | +0.00094 | +0.00157 | +0.00164 | +0.00254 | +0.00363 | +0.00338 | +0.00408 | **+0.00526** | +0.00496 | +0.00518 | +0.00506 | **+0.00548** |
| Δtoken rms | 0.0011 | 0.0011 | 0.0015 | 0.0014 | 0.0019 | 0.0027 | 0.0029 | 0.0032 | 0.0038 | 0.0040 | 0.0043 | 0.0042 | — |
| 限位违反% | 2.62 | 2.73 | 2.63 | 2.63 | 2.58 | 2.51 | 2.55 | 2.52 | 2.46 | 2.47 | 2.45 | 2.46 | — |

cond 门禁 26/26 PASS（跨噪声 Δtoken cosine 0.86–0.96，dJ 恒正）。

### 3.2 E3（lr=1e-4、DDP、bs64）val ΔJ 曲线

| it | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| ΔJ | +0.00023 | +0.00239 | +0.00036 | +0.00144 | **+0.00372** | +0.00285 |
| Δtoken rms | 0.0044 | 0.0049 | 0.0051 | 0.0049 | 0.0049 | 0.0053 |
| 限位违反% | 2.32 | 2.04 | 2.32 | 2.20 | 2.19 | 2.19 |

### 3.3 E4 归因（最重要的证据）

| 对照 | token 变化 rms | 说明 |
|---|---|---|
| proj_out 单独（关 LoRA） | **3.053e-2** | 占绝对大头 |
| LoRA 边际（全开 vs 关 LoRA） | **3.212e-3** | 小一个数量级 |
| 整包 vs pristine | 3.275e-2 | ≈ proj_out 主导 |

WM 侧对应：pristine J = 1.49194 →「只留 proj_out」1.52432（**+0.0324**）→「全开」1.52938（**+0.0374**）。

### 3.4 观测量（训练日志）

* `lora_B` absmean **1.79e-4**、rms 2.28e-4（lr=1e-5 训 600 步后）；`lora_A` absmean 1.13e-2（≈初始化量级）；
* 梯度范数（clip 前）：典型 0.5–2.0，偶发 **9–108**（clip=1.0，所以尖峰步被强剪）；
* lr 调度：warmup 20 → cosine；lr=1e-5 预算 800 步时 **第 685 步 lr 已降到 1.47e-6**（峰的 15%）。

---

## 4. 已经**排除**的解释（每条都有证据）

1. **"梯度根本没接通 / 图被 detach"** —— 排除。四个门禁测试全绿：
   `test_four_step_fm_grad`（4 步 FM 的图存在、梯度回到 LoRA）、
   `test_sonic_wm_grad_to_lora`（**真实** A′ WM + 真实限位下，J20 的梯度回到 LoRA，且只有 LoRA/proj_out 有梯度）、
   `test_lora16_dummy`（恒等/存取）、真实 FT8000 审计（216 层、23,072,768、zero-LoRA `max|Δ|=0.0`）。
2. **"`also_train` 是 no-op，proj_out 其实没训"** —— 曾经是 bug，**已修**；E1–E4 都是修好之后跑的。
   （副作用：更早的 S3 实验里"LoRA+proj_out"其实只训了 LoRA，那批结论需要重新解读。）
3. **"`l_same` 的 sqrt(0) 让首步 NaN"** —— 曾经是 bug，**已修**（平方和 +1e-12）；
   否则 zero-LoRA 第一步 `e_same≡0` 会让所有 LoRA 梯度变 NaN。
4. **"server 端 LoRA scale 装错（alpha 默认 None ⇒ scale=1 而非 2）"** —— 已修（从 Adapter meta 读 rank/alpha）；
   但这条只影响闭环，不影响 WM 侧数字。
5. **"batch 太小"** —— 部分排除。E3 把有效 batch 提到 64（4×），Δtoken 只从 0.0042 涨到 0.0053，
   ΔJ 反而更震荡。**batch 不是主因**（但可能是"震荡"的原因之一：每步窗口全换 + 噪声新采）。
6. **"LR 太小"** —— **证据不支持**。lr 提 10× 后 ΔJ 没有系统性上升（见 3.2），
   而 token 位移只 +26%。⇒ 步长不是绑定约束。

---

## 5. 仍然**开着**的假设（需要 reviewer 帮排序/设计判别实验）

| # | 假设 | 支持证据 | 反证/未知 | 判别实验（建议） |
|---|---|---|---|---|
| H1 | **WM 对 LoRA 子空间的梯度信噪比太低**：ΔJ 的效应量（±0.003）与"换一批窗口+换 flow 噪声"的采样方差同阶 | E3 大 lr 反而震荡；E2 小 lr 靠"隐式平均"才爬；E4 LoRA 位移比 proj_out 小 10× | 未直接量过方差 | 固定同一批窗口/噪声，重复采样估计 ΔJ 的 batch 方差；做 **baseline/控制变量**（同窗口的 ΔJ 期望） |
| H2 | **WM 本身在策略新分布上失真**，梯度方向不可用（不只是方差大，而是有偏） | 旧 A′ 上 WM 的 20 步 root 误差 4.59 cm；策略 token 位移 ~3e-3 已离开训练分布（待量） | 尚未用真 Isaac 数据量 | `probe_wm_gradient_truth.py`：WM 反传方向 vs 真物理 ±扰动 rollout 的符号正确率（仓库已有工具） |
| H3 | **proj_out 的高杠杆"吃掉"了改进**：它贴在输出端、全秩，容易在 WM 上刷分，而 LoRA 的贡献被其残差掩盖 | E4 归因；proj_out 只训 600 步就改变 token rms 3e-2 | 未做 proj_out 冻结对照的完整训练 | **C2 纯 LoRA**（`--lora-also-train ""`，200–400 步）与 C1 同预算对照 |
| H4 | **目标函数（WM 期望回报）本身饱和/可刷**：J 的改进集中在"少违反限位"这种容易项，而非任务进展 | 限位违反率稳定下降（2.77→2.19%），但 ΔJ 与 vio 不同步（E3 中 vio 更好而 ΔJ 更差） | 无 | 分解目标：只优化 `-J` / 只优化物理项 / 两者；看哪个先饱和 |
| H5 | **LoRA 只挂在 attn1/attn2/ff，输出端不可达**，rank16 的容量不足以改变 40×64 token 的行为 | 位移 3.2e-3 ≪ A 的 Δ 0.078 | 未做更大 rank / 加 attn 位置编码层 | rank 32/64、或把 LoRA 挂到 `action_encoder/action_decoder` 的对照 |
| H6 | **评测口径的残余偏差**：`J_base` 随 proj_out 漂移（+1.3%），使小效应被"共同漂移"污染 | 1.50409→1.52432 | — | 冻结 proj_out 再评（或用 pristine FT8000 每次重算） |

---

## 6. 想问 reviewer 的具体问题

1. **在"冻结世界模型的回报"作为唯一目标时，LoRA 的有效学习率/步数/batch 该怎么定？**
   我们观测到 lr 从 1e-5 → 1e-4 只让 token 多走 26%、ΔJ 却更震荡，
   这更像是"梯度噪声主导"而不是"步长不足"。有没有比"调 lr/batch"更根本的手段
   （例如跨窗口 baseline、方差缩减、SVRG/EMA 梯度、或对 Δtoken 做 trust-region 投影）？
2. **如何干净地把"LoRA 的贡献"与"proj_out 的贡献"分开？**
   我们已做 token 级分解（proj_out 3.05e-2 vs LoRA 3.2e-3）和 ΔJ 分解（+0.0324 vs +0.0051），
   但这只是**事后归因**。要做**同预算对照**的话，C2 纯 LoRA 训练多少步、用什么 lr 才公平？
3. **要不要放弃"让 LoRA 学行为"这个假设？**
   如果 H3/H5 成立，那更合理的用法可能是：LoRA 只做"部署域适配"（视觉/状态分布），
   行为改进交给 DeltaTokenNet（A/B 那条线，val ΔJ +0.0282）。
   我们是否应该把 C 的目标改成"LoRA 适配 + Δ 网行为"的组合，而不是二选一？
4. **如何在不用完整任务闭环的前提下，判断"WM 梯度方向可用"？**
   仓库已有 `probe_wm_gradient_truth.py`（plan/score）与 `isaac_probe_action_rollout.py`，
   但需要 reviewer 帮定：窗口数、扰动幅度、符号正确率阈值、以及"LoRA 子空间"要不要单独探。
5. **我们的 ΔJ 效应量（+0.005）在 96 窗口上到底显不显著？**
   当前只报了均值；建议补 episode 聚类 bootstrap CI（我们已在 V7 的门禁里实现了这套统计，
   可以复用到 Actor 评测）。是否需要把"显著"作为下一轮的验收门槛？

---

## 7. 可复现的入口（代码/命令/数据）

```bash
# 环境
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
source scripts/harness03_env.sh

# 单卡训练（E1/E2 用的就是这个）
python3 scripts/train_sonic_lora16_ft8000_bptt.py --out <OUT> --updates 800 \
  --lr 1e-5 --lr-total-updates 800 --min-lr 1e-6 --warmup 20 \
  --ft8000-checkpoint $COMIMIC_MAIN/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vlm-path $COMIMIC_MAIN/groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --limits-npz runs/three_track_v6/physical/limits_standard29.npz \
  --physical-weights runs/three_track_v6/calibration/lora16_weights.json \
  --eval-set runs/temporal_ac_bptt_v5/eval_set --device cuda:0 \
  --batch 16 --vla-batch 16 --fm-grad-backend checkpointed \
  --eval-every 50 --eval-batch 16 --cond-check-every 25 --save-every 50 --log-every 5

# 双卡 DDP（E3）：每卡 batch 32 = 有效 64
torchrun --nproc_per_node=2 --master_port=29541 scripts/train_sonic_lora16_ft8000_bptt.py \
  --out <OUT> --updates 400 --ddp 1 --lr 1e-4 --lr-total-updates 400 --min-lr 1e-5 \
  ... --batch 32 --vla-batch 32 --fm-grad-backend checkpointed

# 纯 LoRA 对照（C2，尚未跑）：把 proj_out 冻回去
... --lora-also-train ""

# 归因分解（E4）
python3 /tmp/cdbg/decomp2.py <ADAPTER_CKPT>     # 见 docs 里的等价实现：sampler 采 proj-only / full，比 rms

# 门禁
python3 -m pytest tests/test_lora16_dummy.py tests/test_four_step_fm_grad.py \
  tests/test_sonic_wm_grad_to_lora.py tests/test_physical_bptt_losses.py -q
python3 scripts/audit_ft8000_lora16.py --ft8000 <FT8000> --vlm <VLM> --out <AUDIT>
```

**关键产物**

| 内容 | 路径 |
|---|---|
| E1 pilot（200 步） | `runs/three_track_v6/lora16_ft8000/train800/`（前 200 步） |
| E2 主跑（700 步停） | `runs/three_track_v6/lora16_ft8000/train800/{train.jsonl,val.jsonl,checkpoint_lora_0700.pt}` |
| E3 DDP 高 lr（~340 步停） | `runs/three_track_v6/lora16_ft8000/train400_lr1e-4_ddp/` |
| 真实 FT8000 审计 | `runs/three_track_v6/lora16_ft8000/audit/manifest.json` |
| C 的 λ 标定 | `runs/three_track_v6/calibration/{lora16.json,lora16_weights.json}` |
| 物理限位 | `runs/three_track_v6/physical/limits_standard29.npz` |
| 问题/参数总表 | `docs/TRACK_ISSUES_PARAMS_RESULTS_20261108.md` |

---

## 8. 请 reviewer 注意的边界

* 不要建议"改旧实验目录"：`runs/three_track_v6/`、`runs/a_prime_sonic_exc/` 是**只读基线**；
  新实验一律落 `runs/three_track_v7/`。
* 不要假设我们已经做过"闭环成功率"评测（C 的闭环尚未做；A 做过 18 集 = 9/18）。
* 任何"LoRA 有效/无效"的结论都必须区分三件事：
  **(i) 与 pristine FT8000 比**（含 proj_out）、**(ii) LoRA 边际**（同一 proj_out）、
  **(iii) token 级位移量**（3.2e-3 vs proj_out 3.05e-2）。
* 我们的 ΔJ 是"冻结 WM 的想象回报"，**不是真机成功率**；两者不可互相替代。
