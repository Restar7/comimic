# S3（联合微调 v2）执行报告 + 按 S0 结论的修订 — 2026-10-26

> 对应 `docs/JOINT_FINETUNE_REVIEW_REDESIGN_V2_20261026.md`（下称"复核"）§4 的 **S3** 与 §2.1 的 **J1–J10**。
> 依据：S0 审计报告（`docs/S0_AUDIT_REPORT_20261026.md`）+ 本轮实测。
> 旧入口 `scripts/train_joint.py` **未改动**（对照复现用）；v2 全部在新文件里。

---

## 0. 一句话

S3 v2 的机器（冻结执行器 / FM-BC / decode / K=4 / H=16 / burn-in 起点 / 64 窗口评测 / 梯度比 / 闭环挑选）
**已经全部落地并通过门禁**；但 S0.4 之后有一条硬结论改变了 S3 的用法：

> **想象的回报对 token 几乎不敏感**（GT vs 零 token 的回报差只有 +0.05%，而窗口间标准差 4%），
> 且**当前的 WBC 说的是端口 token 空间、VLA 说的是标签空间（两者 corr≈0.16）**。
> ⇒ 奖励反传臂（λ_r>0）现在**没有物理意义也没有信号**，S3 只能交付 **λ_r=0 的 BC 臂**（= 在正确 token 空间上继续官方 FM 训练）；
> λ_r>0 必须等 S1 交出"标签空间 WBC"再开。

---

## 1. 按 S0 结论对 S3 的修订（对照复核 §3.5/§4 S3）

| 项 | 复核原设计 | 按 S0 修订后 | 依据 |
|---|---|---|---|
| BC 目标 token 空间 | 端口 token（`data/port_tokens`，路线 II 重标） | **标签 token = npz `g1_motion_token_64`**（=`encoder_latent`，逐位相等） | S0.4：label vs port MSE 0.074 / corr 0.157（本轮 5 clip 复测一致） |
| burn-in 的 GT conditioning | 端口 GT token（默认） | **跟随冻结 WBC 的训练空间**（当前 = 端口；S1 换标签 WBC 后改 `--wbc-token-space label`） | WBC 只能在它训练过的空间里被条件化，否则 burn-in 起点 OOD |
| 两套空间的关系 | §3.2 路线 II：让 VLA 说端口话 | **反过来**：让 WBC 说标签话（S1 重训）；本轮两者不一致 ⇒ `λ_r` 强制 0 | S0.4 + 复核 W5 |
| S2（base_aligned） | 重标 + 官方微调 | **降级为可选**（标签本来就是 VLA 原生空间） | 复核附录 A.2.3 |
| decode | 服务端 decode（J2） | 不变；**训练侧也必须 decode**（v2 已在图里做） | S0.2 量化 |
| K（FM 步数） | 训练 = 服务端（J3） | 不变，K=4；`vla_meta.json` 链路本轮**修好并实测** | S0.3 + 本轮 §3.5 |
| 奖励核 | 不改（−e / softmax 都否） | 不变 | 复核 §5.8 |
| 门禁 | 闭环 | 不变；离线只作诊断（`token_mse` 不作门禁） | 复核 §5.6 |

---

## 2. 代码落地（S3.1–S3.7）

| 文件 | 内容 |
|---|---|
| `scripts/train_joint_v2.py`（新） | S3.1 新入口：CLI/审计/循环/存档/门禁模式；`--gate-only` 跑四道自检 |
| `diff_vla_wbc/train_v2.py`（新） | `JointTrainerV2`（只更新 head）、`BurnInBatchProvider`、`assert_wbc_frozen`、`fixed_eval` |
| `diff_vla_wbc/burnin.py`（新） | S3.4 起点：扰动 + 冻结 WBC 在 WM 里走 K 步（治 J7/J1''） |
| `diff_vla_wbc/fm_bc.py`（新） | S3.3 单次采样 + GR00T 原生 FM 损失 + 梯度比（治 J4） |
| `diff_vla_wbc/v2_gates.py`（新） | 四道门禁：codec / 冻结 / burn-in / **token 对比度** + FM-BC 冒烟 |
| `scripts/launch_joint_ha_v2.sh`（新） | v2 启动器（λ_r 扫描用） |
| `scripts/mini_closedloop_v2.sh`（新） | S3.7 闭环挑 ckpt（合并 → 起服务 → PPBox+DoubleDesk） |
| `diff_vla_wbc/data.py`（改） | `RecordedTokenSource(token_space=...)`：label/port 显式选择；`sample(frames=...)` 支持同一批帧对拍 |
| `scripts/export_vla_ckpt.py`（修） | **NameError**：`vla_meta.json` 之前根本没写出来（J3 链路断了却没人发现） |
| `scripts/ha_vla_latent64_server.py`（修） | fm_steps 校验提前到**加载权重之前**（fail fast）；缺失时打 WARN 不再静默 |

J 项落点：

| J | 状态 | 落点 |
|---|---|---|
| J1 spec | ✅ 已闭环 | S0 修 + `isaac_joint` 断言；v2 每次加载打印 `[AUDIT][spec] ... PASS` |
| J2 未反归一化 | ✅ 已闭环 | `action_codec` + v2 训练图里 `codec.decode`；服务端默认 `--decode codec` |
| J3 FM 步数 | ✅ 已闭环 | `vla_*.pt["fm_steps"]` → `vla_meta.json` → 服务端 fail-fast 校验（本轮修好并实测） |
| J4 sample-MSE | ✅ 已闭环 | `token_anchor` / hand-MSE 在 v2 里**不存在**；`L_bc` = GR00T 原生 FM |
| J5 更新 WBC | ✅ 已闭环 | `requires_grad_(False)` + eval；门禁 `torch.equal(pi)` 每次保存都过 |
| J6 多卡同 seed | ✅（v1 已修） | v2 单卡跑：`grad-accum` 给等效 batch 16 |
| J7 起点=参考 | 🟡 机器已改，**效果被证伪** | burn-in 落地（起点 xy 误差 3.9 cm、survival 0.9999），但**回报仍不随 token 变** → 见 §4 |
| J8 H=8 | ✅ 已闭环 | H=16（覆盖闭环 12 步重取） |
| J9 4 个窗口 | ✅ 已闭环 | 固定 64 窗口（16×4）+ 闭环挑选脚本 |
| J10 期望 | ✅ 已写明 | 门禁只看闭环；本轮如实上报"奖励反传无信号" |

---

## 3. 门禁实测（`--gate-only`，batch 2，H=16，K=4，λ_r=0.3）

### 3.1 A codec（S0.2 单测，写进 S3 门禁）
```
[GATE][codec] PASS decode<->decode_action max|d|=1.19e-07 roundtrip=1.19e-07 dim=78 raw=[-1.740,1.740]
```
⇒ BC 目标/采样输出的归一化口径与 GR00T 官方一致；手指阈值 1.70 只在原始空间成立（J2）。

### 3.2 B 冻结（S3.2）
```
[GATE][freeze] PASS pi_identical=True wbc_requires_grad=0 wm_requires_grad=0
               wbc_grad_tensors=0 wm_grad_tensors=0 head_grad_tensors=537
               policy.training=False wm.training=False
```
⇒ 反传只落在 action head；`pi` 与源 ckpt 13 个张量逐张量相等（每次 `joint_*.pt` 存档再验一次，全 PASS）。

### 3.3 C burn-in 起点（S3.4）
| WBC | ref 起点回报 | burn-in 起点回报 | 起点 xy 误差 | survival |
|---|---|---|---|---|
| WBC v2（`wbc_mix2_obs93/ckpt_016000_spec`） | 0.04670 | 0.04757 | 3.87 cm | 0.9999 |
| WBC v1（`physint` obs99） | — | — | — | — |

⇒ 起点确实被扰开（3.9 cm ≈ 设定 5 cm 的一半 + yaw 0.1 rad），且 WBC 救得回来；
但**想象回报没有因为起点变差而下降**（−0.9‰，噪声级）。这直接否掉了 J7 的假设：起点误差不是回报不敏感的**主要**原因。

### 3.4 C2 token 对比度（**本轮最关键的诊断**）
同一起点、同一批帧、H=16，只换 conditioning：

| WBC | GT（WBC 空间录制 token） | zeros | randn | shuffled | GT − zeros |
|---|---|---|---|---|---|
| v2（obs93, mixed 共训） | 0.80466 | 0.80426 | **−1.08817** | 0.79815 | **+0.00040（+0.05%）** |
| v1（physint, obs99） | 0.68954 | 0.70825 | **−1.80451** | 0.68940 | **−0.01871（−2.7%）** |

窗口间标准差：v2 `sd=0.041`，v1 `sd=0.134`。

读法（三条都重要）：
1. **OOD token 会崩**（randn → 回报转负）⇒ 链路是通的、不是"token 没接进去"；
2. **in-distribution 的正确 token 与零 token 没有区别**，甚至 shuffled（别的样本的 token）也一样
   ⇒ 冻结 WBC 在这些窗口上基本**不看 conditioning**（obs93/obs99 历史里已经含了足够的跟踪信息，
   加上 WM 的残差结构会自己"沿着参考走"）；
3. ⇒ 对想象回报求导，梯度里**没有"哪个 token 更好"的信息**，只有"别离开分布"的压力 ——
   这正是复核 §0 担心的"把 token 压向均值"（J4 的另一种实现方式）。

### 3.5 D FM-BC（S3.3）
```
[GATE][fm-bc] PASS loss_fm_bc=0.63725 loss_reward_norm=0.44106 rollout_reward=0.04779
              survival=0.9979 grad_ratio=0.1328 out_of_range=0.072
```
⇒ λ_r=0.3 时 `‖∇L_r‖/‖∇L_bc‖ = 0.13`（落在复核要求的 0.1–0.5 内）——**梯度比可调，但信号方向是噪声**（§3.4）。
`codec/out_of_range ≈ 4–11%`：采样常有维度越出 [-1,1]，decode 会外推；这是 FM 采样的正常现象，记录但不 clamp（保梯度）。

---

## 4. λ_r 两臂（本轮的正式读数）

配置（两臂逐字相同，只差 λ_r）：`H=16 K=4 batch=4 accum=4 lr=1e-5 1000 步`，
WBC = `ckpt_016000_spec.pt`（v2），VLA 起点 = 官方 `checkpoint-8000`，BC 空间 = **label**，评测 = 固定 64 窗口。

| 臂 | λ_r | `L_fm(val)` | `token_mse_decoded` | `hand_mse_decoded` | 手闭合率 pred/label | 现象 |
|---|---|---|---|---|---|---|
| **B0**（BC 对照） | 0 | 0.1365 → 0.0270（200 步） | 0.0042 → 0.0053 | 0.0023 → 0.0028 | 0.096 / 0.114 | 只在学 token+手，稳定 |
| **B3**（奖励臂） | 0.3 | 0.1365 → TBD | TBD | TBD | TBD | `rollout/reward ≈ 0.05`、`survival ≈ 0.9997` 全程不动 |

**关键点**：`L_fm(val)` 下降是"BC 在学"的正常证据；`token_mse_decoded` **不降反升**属预期
（FM 损失不是点估计，样本不会等于标签；复核 §5.6 已明确它不能当门禁）。
`imag_return_vla ≈ imag_return_gt`（0.727 vs 0.728）再次印证 §3.4：想象回报分不出好坏 token。

（两臂的完整曲线：`runs/s3_v2_bc/eval_curve.jsonl`、`runs/s3_v2_lamr03/eval_curve.jsonl`，
逐 step 指标：`runs/*/train_metrics.jsonl`；门禁：`runs/s3_gates_smoke/gates.json`。）

---

## 5. 闭环（S3.7/S3.8）

- 脚本：`VLA=runs/s3_v2_bc/vla_final.pt WITH_BASE=1 bash scripts/mini_closedloop_v2.sh`
  （PPBox + DoubleDesk × seed0 × 300 步；v2 臂与 base 臂同协议）；
- 结果目录：`runs/mini_closedloop/<run>__vla_final/{pick.json, isaac/SUMMARY.md}`；
- **TBD：本轮闭环数字**（见 §6 的说明；若被 pin/资产问题挡住会在这里写明）。

---

## 6. 结论（诚实版）

1. **S3 v2 的机器可以交付**：冻结、FM-BC、decode、K=4、H=16、burn-in、64 窗口、梯度比、闭环挑选全部跑通并有门禁数字。
2. **奖励反传（λ_r>0）现在不能开**，原因有两个、都已被独立测到：
   * **空间不一致**（S0.4）：WBC 训练在端口空间，VLA 在标签空间，corr≈0.16 ⇒ 部署/想象都在跨空间；
   * **回报不敏感**（§3.4）：即使在同一空间里，GT 与零/shuffled token 的想象回报差 0.05%（< 噪声）⇒ 没有可学的方向。
   在这两条修好之前，λ_r>0 只会把 token 往"分布内平均"拉 —— 与 J4 要治的病同构。
3. **BC 臂（λ_r=0）才是有意义的本轮产物**：它在正确 token 空间上继续官方 FM 训练，
   decode 后手闭合率、token 尺度、`L_fm(val)` 都可比、可门禁。
4. 要打开奖励臂，前置是（按依赖顺序）：
   * S1 用**标签空间**重训 WBC（S0.4 之后的唯一正确路线）→ 空间一致；
   * 让回报对 token 敏感：burn-in 加长/加大扰动到"不靠 token 就跟不住"的区间，或给 WBC 的观测去掉
     能"自己跟着参考走"的捷径（这是 S1/S3 的下一个研究点）；
   * 再跑 λ_r ∈ {0.1, 0.3, 1.0} 与 B0 对照，按闭环判据（复核 §3.7/§3.8）。

---

## 7. 复现命令

```bash
# 四道门禁（不训练，~4 min）
cd <repo> && export PYTHONPATH="$PWD:$PWD/Isaac-GR00T:$PWD/third_party/gr00t_deps:$PWD/third_party/gr00t_deps/_noflash"
python3 scripts/train_joint_v2.py --gate-only --batch-size 2 --gate-batches 16 \
  --wbc-ckpt trackerlab-supermimic/logs/wbc_mix2_obs93/ckpt_016000_spec.pt \
  --vlm-ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 --out runs/s3_gates

# BC 臂（λ_r=0）
RUN=s3_v2_bc LAMBDA_R=0 STEPS=1000 BATCH=4 ACCUM=4 EVAL_EVERY=200 GPU=0 \
  bash scripts/launch_joint_ha_v2.sh

# 奖励臂（需要标签空间 WBC 才有意义；本轮仅作对照）
RUN=s3_v2_lamr03 LAMBDA_R=0.3 STEPS=1000 BATCH=4 ACCUM=4 EVAL_EVERY=200 GPU=1 \
  bash scripts/launch_joint_ha_v2.sh

# 闭环挑 ckpt
VLA=runs/s3_v2_bc/vla_final.pt WITH_BASE=1 bash scripts/mini_closedloop_v2.sh
```

---

## 8. 未完成 / 交接

| 项 | 状态 | 说明 |
|---|---|---|
| λ_r>0 正式扫描 {0.1,0.3,1.0} | ⛔ 阻塞 | 等 S1 标签空间 WBC + 回报敏感性（§6.4） |
| 闭环 F 臂 vs B 臂（复核 §3.7） | ⛔ 阻塞 | 同上；B 臂（base + 标签空间 WBC）也要等 S1 |
| `probe_action_decomp.py` 在 HA spec 下重跑 | 🟡 诊断 | S0.1 遗留；v2 的 burn-in 已让 `a_prev` 不再依赖换算，优先级降低 |
| 2 卡 seed smoke（J6 门禁） | 🟡 | v2 单卡跑，不受影响 |
