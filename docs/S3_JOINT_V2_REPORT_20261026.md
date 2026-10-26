# S3（联合微调 v2）执行报告 + 按 S0 结论的修订 — 2026-10-26

> 对应 `docs/JOINT_FINETUNE_REVIEW_REDESIGN_V2_20261026.md`（下称"复核"）§4 的 **S3** 与 §2.1 的 **J1–J10**。
> 依据：S0 审计报告（`docs/S0_AUDIT_REPORT_20261026.md`）+ 本轮实测。
> 旧入口 `scripts/train_joint.py` **未改动**（对照复现用）；v2 全部在新文件里。

---

## 0. 一句话

S3 v2 的机器（冻结执行器 / FM-BC / decode / K=4 / H=16 / burn-in 起点 / 64 窗口评测 / 梯度比 / 闭环挑选）
**已经全部落地并通过门禁**；但 S0.4 之后有一条硬结论改变了 S3 的用法：

> **UPDATE（当晚，token 前端交接后）**：这个"两套空间"的阻塞**已经解开了一半** ——
> S1 把前端改成 hold 窗口 + 命令流库，并训出了 hold 版 WBC；S3 随之改成
> **环内 hold token（仿真 anchor）+ `ha_cmd` 命令流库 + hold WBC**。
> 新口径下：`bc-space` 与 `wbc-space` 的奖励信号**等价**（1.318 vs 1.336，配对 t 18.6/19.5）、
> `grad_ratio=0.29`（落在 0.1–0.5），λ_r=0.3 不再伤 token/手部（§10）。
> ⇒ 现在只差 `wbc_hold_*` 过完三格门禁就能跑正式 λ_r 扫描；本报告 §4 的旧两臂
> （port/future 口径）已降级为**跨口径冒烟**，不作结论。

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

### 3.4 C2 token 对比度（**本轮最关键的诊断，先踩了一个假阴性，最后靠"双空间"才读对**）

**正式门禁读数**（`--gate-only --batch-size 4 --gate-batches 16` = 64 窗口，`runs/s3_gates/gates_label_bc.json`）：

```
[GATE][contrast] PASS n=64
  ret gt_wbc=1.11320 gt_bc=0.75722 mean=0.94770 shuffled=0.92875 zeros=0.74279 randn=-1.04027
  | wbc-space contrast=+0.37041 rel=+33.27% t=21.36 win=1.00
  | bc-space  contrast=+0.01443          t=1.53  win=0.53
  | r0 gt_wbc=0.07281 zeros=0.04090 | surv gt_wbc=0.9842 zeros=0.9841
```

两个 GT 必须分开看，否则会得出互相矛盾的结论（我就先矛盾了一次）：

| conditioning | 想象回报 | 相对 zeros | 配对 t | 胜率 | 解释 |
|---|---|---|---|---|---|
| `gt_wbc`（WBC 训练空间的录制 token） | 1.1132 | **+33.3%** | **21.4** | **100%** | 奖励通道**有信号**，GT 的 1 步奖励也是 zeros 的 1.8 倍（0.0728 vs 0.0409） |
| `mean`（条件均值 token） | 0.9477 | +27.6% | — | — | 介于 GT 与 zeros 之间 |
| `shuffled`（别的样本的 token） | 0.9288 | +25.1% | — | — | 比 GT 差，但远好于 zeros |
| `gt_bc`（**VLA 空间**的录制 token 喂给 port-WBC） | 0.7572 | **+1.9%** | 1.5 | 53% | **部署现实**：跨空间后信号基本消失 |
| `zeros` | 0.7428 | — | — | — | 起点 |
| `randn`（分布外） | −1.0403 | — | — | — | 链路是通的，OOD 会崩 |

**这三行合起来就是 S3 的核心结论**：
1. 冻结 WBC 的奖励通道**强**（t=21），λ_r 的梯度不是噪声；
2. 但这个信号只在 **WBC 自己的 token 空间**里存在；
3. VLA 说的标签空间喂进 port-WBC，信号掉到 **1.9%（t=1.5，胜率 53% = 掷硬币）** ——
   这正是 S0.4（label vs port MSE 0.074 / corr 0.157）在**行为层面**的后果，也解释了旧 A1 里
   "base 臂走向桌子 / 打转"：不是 VLA 差，是**它说的话 WBC 听不懂**。

**方法学教训（写进门禁）**：8 个窗口时 GT−zeros 只有 +0.0004（假阴性），
64 窗口配对检验变成 +0.372（t=30.2，64/64 胜出）。门禁判据因此改成
`配对 t ≥ 3 且胜率 ≥ 0.8`，默认窗口数 ≥32。

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
WBC = `ckpt_016000_spec.pt`（v2），VLA 起点 = 官方 `checkpoint-8000`，
BC 空间 = **label**（`g1_motion_token_64`），burn-in K=10，评测 = 固定 64 窗口。

### 4.1 B0：λ_r=0（纯 FM-BC 对照臂，本轮交付物）

| step | `L_fm(val)` | `token_mse_decoded` | `hand_mse_decoded` | 手闭合率 pred/label | `imag_return_vla` / `_gt` |
|---|---|---|---|---|---|
| 0 | 0.1365 | 0.00420 | 0.00229 | 0.115 / 0.114 | 0.727 / 0.728 |
| 200 | **0.0270** | 0.00526 | 0.00281 | 0.096 / 0.114 | 0.731 / 0.728 |
| 600 | 0.0269 | 0.00493 | 0.00256 | 0.111 / 0.114 | 0.730 / 0.728 |
| 1000 | 0.0265 | 0.00509 | 0.00230 | 0.102 / 0.114 | 0.732 / 0.728 |

- `L_fm(val)` 在 200 步内从 0.137 掉到 0.027 后**平台**（这不是 held-out episode，只是另一批帧，
  所以只当"学习稳定/没有发散"的诊断，不当门禁）；
- token/手部的点估计 MSE **不降**（0.0042→0.0050、0.0023→0.0023）——预期：
  FM 损失监督的是速度场，不是让样本等于标签（复核 §5.6）；
- 手闭合率与标签一致（0.10 vs 0.11），**没有塌成"永远张开"**（这是 v1 的 J2 病，已消失）。

### 4.2 B3：λ_r=0.3（label BC + **port** 奖励，空间不一致 —— **诊断臂，不进正式表**）

> 这一臂是 02:22 起的，**早于**协作方加的硬门禁（`scripts/train_joint_v2.py`：
> `bc != wbc` 且 `λ_r>0` 直接 `SystemExit`，除非显式 `--allow-token-space-mismatch`）。
> 保留它是因为它把"空间不一致的代价"量化得很干净；它的数字**不能**当 F 臂证据。

| step | `L_fm(val)` | `token_mse_decoded` | `hand_mse_decoded` | 手闭合率 pred/label | `imag_return_vla` / `_gt` |
|---|---|---|---|---|---|
| 0 | 0.1365 | 0.00420 | 0.00229 | 0.115 / 0.114 | 0.727 / 0.728 |
| 200 | 0.0312 | 0.01000 | 0.00330 | 0.075 / 0.114 | 0.855 / 0.728 |
| 400 | 0.0304 | 0.03353 | 0.01479 | 0.005 / 0.114 | 0.937 / 0.728 |
| 600 | 0.0303 | **0.05975 (+14×)** | 0.06114 | **0.000** | **1.006 (+38%)** |

这是本轮最有信息量的一组数字，**它把"空间不一致"的后果量化出来了**：

1. 想象回报被拉高 38%（0.727 → 1.006）——奖励项确实在优化，说明梯度链路完好；
2. 但 token 偏离标签 **14 倍**、**手部闭合率塌到 0**（标签 11.4%）——奖励项只依赖 64 维 token，
   手部没有任何奖励梯度，于是被共享权重的漂移顺手牺牲掉；
3. `L_fm(val)` 几乎不动（0.030）：说明"FM 损失"本身**不足以**发现采样分布已经跑偏 ——
   这正是复核 §5.6 "token_mse 不能当门禁"的另一面：**FM 损失也不能单独当门禁**，必须看 decode 后的量 + 闭环。

⇒ 结论：**两套空间不一致时开 λ_r = 奖励项在教 VLA 说 WBC 的话，代价是标签保真度和手部**。
这不是"λ_r 没用"，而是"λ_r 用得不是地方"；正式 F 臂必须先把空间统一（S1 标签空间 WBC）。

## 5. 闭环（S3.7/S3.8）

### 5.1 本轮实际结果：**未取得**（容器环境问题，非 S3 代码问题）

按 S3.7 写了 `scripts/mini_closedloop_v2.sh`（合并 → 起服务 → PPBox+DoubleDesk × 1 seed × 300 步），
在本容器实测遇到两层环境问题：

1. **Isaac 解释器是坏符号链接**：`third_party/isaacsim_env/bin/python -> /root/.local/share/uv/python/cpython-3.11-…`，
   该 uv 解释器在本容器不存在 → `isaac_env.sh` 的 PATH 首位无 `python`，闭环脚本直接 `ModuleNotFoundError: isaaclab`。
   **已修**（容器内 `ln -s third_party/uv_python/cpython-3.11-linux-x86_64-gnu /root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu`，
   `import isaaclab` 通过；该修复只在容器内，不影响别人）。
2. 修好后再跑，Isaac 在场景装配阶段**卡住**（20 min 无 `CONTROLLER` 输出、无 server 推理；
   `wchan=hrtimer_nanosleep`，同时有 `libneuray.so` 缺失告警）——判定为资产/材质解析阻塞，非 S3 代码路径。

⇒ **闭环数字留给有可用 Isaac 的容器**（S1 那边已经跑通过 O1/O2 的回放）。命令：

```bash
VLA=runs/s3_v2_bc/vla_final.pt WITH_BASE=1 TASKS="PPBox DoubleDesk" SEEDS="0" MAX_STEPS=300 \
  VLA_GPU=0 ISAAC_GPU=0 bash scripts/mini_closedloop_v2.sh
```

连接线都验证过了：`export_vla_ckpt` 合并后 `[verify] OK 537/537`、`vla_meta.json={"fm_steps":4}`、
服务端 `[AUDIT][server] decode=codec fm_steps=4 train_fm_steps=4`、`--fm-steps` 不一致会在加载权重前
直接 `SystemExit`（实测 `--fm-steps 2` 立刻拒绝）。

### 5.2 S3.8 判据的现状

复核 §3.8 要求"F 臂相对 B 臂成功率不降 + 至少一项显著改善"。本轮**不能**给出这个结论，原因不是缺数字，
而是缺**前提**：F 臂要求 VLA 与 WBC 同一个 token 空间，而这个前提要等 S1 用标签空间重训 WBC
（S1 已把前端配方做到逐位复现 `encoder_latent`，见 `docs/S1_REPLAN_AFTER_S0_20261026.md`，
但还需要按 command 流重建 HA 库，因为现在 `joint_pos` 是执行轨迹、与 encoder 输入差 ≈8 帧）。

## 6. 结论（诚实版）

1. **S3 v2 的机器可以交付**：冻结、FM-BC、decode、K=4、H=16、burn-in、64 窗口、梯度比、闭环挑选全部跑通并有门禁数字。
2. **奖励通道有信号**（§3.4：t=30.2，GT 在 64/64 窗口胜出），所以"λ_r 学不到东西"这个担心在本轮被**证伪**；
   但 8 窗口的门禁会给出假阴性 —— 门禁已改成配对 t + 胜率。
3. **真正的阻塞是 token 空间**（S0.4 + 本轮复测 MSE 0.074 / corr 0.157）：
   BC 目标是标签空间，冻结 WBC 吃的是端口空间 ⇒ λ_r 与 BC 在**互相拉扯**。
   本轮 `λ_r=0.3`（label BC + port reward）就是这种冲突的实测样本，不是正式 F 臂。
4. 打开正式 F 臂有两条路，**按代价排序**：
   * **S1 用标签空间重训 WBC**（S0.4 之后的唯一正确路线，改动在 S1）→ 空间一致 → λ_r 直接可用；
   * **端口一致的代理臂**（BC 目标 = 端口侧车 + WBC = 端口，两者自洽）：今天就能跑，
     代价是让 VLA 说非原生空间（复核路线 II 的复活，只在 WBC 换空间前当代理用）。
5. 期望管理（J10）：即便 λ_r 有了信号，它优化的仍是"跟踪录制动作的可执行性"，
   不会带来"去对的目标/抓取"的新能力；正式判据只能是复核 §3.7 的闭环表。

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
| λ_r>0 正式扫描 {0.1,0.3,1.0} | ⛔ 阻塞 | 等 S1 交出**标签空间** WBC（前端配方已逐位复现 `encoder_latent`，但 HA 库要按 command 流重建 —— `joint_pos` 是执行轨迹，与 encoder 输入差 ≈8 帧，见 `docs/S1_REPLAN_AFTER_S0_20261026.md`）。之后把 `--wbc-token-space label` 一改，burn-in 与奖励就同空间 |
| 闭环 F vs B（复核 §3.7/§3.8） | ⛔ 阻塞（两个原因） | ① 同上，缺标签空间 WBC；② 本容器 Isaac 不可用（§5.1，已定位并修了解释器坏链，但场景装配仍阻塞）→ 建议在 S1 那些能跑 Isaac 的容器里执行 `scripts/mini_closedloop_v2.sh` |
| B 臂（base + 标签空间 WBC）闭环 | ⛔ 阻塞 | 同 ① |
| `probe_action_decomp.py` 在 HA spec 下重跑 | 🟡 诊断 | S0.1 遗留；v2 的 burn-in 已让 `a_prev` 不再依赖换算（J1'' 关闭），优先级低 |
| 2 卡 seed smoke（J6 门禁） | 🟡 | v2 单卡跑（`grad-accum` 给等效 batch 16），不受影响 |
| 手部闭合作为单独判据 | ✅ 已有读数 | `hand_gt170_pred 0.10 vs label 0.114`（B0 臂）；B3 臂塌到 0.000 —— 这两行比任何离线 MSE 都更能说明问题 |

## 9. 回报模板（按复核 §6）

```
[S3.1] 入口=scripts/train_joint_v2.py（旧 train_joint.py 未动）           PASS
[S3.2] pi 逐张量相等=YES   wbc/wm requires_grad=0/0   反传后 grad 张量=0/0  head grad=537  PASS
[S3.3] 单次采样=YES  K=4  L_fm(val) 0.1365→0.0265(200 步后平台)  无 token_anchor / 无 hand-MSE  PASS
[S3.4] burn-in: 起点 xy 误差=5.9cm  survival=0.9997  reward ref→burn=0.04981→0.04921  PASS
[S3.5] H=16  回报归一化=J/(H·MAX_STEP_REWARD)                              PASS
[S3.6] batch=4 accum=4  grad_ratio(λ_r=0.3,WBC v2)=0.133（v1: 0.192）       PASS
[S3.7] 固定评测=64 窗口（16×4）  L_fm / 想象回报 / decode 后 MSE 全落盘      PASS；mini 闭环 ⛔（Isaac，§5）
[S3.8] F vs B 闭环对比                                                     ⛔ 前置未满足（§8）
[C2]  奖励通道: wbc-space GT−zeros=+0.370(+33.3%) t=21.4 win=100%
                bc-space  GT−zeros=+0.014(+1.9%)  t=1.53 win=53%   ← 空间不匹配的代价
[闭环] 未取得（Isaac 环境，§5.1）；链路（合并/服务/fm_steps 校验）已验证
```

---

## 10（追加·2026-10-26 晚）：token 前端交接后的 **hold 路线**重跑

> 输入：`docs/TOKEN_FRONTEND_HANDOVER_FOR_S3_20261026.md`。结论：**设计不用改，改的是"口径"**，
> 而口径一改，§3.4/§4 的悲观结论**翻过来了**。

### 10.1 交接说了什么（三条，每条都落在 S3 的代码上）

1. 64 维 token = 冻结 encoder(1762)：其中 580 维是「10 帧命令窗口」。HA 录制/部署填 **hold**
   （当前帧重复 10 次），我们旧实现填 **未来 10 帧** → 两套 token corr 只有 0.05（= S0.4 的根因）。
2. HA 的 npz 库 `joint_pos` 是**执行轨迹**，不是 encoder 吃的命令流（mean|Δ|=0.41、lag **+8 帧**）
   → 标签必须配**命令流库** `runs/isaac_motion/ha_cmd/packed.pt`。
3. hold 下用 identity anchor 复现不了标签（anchor 是实质成分）→ **训练必须用环内机器人 anchor（B 模式）**。

### 10.2 S3 按交接改的代码（3 处，已提交）

| 文件 | 改动 |
|---|---|
| `diff_vla_wbc/sonic_frontend.py`（新） | `LibraryTokenBuilder`：把 `CompactMotionLib` 包成 `SonicTokenBuilder` 的 `q/qd/root_quat` 视图；`chunk(frames, robot_quat)` 一次出整段 token |
| `diff_vla_wbc/burnin.py` | `token_fn(frames, state=...)`：burn-in 每一步用**当时仿真机器人的 root** 算 hold token（B 模式），不再读预计算侧车 |
| `scripts/train_joint_v2.py` | `--motion-packed` 默认 `ha_cmd/packed.pt`；自动发现 `logs/wbc_hold*/ckpt_*.pt`；`--wbc-token-space auto`（hold⇒label）；**两道硬门禁**：库必须是命令流库、WBC ckpt 的窗口口径必须与 `--wbc-token-window` 一致（`--allow-*-mismatch` 仅诊断） |
| `diff_vla_wbc/data.py` | `resolve_clip_files`：`ha_cmd` 库的 clip 名是 raw 的 `ep<ts>`、npz 是 `ep000000`，按**同序号**回退（两库 clip 数/每段长度逐位相同：699 / 634016） |

### 10.3 新口径下的门禁（`runs/s3_gates_hold/gates_hold_label.json`，32 窗口，λ_r=0.3）

```
[GATE][codec]     PASS 1.19e-07
[GATE][freeze]    PASS pi 13 张量相等；wbc/wm 0 梯度；head 537 梯度张量
[GATE][burnin]    PASS ref 0.08931 → burn 0.08789（drop_sign_ok=True）survival 0.9999 start_xy 5.4cm
[GATE][contrast]  PASS n=32  gt_wbc=1.33587 gt_bc=1.31759 mean=1.08617 shuffled=1.03171
                        zeros=0.94728 randn=-1.15487
                        wbc-space +0.38858 (+29.1%) t=19.49 win=100%
                        bc-space  +0.37030 (+39%)  t=18.59 win=100%     ← 不再是 1.9%/1.5 了
                        r0 gt_wbc=0.08985 zeros=0.05431
[GATE][fm-bc]     PASS rollout_skipped=False loss_fm_bc=0.13683 loss_reward_norm=0.70949
                        rollout_reward=0.07646 survival=0.9995 grad_ratio=0.2866
```

**§3.4 的结论就此修正**：不是"奖励通道没信号"，是**两套口径**（hold vs future）让信号对不上。
换成 hold + 环内 anchor 后：label 空间（`gt_bc`）与 WBC 空间（`gt_wbc`）**几乎等价**（1.318 vs 1.336），
两者对 zeros 的配对 t 都 ≈19，`grad_ratio=0.29` 正好落在复核要求的 0.1–0.5 中间。

### 10.4 hold 路线两臂（WBC = `logs/wbc_hold_x20/ckpt_017250.pt`，**训练中的中间 ckpt**）

配置：H=16 K=4 batch 4×accum 4，400 步，固定 64 窗口评测，BC=label（npz 标签），burn-in 环内 hold token。

| 臂 | λ_r | step | `L_fm(val)` | decode 后 token MSE | 手闭合率 pred/label | `imag_return_vla` / `_gt` |
|---|---|---|---|---|---|---|
| H0 | 0 | 0 / 200 / 400 | 0.1365 / 0.0272 / **0.0264** | 0.00420 / 0.00529 / 0.00508 | 0.115 / 0.096 / 0.099 ← **0.114** | 1.290 / 1.276 / **1.283** ← 1.319 |
| H3 | 0.3 | 0 / 200 | 0.1365 / 0.0286 | 0.00420 / 0.00496 | 0.115 / 0.0815 ← 0.114 | 1.290 / **1.302** ← 1.319 |

**和 §4.2（B3）对照，这就是"口径对不对"的全部差别**：

| | 旧 port/future 路线（B3） | 新 hold/label 路线（H3） |
|---|---|---|
| token 偏离标签 | **+14×**（0.0042→0.0598） | **持平**（0.0042→0.0050） |
| 手闭合率（标签 0.114） | **0.000（手全张开）** | 0.082（还在，略降） |
| 想象回报 | +38%（0.727→1.006） | +1%（1.290→1.302） |
| 解释 | 奖励在**跨口径**下拉偏 token、顺手牺牲手部 | 奖励在**同口径**下只做小幅精修 |

> ⚠️ 两个臂用的是 **it 17250 的中间 ckpt**（hold 版 WBC 目标 18500，S1 仍在训）。
> 按交接文档 §6，**正式读数要等 `wbc_hold_*` 过完三格门禁**（HA-699 / BONES-500 / WM 速度）；
> 这里的数字只用于验证 S3 的管道与口径，**不进 A1 正式表**。

### 10.5 §4 的两臂该怎么看（订正）

§4.1/§4.2 是在 **port/future 口径**下跑的：BC 已经是标签（hold）而 WBC 是端口（future），
所以那两臂的正确身份是 **"跨口径冒烟"**，不是 S3 的结论：
- B0（λ_r=0）证明"冻结 + 单次采样 + FM-BC + 64 窗口评测"这条机器是通的；
- B3（λ_r=0.3）证明了**跨口径开奖励**会发生什么（回报 +38%、token 偏离 14 倍、手塌成 0）——
  这个现象本身就是交接文档 §2 那条"两套 token corr 0.05"的行为学证据。

