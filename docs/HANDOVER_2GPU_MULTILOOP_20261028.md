# 双卡 4090 多轮闭环：修复 / **九轮实跑** / 修正后的结论（2026-10-28 夜）

> 承接 `docs/HANDOVER_4090_COMMANDS_20261028.md`、`docs/HANDOVER_STATUS_AND_PROBLEMS_20261028.md`。
> 本文是这一整晚在 2×4090 上**实际跑完**的记录。**结论数字以 `runs/S3_loop/sweeps/MASTER.md` 为准**，
> 那里的每一行都带了"哪个 head 在哪张卡"。

---

## 0. 一句话

**采集 / 训练 / 闭环评测三段都做成了"每阶段两卡全用上"，P0 全部修掉，WM v6 过门，跑了 9 个轮回。**
结论：**没有任何一个训练过的 head 能在同场配对里优于官方底模 π₀**（9 轮 / 7 次 π₀-vs-head 配对 sweeps，n=42），
但**闭环被证明可以"持续跑而不掉分"**（第 9 轮的温和剂量），而且**部署端拿到了真实收益**
（`--sample-avg 4`，−0.08 m）。π₀ 本身很稳：**0.322 ± 0.016 m**（avg1，7 次会话）。 π₀ = **0.322 ± 0.016 m**（7 次会话，极稳）；
最好的候选 r5 主臂 0.284（−0.04，不显著），其余 0.38–0.53。
合并 5 次 sweep 的配对差（n=30）：**候选平均比 π₀ 差 0.121 m（t=+2.88），只有 8/30 更好**。
同一策略放两卡对照 −0.004 m ⇒ **装置无偏，这不是量测问题**。

**唯一真正见效的杠杆在部署端，不在训练**：给 VLA server 加 `--sample-avg N`（同一观测采 N 次
求平均，降采样噪声），同一个 π₀ 只改这个参数：**0.363 → 0.307 m（配对 −0.082 m、约 −22%，
两次会话同向，SD 0.138→0.092）**。训练侧 8 轮全败，部署侧一个开关就拿到 0.08 m。

**第 9 轮补上了最后一块**：用"剂量梯度"找到了**不伤**的训练点 ——
`fm-lr 3e-6 × 20 步`（权重变化只有常规配置的 ~1/170）：配对 vs π₀ **0.260 vs 0.240（+0.020，t=+0.94）
= 统计上打平**，且 SD 只有 0.065–0.079（比之前所有候选的 0.12–0.19 小一半）。
⇒ **闭环现在可以"多轮持续跑而不掉分"**；要涨分请用部署端 `--sample-avg 4`。

**⚠️ 勘误**：本文的第一版曾写"π₄ 比 π₀ 好 0.204 m、−39%、t=2.94"。**那是读错了臂** ——
那两次 sweep 里 π₀ 跑在 GPU0（槽位名 `main`）、候选跑在 GPU1（`ctrl`），我把两者读反了。
`summary.json` 现在**写死了每个槽位加载的 head**（`heads.json`），这个坑已经堵上。

这一晚真正的产出不是"涨点"，而是：
1. **一套无偏、可复现的双卡闭环测量装置**（同策略两卡对照差 −0.004 m；π₀ 跨会话 SD 仅 0.018）；
2. **一条被证明走不通的路**（"加 λ_BC 锚定"在 `V3=1` 下压根不生效），
   和**一条已知的天花板**（标签锚定只能把漂移补回来，补不了 GT 那 0.2 m）；
3. **三处会误导后续所有判断的坑**（见 §7）。

---

## 1. 这台机器的口径（已现场核对）

| 项 | 实测 |
|---|---|
| GPU | 2 × RTX 4090，49140 MiB/张 |
| 驱动 / CUDA | **570.124.06 / 12.8** |
| Isaac 栈 | isaacsim 5.0.0 + IsaacLab 2.2.0（`third_party/isaacsim_env`），torch 2.7.0+cu128 |
| 训练 python | `/usr/bin/python3` = **3.12.3**（有 torch；**没有** av/skimage 的编译扩展） |
| 冒烟（30 步 GT） | `episode_start` / `init from library: row=0` / `episode_end` / `dump n=30` 全有，`Warp CUDA error`=0 |

显存实测：一个 Isaac ≈ 4.5–7 GB，一个 VLA server ≈ 9–10 GB，一路训练 ≈ **30 GB/卡**。
⇒ 两卡各放一路训练没问题；但训练（30）+ Isaac（7）+ server（10）> 49，**不能挤同一张卡**。

---

## 2. P0 修复（这次的根因和文档写的不一样）

### P0-1 `av` / `skimage`：不是"没装"，是 **cp311 副本遮蔽**

- 现象：训练在 `load_gr00t_processor` 就崩：
  `albumentations → skimage.measure.label → ModuleNotFoundError: skimage.measure._ccomp`。
- 真根因：`third_party/skimage_py311/` 的编译扩展是 **`*.cpython-311-*.so`**，
  而本机训练 python 是 **3.12**；启动器把该目录放在 `PYTHONPATH` 最前面，
  **遮蔽**了系统里可能存在的版本 ⇒ 3.12 永远加载不了 `_ccomp`。
  （`third_party/av_py311` 没事：av 的 wheel 是 **abi3**，版本无关。）
- 修复：`scripts/deps_env.sh`——`av` 和 `skimage` **分别探测**，能用系统版就一个都不加。
  **必须分开探**：合在一起会因 `av` 缺失把 skimage 一起退回，等于白修（第一版就是这么错的）。
- 另给 py3.12 装了 `scikit-image 0.26.0 / imageio / tifffile`。
  **镜像坑**：`pip install av scikit-image` 在 nexus 上会卡在依赖解析（>20 min 无进展），
  用 `pip download --no-deps` + 直连 wheel 30 s 就完事。

### P0-2 存盘：`SAVE_EVERY=50` + 不要 `| tee`

- `scripts/launch_joint_ha_v2.sh` 新增 `TEE=0`：后台跑时 `exec python3 ... >> "$LOG"`，
  不再无条件 `| tee`（tee 死于 SIGPIPE 会带走训练，`runs/s3_r1` 断在 step 58 就是这个）。
- 实测 150 步 × 3.4–3.7 s/步 ≈ 9–10 min，每 50 步必出 `joint_/vla_*.pt`。

### P0-3 `CompactMotionLib` 与 trackerlab `MotionLib` 的 API 不同（第 4 轮踩到）

- 现象：第 4 轮第 1 步 `AttributeError: 'CompactMotionLib' object has no attribute 'clip_last'`；
  按提示改 `clip_start+clip_len` 后，下一个错是 `no attribute 'frame_clip'`（它有 `frame_bin`）。
- 原因：`diff_vla_wbc/v3_step.py::_ref_states` 的"快速路径"用了 trackerlab `MotionLib` 的
  `clip_last` / `frame_clip`；本机 reference 库是 **`CompactMotionLib`**。
- 修法（已改，两种 lib 都能跑）：`frame → bin → clip`，
  `last = clip_start[clip] + clip_len[clip] - 1`；有 `clip_last` 时仍走原路径。
- **同时记一笔**：这次期间仓库里有**并发改动**（`git status` 显示 23 个文件 modified，
  `v3_step.py`/`isaac_joint.py` 的 mtime 是 19:29，不是本会话改的），
  第 1–3 轮用的是改动前版本、第 4 轮加载了改动后版本 ——
  **多人/多会话同仓跑长任务，先 `git stash` 或开独立分支，否则归因必被污染。**

---

## 3. 让"多轮"成立的两个补丁（原来缺）

### 3.1 `--init-vla`：π_k → π_{k+1} 链式续训

`scripts/train_joint_v2.py` 新增 `--init-vla`（启动器 `INIT_VLA`）：在 `apply_lora` **之后**、
构造 `JointTrainerV3` **之前**把上一轮 `vla_*.pt` 的 `fm` 装进去，并硬校验
`unexpected == 0`、`lora_B` 不全 0，打印 `saved_step`。
（必须在这个时刻：`TRUST_CENTER=prev` 的 π_k 快照就是构造 trainer 时拷的。）

实测：`[AUDIT][init-vla] loaded ...: fm=432 missing(frozen)=1031 unexpected=0
lora_B_absmean=0.000386 saved_step=100`。**没有它，每一轮都从官方底模重来。**

### 3.2 server 的 LoRA 配置必须显式对齐

`ha_vla_latent64_server.py` 默认 `--lora-targets attn1,attn2,ff`，trainer 默认是 **`attn1,ff`**。
`scripts/start_vla_server.sh` 新增 `LORA_RANK/LORA_TARGETS/LORA_ALSO_TRAIN` 透传，
driver 固定传 `32 / attn1,ff / proj_out`。
实测 P2-2 门：`[server][head] loaded ... fm=432 lora_layers=216 unexpected=0` + `listening`。

### 3.3 `--token-bc-weight`：把 v3 缺的 token 级监督补回来（本轮新加）

v3 的设计是"token 只由奖励+hinge 监督"（`token_bc=OFF`），**没有任何一步告诉 head
"这个状态下正确 token 是什么"**。这次给它补上：

`diff_vla_wbc/v3_step.py` 新增 `V3Config.token_bc_weight`，在 `train_step` 里用
**B dump 自己的 `port_tokens`（该帧的命令流 token）** 当标签，算一次 GR00T 原生
flow-matching 回归（只监督前 64 维 token，手部仍走 `lambda_hand` 分支）：

```python
tgt, mask = bc_target(codec, batch.port_tokens, batch.hand_ref, head_dim=..., ...)
loss_token_bc = fm_bc_loss(head, bb_out, act_in, tgt, tok_mask)
loss = loss + token_bc_weight * loss_token_bc
```

链路：`--token-bc-weight`（trainer）← `TOKEN_BC`（launcher）← `TOKEN_BC`（driver）。
审计行会打 `token_bc=<值>`；metrics 里有 `loss/token_bc`。
语义 = **on-policy 自监督**：状态是真实漂移状态，标签是参考 token。

---

## 4. 门

| 门 | 结果 | 证据 |
|---|---|---|
| 环境 | **PASS** | 30 步冒烟，`Warp CUDA error`=0 |
| P0-1 / P0-2 / P0-3 | **PASS** | 6 轮训练全部过 step 1，ckpt 每 50 步落盘 |
| **G0-b（WM v6）** | **PASS** | r0 训 / r1 早停 / **r2 最终门**：20 步关节 **0.218 → 0.181 rad（−16.8%）**，root **8.7 → 6.9 cm（−20.6%）** |
| critic（P1-2） | **FAIL** | B 留出 `v_ev=+0.2888` < 0.30 ⇒ 继续 `--no-critic` |
| G1 | **PASS** | t=0 起点对齐；评测 6/6 `episode_end`、无摔；采集只有 Boxing 一集摔（87.5°），已记录未丢弃 |
| G4 | **PASS** | 无 NaN；`codec/out_of_range` ≤ 0.13；`act_frac_out` 不贴 1 |
| **G5（本轮核心）** | **FAIL** | 合并 **7 次** π₀-vs-head sweep（**n=42**）：head 比 π₀ **差 +0.084 m（t=+2.61）**，只有 13/42 更好 |

WM v6 的 `pi` 与 `WBC_A24000_wmvel_cmd.pt` **逐位相同**（只有 `wm` 变了）⇒ 闭环基线不受影响。

---

## 5. 九轮实跑（每个阶段两卡都满）

`scripts/s3_loop.sh {train|eval|collect|report}`：
**train** = GPU0 主臂 / GPU1 对照臂；**eval** = 每卡一台 server + 一个 Isaac；
**collect** = GPU0 server+Isaac、GPU1 Isaac（`JOBS=2/NUM_GPUS=2`）。

| 轮 | 起点 π_k | 配置 | 步 | 评测（6 seed 配对） | 采集 |
|---|---|---|---|---|---|
| **1** | π₀ 底模 | center=prev, ACT_EPS 0.6, λ_R 0.3/0, λ_BC=0 | 300 | 单集 0.385/0.419（**不能排名**） | → **B_r2**（6 集） |
| 1-scan | π₁ 中间 ckpt | — | — | 单集 ckpt50 0.383 / ckpt100 0.224 | — |
| **2** | π₁@100 | center=prev + λ_BC=1.0（**空转**） | 150 | 单集 0.322/0.175 | — |
| **3** | π₂ 对照 | **center=label, ACT_EPS 0.3, token hinge 0.05**, λ_R 0.3/0 | 150 | **π₃ 0.408 vs π₀ 0.302（更差 +0.106）** | → **B_r3**（6 集） |
| **4** | π₃ 主臂 | 同上 | 150 | **π₄ 0.527 vs π₀ 0.323（更差 +0.204）** | — |
| **5** | π₄ 主臂 | **收紧：ACT_EPS 0.2, token hinge 0.03** | 150 | **π₅ 0.310 vs π₀ 0.350（好 −0.039）**；臂间 λ_R0.3 0.241 / λ_R0 0.300 | → **B_r4**（6 集） |
| **6** | π₅ 主臂 | 同第 5 轮 | 150 | **π₆ 0.456 vs π₀ 0.324（更差 +0.132）**；臂间 λ_R1.0 0.296 / λ_R0.3 0.405 | — |
| **7** | π₅ 主臂 | **新加 token BC=1.0**，锚定关（ACT_EPS 3.0，trust 0） | 150 | **π₇ 0.382(仅BC) / 0.413(BC+奖励) vs π₀ 0.325（都更差）** | — |
| **8** | π₅ 主臂 | 保守版：**只训 B_r1 干净数据**、`fm-lr 5e-5`、100 步、token BC=1.0 + 紧锚定 | 100 | **π₈ 0.504 vs π₀ 0.304（更差 +0.201，t=+2.80）** | — |
| **10** | π₉ 主臂 | 温和剂量 lr 3e-6×20 步 + token BC，数据含新采的 B_r5（avg4 采） | 20 | **π₁₀ 0.272 vs π₀ 0.257（+0.016，t=+0.31）⇒ 打平** | → **B_r5**（6 集，avg4 采集） |
| **9** | π₅ 主臂 | **剂量梯度**：GPU0 = lr 3e-6 × 20 步；GPU1 = lr 1e-5 × 60 步（都 token BC + 紧锚定，`--sample-avg 4`） | 20 / 60 | 臂间：gentle 0.232 / medium 0.277；**配对 vs π₀：gentle 0.260 vs π₀ 0.240（+0.020，t=+0.94）⇒ 打平** | — |

**读法**：只有第 5 轮不亏，第 6 轮又把它亏回去了；第 7、8 轮特意换了信号（token BC）
和更保守的超参（更小 lr / 更干净的数据），**仍然更差**。⇒
**在这个数据规模与 lr 下，head 的任何微调都是净损害**，8 轮里没有例外。

**补测（2026-10-29 02:33）：把当年只有单集的 r1/r2 补上 6-seed 配对。**
r2 对照（`center=prev`、**完全没有 token 拉力**、λ_R=0、150 步）当年单集是 **0.175**（全场最低），
同场配对后：**r2-ctrl 0.321 vs π₀ 0.299（+0.022，t=+0.25，3/6）⇒ 打平**。
⇒ **那个 0.175 是抽样运气**；"完全不拉 token" 也只是持平，不会变好。
（r1 的 300 步大剂量只会更差，不再补测。）

**采集侧的同起点对照（同一行 f49628、600 步）**：

| 用谁采 | root 均值 | p50 | p90 | tilt max |
|---|---|---|---|---|
| B_r2（π₁） | 0.852 | 0.492 | 2.011 | 15.4° |
| B_r3（π₃） | 0.760 | 0.359 | 1.965 | 8.9° |
| B_r4（π₅） | 0.752 | 0.687 | **1.377** | 13.6° |

均值变化不大但**尾巴收窄**（p90 2.01→1.38）；单集，只能当信号。

训练侧 canary（`eval_curve.jsonl` 的想象回报）几乎没有区分度（第 1 轮 `imag_delta` +0.0009/−0.0007）
⇒ **canary 不能当门**。

---

## 6. 评测口径（这一节是本次最值钱的东西）

### 6.1 单集不可信

同一配置（row 0 / seed 0 / 300 步 / 同一 ckpt）π₀ 实测过：**0.130 / 0.163 / 0.421 / 0.200（旧文档）**。
原因是闭环里 VLA 的 flow-matching 采样**本来就有随机性**，300 步里 15 次重规划，误差会累积。

**最狠的一条证据**（可复现）：把**同一个 π₀** 同时放到两张卡、同 seed=0、同 row=0，
两边的结果是 **0.588 vs 0.285（差 0.303 m）** —— 见 `runs/S3_loop/sweeps/selftest_heads/`。
`n=1` 的比较连"同一个策略跟自己比"都能差出 0.3 m，所以任何单集结论都没有意义。

### 6.2 会话方差分层

| 对象 | 跨会话极差 | 说明 |
|---|---|---|
| π₀（底模） | 0.302–0.350（SD **0.018**） | 很稳 |
| 训练过的 head | 0.24–0.53 | 候选**又差又不稳** |
| 同一策略放两卡 | 配对差 **−0.004** | 装置本身无偏 |

⇒ **π₀ 可以跨会话比；候选必须同场配对 + 多会话。**

### 6.3 权威总表

见 **`runs/S3_loop/sweeps/MASTER.md`**（每次 sweep 的槽位→head 对应写在 `heads.json`）。摘要：

| 策略 | 会话数 | 均值 | 各次 |
|---|---|---|---|
| r5 主臂（λ_R=0.3） | 2 | **0.276** | 0.241 / 0.310 |
| **base(π₀) @avg1** | 7 | **0.322** | 0.328 / 0.324 / 0.302 / 0.323 / 0.350 / 0.324 / 0.304 |
| **base(π₀) @avg4** | 3 | **0.283** | 0.293 / 0.321 / 0.240 |
| r6 主臂（λ_R=1.0） | 2 | 0.376 | 0.456 / 0.296 |
| r7（token BC） | 2 | 0.397 | 0.413 / 0.382 |
| r3 | 1 | 0.408 | — |
| r9 gentle（lr 3e-6×20） | 2 | **0.246** | 0.232 / 0.260 |
| r8（保守版） | 1 | 0.504 | — |
| r4 | 1 | 0.527 | — |

### 6.4 口径结论

- **G5 没通过**：head 平均比 π₀ 差 **0.070 m（n=54，t=+2.55）**，只有 18/54 更好。
  最好也只是"打平"：第 9 轮温和剂量 +0.020（t=0.94）、第 10 轮 +0.016（t=0.31）、
  无 token 拉力的 r2-ctrl +0.022（t=0.25）——**全是打平，没有一个变好**。
- 旧文档"GT 0.082 vs VLA 0.200 = 2.4×"是两边各 n=1：GT 6 seed 是 **0.108±0.038**，
  VLA 的 6 seed 是 **0.325±0.018** ⇒ **真实差距约 3×**，且 VLA 侧比原先以为的更稳、更差。
- 要分辨 −0.05 量级的效果：**每策略每会话 6 集 + 至少 3 次会话**（本次做够了才看见真相反）。

---

## 7. 三个会误导后续判断的坑

### 7.1 `LAMBDA_BC` 在 `V3=1` 下是**空转**（第 2 轮白跑）

`[AUDIT][v3] ... token_bc=OFF`——v3 的设计就是"token 只由奖励（+critic）监督"，没有 token BC。
更麻烦：**唯一**能把 token 拉向标签的 `trust_eps/trust_weight` hinge，
在 `TRUST_CENTER=prev` 时被**强制置 0**（`v3_step.py:136`）。
⇒ "加 BC 权重把 head 锚住"这条路不存在；要么 `TRUST_CENTER=label` + token hinge，要么改 v3。

### 7.2 锚定会饱和，而且它只补得回"漂移那一截"

用 loss 分项能定量看：`center=prev` 时奖励项梯度 ≈ 其它可训项的 **4 倍**（主导）；
换 `center=label` + 紧 hinge 后锚定项占 **95%+**，`act/rmse` 从 0.87 一路压到 0.41–0.53。
但到 ~0.45 就压不动了（B 起点里很多是"已漂移状态"，从这些状态出发标签 rollout 本身就远），
于是**绩效不再提升** —— 这与 π₃/π₄/π₅ 都停在 0.30 上下一致。
机制上它不是"学会新行为"，而是**在分布外状态上补足已有能力**，增益上限就是"把漂移补回来"。

### 7.3 加 token BC 也没救：小数据上的微调是"训练集在降、泛化在升"

第 7 轮把 token BC 加上（这是 v3 原来完全缺的监督），训练侧确实在降：
`loss/token_bc` **0.075 → 0.026**。但同一时间：
- 留出评测的 `token_mse_decoded` **0.00848 → 0.01054（变差）**；
- 想象的 `imag_return` **0.8132 → 0.7997（变差）**；
- 闭环 e_root **0.382–0.413 vs π₀ 0.325（更差）**。

⇒ **train loss 降、held-out 和闭环都在降级 = 过拟合**（训练集只有 18 条 B 轨迹、
而且是在同一批起点上反复采样）。第 8 轮把 lr 降到 5e-5、只用干净的 B_r1、步数砍到 100，
in-distribution 不再退化（token_mse 0.00603→0.00594），但**闭环仍然更差（0.504 vs 0.304）**。

结论：**不是"信号选错了"，是"在这个数据规模/lr 下微调本身就伤"。**

### 7.5 真正见效的是**部署端多样本平均**（不是训练）

闭环噪声的一大来源是 flow-matching 的初始噪声：同一 π₀、同一 seed，单集会从 0.13 抖到 0.42。
给 server 加 `--sample-avg N`（同一观测采 N 次、在解码后的 token 上求平均，N=1 即旧行为）：

| 配置（**同一个 π₀**） | 会话均值 | 配对效应 |
|---|---|---|
| `--sample-avg 1` | 0.363（0.391 / 0.335） | 基线 |
| `--sample-avg 4` | **0.307**（0.293 / 0.321） | **−0.082 m（n=12，t=−1.41，7/12 同向）** |
| `--sample-avg 8` | 0.269（单次会话） | 相对 avg=4 再降 ~0.02，**在 4 附近就饱和** |

- SD 从 **0.138 → 0.092**，与"方差按 1/√N 降"的机制一致 ⇒ 这不是碰巧。
- 成本：每次重规划多 N−1 次前向（单次几十 ms，300 步里 15 次重规划 ⇒ 可忽略）。
- **这是本次唯一在统计上站得住、且机制清楚的正向改动。**

**同一条线上的另一个旋钮是"步数"，但它反着来**：把 Euler 步数 K 从 4 加到 8
（同样 avg=4、同一个 π₀）：**0.284 → 0.368（配对 +0.083，K8 只有 2/6 更好）**。
原因很直接：模型是按 K=4 训的，采样轨迹就是按 4 步标定的，加大步数等于换了采样器。

⇒ **部署端定稿：`--sample-avg 4` + `--fm-steps 4`（不要动 K）。**

### 7.6 最刺眼的一条：**留出 token_mse 与闭环 e_root 方向相反**

把每一轮训练结束时 `eval_curve.jsonl` 的 `token_mse_decoded`（留出、in-distribution）
与该 head 的闭环 e_root 并排看：

| 策略 | token_mse（留出，越小越好） | 闭环 e_root（越小越好） |
|---|---|---|
| π₀ 底模 | 0.00848 | **0.322** |
| π₃ | **0.00720**（比 π₀ 好） | 0.408 |
| π₄ | 0.00929 | 0.527 |
| π₅ 主臂 | 0.00990 | **0.241**（最好） |
| π₅ 对照 | 0.00980 | 0.300 |
| π₆ 主臂 | 0.00953 | 0.456 |
| π₇ 主臂 | 0.01054（最差） | 0.413 |
| π₈ | **0.00594**（最准） | 0.504（最差之一） |

- **Spearman ρ = −0.38（n=10，未达显著）** —— 方向是负的，而且**两个极值都在反方向**：
  token 最准的 π₈ 闭环最差，token 最差的 π₅ 闭环最好。
- 含义：**"把 token 预测得更准"和"闭环走得更好"在这是两件事，甚至互相拉扯。**
  这解释了为什么 8 轮训练（奖励 / 锚定 / token BC，全都在压 token 误差或想象回报）**全部变差**。
- 机制猜想：记录里的 token 是"命令流 encoder_latent"，而闭环里真正好用的 token 是模型
  自己带补偿的那一版；把 head 往参考 token 上拉，正好**抹掉了学到的补偿量**。
- 这条不显著（n=10），但它是目前唯一能同时解释 8 轮失败的假设，**下一步必须直接验**
  （固定状态比 π₀ 与微调头的 token 差异 + 20 步 WM 展开，不跑 Isaac）。

### 7.4 别信槽位名：`main/ctrl` 只是 GPU 位置

`summary.json` 原来只有 `main/ctrl` 两个键。我在 `pi0_vs_pi3main`、`pi0_vs_pi4main` 里
把 **π₀ 放 GPU0、候选放 GPU1**，然后又按"main=候选"去读，得出"π₄ 好 0.204 m"的**相反结论**。
现在：`s3_eval_sweep.sh` 起 server 前就把 `heads.json` 写盘，`summary.json` 里直接带
`label`/`head` 路径，末行打印也是"`<策略名> − <策略名>`"。**这一条以后不会再犯。**

---

## 8. 目前最好的配方（π₅，唯一不亏的一组）

```bash
ROUND=5 STEPS=150 \
TRUST_CENTER=label ACT_EPS=0.2 TRUST_EPS=0.03 TRUST_WEIGHT=1.0 \
LAMBDA_R=0.3 \
INIT_VLA=runs/S3_loop/r4/train_main/vla_000150.pt \
B_GLOB='runs/B_r*/vla_*_f*_r*/dump/*.pt' \
  bash scripts/s3_loop.sh train
```

注意：**同一配方从 π₅ 再训一轮（第 6 轮）就会回吐**（0.284 → 0.386）⇒ 不能无脑连训。

---

## 9. 产物清单

| 路径 | 内容 |
|---|---|
| `runs/S3_loop/r{1..8}/train_{main,ctrl}/vla_*.pt` | 八轮的 head（每 50 步一个 + `vla_final`） |
| `runs/B_r1/` | 26 集（π₀ 时代，六类任务 × 多个 reset） |
| `runs/B_r2/`、`runs/B_r3/`、`runs/B_r4/` | 每轮**两卡并行**采的 B 数据（各 6 集 × 600 步） |
| `runs/S3_loop/sweeps/MASTER.md` | **权威评测总表**（含槽位→head 映射） |
| `runs/S3_loop/sweeps/*/summary.json` + `heads.json` | 每次 6 seed 配对的原始数字与身份 |
| `runs/S3_loop/sweeps/gt_baseline/` | GT 6 seed：**0.108 ± 0.038** |
| `runs/S3_loop/ROUNDS.md` | 所有单集 eval dump 的汇总 |
| `runs/S1_handover/WBC_A24000_wmvel_v6.pt` | WM 刷新件（**已过 G0-b**） |

脚本（新增/改动）：

| 脚本 | 作用 |
|---|---|
| `scripts/deps_env.sh` | **新**：av/skimage 分别探测（P0-1） |
| `scripts/s3_loop.sh` | **新**：多轮 driver，`train/eval/collect/report`，每阶段两卡全用；`TOKEN_BC`/`FM_LR` 可调 |
| `scripts/s3_eval_wave.sh` | **新**：两卡并行单波评测（head 可传 `base`） |
| `scripts/s3_eval_sweep.sh` | **新**：两卡并行**多种子**配对扫描，**带 heads.json 身份记录** |
| `scripts/s3_round_table.py` | **新**：eval dump → markdown 表（口径同 `plot_drift`） |
| `scripts/train_joint_v2.py` | 新增 `--init-vla`；`--token-bc-weight` |
| `scripts/launch_joint_ha_v2.sh` | 新增 `INIT_VLA` / `TEE=0` |
| `scripts/start_vla_server.sh` | 新增 `LORA_RANK/TARGETS/ALSO_TRAIN` 透传 |
| `diff_vla_wbc/v3_step.py` | `_ref_states` 兼容 `CompactMotionLib`；**新增 token BC 损失** |
| `scripts/ha_vla_latent64_server.py` | **新增 `--sample-avg N`**（部署端多样本平均，本次唯一有效杠杆） |
| `runs/S3_loop/sweeps/avg1_vs_avg4`、`avg1_vs_avg8` | 采样平均的 6-seed 配对证据（同一 π₀，只改 N） |

---

## 10. 下一步（按收益排序）

**已落地、直接用**：
0. **部署端 `--sample-avg 4`**：free 的 ~0.08 m（−22%），两次会话同向、机制清楚。
   采集 / 验收 / 上线都建议开；想再压可试 8（增益递减）。

**训练侧（当前全败，按顺序来）**：
1. ~~先证明"微调能不伤"~~ **已找到（第 9 轮）**：`fm-lr 3e-6 × 20 步` ⇒ 配对 vs π₀ +0.020（t=0.94），
   **打平、不伤**，SD 也最小（0.065）。**以后每轮都从这档起步**，想加大剂量必须过配对门。
   注意：同一配置在"臂间扫描"里曾测到 0.232（比 π₀ 好），但同场配对就回到 0.260 ——
   **没有同场 π₀ 的数字一律不算数**。
2. **数据量是硬约束**：每轮只有 6 集、全部加起来 18–44 条轨迹，还被同一批起点反复采样
   （lr 1e-4 × 150 步 = 权重相对变化 ~10%）。先把 B 数据堆到 **≥60 集**
   （两卡并行 6 集/10 min ⇒ 约 1.5 h），再动 lr。
3. **查"为什么一训就坏"**：最可疑的是 **token→动作的标定被移动**。
   记录里的 token 是"命令流 encoder_latent"，而闭环里好用的可能是模型自己那一版；
   把 head 往参考 token 上拉，正好抹掉模型学到的补偿量。
   验证：固定同一状态，比较 π₀ 与微调后 head 的 token 及 20 步 WM 展开误差（不跑 Isaac，10 min）。
4. **评测预算写死**：6 seed × 2–3 次会话才有 0.05 分辨率；
   单次 6 集会因候选的会话方差（0.24–0.53）误判 —— 本次就是靠合并 n=30+ 才看清真相。
5. **O1（ORACLE 三臂）**：按本次口径也要 6 seed × 3 会话，它才是"token 有没有权限"的判据。
6. 训练提速：两卡目前是"两条臂并行"而不是 DDP（trainer 没有分布式），
   实测 3.4–3.7 s/步，300 步 19 min，**不是瓶颈**，暂不必做。

## 11. 下次怎么接着跑（照抄即可）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
export COMIMIC=$PWD
bash scripts/isaac_env.sh check
COMIMIC=$PWD bash -c 'source scripts/deps_env.sh; echo $PYTHONPATH'   # av+skimage 应都来自系统

# 0) 先看权威结论
cat runs/S3_loop/sweeps/MASTER.md
cat runs/S3_loop/sweeps/gt_baseline/summary.json

# 1) 训练一轮（当前最好配方；改 INIT_VLA / ROUND 即可）
ROUND=7 STEPS=150 TRUST_CENTER=label ACT_EPS=0.2 TRUST_EPS=0.03 TRUST_WEIGHT=1.0 \
  INIT_VLA=$PWD/runs/S3_loop/r5/train_main/vla_000150.pt \
  B_GLOB='runs/B_r*/vla_*_f*_r*/dump/*.pt' \
  bash scripts/s3_loop.sh train

# 2) 配对评测：π₀ 必须同场占一个卡；**两边都开 sample-avg 4**（当前最好的部署配置）
TAG=pi0_vs_pi7_avg4 SEEDS="0 1 2 3 4 5" START_FRAME=0 \
  HEAD_MAIN=base HEAD_CTRL=$PWD/runs/S3_loop/r5/train_main/vla_000150.pt \
  EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
  bash scripts/s3_eval_sweep.sh
cat runs/S3_loop/sweeps/pi0_vs_pi7/heads.json    # 先确认谁在哪个槽位，再读 summary.json

# 3) 采集（两卡并行，6–12 min 出 6 集）
ROUND=7 NEXT_ROUND=5 \
  HEAD_COLLECT=$PWD/runs/S3_loop/r7/train_main/vla_000150.pt \
  bash scripts/s3_loop.sh collect

# 4) 汇总
PYTHONPATH=. python3 scripts/s3_round_table.py --out runs/S3_loop/ROUNDS.md
```

**四条纪律**（这一晚踩出来的，每条都花了真金白银）：

1. **判优劣 = 同场配对 + 多种子 + 多会话**；单集数字连看都别看（π₀ 自己在 0.13–0.42 之间跳）。
2. **读评测前先看 `heads.json`**：`main/ctrl` 只是 GPU 位置，不是策略名。
3. **改 `.py` 前确认没有训练在跑**：bash 边读边执行，改运行中的脚本会 `unexpected EOF`；
   并发改被 import 的模块会让"下一轮第 1 步就崩"（P0-3 就是这么来的）。
4. **server 端口必须校验 head**：只探端口通不通，旧 server 会让整轮评测串台（征兆是 `就绪（0s）`）。
5. **train loss 不是指标**：本次第 7 轮 `loss/token_bc` 降了 3 倍，但 held-out token_mse 和闭环
   **同时变差**。只有"held-out + 闭环配对"能判好坏，train loss 最多用来查"有没有在学"。

---

# 附录 B：2026-10-29 三项实测结论（WM 混合 / critic 消融 / 起点筛崩溃）

## B.1 世界模型"混合训练"——实测是**否答案**

问题：WM 刷新现在喂的是什么？把录制数据（`transitions.pt`）和 B 线采集数据混起来会不会好很多？

**先纠正前提**：刷新**从来不是只用 B 线**。实测配比（`runs/S3_pipeline/big1/4_wm_refresh.log`）：

| 数据流 | 量 | 每批占比 |
|---|---|---|
| `--packed` → `ref_action_lib.pt`（录制/GT 分布） | 634,016 帧 | 主料 n |
| `--traj-dump`（B 线 76 条**真实闭环轨迹**，带完整多步目标） | `--traj-frac 0.2` | 0.2n |
| `--extra-transitions`（B 线单步 `(s,a,s')`） | **对我们是死路** | 0 |

`--extra-transitions` 无效的原因在代码里写死了（`finetune_wm_velocity_policy.py:268`）：

```python
if K > 1:
    # extra 里只有 **一步** (s,a,s')；多步目标不存在，硬拼会教 WM"零动力学"
    print("[extra] K>1：跳过 on-policy 混合（它只有一步目标）")
```

我们跑 `--rollout-steps 3`（K=3），所以这个开关只会打一行警告，**一条样本都进不去**。
B 线数据唯一正确的通道是 `--traj-frac`（带多步目标），已经在用。

**唯一没试过的变量是 B 的权重**。实测把 0.2 提到 0.5（每批 B 样本 51→128 条），
同一份留出集（`traj_big1_val`，SitSofa 留出）上的曲线：

| k | base（不刷新） | frac 0.2 | frac 0.5 |
|---|---|---|---|
| 1 | 0.010 rad / 0.1 cm | 0.011 / 0.1 | 0.011 / 0.1 |
| 5 | 0.054 / 1.0 cm | 0.055 / 1.0 | 0.056 / 1.0 |
| 10 | 0.096 / 2.6 cm | 0.097 / 2.6 | 0.098 / 2.6 |
| 20 | **0.142 / 5.4 cm** | 0.141 / 5.9 | **0.142 / 5.6** |

**k=20 关节误差一动没动（0.141 vs 0.142），k=1 还略差。** 留出 20 步 RMSE 也只在
0.2356 → 0.2291 → 0.2284 rad 之间晃（−3%）。

结论：WM 的短板**不是数据分布**，是**误差累积**——k=1 只有 0.010 rad，到 k=20 爬到 0.142，
这条曲线由动力学/接触建模误差决定。往训练里灌更多 B 数据压不下去。混合比例不是杠杆。

产物：`runs/S1_handover/wm_big1_mix50.pt`（对照 `wm_big1.pt`），日志
`runs/S3_pipeline/wmmix/A_frac50.log`。复现命令见 §B.4。

## B.2 critic 消融：把"两条臂"做成真配对

既然 WM 不是杠杆，下一步要排掉的是**critic 这个裁判**。证据链指向它：

- 训练到 600 步比 200 步**更差**（0.377 vs 0.271，配对 +0.107，t=+1.86）；
- 同期 raw imagined reward **平的**（0.0272→0.0271），而 `boot` 从 −0.069 涨到 +0.044
  ⇒ 策略在优化 critic，不是在优化奖励；
- 离线 critic 的 v_ev 只有 0.29，而固定 32 窗口评测集上 `imag_return_vla` 的
  分辨率是 ±0.047/√N（N=32 时 0.0083），策略与 GT 的真实差只有 +0.0019
  ⇒ **信号低于裁判的噪声地板**。

所以要同会话配对测：`main` = critic 开（imagined GAE + grad-through-state），
`ctrl` = critic 关（纯奖励项）。为此把 critic 从"全局"改成"按臂"：

```bash
# s3_loop.sh / s3_pipeline.sh 新增（留空 = 继承全局，默认行为不变）
CRITIC_ON_MAIN=1 CRITIC_GRAD_STATE_MAIN=1 \
CRITIC_ON_CTRL=0 CRITIC_GRAD_STATE_CTRL=0
```
关键修复：`--critic-grad-through-state` 原来拼在**全局** `extra` 里，
关掉 critic 的那条臂照样会拿到它（消融就废了）；现在改成按臂拼接。

评测用 `EVAL_MODE=ablation`：不再拿 π₀ 当对照，而是**本轮两臂互比**，
同会话、同种子、`--sample-avg 4` 两边一致，标签显式写成 `critic_ON` / `critic_OFF`
（`LABEL_MAIN`/`LABEL_CTRL` 覆盖，避免再犯"main/ctrl 是 GPU 槽位不是策略名"的错）。

## B.3 起点筛选（B 线）崩溃的根因 —— 空 episode

征兆：训练第 1 步就崩

```
RuntimeError: random_ expects 'from' to be less than 'to', but got from=0 >= to=0
  diff_vla_wbc/b_start.py:253 in sample → torch.randint(0, len(ts), ...)
```

根因不是阈值太严，而是**清理逻辑被条件包住**：`self.episodes = [e for e in ... if e["ts"]]`
原来嵌在 `if self.max_err is not None:` 里，于是

- 没设"实际漂移"阈值时完全不清理；
- 或者空集是**别的原因**（太短 / stride / horizon 不够）造成的，也不清理。

`B_START_MAX_ERR=0.3`（瞬时误差筛，即奖励核还有梯度的区间）在 91 集上会把
DoubleDesk / SitSofa / Boxing 的**整集**筛空（日志里成片
`WARN ... 全部时刻都超过 max_err`），于是 `ts` 为空 → 采样直接崩。

修复（`diff_vla_wbc/b_start.py`）：清理改成**无条件**执行（没空集时是 no-op），
且 `sample()` 只在有可用时刻的 episode 里抽（空集权重 0），再加一条"清完为空"的明确报错。

**两个阈值不是一回事**，别再混：

| 参数 | 落到 | 含义 |
|---|---|---|
| `--b-start-max-err` | `max_e_inst` | **瞬时** root 误差（决定奖励核有没有梯度；e>0.5 梯度≈0） |
| `--b-self-err-max` | `max_err` | 接下来 horizon 步的**实际**漂移（默认自动取 p25） |

## B.4 独立验收器：`scripts/s3_verify_round.sh`

把"八个阶段一个都不能漏"变成一条能跑的检查（只读，不训练不采集）。它不信日志，只查：

1. `state.json` 里 8 个阶段齐全、时间戳单调递增；
2. 每个阶段记录的产物**真的存在**；
3. 四项一致性断言：
   - `gate` 用的 WM == `wm` 阶段刷出来的那个（不许拿上一轮的 WM 当裁判）
   - `joint` 的 `args.json: wbc_ckpt` == gate 产物（训练真用了新 WM）
   - 起点筛阈值已记录（并区分瞬时/实际两个量）
   - eval `summary.json` 两臂 `per_seed` 等长且 ≥6（配对判决才可信）

```bash
ROUND=big1 bash scripts/s3_verify_round.sh        # 验收某一轮
ROUND=big1 bash scripts/s3_verify_round.sh -v     # 顺带打印每个产物路径
```

## B.5 复现命令

```bash
# 混合权重 0.5 的 WM 刷新（对照 0.2）
SUPERMIMIC_HA_ALIGN=1 PYTHONPATH=. python3 scripts/finetune_wm_velocity_policy.py \
  --ckpt runs/S1_handover/WBC_A24000_wmvel_v6.pt --out runs/S1_handover/wm_big1_mix50.pt \
  --packed runs/isaac_motion/ha_cmd/packed.pt --token-space label \
  --rollout-steps 3 --steps 100 --batch 256 --lr 1e-5 --vel-lambda 5 \
  --traj-dump 'runs/wm_real/traj_big1_train/*.pt' --traj-frac 0.5 --traj-val-frac 0.25 \
  --val-dump 'runs/wm_real/traj_big1_val/*.pt' --val-horizon 20 --val-every 50 \
  --val-max-starts 32 --device cuda:0

# k 曲线（三个 ckpt 一次比完，第一个是基线）
SUPERMIMIC_HA_ALIGN=1 PYTHONPATH=. python3 scripts/eval_wm_multistep.py \
  --ckpt runs/S1_handover/WBC_A24000_wmvel_v6.pt \
         runs/S1_handover/wm_big1.pt runs/S1_handover/wm_big1_mix50.pt \
  --dump 'runs/wm_real/traj_big1_val/*.pt' --horizon 20 --stride 20 --device cuda:1

# critic 消融整轮（八个阶段全跑；collect 约 45 min，整轮约 100 min）
ROUND=big2 STEPS=200 BATCH=16 ACCUM=4 B_START_MAX_ERR=0.3 \
  CRITIC_ON=1 CRITIC_SOURCE=imagined CRITIC_GRAD_STATE=1 \
  CRITIC_ON_MAIN=1 CRITIC_ON_CTRL=0 CRITIC_GRAD_STATE_MAIN=1 CRITIC_GRAD_STATE_CTRL=0 \
  EVAL_MODE=ablation LABEL_MAIN="critic_ON" LABEL_CTRL="critic_OFF" \
  bash scripts/s3_pipeline.sh all
ROUND=big2 bash scripts/s3_verify_round.sh
```

---

# 附录 C：critic 消融判决（`big2` 轮，2026-10-29 11:01）

## C.1 判决

同一会话、同 6 个 reset 种子、两臂各自带本轮 WBC 执行器、两边都 `--sample-avg 4`：

| 臂 | 每种子漂移 (m) | 均值 | SD |
|---|---|---|---|
| `critic_ON` | 0.149, 0.154, 0.275, 0.237, **0.433, 0.556** | 0.301 | **0.163** |
| `critic_OFF` | 0.274, 0.256, 0.180, 0.392, 0.341, 0.259 | 0.284 | **0.074** |

配对（OFF − ON）= **−0.017 m**，sem=0.072，**t=−0.24**，OFF 更好 3/6。
逐种子差：`[+0.125, +0.102, −0.095, +0.155, −0.092, −0.297]`。

**结论：均值不可分辨（|t|<2），但开 critic 会把方差放大 2.2 倍**（SD 0.163 vs 0.074），
最差两个种子 0.433 / 0.556 都出现在 critic_ON 臂。

## C.2 机制证据（为什么）

同轮 `train_metrics.jsonl`，两臂 raw reward 轨迹**几乎逐位相同**，唯一差别是 `boot`：

| step | critic_ON reward | critic_ON boot | critic_OFF reward | critic_OFF boot |
|---|---|---|---|---|
| 1 | +0.18813 | −0.00047 | +0.18813 | 0 |
| 51 | +0.17735 | −0.08783 | +0.17735 | 0 |
| 101 | +0.16158 | −0.11229 | +0.16121 | 0 |
| 151 | +0.17136 | −0.11362 | +0.17130 | 0 |
| 200 | +0.18239 | −0.07833 | +0.18187 | 0 |

`|boot| = 0.113` 相当于 reward 幅值（≈0.17）的 **63%** —— 训练信号里有六成来自一个
`v_ev ≈ −0.15` 的裁判，而 raw reward 一动不动。这解释了"训得越久越差"：
`L = −(ret + boot)`，对 `V(s_H)` 的梯度是常数 `−γ^H`，优化器就在把终点状态推向
critic 低估的方向，与提高奖励无关。

## C.3 离线 critic 的真实水平（比之前记的 0.29 更差）

`fit_critic_real.py --transitions rt_v2_big2.pt --center linear --epochs 12`：

```
[critic] best epoch=1 val v_ev=-0.1462
[critic]   recorded  留出 n=132294  v_ev=-0.1530
[critic]   extra     留出 n= 34644  v_ev=-0.1881
[critic] 门（B 留出 v_ev > 0.30）：FAIL
```

v_ev **−0.146**，低于 mjlab 的门（>0.30）一大截。和 §附录B.2 的分辨率测算一致：
策略与 GT 的真实差只有 +0.0019 m，而估计器噪声 ±0.047/√N —— **信号在裁判的噪声地板以下**。

## C.4 流程图已改成 9 阶段（`s3_pipeline.sh`）

按 mjlab `run_j1_rounds.sh` 第 7 步的口径加了 **critic 门**，现在是：

```
collect → score → transitions → wm_refresh → wm_gate → critic_gate → joint → eval → report
                                                          ↑ 新增
```

- `stage_critic`：`fit_critic_real.py` 在**本轮** `rt_v2_<ROUND>.pt` 上拟合，
  抓 `[critic] best epoch=... val v_ev=`；**退出码 0 = 门过，3 = 门没过（ckpt 已存，不算崩溃）**；
- 写 `runs/S3_pipeline/<ROUND>/critic_gate.json`（v_ev / 门槛 / allow / 规则）；
- `stage_joint` 先 `st_need critic`，**门说不过就强制两臂 `critic=OFF`**，
  不允许绕过（`CRITIC_ON_*` 被覆盖并打印原因）；
- `state.json` 在 collect 阶段写 `pipeline_version: "v9"`；验收器对 v8 老轮次不要求新阶段，
  对 v9 要求，并**校验"门的决定 == 训练 args.critic"**。

同时修掉一个会误导人的报告 bug：`stage_report` 原来写死 `π0 / 候选`，
消融模式下（两臂都是本轮产物）会把人带沟里。现在直接用 `heads.json` 的真实标签。

## C.5 下一步（`big3`）

`big3` 用 9 阶段流程跑全流程：门槛 0.30，`v_ev=−0.146` ⇒ 自动关 critic，纯奖励项
（`lambda_r=0.3`）训 200 步，再和 π₀ 做 6 种子配对。这一轮才第一次回答
"**在没有不可信裁判的情况下，训练能不能真的把漂移压下去**"。

---

# 附录 D：对标 `docs/HANDOVER_TO_ISAACLAB_MUJOCO_JOINT_20261029.md` 的四处差距（2026-10-29）

用户口径："和 mujoco 比差哪里了？虽然现在 mujoco 也变差了，但只是因为世界模型训练有点问题，
理论上应该挺好的。"下面逐条对数字，**四处差距里有一处是我们自己引入的错**。

## D.1 我们一直在**解冻 WBC**（最严重，且是 mjlab §10.1 明确标过的坑）

```
big1: train_wbc=True  wbc_lr=1e-06 steps=600
big2: train_wbc=True  wbc_lr=1e-06 steps=200
big3: train_wbc=True  wbc_lr=1e-06 steps=200
```

mjlab §10.1 实测：

| WBC | root_pos p50 |
|---|---|
| 冻结（W1 ckpt） | **4.16 cm** |
| 解冻 800 步后 | **89.2 cm**（21×） |

机理（他们的原话）：J1 的 reward 是 **WM 想象里的 tracking**，梯度穿过 WBC 时，
它只需要"让 WM 里的 reward 变大"，于是**学会骗 WM**；当时 WM 在部署分布上 qd_corr 只有 0.638。

我们不仅解冻了，还把 `joint_final.pt`（解冻后的执行器）直接拿去做闭环评测。
这直接解释"训得越久越差"：big1 600 步 **0.377** vs 200 步 **0.271**。

## D.2 我把 critic 门用错了对象（我的错，已修）

我们训练用的 **imagined** critic 的 `critic/v_ev`：

| step | 1 | 101 | 200 |
|---|---|---|---|
| `critic/v_ev` | 0.4803 | **0.9974** | **0.9946** |

与 mjlab 的 0.9963 同级 —— **裁判本来是健康的**。

但 `s3_pipeline.sh` 第一版第 6 阶段拿 `fit_critic_real.py`（**real 监督**，`v_ev=−0.146`）去判，
于是 `big3` 被误判关掉了裁判。回看 mjlab `run_j1_rounds.sh` 第 7 步：

```bash
if [[ "$CRITIC_SOURCE" == "imagined" && "$CRITIC_MODE" != "off" ]]; then
    CRITIC_ARGS=(--critic --critic-source imagined)   # GAE 路线：不做 prefit、不过 v_ev 门
else
    ... fit_critic_real.py ...   # v_ev>0.30 只对 real 这条路
```

`v_ev>0.30` 是给 **real** critic 的判据（他们 §10.2 也说 real 结构性不可学：0.088–0.152，
`ret` 变异系数只有 13%、`corr(ret, 剩余步数)=0.16`）。**已改成 imagined 直接放行。**

## D.3 WM 在**部署分布**上差一个量级

| | mjlab（sim 分布） | 我们（留出 B 线部署分布） |
|---|---|---|
| `qd_corr` k=1 | **0.9925–0.9933** | **0.736** |
| k=20 关节 RMSE | — | 0.230 rad / root 5.4 cm |

他们 qd_corr 0.638 时就崩了，结论是"**要用想象 reward 解冻 WBC，先保证 WM 在部署分布上准**"。
我们 **0.736 且解冻着 WBC** —— 同时踩两条。

## D.4 奖励核：root 项权重只有一半、核更钝（和我们猜的"饱和"方向一致）

从 `WBC_A24000_wmvel_v6.pt` 的 `reward_cfg` 实际读到 vs mjlab §1：

| 项 | 我们 | mjlab | 差异 |
|---|---|---|---|
| `w_root_pos` / `w_root_ori` | **0.5 / 0.5** | **1.0 / 1.0** | root 权重差 2× |
| `root_pos_std` / `root_ori_std` | **0.3 / 0.4** | **0.2 / 0.25** | 我们更钝（4 cm 处不够敏感） |
| body 四项 | 1.0（std 0.3/0.4/1.0/3.14） | 4×1.0 | 一致 |

他们明确记录"root 项原来在 4 cm 处饱和"，修法就是 root 加倍 + std 收紧。
而闭环漂移主要由 root 决定 ⇒ 我们的奖励对最该压的那一维最不敏感。

实测奖励轨迹也印证：

| | 起 | 末 | 变化 |
|---|---|---|---|
| mjlab `rollout/reward` | 0.0837 | **0.0914** | **+9%（还在爬）** |
| 我们 `loss/reward_raw` | 0.188 | 0.182 | **−3%（平）** |

我们起手就在上限（≈0.2）的 90%+：误差本来就小 ⇒ `exp(−e/std²)≈1` ⇒ token 改不动 reward。

## D.5 已应用的代码修改（只改代码，未开机时）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `scripts/s3_pipeline.sh` | `TRAIN_WBC_MAIN/CTRL` 默认 **0**（冻结）；要开必须显式置 1 并打印警告 |
| 2 | `scripts/s3_pipeline.sh` | 评测执行器默认 **冻结的 `WBC_A24000_wmvel_v6.pt`**；只有 `TRAIN_WBC_*=1` 才用 `joint_final.pt`（消融分支同步修） |
| 3 | `scripts/s3_pipeline.sh` | `stage_critic`：`CRITIC_SOURCE=imagined` **直接放行**（写 `critic_gate.json` 记 `source/allow/rule`）；只有 `real` 源才拟合 + 过 `v_ev>0.30` |
| 4 | `scripts/s3_pipeline.sh` | `H` 默认 **5 → 8**（§10.3 实测 H=5 太短）；注释写明代价（k=8 的 WM 误差介于 k=5 的 1.0 cm 与 k=10 的 2.6 cm 之间） |
| 5 | `scripts/s3_pipeline.sh` | 联合微调横幅按**实际** critic 状态打印（不再写死"想象的 GAE"） |
| 6 | `scripts/train_joint_v2.py` | 新增 `--reward-root-weight` / `--reward-root-pos-std` / `--reward-root-ori-std`，就地改写 trackerlab 的 `mdp.cfg`（`mdp.reward` 每次调用都读它，saturating/linear 两种模式都生效），并打印 `[AUDIT][reward] root 覆盖 旧 -> 新` |
| 7 | `scripts/launch_joint_ha_v2.sh`、`scripts/s3_loop.sh`、`scripts/s3_pipeline.sh` | 透传 `REWARD_ROOT_WEIGHT` / `REWARD_ROOT_POS_STD` / `REWARD_ROOT_ORI_STD` |

验证情况：四个脚本 `bash -n` 通过，`train_joint_v2.py` `py_compile` 通过；**未开机时，未跑**。

## D.6 还没解决、必须下一轮才能判的

1. **奖励饱和**：即使把 root 项改成 1.0/0.2，如果想象里误差仍然很小，`exp` 核还是会贴 1。
   备选是 `REWARD_MODE=linear`（同样的跟踪误差、没有死区），但它改的是"核的形状"，需要单轮对照。
2. **WM 部署分布精度**（0.736 vs 0.9925）：我们试过混合比例（无效）和 80 倍预算（门口径变差），
   还没试的是**在部署分布上按 qd_corr 定向增训**，或换 WM 结构。
3. **WBC 解冻的前置条件**：等 WM 在部署分布上足够准之前，`TRAIN_WBC=1` 不应该开。

---

# 附录 E：A 线（WBC-only 真物理）——补上 mjlab 第 1 步（2026-10-29）

## E.1 为什么必须补

对标 `docs/HANDOVER_TO_ISAACLAB_MUJOCO_JOINT_20261029.md` 后确认：mjlab 的 WM 之所以
`qd_corr 0.9925 / qd_ratio 1.006`，是因为它有**第 1 步：只跑冻结 WBC、场景里没有 VLA、
1024 env 向量化跑的 ~100 万条真转移**。我们 S3 的 WM 原来吃的是：

| 数据 | 量 | 性质 |
|---|---|---|
| `packed.pt` 的 `ref` | 634,016 帧 | **运动学参考**（不是仿真动力学） |
| `ref_action_lib.pt` 的 `actions` | 634,016 | 录制 npz 的 `decoder_target_action` |
| B 线闭环 dump | 每轮 42 集 | 真物理，但被 VLA 噪声污染、且是漂移后的分布 |

**实测证据**：同一个 WM，在运动学参考上 `qd_corr=0.9689`，在 A 线真物理数据上只有
**0.4976（qd_ratio 1.21）** —— 它从没学过真动力学。

## E.2 工具链

| 脚本 | 作用 |
|---|---|
| `scripts/isaac_a_collect_batch.py` | 多 clip、每 env 独立 `(clip, start)`、向量化跑冻结 WBC，存 `(s,a,s',全局库行号,clip_id)` |
| `scripts/run_a_collect_batch.sh` | 启动器（4 波 × 512 env × 300 步 ≈ 61 万条，**7 分钟**） |
| `scripts/filter_a_line.py` | 清洗：只排**物理退化**（炸开/倒地/NaN），输出 `valid_len`；保留 63.5%（39 万行，中位有效长度 218 步） |
| `finetune_wm_velocity_policy.py --a-line` | WM 刷新直接吃 A 线（`actions` 已是归一化动作，按 `num_envs` 解交错，按 `valid_len` 截断） |

流水线加了 `a_collect` 打头（10 阶段），A 线已存在时自动跳过。

## E.3 这一链上抓到的 5 个真 bug（**都是"静默错"**，不看数字发现不了）

| # | bug | 现象 | 根因 |
|---|---|---|---|
| 1 | **漏 `robot.write_data_to_sim()`** | 采集出的机器人 30 步内倒地，90% 行是垃圾 | `set_joint_position_target` 只写缓冲，不进 PhysX ⇒ WBC 的动作从未生效；**旧采集器 `isaac_g1_collect.py` 也漏了**，它那些"完美站立"（z 恒 0.79、up 恒 1.0）其实是机器人**僵住不动** |
| 2 | token 每步都换 | 同上 | WBC 契约是 **20 步一个 hold 窗口**（`--exec-steps`），每步换 token 是它没见过的输入 |
| 3 | 状态列索引错 | "509/512 env 关节角超限"是假警报 | 布局是 `[p(3),quat(4),q(29),v(3),w(3),qd(29)]`，关节角是 **7:36** 不是 13:42 |
| 4 | A 线加载嵌在 `if a.traj_dump:` 里 | A 段（`--traj-dump ""`）**根本没加载 A 线**，两次跑出逐位相同的 loss | 同名变量 `perm` 也定义在里层，重构时把两者都提到顶层 |
| 5 | 短轨迹验证段越界 | CUDA `index < sizes[i]` 断言崩 | `_tr = max(K+20,…)` 可超 `_n`；且 val 段比 H 还短时 `idx+H` 越界 |

还有两个**流程级**问题：
- **观测历史全零初始化**（应填当前状态）——已改，但实测对结果无影响（保留正确写法）；
- **`pkill -f <脚本名>` 会杀掉自己**（命令行里含同名字符串）→ 必须用 `"w""m_velocity"` 这种拼接模式，或按 PID 逐个杀。

## E.4 训练预算对比（mjlab vs 我们）

| | mjlab | 我们（改前） | 我们（现在） |
|---|---|---|---|
| 步数 | 1000 | 100（50 步早停） | 100（**K=20** 自回归） |
| batch | 2048 | 256 | 128 |
| 多步展开 K | — | **3** | **20** |

## E.5 「A 先训到收敛，再 A+B 混合」——**实测反答案**

用户口径（也是 mjlab 的课程）：先用 A 线把 WM 训稳（3000 步），再按 25% B 混合。
四种方案都跑了，判据用**部署留出的 20 步关节 RMSE**（真正进奖励的口径）：

| 方案 | k=1 | k=10 | **k=20** | k=20 root | 门 |
|---|---|---|---|---|---|
| base | 0.011 | 0.104 | 0.154 | 6.5 cm | — |
| 单阶段 K=3（A 混在里面 100 步） | 0.011 | 0.100 | 0.147 | 6.5 | ✅ |
| A 训 200 → A+B 3:1 | 0.014 | 0.123 | 0.186 | 6.7 | ❌ |
| **A 训 3000 → A+B 1:1** | **0.038** | **0.280** | **0.427** | 8.6 | ❌ +176% |

但 `qd_corr` 恰好相反：A3000 那个是 **0.750/0.550/0.314/0.231**，比 base 还好。

**结论：不是 A 线没用，是"用哪个口径验收"决定了结论。**
mjlab 的门是 **1 步速度相关**，按那个口径 A 先训是对的；想象训练里奖励吃的是
**20 步累积位置**，按这个口径 A 猛训会毁掉位置积分。⇒ 我们锁位置口径，A 线走**单阶段混合**。

## E.6 真正的赢法：**K=20 多步自回归**

`--rollout-steps 20`（batch 128），其余不变：

| k | base | K=3（旧） | **K=20（新）** | 变化 |
|---|---|---|---|---|
| 1 | 0.011 | 0.011 | **0.009** | −16.6% |
| 5 | 0.059 | 0.057 | **0.051** | −13.1% |
| 10 | 0.104 | 0.100 | **0.093** | −10.8% |
| 20 | 0.154 | 0.147 | **0.141** | −8.9% |
| 20 root | 6.5 cm | 6.5 | **5.8** | −10.8% |
| `qd_ratio` | 1.31 | 1.08 | **0.93** | — |

**这是目前唯一在每一个视界都变好的 WM 改法**，已设为流水线默认（`WM_ROLLOUT_STEPS=20`）。
c1 轮的实跑甚至更好：k=20 **0.138（−10.7%）**、root 5.8 cm（−10.1%）。

## E.7 为什么闭环还是打平（c1：0.333 vs π₀ 0.323）

WM 门口径大幅改善（−10.7%），但：

- 训练端 imagined reward 只 **+0.8%**（仍平）；
- 闭环配对差 **+0.010**（打平）。

⇒ **世界模型变准了，但 200 步训练没能把它变成闭环改善**。`STEPS=200` 是从 big1 时代
沿用的值，而那时 WBC 解冻 + critic 乱来 + H=5，三个因素混在一起，**干净配置下训练预算
从没单独标定过**。这是下一轮要扫的变量（600 / 1200 步）。

## E.8 评测噪声：单轮判决不可信的量化证据

同一个 π₀、**同一个种子**，两个不同场次的绝对数：

| 种子 | π₀（big5 场） | π₀（big6 场） |
|---|---|---|
| r0 | 0.355 | **0.214** |
| r1 | 0.528 | **0.706** |

差 **14 cm**。所以单轮 −7 cm / +0.1 cm 都是这个噪声里的抽样。
⇒ 正确做法：**多个独立场次的配对差汇总**（10 轮 ⇒ sem 从 0.045 降到 ~0.014）。

---

# 附录 F：多轮闭环的"订死"清单（2026-10-31）

今天在跑 10 轮累积时踩到的**静默错**，逐条钉成守卫。凡是"配置说一套、代码做一套"的，
一律在流程里加断言，**不允许静默退回默认值**。

| # | 静默错 | 症状 | 守卫（现在） |
|---|---|---|---|
| 1 | 采集只要**有任何一集**就跳过 | 中断留下的 8 集被当成 42 集，那一轮没有新数据（big6/d3 都中过） | `have >= 42-1` 才跳过；不完整则 `rm -rf` 重采；采集后硬断言 `have >= 41`，否则**不许进入下一阶段** |
| 2 | **A 线文件数被硬编码 `/4`** | A 线从 4 波加到 8 波后，B 占比从 25% 悄悄掉到 **19%** | 文件数从目录里 `ls \| wc -l` 数出来；阶段2 结束后**从日志核对实际配比**，偏离设计 0.6~1.8× 就中止 |
| 3 | **注释插进 env 续行链** | `\` 被注释吃掉 ⇒ `RUN_LABEL` 等变量静默丢失，训练跑到默认目录 `r99`（今天犯了 3 次） | 启动自检扫自己的 joint env 链，发现注释行立刻退出；训练后再断言输出目录存在 |
| 4 | `posture_weight` 从来没传 | "歪/快倒"惩罚**一直没生效**（用户口径：该项应删除） | 代码里整段删除；pipeline 恒传 0；训练后断言 `posture_weight==0 / tail_weight / root_ori_mode` 与设定一致 |
| 5 | `trust_center=prev` 把 `trust_eps` 强制置 0 | token 空间的信任域**整项被关掉**（v3 里 token 只有奖励+hinge 两个信号） | 改 `trust_center=self`（eps 0.2 / w 1.0）；AUDIT 行打印实际生效的全部权重 |
| 6 | `self_tokens` 行数少 1（缺标签的窗口不 append） | `tokens - prior_t` 尺寸不匹配直接崩（16 vs 15） | 缺标签用 **0 占位**（长度恒为 B）+ `self_tokens_has` 掩码；另加行数不符的兜底退回 |
| 7 | 奖励里 `acos` 在完美跟踪处导数发散 | NaN 梯度，NaN-guard 第 19 步中止训练 | 改无奇异形式（up 向量差平方 / `2·asin`）+ **梯度有限性回归测试** |
| 8 | WM 门 FAIL 时静默回退 base | 拿没验证过的 WM 当裁判，等于白跑 | 门 FAIL **直接中止**，不再回退 |
| 9 | A 线短轨迹切分越界 | CUDA `index < sizes[i]` 断言崩 | `_tr = min(_n-1, …)`；val 段 ≤ H 时跳过；`--val-max-traj` 限制验证轨迹数 |

## 每轮开跑前的**配置指纹**（`preflight`）

现在每轮 `all` 的第一件事是打印并校验：

```
===== 配置指纹（d3）=====
  A 线      : 8 个文件 / 1228800 行
  B 线      : 全池 runs/B_*/vla_*_f*_r*/dump/*.pt
  WM 三阶段 : 场景 3000 / 混合 3000 / k=20 自回归 1500
  WM 配比   : A:B = 3:1（B 目标 25%）
  VLA 训练  : 800 步, H=8, batch 16×4, 冻结 WBC
  奖励      : root_ori=split, tail=2.0, posture=0(已删)
  信任域    : trust_center=self, eps=0.2
  新数据权重: 3.0×
  评测      : 12 集/臂
[preflight] 数值范围: ✅
```

并且对 `STEPS / B_FRESH_WEIGHT / WM_AB_RATIO / TAIL_WEIGHT` 做**范围校验**（例如 STEPS 必须在
50~5000、新数据权重 1~20），任何一项越界直接中止 —— 这样"某次改参数改出离谱值"也会立刻暴露。

## 本轮累积曲线（截至 d2b）

| 轮次 | 累计步数 | 候选均值 | 配对差 | t | 更好 | 候选 sd |
|---|---|---|---|---|---|---|
| d1 | 800 | 0.200 | −0.084 | −2.25 | 9/12 | 0.087 |
| d2 | 1000 | 0.191 | −0.112 | −2.86 | 11/12 | 0.068 |
| **d2b** | **1600** | **0.153** | **−0.123** | **−3.98** | **12/12** | **0.033** |

（d3 因采集被跳过仅 8 集，作废重跑；重跑时同时把 A 线从 4 波加到 8 波、B 配比修回 28%。）
