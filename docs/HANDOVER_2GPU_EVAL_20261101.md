# 2×4090 闭环评测交接：最佳 checkpoint 系列 + 开跑命令

> 日期：2026-11-01 08:50
> 目标：把当前最好的一套（策略 + 执行器）拿到**另一台 2×4090** 上跑闭环仿真。
> 当前状态：**可以开跑**；`h20r3`（act_weight=0.25）还在 8 卡机上训练，完成后若有显著提升，
> 按 §8 替换 primary 即可。

---

## 1. 一句话：用哪套

**最佳系统 = `h20r2` 策略 + `v7` 执行器。**

| 角色 | 文件 | 说明 |
|---|---|---|
| **primary 策略** | `policy_primary_h20r2.pt` | 最新完整轮；相对 B 点估计 −0.60 cm（32 集，CI 跨 0，未显著） |
| **primary 执行器** | `exec_primary_v7.pt` | P2-lite；相对 v6 均值 −1.77 cm（32 集，CI [−3.78,−0.18]，尾部驱动） |
| **fallback 策略** | `policy_fallback_B.pt` | 有历史显著证据：B vs d2b −1.36 cm（16 集，CI [−2.05,−0.74]） |
| **fallback 执行器** | `exec_fallback_v6.pt` | 所有历史读数的基线执行器；最稳 |
| **anchor 策略** | `policy_anchor_d2b.pt` | d2b，所有配对的历史锚点 |
| **训练用 WM（不是评测用）** | `wm_h20r2.pt` | 固定 134 集留出 k=20 root 3.8 cm / joint 0.061，比旧版好 |

> **不要用**：`h20r2_wbc`（8.5 的 vA，16 集 wash）、`h20r2_jointwbc`（Flow B 联合，16 集略差）。
> 它们只作消融证据，不是部署候选。

### 证据（都是配对闭环，不是训练内指标）

| 比较 | 执行器 | n | 配对差 | 95%CI | 更好 |
|---|---|---|---|---|---|
| B vs d2b | v6 | 16 | **−1.36 cm** | [−2.05, −0.74] | 14/16 |
| h20r2 vs B | **v7** | 32 | −0.60 cm | [−1.67, +0.50] | 21/32 |
| v7 vs v6（B 策略） | — | 32 | **−1.77 cm** | [−3.78, −0.18] | 17/32 |
| 8.5 vA vs v7（h20r2 策略） | — | 16 | −0.26 cm | [−1.57, +1.02] | 8/16 |
| Flow B vs h20r2+v7 | — | 16 | +0.44 cm | [−0.41, +1.34] | 5/16 |

注意：**B vs d2b 的 −1.36 cm 是 v6 执行器**；h20r2 vs B 是 v7。两条不能直接相加。
2 卡机要做的第一件事就是**用同一 v7 执行器**把 `h20r2 vs B`、`B vs d2b`、`h20r2 vs d2b` 各跑一遍。

---

## 2. checkpoint 清单与校验

目录：`runs/DEPLOY_2GPU/`（本机软链；拷贝时用 `rsync -L` 变成实体文件）。

| 部署名 | 实际来源 | 大小 | sha256（前 16 位） |
|---|---|---|---|
| `policy_primary_h20r2.pt` | `runs/S3_loop/h20r2_joint/train_main/vla_final.pt` | 129M | `8254df6fcaa9fe12` |
| `policy_fallback_B.pt` | `runs/S3_loop/h20_8g_b_joint/train_main/vla_final.pt` | 129M | `738f682e977883c4` |
| `policy_anchor_d2b.pt` | `runs/S3_loop/d2b/train_main/vla_final.pt` | 129M | `668d1cb91f5ff5ec` |
| `exec_primary_v7.pt` | `runs/S3_loop/h20_8g_b_wbc_joint/train_main/joint_final.pt` | 52M | `2bce00542f4b5f75` |
| `exec_fallback_v6.pt` | `runs/S1_handover/WBC_A24000_wmvel_v6.pt` | 52M | `b5671c7d64c23c7b` |
| `wm_h20r2.pt` | `runs/S1_handover/wm_h20r2.pt` | 52M | `b47efe4355752d58` |

完整哈希：`runs/DEPLOY_2GPU/SHA256SUMS`。复制后校验：

```bash
cd <repo>/runs/DEPLOY_2GPU && sha256sum -c SHA256SUMS
```

复制（跟随软链）：

```bash
rsync -avL runs/DEPLOY_2GPU/ user@2card:<repo>/runs/DEPLOY_2GPU/
```

---

## 3. 2 卡机还需要什么（不在 DEPLOY_2GPU 里的）

| 依赖 | 本机路径 | 说明 |
|---|---|---|
| 同一份 repo | `<repo>` | 必须有 `scripts/`、`diff_vla_wbc/`、`third_party/` |
| 底模 VLM | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | **6.5 GB**（两个 safetensors + config）；`vla_final.pt` 只是 head delta，必须配这个底模 |
| Cosmos 权重 | `groot_repro/nvidia/Cosmos-Reason2-2B-weights` | 本机是指向 HF cache 的软链；2 卡机要保证 `VLM_PATH` 可读 |
| 动作库 | `runs/isaac_motion/ha_cmd/packed.pt` | 86 MB；评测用的参考帧库 |
| Isaac 环境 | `third_party/isaacsim_env` + `third_party/isaaclab_220` + assets | 跑 `source scripts/isaac_env.sh check` 必须通过 |
| 机器人 USD | `third_party/HumanoidArena/.../g1_29dof_with_dex3_rev_1_0.usd` | 见下面的 env |

如果 2 卡机挂的是**同一份 GPFS**，直接按绝对路径用；否则至少同步 `DEPLOY_2GPU/` + 上表前四项。

---

## 4. 环境（每条命令前都要有）

```bash
cd <repo>
export COMIMIC=$PWD
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD=$PWD/third_party/HumanoidArena/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd
source scripts/isaac_env.sh check     # 确认 torch / isaacsim / isaaclab OK
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

> `SUPERMIMIC_HA_ALIGN=1` 和 USD 两个变量**缺一不可**；缺了会出现 knee/ankle/elbow 的 spec 不一致，
> 所有闭环数字作废（我们踩过）。

---

## 5. 开跑：2 卡上的评测命令

评测协议（和历史可比，**不要改**）：

- `TASK=DoubleDesk`、`START_FRAME=0`、`MAX_STEPS=300`
- `EXTRA_MAIN/CTRL="--sample-avg 4"`、`SRV_SEED=0`
- 两槽位**同一执行器**；main/ctrl 同 reset seed、同 server seed，**只看配对差**
- 2 卡：`GPU_MAIN=0`、`GPU_CTRL=1`；**不要**用 `CUDA_VISIBLE_DEVICES`

### 5.1 先跑 12 集 sanity（~18 min）

```bash
TAG=h20r2_vs_B_2gpu SEEDS="0 1 2 3 4 5 6 7 8 9 10 11" START_FRAME=0 MAX_STEPS=300 \
  HEAD_MAIN=$PWD/runs/DEPLOY_2GPU/policy_fallback_B.pt \
  HEAD_CTRL=$PWD/runs/DEPLOY_2GPU/policy_primary_h20r2.pt \
  SM_CKPT_MAIN=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
  SM_CKPT_CTRL=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
  LABEL_MAIN=B LABEL_CTRL=h20r2 \
  EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
  PORT_MAIN=18150 PORT_CTRL=18151 GPU_MAIN=0 GPU_CTRL=1 SRV_SEED=0 \
  bash scripts/s3_eval_sweep.sh
```

输出：`runs/S3_loop/sweeps/h20r2_vs_B_2gpu/summary.json`。

### 5.2 32 集主读数（~50 min，2 卡顺序跑）

一条 sweep 直接跑 32 个 seed（每槽位 32 集顺序）：

```bash
TAG=h20r2_vs_B_2gpu_32 SEEDS="$(seq -s ' ' 0 31)" START_FRAME=0 MAX_STEPS=300 \
  HEAD_MAIN=$PWD/runs/DEPLOY_2GPU/policy_fallback_B.pt \
  HEAD_CTRL=$PWD/runs/DEPLOY_2GPU/policy_primary_h20r2.pt \
  SM_CKPT_MAIN=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
  SM_CKPT_CTRL=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
  LABEL_MAIN=B LABEL_CTRL=h20r2 \
  EXTRA_MAIN="--sample-avg 4" EXTRA_CTRL="--sample-avg 4" \
  PORT_MAIN=18150 PORT_CTRL=18151 GPU_MAIN=0 GPU_CTRL=1 SRV_SEED=0 \
  bash scripts/s3_eval_sweep.sh
```

如果 2 卡机想用 `eval_h20_8gpu.sh` 的封装（它支持 `N_SWEEPS=1`）：

```bash
HEAD_MAIN=$PWD/runs/DEPLOY_2GPU/policy_fallback_B.pt \
HEAD_CTRL=$PWD/runs/DEPLOY_2GPU/policy_primary_h20r2.pt \
SM_CKPT_MAIN=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
SM_CKPT_CTRL=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
TAG_PREFIX=h20r2_vs_B_2gpu LABEL_MAIN=B LABEL_CTRL=h20r2 \
SEED_TOTAL=32 N_SWEEPS=1 GPU_OFFSET=0 BASE_PORT=18150 \
bash scripts/eval_h20_8gpu.sh
```

### 5.3 另外两条对照（建议同一天跑完）

```bash
# B vs d2b（历史显著效应，换成 v7 执行器）
HEAD_MAIN=$PWD/runs/DEPLOY_2GPU/policy_anchor_d2b.pt \
HEAD_CTRL=$PWD/runs/DEPLOY_2GPU/policy_fallback_B.pt \
SM_CKPT_MAIN=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
SM_CKPT_CTRL=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
TAG_PREFIX=B_vs_d2b_2gpu LABEL_MAIN=d2b LABEL_CTRL=B \
SEED_TOTAL=32 N_SWEEPS=1 GPU_OFFSET=0 BASE_PORT=18150 \
bash scripts/eval_h20_8gpu.sh

# h20r2 vs d2b（最佳系统 vs 锚点）
HEAD_MAIN=$PWD/runs/DEPLOY_2GPU/policy_anchor_d2b.pt \
HEAD_CTRL=$PWD/runs/DEPLOY_2GPU/policy_primary_h20r2.pt \
SM_CKPT_MAIN=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
SM_CKPT_CTRL=$PWD/runs/DEPLOY_2GPU/exec_primary_v7.pt \
TAG_PREFIX=h20r2_vs_d2b_2gpu LABEL_MAIN=d2b LABEL_CTRL=h20r2 \
SEED_TOTAL=32 N_SWEEPS=1 GPU_OFFSET=0 BASE_PORT=18150 \
bash scripts/eval_h20_8gpu.sh
```

> 三条不要同时开（2 卡只有 2 张卡）；顺序跑，端口相同没问题，每条结束会 kill 自己的 server。

---

## 6. 判读规则（照抄，别自创）

1. **只看配对差**（`ctrl − main`）和自助法 95%CI；**不要**拿不同批 episode 的绝对均值横比。
2. 12 集：只能看方向；32 集：才做“显著/不显著”的判断。
3. `mean<0` 且 `median<0` 且**去掉 |diff|>5 cm 异常集后的核心中位也 <0** ⇒ 才算“普遍提升”；
   只有均值 <0、核心中位 ≈0 ⇒ 只算“尾部保险”。
4. 翻倒数、p90、最后 50 步误差一起报；不要只看均值。
5. 同一策略同 seed 重跑会有 **~4.6%** 的 Isaac 内核残差；这也是为什么必须配对、不能跨 run 比绝对值。

---

## 7. 2 卡机常见坑

| 坑 | 处理 |
|---|---|
| 端口被占 | 用 `PORT_MAIN=18150 PORT_CTRL=18151`；sweep 收尾只杀自己端口（本 repo 已修）；若 2 卡副本是旧的，**一次只开一条 sweep** |
| 用了 `CUDA_VISIBLE_DEVICES` | 不要用；用 `GPU_MAIN/GPU_CTRL`（sweep 固定按 GPU 索引起 server） |
| 24 GB 版 4090 | 每个槽位 = Isaac sim + VLA server 同一张卡；标准 24 GB 可能 OOM；48 GB 改卡没问题 |
| 底模不一致 | `VLM_CKPT` 必须是 `checkpoint-8000`；换底模后 `vla_final.pt` 不可比 |
| USD/HA_ALIGN 没设 | spec 会错位（knee/ankle/elbow），数字全废 |
| server 采样没种子 | 必须 `SRV_SEED=0`；否则同策略同 seed 会跳 0.064/0.143/0.152 |
| 改 `MAX_STEPS` | 不要改；历史全是 300 步 |
| 只看 `summary.json` 的绝对 mean | 不同 seed 集不可比；看 `paired_vs_ctrl` |

---

## 8. 当前状态与后续替换

- `h20r3`（`act_weight=0.25`，起点 h20r2，执行器 v7，无 8.5）正在 8 卡机上跑完整轮；
  完成后会做 **h20r2+v7 vs h20r3+v7** 的 32 集配对。
- 如果 h20r3 **显著更好**（mean<0、median<0、核心中位<0、CI 不含 0）：
  - 把 `runs/S3_loop/h20r3_joint/train_main/vla_final.pt` 加进 `DEPLOY_2GPU/`，命名
    `policy_primary_h20r3.pt`，哈希追加到 `SHA256SUMS`；
  - 2 卡机把 §5 的 `HEAD_CTRL` 换成它即可，执行器继续用 **v7**。
- 如果 h20r3 不显著/更差：保持 **h20r2 + v7** 为 primary，B+v7 为 fallback；act_weight 不再降。
- 2 卡机现在就可以先跑 §5.1/§5.2，不必等 h20r3——h20r2 的 ckpt 已稳定。

---

## 9. 文件索引

| 文件 | 作用 |
|---|---|
| `runs/DEPLOY_2GPU/` | 6 个部署 ckpt 软链 + `SHA256SUMS` |
| `scripts/s3_eval_sweep.sh` | 2 卡单 sweep（main/ctrl 各 1 卡） |
| `scripts/eval_h20_8gpu.sh` | 8 卡封装；2 卡用 `N_SWEEPS=1` |
| `scripts/merge_eval_sweeps.py` | 多条 sweep 合并 + 自助 CI |
| `docs/METHOD_LOCK_H20_20261101.md` | 锁定配方与验收判据 |
| `docs/H20_8GPU_JOINTFT_20261101.md` | 两轮结果与证据 |
| `docs/P2_TWO_PATH_PLAN_20261101.md` | Flow A/B 的完整设计 |
