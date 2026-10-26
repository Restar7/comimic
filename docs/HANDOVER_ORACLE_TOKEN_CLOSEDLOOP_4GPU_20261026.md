# 交接：**oracle token 闭环回放**（4×4090）—— 验证"想象里的增益能不能搬到真机"

> 这是整个 S3 联合微调**最便宜、最决定性**的实验（30–60 min，4 卡能并行）。
> 它不训练任何东西；它只回答一个问题：
> **在可微 WM 里比 GT 高 30% 的 token，拿到 Isaac 闭环里是不是也更好？**

---

## 0. 为什么要做它（背景）

我们已实测（`runs/S3_gates/headroom.json`）：

| 场景 | GT token | VLA(官方 ckpt) | 零 token | token 区分度 |
|---|---|---|---|---|
| 标称 H=16 | 1.2218 | 1.2188 | 1.1439 | +6.4% |
| **5× 扰动 H=32** | 1.2323 | 1.1880 | 0.7419 | **+39.8%** |

并且用奖励梯度直接优化 token（oracle）可以从 GT 的 1.797 → **2.413（+34%）**、从 VLA 的 1.877 → **2.413（+29%）**
（`scripts/probe_token_oracle.py`，`runs/S3_gates/oracle.json`）。

**但这全是"想象"里的**。如果 oracle token 在真闭环里也更好 → 说明"可微 WM 奖励"是个有效目标，
联合微调（DR + 约束）值得投；如果 oracle ≈ GT 甚至更差 → 想象增益搬不过去，**训练再多也没用**。

---

## 1. 三臂设计（同一场景、同一 seed、同一手部处理）

| 臂 | token 来源 | 服务 |
|---|---|---|
| **GT** | 录制 `encoder_latent`（= `g1_motion_token_64`，与 hold builder 逐位一致） | `scripts/ha_replay_token_server.py --npz-tokens ... --key g1_motion_token_64` |
| **VLA** | 官方 `checkpoint-8000`（`--decode codec`，K=4） | `scripts/ha_vla_latent64_server.py` |
| **ORACLE** | `runs/oracle_tokens/<clip>.npy`（本实验第一步产出） | `scripts/ha_replay_token_server.py --tokens ...` |

**手部必须三臂一致**：GT/ORACLE 臂用 `--hands-npz <同一 npz>`（回放 `g1_hand_joints_14`）；
VLA 臂用手部来自模型输出。若担心不一致，可先把三方手部都置零（张开）跑一轮，再看手部影响。

---

## 2. 前提（先核对，缺一个都别开）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
git log --oneline -1          # 需要 ≥ fd73963（v3 + oracle 脚本都在）
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader   # 4×4090
```

| 用途 | 路径 | 校验 |
|---|---|---|
| 执行器（闭环也用同一个） | `runs/S1_handover/WBC_A20000_wmvel_cmd.pt` | md5 `e0596acb98957d11931a7854bbe043d3` |
| 参考库（命令流） | `runs/isaac_motion/ha_cmd/packed.pt` | 86 MB |
| 录制 npz（token/手/状态） | `data/npz/ha_tokens/<clip>.npz` | 699 个（含 `g1_motion_token_64`、`g1_hand_joints_14`） |
| VLA 基线 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | 34 GB |
| SONIC encoder（hold builder） | 自动找；不行就 `SONIC_ENCODER_ONNX=` | `third_party/SuperMimic-Reproduced/data/assets/model_encoder.onnx` |
| Isaac 环境 + 闭环脚本 | `scripts/ha_isaac_closedloop.sh`（它会做契约 pin 校验） | `python3 scripts/pin_ha_closedloop.py --check` PASS |

> **本机（2 卡）Isaac 不可用**（解释器坏链已修，但场景装配卡死），所以这一步必须在能跑 Isaac 的机器上做。
> 不需要任何训练权限，一张卡给 Isaac、一张给 token 服务即可。

---

## 3. 执行步骤

### 步骤 0（5 min）：确认闭环的 **clip ↔ 场景/seed 对应关系**

token 回放是"按 clip 播"，所以必须知道每个 (task, seed) 用的是哪条记录。

```bash
# 用 GT 臂跑一次，服务端会打印 infer 行；同时看 Isaac 日志的 object_reset / episode_seed
python3 scripts/ha_replay_token_server.py \
  --npz-tokens data/npz/ha_tokens/HOI_double_desk__tw__ep000000.npz \
  --key g1_motion_token_64 --hands-npz data/npz/ha_tokens/HOI_double_desk__tw__ep000000.npz \
  --port 18083 --consume 12 &
TASK=DoubleDesk SEED=0 MAX_STEPS=300 SERVER_URL=http://127.0.0.1:18083 \
  SM_CKPT=$PWD/runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
  OUT_ROOT=runs/oracle_probe bash scripts/ha_isaac_closedloop.sh
```
- 若 episode 一开局就明显不合（比如场景是双桌、clip 是搬箱子），换成对应任务族的 clip；
- **务必核对 `--consume`**：provider 每 12 步重取 chunk，若每次取 40 帧却只消费 12 帧，
  旧 GT server 的 `cursor += 40` 会让 token 流快放 3.3×。**用 12 或 40 各跑一次 20 步，看哪个跟得上**，
  之后三臂统一。

### 步骤 1（4 卡并行，每个 clip ~6–10 min）：造 ORACLE token

```bash
export PYTHONPATH="$PWD:$PWD/Isaac-GR00T:$PWD/third_party/gr00t_deps:$PWD/third_party/gr00t_deps/_noflash"
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD="$PWD/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"

CUDA_VISIBLE_DEVICES=0 python3 scripts/build_oracle_tokens.py \
  --clips HOI_double_desk__tw__ep000000 HOI_double_desk__tw__ep000001 \
  --chunk 40 --opt-steps 60 --perturb 3.0 --out-dir runs/oracle_tokens --device cuda:0 &
# 其余三张卡并行跑别的 clip / 任务族
```
产出：`runs/oracle_tokens/<clip>.npy`（(T,64)）+ `.json`（每 chunk 的 `ret_gt → ret_oracle`、漂移）。
**门禁**：`.json` 里 `gain_rel` 中位数 ≥ **+10%**（否则这条 clip 没有 oracle 空间，换 clip 或加大 `--perturb`）。

### 步骤 2（每臂每任务 ~3–5 min）：三臂闭环

```bash
# 臂 1：GT（录制 token + 录制手）
python3 scripts/ha_replay_token_server.py --npz-tokens data/npz/ha_tokens/<clip>.npz \
  --key g1_motion_token_64 --hands-npz data/npz/ha_tokens/<clip>.npz --port 18081 --consume 12 &
# 臂 3：ORACLE
python3 scripts/ha_replay_token_server.py --tokens runs/oracle_tokens/<clip>.npy \
  --hands-npz data/npz/ha_tokens/<clip>.npz --port 18082 --consume 12 &

# 跑（三臂分别用 18080/18081/18082 的 SERVER_URL；VARIANTS 一次跑完更省时间）
SM_CKPT=$PWD/runs/S1_handover/WBC_A20000_wmvel_cmd.pt ISAAC_GPU=0 \
VARIANTS="gt=http://127.0.0.1:18081,oracle=http://127.0.0.1:18082,vla=http://127.0.0.1:18080" \
TASKS="PPBox DoubleDesk" SEEDS="0 1 2" REPEATS=1 MAX_STEPS=300 \
  bash scripts/ha_isaac_batch.sh
```
（VLA 臂：另起 `scripts/ha_vla_latent64_server.py --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 --port 18080 --bf16 --fm-steps 4 --decode codec`。）

---

## 4. 判读（**这就是"能不能提高成功率"的答案**）

汇总 `summary.json` 的 per_variant 表，只看这四个：

| 指标 | 通过线 |
|---|---|
| **成功率** | oracle > gt（同一任务、同一 seeds） |
| **存活步数 / 摔倒率** | oracle 不摔的比例 ≥ gt |
| **到目标最小距离** | oracle 更小 |
| **手闭合事件数** | 三臂一致（说明比的是 token 不是手） |

三种结局的含义：

| 观察 | 结论 | 下一步 |
|---|---|---|
| **oracle > gt ≥ vla** | 想象增益**可迁移** | 立刻投 v3 训练（DR + 前验锚定 + WM 集成不确定性），成功率有希望涨 |
| oracle ≈ gt ≈ vla | 想象增益**搬不过去** | 停：可微 WM 只能做"执行器级精修"，不能提成功率；资源转数据和执行器 |
| oracle < gt | WM 奖励**方向有偏**（模型利用） | 先修 WM（保鲜/标定/集成不确定性），别再谈训练 |

---

## 5. 风险与注意

1. **WM 偏置**：oracle 是在**冻结 WM** 上优化的，天然偏向模型的怪癖。所以本实验里 oracle 用
   `--perturb 3.0`（比训练更温和）并且带 `--anchor-lambda 0.1`，避免一上来就是极端外推。
2. **手部**：三臂必须同处理。oracle 臂用录制手（否则会比"手"而不是"token"）。
3. **`--consume` 语义**：O1/O2 时代没人核对过 token 流的播放速率；本实验先把 `consume=12` 与 `40`
   各跑 20 步对比，**这个不解决，所有回放结论都不可信**。
4. **不改代码**：本实验只新增 `scripts/build_oracle_tokens.py`、`scripts/ha_replay_token_server.py`，
   不动训练/闭环脚本。
5. **别拿它当最终结论**：它测的是"WM 目标的有效性"；训练后的正式表还是要 base vs ft 的多 seed 闭环。

---

## 6. 回传

| 文件 | 说明 |
|---|---|
| `runs/oracle_tokens/*.json` | 每条 clip 的 `ret_gt → ret_oracle`、漂移（先看这个，没空间就别跑闭环） |
| 闭环 `summary.json` / `SUMMARY.md` | 三臂 per_task / per_variant |
| 每个 episode 的 `result.json` | `success / episode_steps / failure_reason`（重点看 fall 率） |
| `logs/*.log` 里的 `[replay] infer` 行 | 证明 token 流播放速率正常 |

一句话：**这张表出来，S3 该继续投训练还是该收手，就有答案了。**
