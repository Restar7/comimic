# 闭环推理修复（2026-11-04）：token 流塌成常量 → 机器人原地站桩

> 状态：**已修复并验收**。修复前 official 闭环净位移 **0.017 m**（原地踏步 12 s），
> 修复后 **1.006 m / 20 s，零摔倒，手能闭合（max 1.826 > 1.70）**；
> 真值 token 上界 **1.223 m**（复现录制里 1.17 m/12 s 的走法）。

---

## 1. 症状（可复现的最小事实）

官方链路：`HA_ALLOW_SONIC=1` → `SonicActionProvider`（GEAR-SONIC **release** decoder）
→ `scripts/ha_vla_latent64_server.py`（我们的 VLA）。`DoubleDesk / seed 0 / 600–1000 步`。

| | base head | FT head(@250) | 参考 |
|---|---|---|---|
| 平面净位移 | 0.017 m | 0.018 m | 1.17 m（录制前 12 s）|
| 翻倒帧 | 0 | 0 | — |
| token `absmean` | 0.093 | 0.090 | 0.134 |
| 相邻帧 `‖Δt‖` | **0.037** | 0.053 | **0.139** |
| 逐维时间 std | 0.022 | 0.023 | 0.119 |

机器人**站着不动但也不倒**，`success=False(timeout)`。VLA 与 FT 的差异被这条前置故障
完全吃掉，任何"训练有没有效"的结论都不可信。

## 2. 四个根因（全部用实验定位，不是猜）

### 2.1 `state46` 布局选错 —— 46 维里 21 维是垃圾（**已修**）

`SonicActionProvider` 调的 `infer_chunk(...)` **不发 `state46`**（只有 supermimic provider 发），
服务端于是用 `--state-layout`（旧默认 `train_v1`）从 64 维 `vla_state` 重建：

```
train_v1: [jp0:22 | left_hand(7) | jp22:29 | right_hand(7) | gravity]   ← 错
legacy  : [jp0:29 | left_hand(7) | right_hand(7) | gravity]             ← 训练数据真实布局
```

证据（抓包 `--log-dir` 与训练数据逐位比对）：

```
训练数据 g46[0:43] = [...,手 0, 1.0, 1.74, -1.57, -1.74, -1.57, -1.74]
train_v1 拼出来    = [...,手 1.2, 0, 0, 0, 0, 0, 0]        max err 1.740
legacy   拼出来    = 逐位一致                                max err 0
```

右侧臂被手顶掉、`left_hand` 位置上塞的是 `jp22:29`。**全程静默**：不报错、不崩，只是机器人不动。

补强证据（checkpoint 自己的 `dataset_statistics.json`）：`left_hand`/`right_hand` 首维均值恒
`0.0`、`std≈0.55`，只有 legacy 分块能对上；`train_v1` 会把 `right_arm` 对成手部统计。

**锁死机制**：`assert_state_layout_matches_training_data()`，在**读 6 GB 权重之前**执行 ——
把 `data/npz/ha_tokens/*.npz::g1_state_46` 拆成 (joints, hands) 再喂回 `state64_to_state46`，
拼回来必须逐位一致，否则 `SystemExit`：

```bash
[LAYOUT][OK] layout=legacy 与训练数据 HOI_double_desk__tw__ep000000.npz 逐位一致（关节 29 + 手 14）
# --state-layout train_v1 →
[LAYOUT][FATAL] ... 最大误差 1.7400 ... 会把 46 维状态拼错 → VLA 输出会退化成常量 token
```

### 2.2 `--fm-steps 4` 把 40 帧 chunk 采样成「平」的（**已修**）

head 的 **row-0 很准**（与 GT 首帧 `cos 0.98 / mse 0.006`），但 **chunk 内部几乎不变**；
而部署是**逐帧消费 chunk 内的行**（`HA_VLA_EXEC_STEPS=20`，一次 fetch 40 行、用 20 行）。

| fm_steps | chunk 内 `‖Δt‖` |
|---|---|
| 4（旧默认） | 0.0350 |
| 20（新默认） | 0.0716 |
| 40 | 0.0774 |
| GT | 0.0897 |

少步 Euler 采样等于「对 flow 欠积分」，输出被平均化 → decoder 收到准常量 token → 保持姿势。
**2026-11-05 修正（重要）**：`--fm-steps` **已改回官方默认 4**。
上一版把默认抬到 20 的理由是"4 步会把 chunk 采平 ⇒ 机器人站桩"，但那次 A/B **同时改了
state46 布局（train_v1→legacy）**，两个修复混在一起、归因错了。用已修布局重测：
`fm=4` 走 **1.165 m**（600 步）、`fm=20` 走 1.134 m（都正常），且 `fm=4` 单次推理
**231 ms** vs `fm=20` **555 ms**。checkpoint 的 `config.json` 里
`num_inference_timesteps = 4`（GR00T N1.7 官方口径）⇒ 保持 4。

### 2.3 手的 14 维恒为 0 → 抓取指令永远不出现（**已修**）

HA 的 64 维 `vla_state` **没有手**（`root_rot6d(6)+dof_pos(29)+dof_vel(29)`），
服务端只能 `--hand-source feedback` 拿自己上一次输出喂自己 ⇒ 自收缩到 ≈0；
而训练数据里手是**真实手指角**，起始帧就是闭手 `[0,1,1.74,-1.57,-1.74,-1.57,-1.74]`
（与 `DEFAULT_HAND_POSE["unitree_g1"]["left"]["close"]` 逐位相同）。

实测：修复前闭环 `hand14_max = 0.045`（闭手阈值 **1.70**）⇒ 永不闭合；
修复后 **`hand14_max` 峰值 1.826、均值 0.298**（数据集标签均值 0.311）。

**改法**（链路上最小改动，口径单一）：
`SonicActionProvider._measured_hands14()` 读 `robot.data.joint_pos` 的 14 个手指关节
（左 7 + 右 7，`_left_hand_idx`/`_right_hand_idx`，顺序 = `DEFAULT_HAND_POSE`）
→ `lerobot_vla_http_client.infer_chunk(..., hands14=)` → 服务端 `state64_to_state46` 用它填手。
布局仍然**只有服务端一处实现**，provider 不重复。

### 2.4 图像/状态本身是对的（排除项，避免以后再查一遍）

* 闭环抓包的 `state64[6:35]` 与录制 `joint_pos` **逐位相同**；重力 `[0,0,-1]` 一致。
* 闭环相机图与录制 mp4 首帧**肉眼与统计都几乎一致**（mean 121.5 vs 119.5，std 70.2 vs 72.9）。
* `TEXTURES=1`（灰模 → 有贴图）**无效**：`‖Δt‖` 0.0369→0.0381 ⇒ 不是灰模 OOD。
* chunk 消费逻辑没有 bug：chunk 内确实在推进（0.027≠0），跨 chunk 跳 0.23。

## 3. 验收（官方链路，`DoubleDesk / seed0 / SONIC release decoder`）

| 跑法 | 步数 | 净位移 | 翻滚帧 | `z` 最低 | `‖Δt‖` | 手 max |
|---|---|---|---|---|---|---|
| 修复前 base | 600 | 0.017 m | 0 | 0.778 | 0.037 | 0.045 |
| 修复前 FT@250 | 600 | 0.018 m | 0 | 0.778 | 0.053 | 0.045 |
| **真值 token（上界）** | 600 | **1.223 m** | 0 | 0.761 | 0.120 | — |
| 修复后 base | 600 | 0.983 m | 0 | 0.756 | 0.072 | 0.06 |
| 修复后 base（官方 20 s） | 1000 | **1.006 m** | 0 | 0.711 | 0.091 | **1.826** |
| 修复后 **round-3 FT（λ_bc=0, 1000 步）** | 1000 | **0.111 m** | 0 | 0.755 | 0.107 | 0.149 |

真值 token 那行证明：**执行器 + 部署接线本身是对的**（把录制 token 喂进去，机器人就按录制走，
1.223 m vs 录制 1.17 m），剩下的差距全在 VLA 的 token 质量上。

## 4. 修好之后的第一条结论：**round-3 arm A（λ_bc=0）是负结果**

同一套修复后的链路、同 seed、同 1000 步，只换 head：

| head | 净位移 | 路径 | 手 max | 关节活动 q_std |
|---|---|---|---|---|
| base（未微调） | **1.006 m** | 2.638 m | **1.826（会闭合）** | 0.147 |
| round-3 FT（1000 步, λ_bc=0） | **0.111 m** | 1.192 m | 0.149（永不闭合） | 0.044 |

FT 把"会走"训成了"原地踏步"，而且把手也训没了。结合训练曲线（`reward_norm` 0.33→0.11、
`act/rmse` 0.02→1.46）说明：**WM 里想象出来的漂移罚被优化掉了，但那是 WM rollout 的分歧被压平，
不是真机变好**；λ_bc=0 + 无 critic 时唯一能降的就是这一项，训久了必然漂。处理建议见 spec §12。

## 5. 还没解决的（别混进来）

1. **DoubleDesk 在 20 s 内不可能 success**：录制本身 41.6 s（2080 帧 @50 Hz），
   而 env `episode_length_s = 20.0`。这条任务不该用 `success` 当判据，
   应该看**跟踪误差 / 阶段进度**（后续在 spec 里改判据）。
2. base head 走到桌边（`x≈-2.8`）后停住，不再继续；真值 token 能继续走。
   这是**模型能力**问题（token 的相位/意图），不是接线问题。
3. round-3 之前的所有闭环对比**全部作废**（都是拼错 state46 + 4 步采样的口径）；
   本文件 §3/§4 的表格是**修好之后**的第一批有效数据。

## 6. 复现命令

```bash
# VLA 服务（会自动做布局自检 + 打印 [LAYOUT][OK]）
PYTHONPATH=$PWD:$PWD/Isaac-GR00T:$PWD/third_party/gr00t_deps:$PWD/third_party/gr00t_deps/_noflash \
python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --head-ckpt runs/joint_v3_round3A/vla_final.pt \
  --port 18088 --device cuda:1 --seed 0

# 闭环（另一张卡；执行器固定为 SONIC release decoder）
TAG=ftfinal HEAD=runs/joint_v3_round3A/vla_final.pt GPU=3 PORT=18088 MAX_STEPS=1000 \
  bash scripts/r3_closedloop_ab.sh

# 体检（摔没摔/走没走/手有没有动）
python3 scripts/analyze_closedloop_r3.py --dump runs/closedloop_r3/ftfinal/dump
```

## 7. 相关脚本（本次新增，都是可独立复跑的诊断）

| 脚本 | 作用 |
|---|---|
| `scripts/r3_closedloop_ab.sh` | 一次跑完"服务 + 官方闭环 + 结果体检"，`SRV_CMD` 可换成真值 token 服务 |
| `scripts/analyze_closedloop_r3.py` | dump 体检：站立高度/翻倒帧/净位移/路径/token 时间变化 |
| `scripts/diag_head_token_variability.py` | 离线：head 在**训练集原帧**上的 token 时间变化（判"模型塌了"还是"输入口径错"）|
| `scripts/diag_state46_layout.py` | 离线：只换 state46 拼法，量化 |Δt| / cos / mse |
| `scripts/diag_2x2_image_state.py` | 离线：图像×状态 2×2 交叉，定位是哪一路 |
