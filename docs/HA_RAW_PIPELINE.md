# HumanoidArena raw → 官方 20k 微调 → 我们的联合微调（完整跑通记录）

> 目标：用 HumanoidArena 的 **motion token** 监督 GR00T-N1.7-3B，走官方微调路径出 20000 步
> checkpoint，再从这个 checkpoint 走我们自己的可微联合微调（反传到 VLA head）。
>
> 关键结论：**本地已有的 LeRobot 版 `HumanoidArenaV3.1_SONIC40` 不能用于监督**——它只有
> 40 维 refpose action + 2 维 binary hand，没有 token、没有手指。带 token 的是 **raw** 那份
> （`Twang2026/HumanoidArena_raw`）。head 不换：官方微调直接用 `UNITREE_G1_SONIC`
> （64 维 token + 7+7 手），出厂的 checkpoint 就能被我们的 `train_joint.py` 加载。

---

## 0. 两条数据发布的差别（实测，不是推断）

| | LeRobot `HumanoidArenaV3.1_SONIC40`（本地已有 3.2 GB） | raw `HumanoidArena_raw`（ModelScope） |
|---|---|---|
| 规模 | 7 任务 × 100 ep，546,460 帧，50 Hz | 7 个任务有 SONIC 数据，699 ep |
| action | `(T,40)` = xy_delta2 + root_z + rot6d6 + 29 关节 + hand_binary 2 | `vla_action (T,40)` 同 |
| state | `(T,64)` = rot6d6 + 关节29 + 速度29 | `vla_state (T,64)` 同 |
| **motion token** | **无** | **`encoder_latent (T,64)`，另有 `encoder_input (T,1762)`** |
| **手指** | **无**（2 维开关） | **`hand_action_left/right (T,7)`** |
| 其他 | front RGB mp4 | `final_body_action_29dof`、`decoder_obs (T,994)`、`robot_qpos/qvel_before_decimation`、`robot_root_position/orientation`、front RGB mp4 |

raw 一个 episode 的 npz 约 113 MB，其中 **95% 是 `vision_depth (T,480,640,1) float16`**（解压 644 MB），
我们不用的部分可以丢。转换后只留 token/手/body/根位姿，2.1 MB/集。

**token 一致性验证（决定性）**：把 raw 自带的 `encoder_input (1762)` 灌进本地
`third_party/SuperMimic-Reproduced/data/assets/model_encoder.onnx`，与它自带的 `encoder_latent` 逐帧比：

```
frame 0/100/500/900/1048   maxabsdiff = 0.00000   cosine = 1.000000
```

即同一 encoder、同一 token 空间，与我们 ft318 的 `g1_motion_token_64` 天然兼容
（尺度：HA std 0.162 / norm 1.29 vs ft318 std 0.204 / norm 1.63）。

## 1. raw 的任务清单（和 LeRobot 版不一样）

| 目录 | 后端/操作员 | episodes | 备注 |
|---|---|---|---|
| `HOI_double_desk` | sonic/tw | 100 | |
| `HOI_grap_cup` | sonic/yb | 99 | LeRobot 版没有这个任务 |
| `HOI_pp_box` | sonic/yb | 100 | |
| `HSI_boxing` | sonic/zz | 100 | |
| `HSI_open_door` | sonic/zz | 100 | |
| `HSI_sit_sofa` | sonic/zz | 100 | |
| `HSI_vision_navi` | sonic/yb | 100 | |
| `HOI_football_v2` | **无 sonic/** | — | 只有 twist2 后端，自动跳过 |

共 **699 episodes / 80.04 GB / 1398 个文件**（100 npz + 100 front RGB mp4 每任务）。

## 2. 下载

```bash
cd /inspire/.../wsc-workspace/comimic
# 清单 + 断点续传 + 16 并发（32 并发反而更慢：服务端限流，实测 23 MB/s vs 47 MB/s）
python3 scripts/fetch_ha_raw.py --list-only      # 只建 manifest.tsv
python3 scripts/fetch_ha_raw.py --workers 16     # 下到 data/ha_raw/
```

- 落在 `data/ha_raw/<task>/<op>/<ep>.npz` 与 `<op>/videos/<ep>_front_rgb.mp4`
- 断点续传用 `.part` + 尺寸校验后 rename，日志 `data/ha_raw/download.log`，完成写 `DOWNLOAD_OK`
- 实测速率 30–47 MB/s，全量 ~35–45 分钟

## 3. 两个转换（一个给官方 finetune，一个给我们的联合微调）

### 3.1 → 我们训练器吃的 npz（`--recorded-tokens`）

```bash
python3 scripts/prepare_ha_tokens_npz.py --workers 8 --out-dir data/npz/ha_tokens
# 199 条 ≈ 50 s / 423 MB；699 条约 2.1 MB/集 → ~1.5 GB
```

字段映射（目标键与 `scripts/prepare_g1_tasks_npz.py` 写的 ft318 npz **完全一致**，可直接互换）：

| ft318 key | raw 字段 | 说明 |
|---|---|---|
| `joint_pos` / `joint_vel` | `robot_qpos/qvel_before_decimation` | 实测值，不是 reference |
| `body_*` | 用 `G1Kinematics` FK（同 ft318 那套） | FK 复算根位姿 MAE < 5 mm，用来卡 joint 顺序/quat 约定 |
| `g1_motion_token_64` | `encoder_latent` | 逐位相等 |
| `g1_hand_joints_14` | `hand_action_left/right` | 左 7 后右 7 |
| `g1_state_46` | 关节29 + 手14 + gravity3 | gravity = Rᵀ[0,0,−1] |
| root | `robot_root_position/orientation` | **xy 保留**（有导航任务）、z 实测（ft318 是脚触地估的） |

### 3.2 → 官方 finetune 的 LeRobot 数据根

```bash
python3 scripts/prepare_ha_sessions.py --workers 8 \
  --ref-parquet /inspire/.../flowbody/FlowBody/dataset/groot_sonic_vla_ft318_frozen_20261023/task_1_meeting_room_milk_basket/2026-07-14-19-15-12/data/chunk-000/episode_000000.parquet
# → data/ha_sessions/<task>__<op>/ + roots.txt（7 根，冒号分隔）
```

- parquet **schema 与 ft318 参考根 `a.equals(b) == True`**（31 列、fixed_size_list dtype 全同），
  做法是直接复用参考根 schema + `data/ha_session_template/{info,modality,stats,relative_stats}.json`
- `stats.json` / `episodes_stats.jsonl` 按 HA 数据**重算**（25 组 mean/std/min/max/q01/q99）
- 视频：raw 的 mp4 硬链到 `videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4`
  （raw mp4 恰好是 1049 帧 @50 fps = `num_frames`，不需要重编码）
- `tasks.jsonl` 的指令是 7 条人工句子（raw 的 `task` 只是环境名，如 `Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody`）

## 4. 官方 20k 微调（stage O）

```bash
bash scripts/launch_ha_official_20k.sh        # 2 卡，20000 步
```

等价裸命令（就是训出现有 20k 的那条，只换数据根）：

```bash
cd .../flowbody/FlowBody/Isaac-GR00T
HF_HUB_OFFLINE 反着用：TRANSFORMERS_OFFLINE=0 + HF_HOME 指向本地缓存
NUM_GPUS=2 GLOBAL_BATCH_SIZE=32 MAX_STEPS=20000 SAVE_STEPS=1000 \
DATALOADER_NUM_WORKERS=8 SHARD_SIZE=1024 NUM_SHARDS_PER_EPOCH=100000 EPISODE_SAMPLING_RATE=0.1 \
bash examples/finetune.sh --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$(cat data/ha_sessions/roots.txt)" \
  --embodiment-tag UNITREE_G1_SONIC --output-dir runs/groot_sonic_vla_20k_ha
```

参考成本：271 ep / 405k 帧，**1×4090 约 4.6 h**；本数据 699 ep / ~60 万帧，步数相同。

## 5. 我们的联合微调（stage ②）

```bash
bash scripts/launch_joint_ha.sh
# 等价：/usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 scripts/train_joint.py
#   --motion data/npz/ha_tokens
#   --supermimic-ckpt .../logs/e9_bones_seed_token_h200x2/ckpt_0024000.pt
#   --vlm --vlm-ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000      # 写死的官方微调产物，见 runs/OFFICIAL_CKPT.json
#   --vlm-dataset --vlm-steps 2 --hand-loss-weight 0.2 --hand-grasp-boost 5
#   --steps 12000 --eval-every 1000 --save-vla-every 4000
```

## 6. 环境坑（2026-10-24 实测修好，不修官方微调起不来）

| 症状 | 根因 | 修法 |
|---|---|---|
| `import numpy` → `No module named '_contextvars'` | `Isaac-GR00T/.venv/bin/python` 被改指向 `/usr/bin/python3.12`，但 `pyvenv.cfg home=` 是 uv 的 3.12.14（二进制与 stdlib 来自不同 prefix） | `ln -sfn /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 .venv/bin/python` |
| `libtorchcodec_core4/5/6.so` 全部 "Could not load" | 系统没有 FFmpeg（torchcodec 0.8 只支持 4–7） | `apt-get install ffmpeg` → 6.1.1 / libavcodec.so.60 |

修好后实测：`numpy 1.26.4 / torch 2.9.0+cu128 / cuda True / gr00t ok`，
`get_frames_by_indices(mp4,[0,100,700]) → (3,480,640,3) uint8`。

⚠️ raw 视频是 **mpeg4 (mp4v)、3.7 Mbps**，比 ft318 的 h264 糊；token 链路不受影响，但练视觉时这是画质上限。

## 7. 时间线（实测）

| 步骤 | 耗时 |
|---|---|
| 下载 80 GB（16 并发） | ~40 min |
| 转 npz（699 条，8 进程） | ~3 min |
| 转 sessions（699 条 + 视频硬链 + FK + stats） | ~10 min |
| 官方 20k（2 卡） | ~2.5–3 h |
| 联合微调 12k（2 卡，带手部 loss） | ~9 h |
