# 交接：GR00T N1.7 VLA + SuperMimic 闭环的 token/anchor 对齐问题

> 写于 2026-10-28。目的：把当前闭环不稳的**可复现事实、代码口径、实验结果、未决问题**整理给另一个 agent 分析。
> 不要把它当成结论；下面区分了「已实测」「代码事实」「假设」。
> 工作区：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`

---

## 0. TL;DR

- **token 空间（64 维 SONIC latent）上下是对齐的**：VLA 的 `action.motion_token` 和 WBC 的 conditioning token 都是同一个 encoder 的 64 维输出。
- **但 token 的 anchor 不对齐**：
  - WBC 训练时 token 是**每一步在线算**的：`anchor = R_robot_current^T R_ref_current`，`token = encoder(command, anchor)`（`trackerlab-supermimic/.../runner.py` + `sonic_token.py`）。
  - VLA 的训练目标是**录制时**算好的 `encoder_latent`，anchor 是**录制时**的 robot/reference 姿态（`scripts/prepare_ha_sessions.py`）。
  - 同一个 64 维向量，在不同 anchor 下代表不同动作。VLA 输出的是 recorded-anchor token，WBC 期待的是 live-anchor token。
- VLA 的输入（图像 + q/dq/gravity + 文本）**不包含 reference heading**，所以它无法凭空推出 live anchor；给 VLA 直接加 anchor 输入不是“顺手加一下”，而是缺一个不可观测量。
- 已实测：teacher-forced/live-token 口径下 WBC 很强（HA 699: `0.8673`，0 failed；BONES 500: `0.8799`，9 failed）；但 Isaac 闭环里喂**录制 token** 会摔（318/256/1470 步），VLA 闭环也会摔/近摔。
- 待分析：**联合微调能否补偿 anchor mismatch**？还是必须单边改（WBC 吃 recorded-anchor token / VLA 输出 command q/qd / 给 VLA 喂 reference）？

---

## 1. 系统与 checkpoint

| 角色 | 路径 | 关键信息 |
|---|---|---|
| VLA | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | `global_step8000`；官方 HA 微调；`config.json` `num_inference_timesteps=4` |
| WBC (C6h) | `runs/S1_handover/WBC_A20000_wmvel_cmd.pt` | iter 20000；`paper_obs93`；`hist=10`；`token_dim=64`；`pi.dec.0` `(2048,994)`；`dec.8` `(29,256)` |
| WBC (24k, “带手”) | `runs/S1_handover/WBC_A24000_wmvel_cmd.pt` | iter 24000；**接口与 20k 完全相同** `(2048,994)->(29,256)`；`provenance.token_k=1`，`token_window=hold` |
| 24k 源 ckpt | `trackerlab-supermimic/logs/wbc_hand_hold_x20c/ckpt_024000.pt` | 与上面同接口；“带手”目前看是训练数据/奖励带手，**policy 不吃手** |
| VLA 服务 | `scripts/ha_vla_latent64_server.py` | `Gr00tVLM` + `sample_model_with_grad` + `ActionCodec`；HTTP latent64 |
| Isaac eval | `third_party/HumanoidArena/isaaclab_twist2_g1/script/eval_scripts/sonic/sim_eval_vla.py` | 官方 HumanoidArena 入口 |
| WBC provider | `.../action_provider/action_provider_supermimic.py` | 把 token + 93 维 obs 喂 policy，输出 29 关节目标 |

> 注意：`WBC_A24000_wmvel_cmd.pt` 的 `pi.dec.0` 仍是 994 = 64 + 10×93，**没有手部输入**；输出 29 维 body。若“带手”指的是 policy 吃手，需要再确认，当前文件不是。

---

## 2. 代码事实：token 是怎么算的

### 2.1 WBC 训练是 live anchor

`trackerlab-supermimic/source/trackerLab/trackerLab/algorithms/supermimic/runner.py`：

```python
# token 每步从 (future reference frames, robot root orientation) 在线计算
TOKEN_MODE = getattr(args, "interface", "motion") == "token"
token_fn = SonicTokenBuilder(motion, dev, k_window=_k) if TOKEN_MODE else None
...
def token_of(frames, robot_quat):
    return token_fn(frames, robot_quat=robot_quat)
```

`trackerlab-supermimic/source/trackerLab/trackerLab/algorithms/supermimic/sonic_token.py`：

```python
q_robot = robot_quat.repeat_interleave(nf, dim=0).reshape(n, nf, 4)
anchor  = quat_to_rot6d(_quat_mul(_quat_conj(q_robot), q_ref))
token   = encoder(command, anchor).detach()
```

即 `anchor = R_robot_current^T R_ref_current`，每一步用**仿真当前 robot root**。

### 2.2 VLA 训练目标是 recorded anchor

`scripts/prepare_ha_sessions.py`：

```text
action.motion_token (64)              encoder_latent
```

`scripts/prepare_ha_tokens_npz.py`：

```text
g1_motion_token_64     encoder_latent     bit-exact SONIC encoder output
token = np.asarray(z["encoder_latent"], dtype=np.float32)
```

`scripts/build_ha_command_lib.py` 明确说 `encoder_latent` 是用 **`encoder_motion_joint_pos_hist`（命令流）** 算的；录制时 anchor 已经烘焙进 token。

### 2.3 anchor 的数学定义

`diff_vla_wbc/sonic_encoder.py`：

```python
q_diff = quat_mul(quat_conj(q_robot_wxyz), q_ref_wxyz)
r = quat_to_rotmat(q_diff)
return r[..., :, :2].reshape(*r.shape[:-2], 6)   # R_robot^T R_ref
```

`build_motion_windows(..., robot_root_quat_wxyz=...)` 用同一个公式。

**结论（代码事实）**：
- WBC 的 token 是 `enc(command, R_robot_live^T R_ref_live)`；
- VLA 的 token 是 `enc(command, R_robot_recorded^T R_ref_recorded)`；
- 两者都是 64 维、同一个 encoder、同一个“空间”，但**anchor 不同**。

---

## 3. 已实测结果

### 3.1 Isaac Lab 官方闭环（VLA-8000 + C6h）

| 实验 | 步数 | 结果 |
|---|---|---|
| 基线 VLA token（无修复） | 266 | **fall**（DoubleDesk，tilt 92°） |
| VLA token + history 热启动 + q_target 限幅 | **2080** | **timeout，不摔**（但近摔；手不闭合） |
| 基线六任务（300 步） | 300 | 0/6；DoubleDesk fall@266，其余 timeout |
| 修复后六任务（300 步，跑到 4 个） | 300 | DoubleDesk/PPBox/Boxing/OpenDoor 全 timeout，**0 fall** |

修复代码（`action_provider_supermimic.py`）：
1. `on_env_reset` 后 `hist` 用第一帧真实观测重复 10 帧（不再全零）；
2. `q_target = default_q + action_scale*a` 后 clip 到 29 个 body 关节的 `joint_pos_limits`；
3. 新增 `SM_INIT_Q_NPY`（reset 用命令参考 q/qd 初始化）；
4. 修了 `_os` 未定义的 NameError；加了 token_space LOCKED 日志。

### 3.2 GT / 录制 token 回放（Isaac，DoubleDesk）

| 初始化 | 手 | 结果 |
|---|---|---|
| 执行轨迹首帧 | 有手 | **fall @318** |
| 执行轨迹首帧 | `SM_HAND=0` | **fall @256**（更早） |
| **ha_cmd 命令参考 q/qd** | `SM_HAND=0` | **fall @1470**（49 s） |
| VLA + 命令参考初始化 | 有手 | **fall @716**（比执行初始化更差） |

- 手不是主因：无手更早摔。
- 命令参考初始化对 GT 有效（318→1470），但对 VLA 有害（2080→716），因为 VLA 训练分布是执行状态，命令参考姿态对它 OOD。

### 3.3 WBC 的 teacher-forced 跟踪（强）

| 数据 | 口径 | 结果 |
|---|---|---|
| HA 699 clips | `evaluate.py`，live token | `norm_return 0.8673`，`n_failed=0` |
| BONES-SEED 500 clips | `evaluate.py`，live token | `norm_return 0.8799`，`n_failed=9/500` |
| C6h BONES 500 | `runs/S1_handover/WBC_A20000_wmvel_cmd_eval.json` | 同上（2026-10-28 00:22 重跑） |

> 关键：teacher-forced eval 用的是 `SonicTokenBuilder`（live anchor）；它不能证明 WBC 对 recorded-anchor token 也稳。

### 3.4 手部

- 长跑 120 次 VLA infer：`hand_gt170=0` **120/120**；手指最大 1.633，HA 阈值 1.70。
- `SM_HAND_SCALE=1.2` 版最大只有 0.556，仍无抓取。
- 所以“抓不住”是 VLA 手部输出问题，和摔倒是两个独立问题。

---

## 4. 当前假设（待验证）

### H1（主假设）：anchor mismatch

WBC 训练/teacher-forced eval 用 live anchor，VLA 闭环用 recorded anchor；机器人转向后两者差一个旋转，WBC 把 token 解释成错误方向的运动 → 转向不稳/摔。

**支持**：
- 命令参考初始化把 GT 从 318 提到 1470（说明 anchor/状态对齐有影响）；
- teacher-forced live token 0.867，recorded token 闭环 318/1470；
- VLA/GT 的 token 都是 recorded `encoder_latent`。

**反证/未决**：
- GT 命令参考 + live 状态仍 1470 摔，说明 WBC 闭环鲁棒性也有上限；
- 还没有跑过**真正的 live-token GT 回放**（provider 目前只支持 http/npy/zeros，没有在 Isaac 里实时 `SonicTokenBuilder`）。

### H2：VLA token 时序/精度

- 离线 corr 0.82–0.97，转向帧 MSE 最差 0.0134；
- 2.5 Hz 独立 chunk，边界可能跳；
- 但 `HA_VLA_EXEC_STEPS=40` 反而更早摔，不能全怪频率。

### H3：WBC 闭环鲁棒性

- GT 命令参考 + 无手也 1470 摔；
- teacher-forced 强 ≠ 闭环抗扰强；
- 需要 tilt/CoM 安全层或 on-policy 微调。

### H4：手/抓取

- 手不闭合是独立问题；无手更早摔，说明手不是站不稳主因。

---

## 5. 为什么“给 VLA 加 anchor”不直接

- anchor 需要 `R_ref`（参考/teleop 的 heading）；
- VLA 输入：图像 + q/dq/gravity + 文本；gravity 对 yaw 不变，图像不含 reference heading；
- 所以 VLA **不可观测** `R_ref`；除非把 planner/teleop 的参考姿态也作为输入（那就不是纯 VLA）。

因此“给 VLA 加 anchor 输入”不是简单的特征拼接，而是一个**可观测性/接口设计**问题。

---

## 6. 候选修复（请另一个 agent 分析优先级）

| 方案 | 改哪边 | 是否需要重训 | 备注 |
|---|---|---|---|
| A. WBC 改吃 **recorded-anchor token** | 只改 WBC | 微调 WBC | 最贴近当前 VLA 输出分布；用 `ha_cmd` 命令 q/qd + 录制 `encoder_latent` 造训练集 |
| B. 给 VLA 输入 anchor/reference | 只改 VLA + planner | 微调 VLA | 需要 planner 提供 `R_ref`；纯 VLA 不可观测 |
| C. VLA 输出 command q/qd，部署时实时算 token | 只改 VLA 输出空间 | 重训 VLA action head | WBC 不动；需要 command q/qd 标签 |
| D. 联合微调（现有 S3） | VLA + WBC | 已做/可继续 | 只能补偿/分布对齐，补不上缺失的 anchor；需要 live-token training 才可能有效 |
| E. tilt/CoM 安全层 | 执行层 | 不用重训 | 最快压住“近摔”，但不解决根因 |

**需要另一个 agent 回答的问题**：
1. H1 的 anchor mismatch 是否真的足以解释 318/1470/716 的差异？有没有更简单的解释（scene/object reset、初始状态、provider 的 history/action 约定）？
2. 联合微调（D）在数学上能否补偿 unobserved anchor？如果不能，为什么？
3. 方案 A 的训练集怎么造：`ha_cmd` 命令 q/qd + 录制 `encoder_latent`，anchor 用录制时的 robot/ref；还是直接用 `data/npz/ha_tokens` 的 `joint_pos` + `encoder_latent`？两者的 0.16 s 错位怎么处理？
4. 方案 C 是否更干净：VLA 输出 29×10 的 command q/qd，部署时用 `SonicTokenBuilder` + 当前 robot quat 实时算 token？需要什么样的标签和 loss？
5. 24000 “带手” ckpt 的 `token_k=1` 是什么语义？它和 20000 的接口完全一样，是否真的“带手”？
6. 是否有必要先做 **live-token GT 回放**（provider 加 `SM_TOKEN_SOURCE=live`）来验证 H1？

---

## 7. 复现命令

### 7.1 环境（必须）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# 关键：/etc/vulkan/icd.d/nvidia_icd.json 是 0 字节；两个变量都要指到 /usr/share
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

> 注意：`VK_DRIVER_FILES=$VK_ICD_FILENAMES` 写在**同一行 export** 会展开旧值（0 字节路径），必须分两行或写绝对路径。

### 7.2 VLA 服务

```bash
PYTHONPATH=$PWD:$PWD/Isaac-GR00T:$PWD/third_party/gr00t_deps/_noflash \
HF_HOME=/inspire/hdd/global_user/liumengfan-253108110079/cache/huggingface \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --vlm-path groot_repro/nvidia/Cosmos-Reason2-2B-weights \
  --port 18090 --device cuda:0 --bf16 --horizon 40 --fm-steps 4 --decode codec
```

### 7.3 Isaac 闭环（VLA）

```bash
TASK=DoubleDesk SEED=0 MAX_STEPS=2080 HA_VLA_EXEC_STEPS=20 \
SERVER_URL=http://127.0.0.1:18090 \
SM_CKPT=$PWD/runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
OUT_ROOT=$PWD/runs/ha_isaac_vla8000_c6h/xxx \
bash scripts/ha_isaac_closedloop.sh
```

### 7.4 GT token 回放

```bash
# 录制 token（command encoder_latent）
python3 -c "
import numpy as np
d=np.load('data/npz/ha_tokens/HOI_double_desk__tw__ep000000.npz')
np.save('/tmp/gt_tokens.npy', d['g1_motion_token_64'].astype(np.float32))
np.save('/tmp/gt_hands.npy',  d['g1_hand_joints_14'].astype(np.float32))
"
# 命令参考 q/qd（ha_cmd packed）
python3 - <<'PY'
import torch, numpy as np
ck=torch.load('runs/isaac_motion/ha_cmd/packed.pt',map_location='cpu',weights_only=False)
i=list(ck['names']).index('HOI_double_desk__tw__ep1776268633456387'); s=ck['starts'][i]
row=ck['ref'][s].float().numpy()
np.savez('/tmp/ha_cmd_q0.npz', q=row[13:42].astype(np.float32), qd=row[42:71].astype(np.float32))
PY

export HA_PIN_BYPASS=1 SM_HAND=0 SM_INIT_Q_NPY=/tmp/ha_cmd_q0.npz
TASK=DoubleDesk SEED=0 MAX_STEPS=2080 HA_VLA_EXEC_STEPS=20 \
SM_TOKEN_SOURCE=npy SM_TOKEN_NPY=/tmp/gt_tokens.npy SM_HANDS_NPY=/tmp/gt_hands.npy \
SM_CKPT=$PWD/runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
OUT_ROOT=$PWD/runs/ha_isaac_vla8000_c6h/xxx \
bash scripts/ha_isaac_closedloop.sh
```

### 7.5 WBC teacher-forced eval

```bash
cd trackerlab-supermimic
python scripts/rls/supermimic/evaluate.py \
  --run_dir logs/bones_seed_token_ha_v2 \
  --ckpt ../runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
  --motion_packed data/bones_seed/bones_seed_compact.pt \
  --clip_limit 500 --headless --device cuda:0
```

---

## 8. 代码/环境现状（重要）

- `action_provider_supermimic.py` 已改：
  - `import os as _os`（修 NameError）；
  - history 热启动；
  - q_target clip（缓存 `joint_pos_limits`）；
  - `SM_INIT_Q_NPY`；
  - `[SuperMimic][LOCKED] token_space=command_encoder_latent` 日志；
  - `token_space` 审计。
- `scripts/ha_vla_latent64_server.py` 已改：
  - `/` 返回 `token_space=command_encoder_latent`；
  - 启动打印 `[AUDIT][token]`。
- `simulation_app.py` 加了 viewport 120 帧超时（headless 容器防卡死）。
- `scripts/ha_isaac_closedloop.sh` 加了 `HEADLESS` / `VGLRUN` 支持。
- **pin 校验当前会失败**：`configs/ha_closedloop_pin.json` 里 `sim_eval_vla.py` 的 sha256 与工作区不一致（19:06–19:25 之间被另一个进程改了 87 行）。目前用 `HA_PIN_BYPASS=1`；正式结果前要重新固化 pin。
- 容器没有 `CAP_SYS_ADMIN`，`nvidia-smi --gpu-reset` 不可用；GPU 被僵尸上下文卡住时只能重启容器。

---

## 9. 产物/日志路径

| 内容 | 路径 |
|---|---|
| 六任务基线视频 | `runs/ha_isaac_vla8000_c6h/deliverables/` |
| 40 s 修复版 DoubleDesk + 对比 | `runs/ha_isaac_vla8000_c6h/deliverables_long/` |
| 修复后 2080 步结果 | `runs/ha_isaac_vla8000_c6h/long_doubledesk/` |
| GT recorded token fall@318 | `runs/ha_isaac_vla8000_c6h/gt_2080/` |
| GT no-hand fall@256 | `runs/ha_isaac_vla8000_c6h/gt_nohand_2080/` |
| GT 命令参考 fall@1470 | `runs/ha_isaac_vla8000_c6h/gt_cmdinit_2080/` |
| VLA + 命令参考 fall@716 | `runs/ha_isaac_vla8000_c6h/long_dd_cmdinit_vla/` |
| HA 699 eval | `runs/S1_ctrl/A_20000_cmd_eval.json` |
| BONES 500 eval（C6h） | `runs/S1_handover/WBC_A20000_wmvel_cmd_eval.json` |
| 24k hand ckpt | `runs/S1_handover/WBC_A24000_wmvel_cmd.pt` |
| 24k hand 源 | `trackerlab-supermimic/logs/wbc_hand_hold_x20c/ckpt_024000.pt` |
| 24k GT 回放（崩溃） | `runs/ha_isaac_gt_hand_24000/`、`logs/s1/gt_replay_hand24000.log` |

---

## 10. 给分析 agent 的一句话

> 请重点判断：**token space 相同但 anchor 不同**（WBC live `R_robot^T R_ref` vs VLA recorded `encoder_latent`）是否足以解释 318/1470/716 的闭环差异；如果不足以，最可能漏掉的是哪个环节（scene/object reset、初始状态、provider history/action 约定、WBC 闭环鲁棒性）？以及**联合微调能否补偿 unobserved anchor**，还是必须单边改接口？
