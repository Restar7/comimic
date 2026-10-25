# 交接：闭环推理的 action_scale 口径错误（已修 + 已锁死）

> 写于 2026-10-25（2×4090 容器）。**影响面：所有走 `action_provider_supermimic.py` 的闭环推理结果作废，
> 必须重跑。联合训练（A1 处理臂 / A2 / A3 / A4 / A5）不受影响，不用重跑。**
> 修完之后这件事**不可能再漏**：锁写在 action provider 模块内部，任何启动方式都绕不过去（§4）。

---

## 1. 一句话
WBC 训练时用 `SUPERMIMIC_HA_ALIGN=1` 的执行器表算 `action_scale`，而闭环推理的 action provider
**没带这个环境变量** → `get_robot("g1")` 退回 BeyondMimic 表 → **同一个 policy 动作被换算成不同的关节目标**
（`hip_pitch` 放大 **2.85×**、`wrist_yaw/pitch` 缩小 **10×**）→ 机器人站不稳、摔倒。

## 2. 证据（都是实测，不是推测）

### 2.1 闭环日志里打出来的就是错的值
```
[SuperMimic] action_scale mean=0.395 max=0.548 (trackerlab g1 contract)
```
| 开关 | action_scale mean | max |
|---|---|---|
| `HA_ALIGN=0`（BeyondMimic，**闭环实际用的**） | **0.3950** | **0.5475** |
| `HA_ALIGN=1`（HumanoidArena，**训练用的**） | **0.3789** | **0.7500** |

### 2.2 逐关节差异（致命的那几个）
| 关节 | 闭环（ALIGN=0） | 训练（ALIGN=1） | 偏差 |
|---|---|---|---|
| `hip_pitch` ×2 | 0.548 | 0.192 | **动作放大 2.85×** |
| `wrist_yaw/pitch` ×4 | 0.075 | 0.750 | 缩小 **10×** |
| `wrist_roll` ×2 | 0.439 | 0.750 | 缩小 41% |

### 2.3 后果
- `runs/ft4000_p0fix_eval/PPBox__seed0__repeat0__20261025_193742/result.json`：
  `"failure_reason": "fall"`、`"terminal_detail": "tilt_deg=104.5 ... mode=hard_tilt"`、`episode_steps: 278`。
- 同一口径下，WBC 官方评测（`evaluate.py`）也踩过：**不带** 变量 → **39/40 clips 判失败**；
  **带上** → **0/40、norm_return 0.9036**。

## 3. 根因
```python
# trackerTask/trackerTask/supermimic/robot.py
HA_ALIGN = bool(int(os.environ.get("SUPERMIMIC_HA_ALIGN", "1" if HAS_HANDS else "0")))
HAS_HANDS = ("dex3" in Path(USD_PATH).name) or ("with_hand" in Path(USD_PATH).name)
USD_PATH  = os.environ.get("SUPERMIMIC_ROBOT_USD") or ".../g1_29dof_rev_1_0.usd"
```
- 默认 USD 是**无手** 29-DoF → `HAS_HANDS=False` → `HA_ALIGN=0`；
- 而 action provider 的 `_resolve_action_scale()` 调 `get_robot("g1").action_scale`，**这个值随 HA_ALIGN 变**；
- 训练脚本 `train_bones_seed.sh` / `launch_sm_finetune.sh` 显式导出了 `SUPERMIMIC_HA_ALIGN=1`，
  闭环脚本（`run_a1_ha_isaac.sh` / `ha_isaac_batch.sh` / `ha_isaac_closedloop.sh`）**都没导出**。
- 之前的 P0-1（默认姿态）与 P0-3（执行器 kp/kd）修的是另外两处，**唯独 action_scale 这条依赖环境变量，被漏了**。

## 4. 修法（三层锁，已生效）
| 层 | 文件 | 内容 |
|---|---|---|
| **① 模块内强制（主锁）** | `third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_supermimic.py` | 模块 import 时**无条件** `os.environ["SUPERMIMIC_HA_ALIGN"]="1"` 并补 `SUPERMIMIC_ROBOT_USD`；`_resolve_action_scale()` 里再校验 `trackerTask.robot.HA_ALIGN`，为 0 时直接 `RuntimeError`；通过时打印 `[SuperMimic][LOCKED] HA_ALIGN=1 action_scale mean=0.3789 max=0.7500` |
| **② 启动脚本 export** | `scripts/ha_isaac_closedloop.sh`、`scripts/ha_isaac_batch.sh`、`scripts/run_a1_ha_isaac.sh` | 在 `COMIMIC=` 之后 export 两个变量（双保险） |
| **③ 评测入口默认** | `trackerlab-supermimic/scripts/rls/supermimic/evaluate.py` | import 前默认 `SUPERMIMIC_HA_ALIGN=1` 并打印审计行 |

**逃逸开关（只给 A/B 用）**：`SM_ALLOW_SCALE_MISMATCH=1`。设了它才会跳过强制锁，
且结果**不得**与正式结果混表（脚本会打印 `[SuperMimic][WARN]`）。

## 5. 验证（两条，必过）
```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic

# ① 模块级：import 后环境变量必须是 1，USD 必须是 HA 的 29dof
COMIMIC=$PWD PYTHONPATH=$PWD python3 -c "
import sys; sys.path.insert(0,'third_party/HumanoidArena/isaaclab_twist2_g1')
import action_provider.action_provider_supermimic, os
print(os.environ['SUPERMIMIC_HA_ALIGN'], os.environ['SUPERMIMIC_ROBOT_USD'])"
# 期望：1 / .../trackerlab-supermimic/data/assetslib/.../g1_29dof_rev_1_0.usd

# ② 跑一集：日志里必须出现 LOCKED 行，且 action_scale mean=0.3789 max=0.7500
#    （mean 0.395 / max 0.548 就是又错了，立刻停）
grep -h "\[SuperMimic\]\[LOCKED\]" runs/<新评测目录>/**/runs/*.log | head
```

## 6. 重跑清单（改完之后）
| 必须重跑 | 原因 |
|---|---|
| `runs/ablations/A1_joint_finetune_ha/*`（base + ft） | 全部在错 scale 下跑的 |
| `runs/ft4000_p0fix_eval/*` 及所有同类逐集评测 | 同上 |
| 任何 `ha_isaac_batch.sh` / `ha_isaac_closedloop.sh` 的产物 | 同上 |
| WBC 的 `evaluate.py` 全片评测 | 同上（我这边已重跑：0/40、0.9036） |
| **不用重跑**：A1 处理臂训练、A2/A3/A4/A5 联合训练、WBC 训练 | 它们走原始动作空间（`act()` 返回 policy 原始输出，`COMIMIC_ACTION_SCALE` 默认 `one`），只有奖励里的动作平滑项与 `HA_ALIGN` 有关（权重 −0.1、实测贡献 −0.003，可忽略） |

### 重跑命令（A1 闭环，ft 臂示例）
```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# 1) 起 VLA server（GPU1）
PYTHONPATH=$PWD/third_party/gr00t_deps/_noflash python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/ckpt_vla_ha_ft4000 --port 18085 --device cuda:1 --bf16 \
  --horizon 40 --fm-steps 2 > /tmp/vla_ft4000.log 2>&1 &
# 2) 起 HA 闭环（GPU0）；锁在 provider 里，但这里也显式给一遍
SM_CKPT=$PWD/trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt \
ISAAC_GPU=0 VARIANTS="ft4000=http://127.0.0.1:18085" \
TASKS="PPBox DoubleDesk" SEEDS="0 1 2" REPEATS=1 MAX_STEPS=300 \
OUT=$PWD/runs/ft4000_p0fix_eval_locked \
  bash scripts/ha_isaac_batch.sh
# 3) 全量 A1（两臂 + 汇总）
TASKS="PPBox DoubleDesk" SEEDS="0 1 2" bash scripts/run_a1_ha_isaac.sh
```

## 7. 不要做的事
1. ❌ 用 `SM_ALLOW_SCALE_MISMATCH=1` 跑正式结果（只能做 A/B，且必须单独归档）。
2. ❌ 把旧目录（`ft4000_p0fix_eval`、`A1_joint_finetune_ha`）的数字和新跑的数字混在一张表里。
3. ❌ 只改脚本不验证 `[SuperMimic][LOCKED]` 行 —— 脚本可以被人绕过，**只有日志能证明**。
4. ❌ 用 `pkill -f train_joint.py` 之类的宽匹配停任务（会误杀自己的 shell）。

## 8. 变更文件（2026-10-25）
```
third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_supermimic.py   # 主锁 + 审计 + RuntimeError
scripts/ha_isaac_closedloop.sh                                                              # export
scripts/ha_isaac_batch.sh                                                                   # export
scripts/run_a1_ha_isaac.sh                                                                  # export
trackerlab-supermimic/scripts/rls/supermimic/evaluate.py                                    # 默认 HA_ALIGN=1 + 审计打印
```
