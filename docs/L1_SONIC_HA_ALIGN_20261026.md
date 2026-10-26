# L1 官方线（官方微调 ckpt + SONIC release decoder）— HA_ALIGN 执行器口径

> 2026-10-26。回答的问题：**把主线/消融用的 `HA_ALIGN=1` 执行器口径（default pose / kp-kd /
> action_scale）套到 SONIC decoder 上，官方微调 ckpt 在 HA 的 7 个官方任务上闭环能到多少。**
>
> 这不是"官方 SONIC deploy 口径"的基线。两者执行器不同（见 §2），**数字不得混表**。

---

## 1. 改动（可开关，默认关）

文件：`third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py`

新增环境变量 **`SONIC_HA_ALIGN=1`**（默认 `0`，官方 deploy 语义完全不动）。打开后：

| 项 | 官方 deploy（`SONIC_HA_ALIGN=0`） | HA_ALIGN（`SONIC_HA_ALIGN=1`） |
|---|---|---|
| default pose | `SONIC_DEFAULT_POS`：hip −0.312 / knee 0.669 / ankle −0.363 / elbow 0.6 | `trackerTask.get_robot("g1").default_q`：hip **−0.2** / knee **0.42** / ankle **−0.23** / elbow **0.87** |
| decoder target | `a * G1_ACTION_SCALE_ISAACLAB + default`（0.3507 / 0.5475 / 0.0745） | `a * action_scale + default`（mean **0.3789** / max **0.7500**） |
| kp / kd | `policy_parameters.hpp`（hip 99 / knee 99 / ankle 48.4 …） | HA 资产表（hip 130 / knee 160 / ankle 60 …，kp mean **71.4**） |
| episode 起点 | 任务 yaml 的机器人 init pose（TWIST2: 肩 ±0.4 / 肘 1.2） | 写 HA 默认姿态（与主线 P0-1 同口径） |
| 审计行 | `[ALIGN] HA_ALIGN=0 (official SONIC deploy spec…)` | `[ALIGN] HA_ALIGN=1 default_q[knee/ankle/elbow]=0.420/−0.230/0.870 action_scale mean/max=0.3789/0.7500 kp mean=71.4` |

映射：`trackerTask` 的 `RobotSpec` 是 `STANDARD_29` 序，SONIC encoder/decoder 是
`HA_V31_29` 序（两者名字集合相同、顺序不同），代码里按名字求 `perm`，不写字面量。

要求：`SONIC_HA_ALIGN=1` 时 `trackerTask.robot.HA_ALIGN` 必须为 True
（`export SUPERMIMIC_HA_ALIGN=1` + `SUPERMIMIC_ROBOT_USD=...g1_29dof_rev_1_0.usd`），
否则 provider 直接 `RuntimeError`（不允许静默退回 deploy 口径）。

---

## 2. 协议（与主线/消融一致）

- 7 个 HA 官方任务：`DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi FootballSingle`
  （`HOI_grap_cup` 本 checkout 没注册/没资产，官方 7 类里也没有）；
- 每任务 **10 集**（seed 0..9），`max_steps=300`，persistent-sim（`ha_fast_eval.sh`
  的 `--episode_batch_json`），pin 契约（场景/16 资产/init pose/decoder=release）；
- VLA：`runs/ckpt_vla_ha_8000_local`（→ `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`），
  `--decode codec --fm-steps 4 --horizon 40`，`--bf16`；指令用 `prepare_ha_sessions.py` 的
  同一张 `TASK_INSTRUCTIONS` 表（`--sv_task_name`）；
- 执行器：`SONIC_HA_ALIGN=1`，release `model_decoder.onnx`（994→29），CPU EP（2–4 ms/帧）。

一条命令（VLA 服务需已就绪）：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
python3 scripts/ha_vla_latent64_server.py \
  --ckpt runs/ckpt_vla_ha_8000_local --port 18100 --device cuda:0 --bf16 \
  --horizon 40 --fm-steps 4 --decode codec > logs/base8000_ha_align_server.log 2>&1 &
TASKS="DoubleDesk PPBox Boxing OpenDoor SitSofa VisionNavi FootballSingle" EPISODES=10 \
  OUT=$PWD/runs/L1_sonic_ha_align bash scripts/run_L1_sonic_ha_align.sh
```

产物：`runs/L1_sonic_ha_align/{<task>/episodes/*.json, videos_*, sim.log}` +
`summary.json` + `SUMMARY.md` + `driver.log`。

**口径校验（每个任务）**：`sim.log` 必须出现
`[SonicActionProvider][ALIGN] HA_ALIGN=1`；没有这一行的结果不能当 HA_ALIGN 臂。

---

## 3. 结果

（跑完后填；脚本自动写 `runs/L1_sonic_ha_align/SUMMARY.md`）

| task | episodes | success | rate | avg_steps |
|---|---|---|---|---|
| DoubleDesk | 10 | 0 | 0.0% | 274.2 |
| PPBox | | | | |
| Boxing | | | | |
| OpenDoor | | | | |
| SitSofa | | | | |
| VisionNavi | | | | |
| FootballSingle | | | | |

---

## 4. 与其它口径的关系（别混表）

| 臂 | VLA | 执行器 | default / scale / kp |
|---|---|---|---|
| **本表 L1-HA_ALIGN** | 官方 `checkpoint-8000` | SONIC release decoder | HA_ALIGN 表（本文件 §1） |
| 旧 L1（10-25 `compare_two_routes/L1`） | 同上 | SONIC release decoder | 官方 deploy 表 |
| L2 主线/消融 | 联合/官方 VLA | SuperMimic WBC | HA_ALIGN 表 |

- 本表与"旧 L1"只差执行器口径（default pose / scale / kp-kd / 起点），VLA 与 decoder 权重相同；
- 本表与 L2 只差执行器（decoder vs WBC），VLA 也不同——headline 对比时按 `FINAL_HANDOVER` 的
  两条路线规则走，不要交叉喂 token。

---

## 5. 备注

- 10-26 14:22 另一个会话重启的"官方微调 continuation"（batch16，占 39/49 GB）按用户指示
  **已停掉**（step ~3/4000，无 ckpt 损失），把卡让给本次闭环；需要时用
  `BASE=runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 MAX_STEPS=4000 NUM_GPUS=1
  GLOBAL_BATCH_SIZE=8 RUN=groot_sonic_vla_ha_official_cont bash scripts/launch_ha_official_20k.sh`
  重新排队（建议 batch 8 以避免 OOM）。
- `SONIC_HA_ALIGN=1` 是**本项目的实验口径**，不是 HA 官方协议；论文/表格里必须显式写执行器。
