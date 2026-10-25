# HA 闭环推理的锁死契约（场景 / 资产 / 机器人初始位姿 / 编解码器）

> **一句话**：以后不管谁（人或 agent）说「跑 HA 闭环推理」，都必须是**同一套场景、同一批资产、
> 同一个机器人初始位姿、同一组仿真参数、同一个 decoder**。这些都固化在
> `configs/ha_closedloop_pin.json`，由 `scripts/pin_ha_closedloop.py --check` 强制校验，
> 漂一点就拒绝启动。
>
> 建立时间：2026-10-25。生成方式：`python3 scripts/pin_ha_closedloop.py --write`。

---

## 1. 强制点在哪

| 入口 | 是否强制 | 行为 |
|---|---|---|
| `scripts/ha_fast_eval.sh`（持久仿真批量，队列用的就是它） | ✅ | 启动前 `--check`，不过就 `exit 9` |
| `scripts/ha_isaac_closedloop.sh`（逐集模式） | ✅ | 同上 |
| `scripts/ha_isaac_batch.sh` | ✅（间接） | 它调上面那个 |
| `HA_PIN_BYPASS=1` | 例外 | 跳过校验并在日志里打 WARN；**这种结果不得与正式结果混用** |

校验耗时约 1 s（要 sha256 约 260 MB 的场景/机器人 USD + 权重）。

---

## 2. 锁了什么

### 2.1 七个任务（`tasks`）

| key | task_id | 机器人 init_pos | init_rot |
|---|---|---|---|
| DoubleDesk | `Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody` | `(-3.7, -2.2, 0.8)` | `(1.0, 0.0, 0.0, 0.0)` |
| PPBox | `Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby` | `(-1.6, -5.0, 0.8)` | `(1.0, 0.0, 0.0, 0.0)` |
| Boxing | `Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody` | `(-1.6, -2.0, 0.8)` | `(0.70711, 0, 0, -0.70711)` |
| OpenDoor | `Isaac-Move-Open-Door-G129-Dex3-Wholebody` | `(-1.6, 0.2, 0.8)` | `(0.70711, 0, 0, 0.70711)` |
| SitSofa | `Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody` | `(-1.6, 2.0, 0.8)` | `(0.70711, 0, 0, -0.70711)` |
| VisionNavi | `Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody` | `(-1.9, -5.2, 0.8)` | `(0.7071, 0, 0, 0.7071)` |
| FootballSingle | `Isaac-Move-Football-Single-G129-Dex3-Wholebody` | `(0.0, -2.0, 0.8)` | `(0.7071, 0, 0, 0.7071)` |

每个任务还锁了：

- **test yaml**（`tasks/common_test_config/base_test/*_sonic_test.yaml`）的 sha256 ——
  里面写着 `wait_for_textures` / `sim.dt` / `decimation` / `deterministic_object_resets`（物体初始位姿范围）；
- **env_cfg**（`tasks/g1_tasks/<task>/*_hw_env_cfg.py`）的 sha256 —— 机器人初始位姿就写在这里；
- **它 import 的 scene 模块**（`tasks/common_scene/*.py`）的 sha256 —— 场景由谁定义；
- **所有被引用的 `assets/**.usd*`**（共 16 个）的 sha256 + 大小 —— 桌子/篮子/锤子/门/沙发/纸箱/仓库/标志牌/足球/球门；
- **`object_resets`** 记录名（yaml 里 `deterministic_object_resets[].record_name`）。

### 2.2 机器人与仿真参数

- `robot.type = unitree_g1_refpose_v3_1`
- `robot.usd_dir = assets/robots/g1-29dof-dex3-base-fix-usd`（4 个 USD 全锁 sha256）
- `robot.cfg = tasks/common_config/robot_configs.py` —— **默认关节角**（`G1RobotJointTemplates`）在这里，随 sha256 锁定
- 禁止 `ROBOT_USD_OVERRIDE` 环境变量（会换机器人资产，`--check` 直接报错）
- `sim.dt = 0.005`、`decimation = 4`、`wait_for_textures = false`（灰模，与 10-24 那批一致）

### 2.3 权重与评测协议

- `weights.sonic_encoder / sonic_decoder` = **release** 变体
  （`GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/`，复现 HA 录制动作残差 1e-5 度，
  见 `docs/SONIC_DECODER_FOUND.md`；**不是** low_latency）
- `weights.vla_default` = `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`
- `weights.wbc_default` = `trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt`
- `protocol`：seeds `0..9`、`repeats_per_seed=1`、`max_steps=300`、`video_fps=30`、headless、
  episode_seed = seed 直通

---

## 3. 怎么用

```bash
# 日常：校验（CI/脚本里都一样）
python3 scripts/pin_ha_closedloop.py --check

# 有意改动（换资产/改初始位姿/换 decoder）后重新固化
python3 scripts/pin_ha_closedloop.py --write
git diff configs/ha_closedloop_pin.json      # ← 必须逐项 review
```

`--check` 的失败样例（实测过，不是设想）：

```
[pin] ❌ 契约校验失败（2 处）：
   [内容变了] DoubleDesk.asset: .../SM_Crate_A14_Gray_01_physics.usd
            0000000000000000 -> 3abede861a60e884
   [初始位姿变了] Boxing.init_pos: '(-9.9, -9.9, 0.8)' -> '(-1.6, -2.0, 0.8)'
```

---

## 4. 不在契约里的东西（别误会成"锁了"）

| 项 | 原因 |
|---|---|
| `HOI_grap_cup`（ArtVIP-Livingroom-GrapCup） | 本 checkout **没注册任务、没有 ArtVIP 资产**，跑不了。官方 7 类里也没有它 |
| `FootballSingle` 的成功率绝对值 | 任务/场景都在契约里，但我们的 VLA 训练集第 7 类是 grap_cup、**没见过 football** → 对两条路线都是 OOD，只能看相对差 |
| 训练数据 / SuperMimic ckpt 的"最优"选择 | 契约只锁**默认路径 + sha**，不锁你换哪个 ckpt 做实验（换 ckpt 不算契约漂移，但要在结果里写清） |
| 贴图 | 当前 `wait_for_textures: false`（灰模）。要开贴图 = 改 yaml = 必须 `--write` 重新固化，**不能偷偷开** |

---

## 5. 变更流程（写死）

1. 改资产/场景/初始位姿/decoder；
2. `python3 scripts/pin_ha_closedloop.py --write`；
3. `git diff configs/ha_closedloop_pin.json` 逐项确认是**有意的**；
4. 在 PR/记录里写一行：改了哪个任务的哪个字段、为什么；
5. **禁止**为了过 `--check` 手改 json 里的 sha。
