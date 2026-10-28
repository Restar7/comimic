# patch: SuperMimic provider —— 在线 token 源 + 采集落盘补强（2026-10-28）

`third_party/HumanoidArena/` 是 gitignored，改动镜像在这里。文件：
`third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_supermimic.py`

## 1. `SM_TOKEN_SOURCE=online`（新增）
- `_make_token_source`：mode `online` → 读 `SM_TOKEN_LIB`（默认 `runs/isaac_motion/ha_cmd/packed.pt`）、
  `SM_TOKEN_START`（默认 0）、`SM_TOKEN_STRIDE`（默认 1）、`SM_HANDS_NPY`。
- `_build_online_token_builder` / `_OnlineLibView`：用 **S1 训练同一个类**
  `trackerLab...sonic_token.SonicTokenBuilder`，`window="hold"`；库视图从 packed.pt 切
  `q=ref[:,13:42]`、`qd=ref[:,42:71]`、`quat=ref[:,3:7]`（布局见 build_ha_command_lib.py）。
- `_next_latent_online(env)`：每步 `frame = 库行(start + k*stride)`，`robot_quat = robot.data.root_quat_w`
  （**实时**），返回 `(1,64)` 广播。**anchor 必须是实时 quat**：冻 anchor 会让机器人按错的世界方向走
  （实测根漂 3.49 m；实时 anchor → 1.0 cm）。
- `_next_latent` 拆成 wrapper + `_next_latent_impl`，wrapper 记录 `self._last_token`。

## 2. `_record_state_dump` 补字段
| 字段 | 含义 |
|---|---|
| `q_target_applied (T,29)` | 真正下发到 PD 的目标（**clip 到关节限位后**，canonical 序、弧度）。`actions` 仍是策略原始输出 `a_norm`（可达 ±8，不能当 WM 标签）|
| `tokens (T,64)` | 每步喂进策略的 token（`SM_DUMP_TOKENS=0` 可关）|
| `meta.cfg` | `SM_TOKEN_SOURCE/LIB/START/STRIDE`、`SM_RESET_*`、`HA_VLA_EXEC_STEPS`、`SONIC_TOKEN_WINDOW`、`SONIC_ENCODER_ONNX` |
| `meta.sm_ckpt` / `meta.task` | 执行器 ckpt 与任务 id |
| `frames` | online 模式下 = **真实参考库行号**（`self._ref_frame`），不再是步数 |

## 3. reset 扰动 `SM_RESET_NOISE`（新增）
`_apply_reset_noise()`，在 `on_env_reset` 末尾调用；`SM_RESET_SEED`（+episode 序号）、`SM_RESET_SCALE`。
量级对齐训练 DR：xy 0.10 m / yaw 0.25 rad / 关节 0.05 rad / 关节速度 0.30。
没有它：不同 seed 逐位相同（`max|Δa|=0.0`），采集等于 n=1，critic 拿不到偏离状态。

## 4. 配套（非 third_party）
- `script/eval_scripts/sonic/sim_eval_vla.py`：`--sm_token_source` 的 `choices` 加 `online`。
  → 改完必须 `python3 scripts/pin_ha_closedloop.py --write`（已做）。
- `scripts/isaac_replay_online.sh`（新）：3 seed 驱动脚本，内含 `env -u LD_LIBRARY_PATH -u LD_PRELOAD`
  （不清会 fork EAGAIN）。
