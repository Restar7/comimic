# A' 全量采集 + WM 更新（2026-11-03 夜）

> 本文是当晚的交接：**修了什么、正在跑什么、跑完怎么接**。
> 细节推导见 `docs/REPLAY_JOINT_ORDER_FIX_20261103.md`（§1–§13）。

---

## 0. TL;DR

| 项 | 状态 |
|---|---|
| 官方 replay 二进制 | ✅ 跑通（根因：`HA_ALLOW_SONIC` 门禁），逐位对齐 |
| 回放关节序 | ✅ 修好（0.264 → 0.031 rad；官方 provider 0.0014 rad） |
| **A' 闭环采集"一采就倒"** | ✅ **修好**（根因：初值关节序；已设为默认） |
| v+l 闭环 | ✅ 跑通（1000 步 / 20 s **不摔**） |
| 任务成功 | ❌ 卡在手指合不上（VLA 手部输出幅度只有数据的 1/5） |
| 全量 A' | 🟡 **正在跑**（3514 窗口 / 4×16 env / σ=0.05 / 20 步段） |

---

## 1. 今晚修的 4 个真 bug

### 1.1 官方 replay 起不来 → `HA_ALLOW_SONIC`

`action_provider/create_action_provider.py:12`：

```python
SONIC_ALLOWED = os.environ.get("HA_ALLOW_SONIC", "0") == "1"
```

不导出时直接 `raise SystemExit`，但 ImageServer 是非守护线程 ⇒ 进程不退、只刷同一帧，
表现为"卡死在 provider 创建前"。**加 `HA_ALLOW_SONIC=1` 即通**。

### 1.2 回放把 SONIC 序的目标写到 standard 槽位

仿真侧关节顺序 = `SONIC_ISAACLAB_JOINT_ORDER`（已打印逐项确认），录制数据同序；
而 ckpt 的 `robot_spec.joint_names` 是 standard 序。旧代码用 `idx_t`（standard）
写 SONIC 目标 ⇒ 值对、槽位错。

修：回放路径统一用 `idx_s`（`_rtgt_ids`）。实测 step0 关节误差 **0.264 → 0.031 rad**；
改用官方 `SonicActionProvider` 驱动后 **0.0014 rad**（与官方二进制逐位一致）。

### 1.3 A' 采集"一采就倒" → 初值关节序

同一个病：`q0[:, idx_t] = 库里的 q`，而库里的 q 是 **canonical 值 + standard 标签**。

| 配置 | 结果 |
|---|---|
| 旧（idx_t 写初值） | z 0.80 → 0.60@160 → **180 步内摔** |
| 新（idx_s 写初值） | **2080 帧全程不摔**（z 0.752–0.798，up ≥ 0.936） |

**已把 `SM_INIT_IDS` 的默认值改成 `sonic`**（要复现旧行为显式设 `ckpt`）。
这一条是"A' 根本采不出来"的根因。

### 1.4 VLA 服务 fp32 + flash-attn → HTTP 500

`FlashAttention only support fp16 and bf16 data type`（VLM 以 fp32 载入、ckpt 配置写的是
flash_attention_2）。正解不是转 bf16（会连带 head 输入 dtype 不匹配），而是：

```python
# diff_vla_wbc/gr00t_vlm.py（checkpoint 分支）
config.use_flash_attention = False   # -> 走 sdpa
```

修完 `/infer` 正常：**首次 989 ms（warmup），稳态 156–194 ms**，返回 `(40,64)`。

### 1.5 附带：`pin_ha_closedloop.py` 的默认 WBC 路径已失效

`DEFAULT_WBC = trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt`
文件已不存在 ⇒ 契约校验必然失败。临时用 `HA_PIN_BYPASS=1`；**该默认值需要修**。

---

## 2. 正在跑：全量 A'（`runs/a_prime_seg20/`）

```bash
bash scripts/run_a_prime_full_seg20.sh
```

| 参数 | 值 | 理由 |
|---|---|---|
| 窗口 | `--steps 200` + **尾窗口** | 每条 clip 全长覆盖（含末尾 <200 帧） |
| 段长 | `--segment 20` | 每 20 步重置到参考；与部署重规划周期、训练 H=20 对齐 |
| 噪声 | `--action-noise-std 0.05 --perturb-hold 20` | A' L1：环内保持 20 步的动作噪声 |
| 并行 | 4 GPU × 16 env | 16 是带相机时的显存上限 |
| 规模 | **3514 个窗口** | 634k 帧全库 |

产物：`runs/a_prime_seg20/off<N>.pt`（每个 16×200=3200 行），运行日志同目录。

**合并**（跑完后）：

```bash
python3 - <<'PY'
import torch, glob
fs = sorted(glob.glob('runs/a_prime_seg20/off*.pt'))
keys = ('states','actions','next_states','frames','clip_ids')
out = {k: torch.cat([torch.load(f, map_location='cpu', weights_only=False)[k] for f in fs]) for k in keys}
meta = torch.load(fs[0], map_location='cpu', weights_only=False)['meta']
out['meta'] = meta; out['n_windows'] = len(fs)
torch.save(out, 'runs/a_prime_seg20/merged.pt'); print(out['states'].shape)
PY
```

---

## 3. 采集口径依据（为什么是 20 步 + 段内重置）

### 3.1 漂移随段长线性增长（实测）

| 段长 | p50 | p90 | max |
|---|---|---|---|
| 5 步 (0.1 s) | 3.3 cm | 4.4 cm | 5.4 cm |
| 10 步 (0.2 s) | 5.4 cm | 9.6 cm | 14.2 cm |
| 15 步 (0.3 s) | 8.1 cm | 14.8 cm | 22.4 cm |
| **20 步 (0.4 s)** | **4.9 cm** | **12.7 cm** | 28.2 cm |

### 3.2 累积漂移不是采集口径

连续 2080 步不重置：0.06 m@20 → 0.30@100 → 0.97@300 → **3.74@1000**。
那是误差积分，**不代表模型差**；采集用 20 步段，段间重置，**不会累积**。

### 3.3 奖励 δ 需要重定

δ=5 cm 时只有 **51%** 的帧在容差内（奖励大面积饱和）；
δ=10 cm 时 **84%**。⇒ **`--reward-delta-pos` 建议 0.05 → 0.10**。

### 3.4 "和参考轨迹比"只在局部有意义

参考距离奖励只在"动作还没走岔"时等价于跟踪误差（≈20 步以内）。
超出就是"两个不同的解"在比，梯度无意义。
⇒ 联合微调必须 **H=20 + 每段重置**；起点最好从 **A' 的真实状态**里采（DAgger），
而不是只用干净的 `clip_start`。

---

## 4. v+l 闭环（跑通了，但任务没完成）

入口：`scripts/ha_isaac_closedloop.sh` + `SM_TOKEN_SOURCE=http` + `scripts/ha_vla_latent64_server.py`。

⚠️ **执行器由 `HA_ALLOW_SONIC` 决定**（`sim_eval_vla.py:204`）：

| 环境变量 | 执行器 |
|---|---|
| 未设 | `SuperMimicActionProvider`（我们的 WBC） |
| `HA_ALLOW_SONIC=1` | `SonicActionProvider`（官方 decoder） |

实测（SuperMimic 执行器，1000 步 / 20 s）：

```
success=False  failure_reason=timeout（不是 fallen，fall_candidate=0 条）
reward 每步 raw_total=0.0
```

录像（`runs/ha_isaac_eval/DoubleDesk__seed0__repeat0__20261103_190643/videos_failure/*.mp4`）抽帧：

| 时刻 | 画面 |
|---|---|
| 0% | 站在 double desk 前 |
| 25% | 已转向放锤子的桌子，双臂抬起 |
| 50% | 前倾伸右手去够锤子 |
| 90% | 走到篮子那张桌子前，**手里是空的** |

⇒ 动作序列对，**没抓住**。原因量化：VLA 的 `hands14` 输出 **±0.33 rad**，
数据里是 **±1.74 rad**（差 5 倍），`hand_binary` 基本恒 0 ⇒ 手指合不上。
手部通路本身是通的（`SM_HAND=1`、`_hand_queue`、14 关节 passthrough、scale=1.0）。

---

## 5. WM 更新（采集跑完后）

1. 合并 → `runs/a_prime_seg20/merged.pt`
2. 用新数据重训/刷新 WM：
   * `scripts/pretrain_wm.py`（从头预训练），或
   * `diff_vla_wbc/wm_refresh.py`（在旧 WM 上刷新）
3. 校验：imagined vs real 的 20 步 headroom（旧基线：GT 0.0694 / VLA-base 0.0674，imag −0.7% vs real −2.8%）
4. 通过后再进联合微调（H=20 + 每段重置 + δ=0.10）

---

## 6. 还没修的（按优先级）

1. **VLA 手部输出幅度**（挡住任务成功）：
   - 先查训练侧 hand 是否做过归一化（若是 `/1.74`，推理端要乘回来）；
   - 否则是 hand head 训练不足（loss 权重）。
2. **`pin_ha_closedloop.py` 的 `DEFAULT_WBC` 指向不存在的 ckpt**。
3. **WBC 跟踪精度**（关节 0.28 rad）—— 决定 A' 数据的天花板。
4. 评测口径：把"累积漂移"换成 **任务成功率**（官方口径）+ **段内漂移**（A' 用）。

---

## 7. 本轮改动的文件

| 文件 | 改动 |
|---|---|
| `scripts/isaac_a_collect_batch.py` | 关节序修正（`_rtgt_ids`/`SM_INIT_IDS` 默认 sonic）、官方 provider 驱动、43 维 action、手部目标、windows 尾窗口、各类对照开关 |
| `scripts/run_a_prime_full_seg20.sh` | **新增**：全量 A' 采集驱动（4 GPU × 16 env，窗口 + 20 步段 + 噪声） |
| `scripts/ha_vla_latent64_server.py` | `attn_implementation="sdpa"` + 诊断 |
| `diff_vla_wbc/gr00t_vlm.py` | checkpoint 分支关掉 `use_flash_attention`（走 sdpa） |
| `third_party/.../sim_main.py` | replay 时跳过整场景 DDS 序列化 |
| `third_party/.../action_provider_supermimic.py` | 手部 chunk 诊断打印 |

---

## 8. 方案复核（2026-11-03 深夜，逐条查过）

复核结论：**方案主体成立，但发现 2 个会让它跑不完/跑坏的问题，已修**。

### 8.1 ✅ 已验证成立的点

| 检查项 | 结论 |
|---|---|
| A' 需要存相机图像吗 | **不需要**。联合微调的 VLA 输入走 `processed_batch(processor, builder, frames)`，图像由 **LeRobot 数据集按 `frames`（库行号）** 取，A' 只提供 (s,a,s') 与行号 |
| 段重置是否真的生效 | ✅ 每段起点漂移**恒为 0.0**（off0 实测 158 段） |
| 帧号语义 | `frames = lib_starts[clip] + start + step`，与库/奖励/图像索引一致 |
| 噪声 | `σ=0.05`、hold=20，与段长同周期；`actions` 实测 std=1.85（WBC 输出无界） |
| 覆盖 | windows 模式已补**尾窗口**，每条 clip 全长覆盖 |

### 8.2 ❌ 问题 1：进程写完盘卡在 Isaac teardown，驱动被堵死

实测：4 个采集进程 19:38:48 全部写完 `.pt`，但**4 分钟不退**（`Rl` 状态），
驱动脚本的 `while` 循环一直挂着 ⇒ 50 分钟的任务会变成几小时。

**修**：`SM_HARD_EXIT=1`（默认）在 `torch.save` + 打印后直接 `os._exit(0)`；
驱动脚本每批加 `timeout 300` 兜底 + 批次结束 `pkill` 残留。

### 8.3 ❌ 问题 2：坏段比例很高（23% 的段内漂移 > 30 cm）

off0 实测（158 段）：段内最大漂移 p50=6.4cm、p90=12.0cm，但

```
> 10 cm: 77%   > 20 cm: 46%   > 30 cm: 23%   > 50 cm: 3%（最差 65 cm）
```

坏段还成簇出现（25–31、43–44 连续段）⇒ 某些片段的参考运动本身就跟不上（或趔趄）。

**修**：新增 `scripts/post_a_prime_seg20.py`，合并后**按段过滤**
（默认丢段内最大漂移 > 30 cm 的整段），并打印分布；WM/奖励只用 `filtered.pt`。

### 8.4 ⚠️ 问题 3：δ=0.05 偏严（复核后确认）

δ=5cm 只有 51% 的帧在容差内；δ=10cm 为 84%。**采完统一用 `--reward-delta-pos 0.10`。**

### 8.5 复核后的执行顺序（已按此跑）

```bash
# 1) 采集（后台，~50 min）
bash scripts/run_a_prime_full_seg20.sh

# 2) 合并 + 过滤坏段
python3 scripts/post_a_prime_seg20.py --max-drift 0.30

# 3) 更新 WM（用过滤后的转移）
python3 scripts/pretrain_wm.py --joint-ckpt runs/S1_handover/WBC_A24000_wmvel_cmd.pt \
    --transitions runs/a_prime_seg20/filtered.pt --steps 6000 \
    --out runs/a_prime_seg20/wm_pre.pt [--no-stage1 --no-stage3]

# 4) 校验：imagined vs real 的 20 步 headroom（旧基线 GT 0.0694 / base 0.0674）

# 5) 联合微调（H=20 + 每段重置 + δ=0.10）
python3 scripts/train_joint_v2.py ... --reward-delta-pos 0.10
```

### 8.6 仍未解决（不阻塞本次采集）

1. VLA 手部输出幅度只有数据的 1/5（±0.33 vs ±1.74 rad）→ 挡住任务成功；
2. `pin_ha_closedloop.py` 的 `DEFAULT_WBC` 指向不存在的 ckpt；
3. WBC 跟踪精度（0.28 rad）决定 A' 的天花板。

---

## 9. ❌❌ 最严重的问题：699 条 clip 跨 7 个任务，不能塞进同一个场景

复核时用已完成分片做了一次后处理，结果**爆表**：

```
[merge] 16 个分片 -> 50251 行
[filter] 段数=2512  段内最大漂移: p50=23.7cm  p90=51.9cm  max=107.9cm
         > 30cm 的段占 36.3%
```

对照：**同一套参数、只跑 DoubleDesk 的 100 条 clip 时，段内漂移 p50 只有 4.9–6.4 cm**。

原因：库里的 699 条 clip 分属 7 个任务，而我第一版驱动把**全部 clip 塞进 DoubleDesk 场景**
（`--ha-task ... DoubleDesk`），于是 boxing / open_door / sit_sofa / vision_navi 的参考动作
在一个摆着双桌的场景里执行 ⇒ 必然"跑偏"。

### 9.1 正确的任务分组

| 前缀 | clip 区间 | 任务 ID | 本 checkout |
|---|---|---|---|
| `HOI_double_desk` | [0,100) | `Isaac-Move-PickPlace-DoubleDesk-...` | ✅ |
| `HOI_grap_cup` | [100,199) | — | ❌ **任务未注册 + 无 ArtVIP 资产**（见 `HANDOVER_ISAAC_CLOSEDLOOP.md §11`） |
| `HOI_pp_box` | [199,299) | `Isaac-Move-PickPlace-Box-...` | ✅ |
| `HSI_boxing` | [299,399) | `Isaac-Move-Boxing-Bag-...` | ✅ |
| `HSI_open_door` | [399,499) | `Isaac-Move-Open-Door-...` | ✅ |
| `HSI_sit_sofa` | [499,599) | `Isaac-Move-Sit-Sofa-...` | ✅ |
| `HSI_vision_navi` | [599,699) | `Isaac-Move-SmallWarehouse-VisionNavigation-...` | ✅ |

⇒ **可采 6 个任务 / 600 条 clip；GrapCup 的 99 条在本 checkout 采不了。**

### 9.2 驱动已重写

`scripts/run_a_prime_full_seg20.sh` 现在：

* 按任务分组循环，每个任务用自己的 `--ha-task` + `--ha-yaml`；
* 新增 `--clip-start/--clip-end` 限定分组内的 clip；
* 断点续跑（`<task>_off<N>.pt` 已存在则跳过）、每批 `timeout 300`、支持 `SMOKE=1`；
* 上一版把全部 clip 混跑的产物**已全部删除**（数据无效）。

冒烟验证：`SMOKE=1 TASKS=DoubleDesk` → `smoke_DoubleDesk_off0.pt`（400 行，1 条 clip，2 个窗口）。

### 9.3 这次复核暴露的教训

**"全量 699 条"这个前提本身是错的**：任务与场景一一绑定，
`Ha_task` 不是一个全局参数，而是**每条 clip 的属性**。
后处理脚本的"段内漂移"分布正好是发现这个错误的探针——
第一版 p50 23.7cm 和单任务 4.9cm 差 5 倍，一眼就能看出场景错了。

---

## 10. ❌❌❌ 致命 bug：历史所有 A' 数据**没有转移**

### 10.1 现象

```
runs/calib/apr_desk16.pt        max|states − next_states| = 0.0
runs/calib/cl_full2080.pt       max|states − next_states| = 0.0
runs/calib/a_prime_seg20.pt     max|states − next_states| = 0.0
```

**`states` 和 `next_states` 是同一个张量。**

### 10.2 根因（`isaac_a_collect_batch.py` 主循环）

```python
a = step_block(step, noisy=True)   # ← 内部已经 sim.step() 推进了仿真
q  = robot.data.joint_pos[...]      # ← 读到的已经是 s'
S.append(state)                     # s'
S2.append(同一个 state)              # s'   ← 又存了一遍
```

⇒ 每行是 `(s', a, s')`，**没有 (s,a,s')**；
且每行状态比它的 `frames` 标签**超前一步**（这解释了"段起点绝对漂移 7.6 cm"
——那 7.6cm 不是重置失败，是标签与状态错位一帧）。

### 10.3 修法

```python
_s_before = <state now>            # 步前抓 s
a = step_block(step, noisy=True)   # 推进到 s'
state     = <state now>            # 步后抓 s'
S.append(_s_before); S2.append(state)
```

修后实测：`DoubleDesk_off0.pt  max|s − s'| = 125.18` ✓（有真实转移）

### 10.4 影响面（最要紧的一条）

`next_states` 被这些下游用到：

* `scripts/build_real_transitions_v2.py`
* `scripts/concat_transitions.py`
* `scripts/bline_to_transitions.py`
* `scripts/pretrain_wm.py`（经 transitions 文件）

⇒ **任何用这些 dump 训过的 WM，看到的都是 "s' == s"**，
即"动作不改变状态"的世界模型。这足以解释：
WM 的速度尺度异常、critic 的 GAE 目标退化、imagined headroom 与 real 对不上。

### 10.5 复核后的动作

1. 采集器已修（默认走修正后的抓取顺序）；
2. 旧 A' 分片**全部删除**（无效数据）；
3. 全量重跑（6 个任务 / 600 条 clip，20 步段，σ=0.05）；
4. WM 训练必须用**修好之后**的 transitions。

### 10.6 这次复核一共抓到 5 个真 bug

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | **states == next_states** | WM 学 (s,a,s) | ✅ 已修 |
| 2 | 699 条 clip 跨 7 任务塞进同一场景 | 段内漂移 p50 23.7cm | ✅ 按任务分组 |
| 3 | Isaac teardown 卡死 | 50min 任务变数小时 | ✅ 硬退出+超时+续跑 |
| 4 | 初值关节序 idx_t | A' 180 步内必倒 | ✅ 默认 idx_s |
| 5 | `pin_ha_closedloop` 默认 ckpt 失效 | 闭环启动被拒 | ✅ 已修+复检通过 |

附加：**GrapCup 任务在本 checkout 不存在**（无注册/无资产）⇒ 全量是 **600 条**，不是 699。

---

## 11. 更正：之前的"段内漂移 23.7cm / 33% 坏段"是**错误口径**

### 11.1 错在哪

分片里的行是 **16 个 env 交错**存的：

```
row_i  = env(i % num_envs),  step(i // num_envs)
```

而我先前按"连续 20 行 = 一个 20 步段"去统计 ⇒ **把 16 个不同 env 的行拼成了一个段**，
所以算出来的"段内漂移"和"坏段比例"全是假的。

### 11.2 正确口径：逐行（s→s' 与参考帧 f→f+1 的位移差）

```
[filter] 逐行漂移: >2cm 9.8% | >5cm 0.8% | >10cm 0.1% | >30cm 0.1%
[filter] 阈值 30cm -> 丢弃 0.1%，保留 105890/105992
```

**90% 的行，每步位移误差 < 2 cm** —— 数据质量其实是好的。

### 11.3 受影响的结论

| 之前写的 | 更正后 |
|---|---|
| off0 "23% 坏段"、"p50 23.7cm" | ❌ 错误口径，作废 |
| 6 个任务混跑导致漂移爆表 | ⚠️ 结论仍成立（不同任务的参考动作不该在错场景里执行），但**数字不作数**；改用逐行口径重新评估 |
| 过滤阈值 30cm 丢 33% | ❌ 实际只丢 0.1%（阈值可以收紧到 5cm） |

### 11.4 现在的过滤口径

`scripts/post_a_prime_seg20.py` 已改为**逐行**：

```
err_t = || (p_{t+1} - p_t) - (ref[f+1] - ref[f]) ||
keep  = err_t <= --max-drift          # 默认 0.30，建议 0.05
```

**建议阈值改为 5cm**（丢掉 0.8% 的越界行），既保住数据量又剔除异常。

---

## 12. 收尾一轮查出的 4 个问题（2026-11-03 夜）

### 12.1 覆盖率其实只有 583/600 —— 两个漏采 bug

补洞前 `filtered.pt` 只覆盖 **583** 条 clip（本 checkout 能跑的只有 600 条，GrapCup 那 99 条没注册）。
差的 17 条不是"clip 太短"，是两条独立的 bug：

**(a) 窗口分片的尾批被 dismiss。** 驱动循环写成

```bash
while [[ $off -lt $TOTAL ]]; do
  if [[ $((off + PER)) -gt $TOTAL ]]; then break; fi   # ← 尾批直接不跑
```

⇒ 每个任务最后不足 16 个窗口的尾巴永远不采（6 个任务共 37 个窗口）。
修法：采集脚本加 `--allow-partial`（尾批用实际窗口数建 env，`n` 以 plan 为准），
新增 `scripts/run_a_prime_tail.sh` 专门补尾批。

**(b) 并行 Isaac 实例撞共享内存 → 退出码 0 但没产物。**
日志里是

```
FileExistsError: [Errno 17] File exists: '/isaac_multi_image_shm'
```

根因：`image_server/shared_memory_utils.py` 在 **import 时**执行
`SHM_NAME = resolve_shm_name()`，默认名写死；异常发生在 ImageServer 线程里，
进程照样退出 0 ⇒ 多卡驱动只看 rc 就把这个分片当成功，`resume` 又只判"文件存在"
⇒ **VisionNavi off192/off224 两个满批分片永久丢失**（clip 640/641/647/648）。
修法：照抄官方 `script/eval_scripts/sonic/sim_eval_vla.py:185`，在 import 前设
`ISAAC_MULTI_IMAGE_SHM_NAME=isaac_multi_image_shm_<pid>_<rand>`。

**(c) 一条 clip 被长度门槛踢掉。** `build_plan` 里是 `if L < steps + 2: continue`，
而 clip 385 的 `L=271`、`steps=270` ⇒ 合法窗口（起点 0 和 1）被误杀。
改成 `L < steps` 后单独补跑，覆盖率达到 **600/600**。

结果：

```
[merge] 135 个分片 -> 551209 行
[filter] 逐行漂移 p50=0.79cm p90=1.87cm
[filter] →5cm 丢弃 0.5%，保留 548226 行
[cover] 覆盖 clip 数 = 600 / 699
```

### 12.2 ❌ WM 刷新是**不可复现**的，跨轮次比 qd_corr 没意义

先说现象：同一份数据跑两次 `pretrain_wm --steps 6000`：

| 轮次 | 数据行数 | 日志 qd_corr |
|---|---|---|
| 第一轮（583 clip） | 529,641 | 0.421 → **0.700** |
| 第二轮（600 clip） | 548,226 | 0.363 → **0.512** |

数据只差 3%，结果差 0.19。查到根因：`refresh_wm` 的 minibatch 采样用的是

```python
g = generator or torch.Generator(device="cpu")   # ← 没播种，每次都不一样
```

`pretrain_wm` 从来没传过 `--seed`。**4 个 seed 的同一份数据实测**（固定评测集、batch=65536、seed=20261103）：

| ckpt | qd_corr | qd_ratio | q_mse | pos_mse |
|---|---|---|---|---|
| `WBC_A24000_wmvel_cmd.pt`（A' 之前的基线） | 0.3687 | 0.9417 | 1.89e-03 | **5.04e-06** |
| 583-clip 那次（运气好） | 0.6735 | 1.0038 | 1.03e-03 | 1.08e-05 |
| seed 0 | 0.5087 | 0.9920 | 1.51e-03 | 1.14e-05 |
| seed 1 | 0.6721 | 0.9759 | 9.98e-04 | 1.08e-05 |
| **seed 2（选中）** | **0.6763** | **0.9960** | 1.02e-03 | 1.13e-05 |
| seed 3 | 0.6740 | 1.0050 | 1.01e-03 | 1.13e-05 |

结论：

* 刷新**确实有效**：qd_corr 0.369 → 0.67x，qd_ratio 0.942 → 0.996（4 个 seed 里 3 个都到这里，
  不是侥幸）。**但 seed 0 只到 0.51**，所以"跑一次看数字"是不可接受的口径。
* 代价是 `pos_mse` 从 5.0e-06 退到 1.1e-05（`vel_weight=5.0` 奖励速度通道，牺牲位置精度）。
  这一项要记着：闭环里如果出现"跟着参考走但慢半拍"，方向就在这。
* 已经把 `--seed` 加进 `scripts/pretrain_wm.py`，并写了
  `scripts/sweep_wm_refresh.sh`（多 seed 扫 + 固定评测集打分 + 自动把最好的拷成
  `wm_pre_full.pt`）和 `scripts/eval_wm_refresh_ab.py`（可复现 A/B）。

### 12.3 ❌ 第 1 轮联合微调：act hinge 全程饱和 ⇒ 300 步白跑 + WM 根本没换

`runs/joint_v3_round1`（H=20, ACT_EPS=0.10, ACT_WEIGHT=0.30, 300 步, 4 卡 DDP）曲线：

| step | L_fm | token_mse | imag_ret_vla | imag_ret_gt | delta_win |
|---|---|---|---|---|---|
| 0 | 0.1521 | 0.00437 | 0.5536 | 0.5535 | 0.50 |
| 250 | 0.1519 | 0.00440 | 0.5525 | 0.5535 | 0.50 |
| 300 | 0.1519 | 0.00441 | 0.5526 | 0.5535 | 0.44 |

**几乎没动**。诊断：`act/rmse` 在 0.2–1.5 之间而 `act_eps=0.10` ⇒ `act/frac_out = 1.00`
（100% 样本出界），`loss/act ≈ 8` 完全盖住别的项，hinge 的梯度方向是"把两条 H=20
想象轨迹的 RMS 差压到 0.1 rad"，这在 H=20 上不可能达成 ⇒ 只产生噪声。

第二个问题（更要紧）：**第 1 轮没加载刷新过的 WM**。`launch_joint_ha_v2.sh` 里
`WM_CKPT` 默认空 ⇒ 不传 `--wm-ckpt` ⇒ FT 用的还是 WBC ckpt 自带那份 `qd_corr=0.369`
的旧 WM。"更新 WM"那一步在 FT 里是装饰品。已修（`run_joint_ft_round2.sh` 显式
`export WM_CKPT=$DIR/wm_pre_full.pt`，4 个 rank 日志都能看到
`[wm] swapped from .../wm_pre_full.pt`）。

### 12.4 第 2 轮配置（对照实验）

| 项 | 第 1 轮 | 第 2 轮 |
|---|---|---|
| WM | WBC 自带的（qd_corr 0.369） | `wm_pre_full.pt`（seed2，qd_corr 0.676 / ratio 0.996） |
| 数据 | 529,641 行 / 583 clip | 548,226 行 / **600 clip** |
| `ACT_EPS` | 0.10（100% 出界） | **0.40**（H=20 轨迹差的合理量级） |
| `ACT_WEIGHT` | 0.30 | 0.30（保持不变，只动 eps） |
| EVAL/SAVE | 250 / 500（没留下中间 ckpt） | **100 / 100** |

### 12.5 第 2 轮结果：**还是没有可测量的变化** —— 说明瓶颈不在 token 策略

`runs/joint_v3_round2`（H=20、ACT_EPS=0.40、WM=seed2 刷新版、548,226 行 / 600 clip）：

| step | L_fm | token_mse | hand_pred/label | ret_vla | ret_gt | delta_mean | win |
|---|---|---|---|---|---|---|---|
| 0 | 0.1521 | 0.00437 | 0.294/0.286 | 0.4461 | 0.4530 | −0.0139 | 0.44 |
| 100 | 0.1521 | 0.00437 | 0.294/0.286 | 0.4460 | 0.4530 | −0.0141 | 0.44 |
| 200 | 0.1521 | 0.00437 | 0.293/0.286 | 0.4466 | 0.4530 | −0.0128 | 0.44 |
| 300 | 0.1522 | 0.00436 | 0.293/0.286 | 0.4478 | 0.4530 | −0.0105 | 0.44 |

act hinge 这次是**健康**的（`act/frac_out` 0.45、`loss/act` 0.4，第 1 轮是 1.00 / 8.0），
但曲线照样不动。把两轮的证据拼起来，原因很清楚：

1. `imag_return_norm_vla` 与 `..._gt` 的差只有 **0.01**，而 `imag_delta_std ≈ 0.06`
   ⇒ **信噪比 < 0.2，win rate 0.44**。奖励侧根本没有可用的梯度方向。
2. 这和 headroom 探针的结论**完全一致**：gt token 0.8534 vs VLA token 0.8490（差 0.5%），
   也就是"就算给最优 token，这个 WM 也最多再多认 0.5%"。
3. 结论：**联合微调（用想象回报）在这条链路上没有可学的东西，不是超参没调好**。
   继续加步数/调 lr 是浪费时间。要动就得动别的：
   * `lambda_hand > 0`（手部现在完全没监督）+ 手部专项 loss；
   * WBC 跟踪精度（A' 里 `q` 与参考的残差 0.28 rad，是闭环漂移的直接来源）；
   * 或者干脆换目标（L_fm / token_mse 的纯蒸馏，不带奖励项）。

### 12.6 ✅ 官方闭环**第一次真正跑通**（600 步 / 12 s 不摔）

之前最后一次官方闭环（`runs/ha_isaac_eval/...20261103_185157`）是 `sim_error`：
`HTTP 500 ... FlashAttention only support fp16 and bf16`。sdpa 修掉之后这次跑通了：

```
[sim_eval_vla] episode_start seed=0 repeat_idx=0 rss_mb=6368.6
[handdbg] chunk#29 hands14 min=-0.067 max=0.064 mean=-0.002
[sim_eval_vla] episode_end ...
success=False  failure_reason=timeout  episode_steps=600  duration=77.1s
[server] infer #30 dt=244.3ms   → 600/20 = 30 次重规划，节奏对
```

* **没摔**（`failure_reason=timeout`，不是 `fall`），600 步 @50Hz = 12 s 全程站立。
* 30 次 `/infer`、每次 ~240 ms ⇒ receding horizon 20 步的执行时间（0.4 s）**够用**，
  推理延迟不是瓶颈。
* `max_reward = 0.0` 是**这个 eval harness 的正常值**（Move-PickPlace 的成功判定走
  `success` 谓词，`reward_manager` 没有 active 项），不要当成"奖励没算"的 bug。

### 12.7 ❗ 剩下的真缺陷：手部自锁在"张开"吸引子（有证据）

数据集里手部命令 `|hand|` 均值 **0.29**、有 4% 的样本 > 1.7（就是抓取瞬间）；
闭环里模型输出的手部只有 **0.05**（30 倍偏小），`hand_gt170 = 0.000`（抓取那一档**从不出现**）。

根因链（已查证）：

1. 训练数据 state46 的手 14 维来自**录制机器人的真实手指关节**；
2. 部署时 `_vla_state46()`（`action_provider_supermimic.py:641`）在 `_hand_enabled` 分支读
   `robot.data.joint_pos[finger_idx]` —— 而手指目标在初始时刻≈0，于是模型看到"手是张开的"；
3. 模型据此输出小幅度手部命令 ⇒ 手指目标仍然≈0 ⇒ 下一步看到的还是"张开"
   ⇒ **正反馈自锁**。数据集里给模型做锚的是"真的在动的手指状态"，部署时这一路是死的。

实测支持这个解释：把 `SM_HAND_SCALE` 从 1 调到 8，手部会**自己涨起来**
（chunk#24→#29：0.05 → 0.75 rad），说明幅度是可放大且方向对的，只是起点卡在吸引子上。
即使是 ×8 也没能完成任务（`success=False, timeout`），说明光放大幅度不够，
要么补手部监督（`lambda_hand>0`），要么把参考手部流接进来（`--sm_hands_npy`）。

**修法（下一步，未做）**：训练侧把 state46 的手 14 维以概率 p 置零（对齐部署分布），
让模型学会**只靠视觉**决定何时闭手；或者部署侧把"上一条手部命令"回灌进 state46
当 `current hand state`（自洽，但需要确认不会削弱视觉条件）。

### 12.8 手部自锁假设**被实验否掉**了（记录一下，别再走这条路）

给 `action_provider_supermimic._vla_state46()` 加了诊断开关
`SM_HAND_STATE=measured|cmd`（`cmd` = 用上一条下发命令代替测量手指角当状态），
其余不变再跑一次 600 步闭环：

| 配置 | 手部命令 `|hand|` 峰值 | 结局 |
|---|---|---|
| 旧行为（measured） | 0.067 | timeout / 600 步 |
| `SM_HAND_SCALE=8` | **0.758** | timeout / 600 步 |
| `SM_HAND_STATE=cmd` | 0.064 | timeout / 600 步 |

⇒ **"状态侧反馈自锁"不成立**（cmd 与 measured 本来就几乎相同，因为手指就是被命令推着走的）。

修正后的结论：手部幅度小**不是手部支路自己的 bug，而是身体不在轨迹上的症状**。
模型在 A' 数据里 `hand_absmean_pred 0.293 ≈ label 0.286`、`hand_gt170 0.072 ≈ 0.071`
—— 数据集条件下手部统计量是**对得上**的；只有闭环里 `hand_gt170 = 0`
（一次抓取都没尝试）。合理的解释是：手没被送到物体那儿，模型就没有"该闭手了"的条件，
于是维持小幅度。这也把优先级重新排回**身体跟踪/漂移**（WBC `q` 残差 0.28 rad），
而不是继续在手部输出上加增益。

顺带记一下 `SM_HAND_SCALE=8` 时手能自己涨到 0.75 rad 而**没把机器人掀翻**
（`_hand_max_delta=0.2 rad/step` 限速起了作用）—— 说明限速那套安全逻辑是有效的，
将来真把大幅度手部接回来可以放心。

---

## 13. 🔒 契约锁死（2026-11-04）：把踩过的坑变成可执行断言

我在这条链路上连续犯了三次同类错误（关节序、ref 布局、循环论证的"读回偏差"），
所以不再靠"记得别犯"，而是把每条不变量写成**会 FAIL 的断言**，并在采集器里做**运行时硬门禁**。

### 13.1 一条命令的契约门

```bash
bash scripts/check_all_contracts.sh                    # 采集/训练/评估/交付之前先跑
python3 scripts/verify_contracts.py --skip-shards      # 只跑离线契约（秒级）
python3 scripts/verify_contracts.py --shards runs/a_prime_sonic
```

当前状态：**22/22 PASS**。

### 13.2 锁的是哪些不变量（每条都对应一次真实事故）

| 断言 | 当初的症状 |
|---|---|
| `ref q 在 [13:42]`（相对容差 1e-2，fp16 量化） | 按扁平布局 `[7:36]` 读 ⇒ 报出 0.38 rad 假跟踪误差（真值 0.14） |
| `ref [7:36] 不是 q`（反证） | 同上，防止有人"改回去" |
| `decoder_q_offset ≡ −SONIC_DEFAULT_POS`（1e-5） | 被当成"原始 q"，实际是 SONIC 默认角（差 6e-8） |
| `obs q 块 = qpos − SONIC_DEFAULT`（1e-5） | SONIC decoder 的 q/qd 用 standard 序 ⇒ 闭环 50 步发散 |
| `target = raw*G1_ACTION_SCALE + SONIC_DEFAULT`（1e-5） | 动作打到错关节 |
| `final_action == target` / `无延时后处理` | 曾猜"缺部署侧动作后处理"，实测不存在 |
| `last_action 在 [674:964]（滞后一帧）` + `不是同期（反证）` | obs 在动作更新前拼接，按同期比会误判 |
| `SONIC 序 ≠ standard 序` + `A∘B == 恒等` | 两个 29 关节序不同，且不是恒等置换 |
| `有真实转移` `max|s−s'| > 1e-2` | 历史数据集 `states == next_states`，**没有任何转移** |
| `没有摔倒` `z_min > 0.25` | 旧 A' 数据 `z_min=0.06`（含摔倒）没人发现 |
| `逐行漂移 p50 < 2cm` | 逐行口径才有效（按段统计会被 16 env 交错骗） |
| `段首复位(正序) < 0.05 rad` **且** `错序读明显更大` | 后者是**反证**：只查"正序小"会被"自洽地写错"骗过去 |

### 13.3 采集器里的运行时硬门禁（不通过就 `SystemExit`，不写数据）

1. **启动自检**：`decoder_q_offset + SONIC_DEFAULT_POS` 必须 < 1e-5；`idx_s != idx_t` 必须成立。
   日志：`[a-line][CONTRACT] ✅ SONIC 契约：q/qd 走 idx_s、目标走 idx_s、offset≡−SONIC_DEFAULT_POS`。
2. **每次段重置后的非循环校验**：同时按"正序/错序"两种索引读回，
   要求 `正序 < 0.05 rad` 且 `错序 > max(3×正序, 0.05)`。这一条直接锁死"每 20 步瞬移到错位姿态"。
3. **落盘前**：`max|s−s'| > 1e-2`（有转移）、`z_min > 0.20`（没摔）、全 `finite`。

### 13.4 下游脚本的门禁

* `post_a_prime_seg20.py` 之后：`check_all_contracts.sh` 会校验 `filtered.pt` 的转移与摔倒。
* `run_joint_ft_round2.sh`：
  - `WM_CKPT` 必须存在（否则第 1 轮那种"没加载刷新过的 WM"会重演）；
  - `ACT_EPS < 0.20` 直接 FAIL（H=20 的轨迹差量级约 0.4，过小 ⇒ frac_out=1.0、300 步白跑）；
  - 开跑前先过 `check_all_contracts.sh`。
* `pretrain_wm.py`：把 `refresh_recipe`（**seed**/steps/batch/lr/vel_weight/kl_anchor/transitions）
  写进产物 —— 以前 `args` 里没有 seed，事后根本分不清两份 ckpt 是不是同一次随机采样。

### 13.5 驱动器层面的两条纪律

* **`rc=0` 不等于成功**：并行 Isaac 撞共享内存会 `rc=0` 但没产物。
  `run_a_prime_sonic_seg20.sh` 只看**产物文件是否存在**，并显式打印"没有产物（失败）"。
  （另外已在采集器 import 前设 `ISAAC_MULTI_IMAGE_SHM_NAME=<pid>_<rand>`，根除这个冲突。）
* **只对在用的数据目录做硬门禁**：历史目录（如 `runs/a_prime_seg20`）只报 `warn`，
  但仍会打印 `z_min`，便于一眼看出"旧数据里有摔倒"。
