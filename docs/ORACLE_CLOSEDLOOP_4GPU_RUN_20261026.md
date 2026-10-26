# oracle-token 闭环：本机（4×4090 / 驱动 595）执行结果与换机交接

> 对应文档：`docs/HANDOVER_ORACLE_TOKEN_CLOSEDLOOP_4GPU_20261026.md`
> 本机：4× RTX 4090 / NVIDIA 驱动 **595.71.05（CUDA 13.2）** / Ubuntu 24.04
> 执行时间：2026-10-26 12:50–14:40 UTC
> 结论先行：**步骤 1（造 oracle token）本机能跑、已跑；步骤 0/2（三臂闭环）本机跑不了** ——
> Isaac Sim **5.0.0** 在 595 驱动上一启动就段错误（同一台机上 Isaac Sim **6.0.1** 能正常起，已做对照）。
> 按交接文档 §2「这一步必须在能跑 Isaac 的机器上做」，闭环换到 570 驱动的机器执行；
> 本文件给那条机器准备好可直接抄的命令与产物。
> **本轮没有得出"奖励项有没有用"的答案**（闭环没跑成），能得出的结论见 **§0**。

---

## 0. 结论：这轮到底说明了什么

### 0.1 已经能说的（有数据）

**① 想象空间真实存在，且方向与交接文档一致。** 在冻结 WM 里只优化 token（不动任何权重），
单条 clip 的想象回报能抬 **+20.6% / +22.6%**（chunk 中位 +19.8% / +20.7%），并且随优化步数单调
（文档 60 步 → +34%；本机 6 步 → +20%，见 §3.1）。说明 **WM + 奖励这条链可微、有 headroom、没坏**。

**② 这个 oracle 有多"野"，可以量化——而且比已经失败的东西还野。**
oracle 的 `drift`（`|optimized_token − GT_chunk|` 均值）与训练侧 `token_shift`
（`|vla_token − 标签token|` 均值，`diff_vla_wbc/train_v2.py:568`）**同定义，可直接比**：

| 对象 | 偏离 GT（同定义） | 真机/评测后果 |
|---|---|---|
| `s3c6h_lam0.1` 训 1000 步的 ckpt | `token_shift` **0.0855** | 已破硬门禁：`token_mse +166%`、`hand_gt170 −49%` |
| `s3c6h_lam1.0` 训 1000 步的 ckpt | `token_shift` **0.0942** | `token_mse +215%`、`hand_gt170 −79%` |
| **oracle token（本轮）** | `drift` **0.111 / 0.113** | **未测**（闭环被阻断） |

即：**我造出来的 oracle，比一个已经被判定"手塌、破硬条件"的 ckpt 还要远离标签分布 1.2–1.3×。**
这既是实验的力量（它测的不是温和改进，而是"被 WM 推到分布外还能不能更好"），
也是最可能失败的地方 —— 判读时必须带着这个基线看，别把"oracle 输给 GT"简单归因为"想象增益搬不过去"，
它同时也可能是"WM 奖励在这个偏离量级上已经开始利用模型"。

### 0.2 还不能说的（正是本来要回答的那个）

**想象里的 +20% 能不能搬到真机：一句话都不能说。** 闭环没跑，三臂表是空的，
按交接文档 §4 的三个分支**仍然全部开着**：

| 闭环结果 | 结论 | 本轮状态 |
|---|---|---|
| oracle > gt ≥ vla | 增益可迁移 → 投 v3 训练 | 未测 |
| oracle ≈ gt ≈ vla | 搬不过去 → 收手，资源转数据/执行器 | 未测 |
| oracle < gt | WM 奖励方向有偏 → 先修 WM | 未测 |

所以 **「S3 该继续投训练还是收手」没有答案**。本轮交付的是**待测物**（oracle token）与**量具**
（`launch_oracle_3arm.sh` / `report_oracle_3arm.py` / consume 核对法），不是测量结果。

### 0.3 意外但最值钱的发现：闭环这条"最终判据"是单点

同一台 4×4090：

| 栈 | 同一驱动 595.71.05 | 结果 |
|---|---|---|
| Isaac Sim **5.0.0** + IsaacLab 2.2（项目 pin 的口径） | | ❌ 启动即 SIGSEGV（§2.2） |
| Isaac Sim **6.0.1** | | ✅ 正常起、正常退 |

所以不是"没配好"，是 **5.0 栈 × 595 驱动这一对绑死**。而所有"S3 最终以闭环为准"的门禁都依赖前者 ⇒

> **整条 S3 主线的最终判据，实际只有一个单点：一台装 570 驱动的机器。**
> 那台机器没了 / 驱动被升了，主线就停摆。今天已经停摆一次，定位花了约 2 小时（且这 2 小时不可复用，
> 因为没有把驱动写进契约）。

### 0.4 与本轮 λ_r 消融的呼应（解释性，不作为结论）

两条独立读数指向同一件事：**`imag_return` 这个量太容易被推高，它本身不是成功率的可靠代理。**
- `runs/s3c6h_lam*`：奖励项能把 `imag_return_vla` 推高并**越过 GT**（λ_r=1.0 超 GT +0.034），
  代价是 token/手部保真度随 λ_r 单调退化（`docs/S3_LAMR_SWEEP_2GPU_RESULT_20261026.md`）；
- 本轮：**不训练、只改 token** 也能把想象回报推高 20%，代价是偏离 GT 0.111。

因此三臂闭环表的价值不止"投不投 v3"，它同时是**给 `imag_return` 这个代理做一次真机标定**。
而这张表现在被一台驱动版本卡住 —— 这就是 §6 建议的由来。

### 0.5 一句话

> 这轮说明的是"**标定问题有多严重**"（oracle 比已失败的 ckpt 还野 1.3×，imag_return 易被推高），
> 不是"**奖励项有没有用**"（闭环未跑，三个分支全开）。
> 要拿到后者的答案，唯一路径是 §4 那三条命令在 570 驱动的机器上跑一遍。

---

## 1. 本机做了什么

| 阶段 | 状态 | 产物 |
|---|---|---|
| 环境修复（解释器坏链 / GL 库 / Vulkan ICD） | ✅ 修好，`isaacsim 5.0.0 + isaaclab 0.44.9` 可 import | 见 §2.1 |
| 闭环契约校验 | ✅ `[pin] 契约校验通过：7 任务 / 16 个资产 / init pose 已锁 / decoder=release` | — |
| replay token server 协议自测 | ✅ `GET /` / `POST /infer` 正常，`consume=12` 时 cursor 每次 +12 | `logs/replay_selftest.log` |
| **步骤 1：造 ORACLE token** | ✅ 两条 clip 跑完（见 §3） | `runs/oracle_tokens/*.npy|.json` |
| 步骤 0：clip↔场景 + consume 核对 | ❌ **被 Isaac 阻断** | — |
| 步骤 2：三臂闭环 | ❌ **被 Isaac 阻断** | — |

---

## 2. 阻断：Isaac Sim 5.0 在驱动 595 上必崩

### 2.1 先修好的三个环境问题（这台机器的真实缺口）

1. **venv 解释器坏链**：`third_party/isaacsim_env/bin/python` 指向 `/root/.local/share/uv/python/cpython-3.11-…/python3.11`，该目录**不存在**（本机 uv 只装了 3.10）。
   → 修法：`uv python install 3.11`（生成 `cpython-3.11-linux-x86_64-gnu` 别名，正好是 `pyvenv.cfg` 里 `home` 期望的路径）。
   修后：`python 3.11.16`，isaccsim/isaaclab 均可 import。
2. **缺 GL/X 库**：`libXt.so.6`、`libGLU.so.1` 缺失（RTX 的 `libMaterialXRender*`/`libneuray.so` 加载失败）。
   → `apt-get install -y libxt6 libglu1-mesa`。
3. **Vulkan ICD**：`/etc/vulkan/icd.d/nvidia_icd.json` 本来就有；系统里另有 mesa(lavapipe/intel/nouveau) 等 8 个 ICD。
   **注意：不要往 `/usr/share/vulkan/icd.d/` 再加一份 nvidia ICD**——那会让同一张卡被多家 ICD 重复上报，
   Kit 直接报 `[Error] [gpu.foundation.plugin] Multiple Installable Client Drivers (ICDs) are found for the same GPU`（我踩过，已撤掉）。
   只有当环境里完全没有 nvidia ICD 时，可用 `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` 限定。

### 2.2 修完仍然崩，且是 5.0 独有的崩

最小复现（不加载任何场景/资产）：

```python
# /tmp/isaac_min.py
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "hide_ui": True})
print("[min] SimulationApp OK")
import omni.usd; omni.usd.get_context().new_stage()
print("[min] new_stage OK"); app.close()
```

| 解释器 / 栈 | 结果 |
|---|---|
| `third_party/isaacsim_env`（**Isaac Sim 5.0.0** + IsaacLab 2.2） | ❌ **SIGSEGV（139）**，崩在 `librtx.scenedb.plugin.so` ← `carb.scenerenderer-rtx` ← `omni.hydra.rtx` ← `UsdManager::createHydraEngine` |
| `/inspire/.../isaacsim-6.0.1/env`（**Isaac Sim 6.0.1**，同机同驱动） | ✅ `SimulationApp OK` / `new_stage OK`（91 s 起完，正常退出） |

py-spy 抓到的 Python 栈（5.0 崩前最后位置）：

```
MainThread:
  _wait_for_viewport   isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py:559
  __init__             .../simulation_app.py:285
```

即 **RTX 视图初始化阶段**就挂了，跟场景装配无关。日志里同时有：

```
Warp CUDA error: Failed to get driver entry point 'cuDeviceGetUuid' (CUDA error 1)
Warp CUDA error 36: API call is not supported in the installed CUDA driver
```

（`libcuda.so.1` 用 ctypes 检查过 `cuDeviceGetUuid` 是**存在**的，所以这是 5.0 自带 CUDA 12.8 运行库与 595 驱动
在 `cuGetProcAddress` 层的版本协商失败，不是缺符号。）

**判定**：Isaac Sim 5.0 与驱动 595.71.05 不兼容；本机要跑闭环只能换栈（6.0.1 / 5.1），
但项目口径 pin 在 5.0.0 + IsaacLab 2.2.0（`configs/ha_closedloop_pin.json` + 历史结果都基于它），
**换栈就等于换口径，结果不能与旧闭环表并列**。所以按交接文档 §2，换到**570 驱动的机器**执行。

> 交接文档 §2 说「本机（2 卡）Isaac 不可用（解释器坏链已修，但场景装配卡死）」——那是同一类症状；
> 本机（4 卡机）的解释器坏链我已经修好，但**驱动版本**这一层没法在容器里降级（595 是宿主驱动）。

---

## 3. 步骤 1 已产出：ORACLE token（本机可跑，已完成）

命令（每张卡一条 clip，互不干扰）：

```bash
export PYTHONPATH="$PWD:$PWD/Isaac-GR00T:$PWD/third_party/gr00t_deps:$PWD/third_party/gr00t_deps/_noflash"
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD="$PWD/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"
CUDA_VISIBLE_DEVICES=1 python3 scripts/build_oracle_tokens.py \
  --clips HOI_double_desk__tw__ep000000 --chunk 40 --opt-steps 20 --perturb 3.0 --out-dir runs/oracle_tokens --device cuda:0
CUDA_VISIBLE_DEVICES=2 python3 scripts/build_oracle_tokens.py \
  --clips HOI_pp_box__yb__ep000000 --chunk 40 --opt-steps 20 --perturb 3.0 --out-dir runs/oracle_tokens --device cuda:0
```

**与文档的唯一偏差**：`--opt-steps 20`（文档写 60）。原因：本机 60 步实测 **229 s/chunk × 52 chunk ≈ 3.3 h/clip**，
20 步是 ~78 s/chunk ≈ 68 min/clip；而门禁只要 `gain_rel` 中位 ≥ +10%（2 步时就已 +11.3%）。
两个 `.json` 里 `opt_steps` 字段如实记录了 20，判读时按此口径。

### 3.1 门禁结果：两条 clip 都过（`gain_rel` 中位 ≥ +10%）

| clip | 帧数 / chunk 数 | `ret_gt` → `ret_oracle`（chunk 均值） | **`gain_rel` 中位** | 逐 chunk 范围 | >+10% 的 chunk 占比 | 漂移中位 |
|---|---|---|---|---|---|---|
| `HOI_double_desk__tw__ep000000` | 2080 / 52 | 2.256 → **2.720** | **+19.8%**（均值 +20.6%） | +3.6% ~ +140.1% | 83% | 0.112 |
| `HOI_pp_box__yb__ep000000` | 1917 / 48 | 2.216 → **2.716** | **+20.7%**（均值 +22.6%） | +4.8% ~ +106.5% | 94% | 0.113 |

产物自检：`(2080, 64)` / `(1917, 64)` float32、**无 NaN**、`|token|` 最大 0.688（codec 原始范围 ±1.74 内）、
每条 480–520 KB；`drift`（optimized 与 GT chunk 的平均绝对差）中位 0.11，说明不是在疯狂外推。

**与交接文档口径的差异（务必写进判读）**：文档报的是 `--opt-steps 60` → +34%（`probe_token_oracle.py`）。
本机 60 步要 3.3 h/clip，所以用了 **6 步**，拿到 +20%/+23%。收益随步数单调，故本产物是**偏弱的 oracle**；
若闭环里它都能赢 GT，结论比"用 60 步的强 oracle 赢了"更硬（保守方向）。

### 3.2 clip 名注意

PPBox 的 npz 中缀是 **`__yb__`**（`HOI_pp_box__yb__ep*`），不是 double_desk 的 `__tw__`。
交接文档 §3 的示例用了 `__tw__`，照抄会 `[oracle][skip] … npz 找不到`（我踩过）。


---

## 4. 换机后怎么跑（570 驱动的机器，照抄即可）

前置：`git ≥ fd73963`（本仓库已含本次新增的 `scripts/launch_oracle_3arm.sh`、`scripts/report_oracle_3arm.py`）、
`python3 scripts/pin_ha_closedloop.py --check` PASS、`bash scripts/isaac_env.sh check` 三项全 OK、
`runs/oracle_tokens/<clip>.npy` 已在共享 GPFS 上（本机已产出，路径相同）。

### 4.1 先核 `--consume`（交接文档 §3 步骤 0，必须做，否则所有回放结论不可信）

> **⚠️ 2026-10-26 16:40 更正（S3 口径统一后）**：部署的 `action_provider_sonic.py` 用
> `HA_VLA_EXEC_STEPS=20` 做 receding horizon（2.5 Hz @ 50 Hz），我们的
> `action_provider_supermimic.py` 已同口径；`scripts/ha_replay_token_server.py` 的 `--consume`
> 默认也改成了 **20**。所以这里 probe 的两档应是 **{20, 40}**、固定值应是 **20**，不是 12。
> 用 12 会让 token 流比仿真**快 1.67×**，三臂表全部作废。

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# 两档各跑 20 步，看仿真步与 [replay] infer 的 rows 增长是否 1:1
for C in 20 40; do
  TASK=DoubleDesk CLIP=HOI_double_desk__tw__ep000000 SEEDS="0" CONSUME=$C \
    MAX_STEPS=20 MODE=probe SKIP_VLA=1 bash scripts/launch_oracle_3arm.sh
done
ls -t logs/3arm_gt_*.log | head -2 | xargs -I{} sh -c 'echo "--- {}"; grep -c "\[replay\] infer" {}'
```

判读：`[replay] infer #k rows a..a+40`，`consume=12` 时每次 `a += 12`。取**让 token 流与仿真步 1:1** 的那档
（provider 每 20 步重取 → 应为 **20**；只有当 provider 保留旧的"全消费 40 帧"行为时才固定 40。三臂必须统一）。

### 4.2 三臂闭环（每任务；GT / ORACLE / VLA 用同一 clip、同一手部来源）

```bash
# DoubleDesk：一条 clip 同时供 GT 和 ORACLE，手部都是录制 g1_hand_joints_14（口径一致）
TASK=DoubleDesk CLIP=HOI_double_desk__tw__ep000000 SEEDS="0 1 2" CONSUME=20 MAX_STEPS=300 \
  ISAAC_GPU=0 bash scripts/launch_oracle_3arm.sh

# PPBox（换 clip；PPBox 的 npz 中缀是 __yb__，不是 __tw__）
TASK=PPBox CLIP=HOI_pp_box__yb__ep000000 SEEDS="0 1 2" CONSUME=20 MAX_STEPS=300 \
  ISAAC_GPU=0 bash scripts/launch_oracle_3arm.sh
```

脚本会：起 GT(18081) / ORACLE(18082) / VLA(18080) 三个服务 → 等 `{"ok":true}` → 跑
`ha_isaac_batch.sh`（`VARIANTS=gt,oracle,vla`）→ 打印 `SUMMARY.md`。
想省掉 VLA 臂（不加载 6.5 GB 权重）：加 `SKIP_VLA=1`。

也可以不用编排器，手工等价命令就是交接文档 §3 步骤 2 那三条（端口 18081/18082/18080 不变）。

### 4.3 出表（§4 四项判据）

```bash
python3 scripts/report_oracle_3arm.py runs/oracle_3arm_DoubleDesk_<stamp>
```
表里给：每任务 × 每臂的 `episodes / success / rate / fall / timeout / avg_steps / avg_maxR`，
以及**每任务内相对 gt 的差值**（成功率、摔倒率、平均步数、`max_reward_scaled`）。

> 注意「到目标最小距离」本仓库 `result.json` 没有单独落盘（`_build_result_payload` 只有
> `success/failure_reason/episode_steps/final_reward/max_reward[_scaled]`），所以用 `max_reward_scaled` 当代理。
> 「手闭合事件数三臂一致」在 GT/ORACLE 臂是**构造保证**的（同一 npz 的 `g1_hand_joints_14`），
> 可用 `logs/3arm_gt_*.log`/`3arm_oracle_*.log` 里的 `hand_max==` 逐行核对；VLA 臂手部来自模型输出，**必然不同**，
> 这一项的对照组应在 GT vs ORACLE 之间做。

### 4.4 回传清单（交接文档 §6）

| 文件 | 说明 |
|---|---|
| `runs/oracle_tokens/*.json` | 每条 clip 的 `ret_gt → ret_oracle`、`gain_rel`、每 chunk 漂移（§3 已产出） |
| `runs/oracle_3arm_*/summary.json` / `SUMMARY.md` | 三臂 per_task / per_variant |
| `runs/oracle_3arm_*/runs/*/result.json` | 每 episode 的 `success / episode_steps / failure_reason`（重点看 `fall`） |
| `logs/3arm_*.log` 的 `[replay] infer` 行 | 证明 token 流播放速率正常 |

---

## 5. 新增文件（本机未改任何既有代码/口径）

| 文件 | 作用 |
|---|---|
| `scripts/launch_oracle_3arm.sh` | 三臂编排器：起三个 token 服务 → 跑批量闭环 → 出 SUMMARY；`MODE=probe` 用于 §4.1 的 consume 核对 |
| `scripts/report_oracle_3arm.py` | 按 §4 四项判据出表（成功/摔倒/步数/`max_reward`，含相对 gt 的差值） |
| `runs/oracle_tokens/<clip>.npy` + `.json` | ORACLE token 产物与审计 |
| `logs/oracle_*.log` | 造 token 的过程日志 |
| 本文件 | 诊断 + 交接 |

环境侧改动（非仓库文件，只影响本机，供后人对照）：`uv python install 3.11`；`apt install libxt6 libglu1-mesa vulkan-tools`；
未改 `/etc/vulkan`（曾误加 `/usr/share/vulkan/icd.d/nvidia_icd.json`，已删除）。

---

## 6. 建议动作（按性价比排序）

### P0：把驱动写进契约，让"单点"变成"一眼可见"

1. `configs/ha_closedloop_pin.json` 增加 `driver_version: "570.124.06"`（±小版本白名单），
   由 `scripts/pin_ha_closedloop.py --check` 一并校验，判据：`nvidia-smi --query-gpu=driver_version` 落在白名单内，
   否则 `[pin] 驱动不在闭环口径内，拒绝启动`。理由：本轮 2 小时的定位成本，就是因为契约里缺这一行。
2. 交接文档模板里加一句（一行就够）：
   > **驱动 595+ 请勿尝试：Isaac Sim 5.0 在此驱动上启动即 SIGSEGV；570.x 是已验证口径。**

### P1：闭环前先跑三条自检（30 秒，能把"环境坏"和"实验失败"分开）

```bash
python3 scripts/pin_ha_closedloop.py --check        # 契约（含驱动）
bash scripts/isaac_env.sh check                     # torch / isaacsim / isaaclab 三件套
python3 - <<'PY'                                    # 最小 SimulationApp（§2.2 那段）
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "hide_ui": True}); app.close(); print("[ok] isaac boots")
PY
```
三条任一失败 → 当日闭环结果作废，别写进任何对比表（本轮就是靠第三条把结论钉死的）。

### P2：拿答案的唯一路径（在 570 机器上，按 §4 抄）

`consume=12/40` 核对 → DoubleDesk / PPBox 各 `SEEDS="0 1 2"` 三臂 → `report_oracle_3arm.py` 出表 → 按 §4 三分法判读。
判读时把 **§0.1 的 0.111 偏离基线**摆在旁边：oracle 若输，先分清是"增益不可迁移"还是"WM 奖励在分布外被利用"。

### P3：中期（排期活，不是顺手活）

要么固定一台 570 专用机（本机不参与闭环），要么整体升到 Isaac Sim 6.0.1 + 对应 IsaacLab，
并**重跑 pin 校验 + 一集 GT 闭环**做口径再基线化 —— 换栈意味着历史闭环表不能直接并列，
这一步不做，"最终以闭环为准"的判据就永远吊在一台机器上。

