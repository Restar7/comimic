# P21 续训交接（refresh#2 联合微调）

> **✅ 已完成（2026-11-11 07:09 UTC）**：`run_p21_resume.sh` 按 §2 跑完 175→300（18.3 min），
> **没有新 best**（best 仍为 it=150，`val_dJ=0.002172`；200/250/300 = 0.00124/0.00175/0.00167），
> 每 25 步权重位移衰减到 0.05%（配方已收敛）。按 §3 规则判定 **停**。
> 结果与下一步见 `comimic-c-wm6/docs/P21_RESUME_RESULT_20261111.md`。以下原文保留作背景。
>
> **闭环最终结果（2026-11-11 15:24，28 集跑完）**：p21 **18/28** vs C **14/28** vs base **15/28**；
> 配对 p21 vs C = **+6 / −2（p=0.289）**，vs base = **+7 / −4（p=0.549）**；p21 失败 = **9 timeout + 1 fall**。
> 分任务：OpenDoor 4/7、Boxing 5/7、SitSofa 5/7、VisionNavi 4/7。§4 是 14:40 的中间快照，最终以上面为准。
>
> 2026-11-11 14:45 ｜ C 线 ｜ 写给接手的人：**先看 §2 的命令，再决定跑不跑**
>
> **一句话**：p21 训练断在 200/300，best=150；续训脚本已就绪（约 15 min）；闭环中间结果 **p21 11/17、C 7/17、base 10/17**（未跑完，见 §4）。

---

## 0. 速览

| 项 | 值 |
|---|---|
| 是什么 | 从 C 基线 LoRA 出发、用 **refresh#2** 当裁判、联合微调 150 步（best） |
| 训练状态 | 计划 300，实际跑到 **200 中断**；best = **update 150**，`dJ = +0.00217` |
| 续训脚本 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/run_p21_resume.sh` |
| 续训方式 | `--resume p21/ft300`（trainer_state 停在 175）→ 续到 300（**125 步**） |
| 输出 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/p21/ft300_resume/`（**不动原 ft300**） |
| 闭环中间结果 | p21 **11/17** vs C **7/17** vs base **10/17**（见 §4，未跑完） |
| 重要边界 | refresh#2 的 `dir_agree` 只有 **77%**；续训**不会**修裁判方向/token 通道 |

---

## 1. p21 是什么

| 项 | 值 |
|---|---|
| 起点 | C 基线 `critic10/lora300_ddp/checkpoint_lora_best.pt`，sha **6942eab9** |
| 裁判 | `p19/refresh2/wm_cl_candidate.pt`，sha **19163963** |
| λ | `--lambda-lim 0.00490`，无 margin，无 recovery |
| 计划/实际 | 300 步 / 跑到 200 中断；best **150** |
| best val | `J 0.56858` vs `J_base 0.56641`，`dJ +0.00217`，CI [+0.00126, +0.00313] |
| token 改动 | 相对 C 约 **2–3%**（`dtok_rms 0.0034`，cos 0.9996）；27% 是 FM 噪声 |
| 产物 | `p21/ft300/checkpoint_lora_best.pt`（update 150）、`critic_0175.pt`、`trainer_state.pt`（175） |

血缘：
```
FT8000（官方 base，权重冻结）
  └─ + LoRA 300 步，裁判 wm_base6，λ 0.0307  →  C 基线（6942eab9）
       └─ + LoRA 150 步，裁判 refresh#2，λ 0.0049  →  p21（update 150）
```

---

## 2. 续训命令（唯一入口）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
bash runs/dual_actor_wm6/c_lora/run_p21_resume.sh
```

### 前置（脚本会自己检查前两条）

1. **闭环评测已结束**：脚本检测到 `c2_driver.py` 在跑会直接退出；
2. **两张卡空出来**（当前评测占着 2 卡；48 GB/卡，评测时每卡只剩 ~27 GB）；
3. 训练**不需要** Vulkan，但脚本仍设了 `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`（见 §5 的坑）。

### 脚本关键配置

| 参数 | 值 |
|---|---|
| 起点 | `checkpoint_lora_0175.pt`（脚本先复制成 `checkpoint_lora_last.pt`） |
| `--resume` | `p21/ft300`（trainer_state 在 175，续到 300） |
| `--out` | `p21/ft300_resume` |
| 裁判 | `p19/refresh2/wm_cl_candidate.pt` + `--strict-wm-hash 19163963` |
| λ | `--lambda-lim 0.00490` |
| critic | `--critic-warm-from p21/ft300/critic_0175.pt` |
| peak budget | `--peak-budget-json p21/ft300/peak_budget.json` |
| 步数 | `--updates 300 --lr-total-updates 300 --warmup 20`（从 175 续，共 125 步） |
| 调度 | `--lr 1e-4 --min-lr 1e-6`，原 cosine |
| 存/评 | `--save-every 25 --eval-every 50` |
| DDP | 2 卡，`--nproc_per_node=2`，端口 29583 |

### 注意

- **不要加 `--strict-resume`**（p14 的坑：fingerprint=None 会 fail-closed）；
- trainer 只认 `checkpoint_lora_last.pt`，脚本已处理；
- 不想续优化器、只想从 best 150 再训 150 步：去掉 `--resume`/`--peak-budget-json`，改
  `--init-lora p21/ft300/checkpoint_lora_best.pt --updates 150 --lr-total-updates 150 --warmup 10`；
- 只有 1 张卡：`--nproc_per_node=1`、`--ddp 0`、`--batch 8`。

---

## 3. 跑完看什么

| 文件 | 看什么 |
|---|---|
| `p21/ft300_resume/val.jsonl` | 200/250/300 的 `dJ` 是否还涨 |
| `p21/ft300_resume/summary.json` | `best.val_dJ` / `best.it` |
| `p21/ft300_resume/checkpoint_lora_best.json` | best update / dJ |
| `p21/ft300_resume/cond_check.jsonl` | `dtok_cos_across_noise` 是否恶化 |

**决策规则（先写死）**：

- `dJ > 0.0025` 且 `cond_check` 的 cos 不明显下降 → 值得花 ~1 h 跑 **8–16 集**闭环（同 §4 的协议）；
- `dJ` 平/噪声大 → **停**，把资源投到 token→动作 或 DAgger；
- 预期：按 p19/p20/p21 的窄带，补到 300 大概率只到 **0.0024–0.0026**，闭环仍在噪声内。

---

## 4. 闭环中间快照（截至 2026-11-11 14:40，**未跑完**）

运行中：`c2/p21_only_0_6/`（p21 only；card0 = OpenDoor+SitSofa，card1 = Boxing+VisionNavi；seeds 0–6；strict 单次、禁止 best-of）。

| 任务 | seeds | p21 | C | base |
|---|---|---|---|---|
| OpenDoor | 0–6 | **4/7** | 2/7 | 4/7 |
| Boxing | 0–6 | **5/7** | 4/7 | 6/7 |
| SitSofa | 0 | **1/1** | 0/1 | 0/1 |
| VisionNavi | 0–1 | **1/2** | 1/2 | 0/2 |
| **合计** | 17 | **11/17（65%）** | **7/17（41%）** | **10/17（59%）** |

配对：**p21 vs C = +5 / −1**；**p21 vs base = +4 / −3**（p 都还不显著）。

> ⚠ **C/base 两列不是同协议重跑**：OpenDoor/SitSofa 取自 `ab_7seed`，Boxing/VisionNavi 取自 `ab2way_0_7`（早期批次，可能早于 `/reseed`）。严格三臂对比需要 C/base 用同一套新协议各跑一遍。
> ⚠ 剩余 SitSofa s1–s6、VisionNavi s2–s6 跑完后，用 `c2/p21_only_0_6/{card0,card1}` 的日志重新出表。

---

## 5. 已知边界 / 坑

1. **refresh#2 的方向**：`dir_agree` 77%（base6 98%、refresh#1 88.5%）——裁判在“方向”上不如老 WM；这是 p21 改动小、闭环提升有限的根因；
2. **token→动作通道**：3% token 只动 **0.028%** 动作；micro-eps 探针 3% token → `dJ 0.008%`；续训不修这个；
3. **val 越限是假指标**：p20 训练窗口 5.8% vs 真机闭环 ticks **0.00%**；不要用 `limit_violation_frac` 判死；
4. **Vulkan ICD**：2026-11-11 发现 `/etc/vulkan/icd.d/nvidia_icd.json` 是 **0 字节**，导致 Isaac 报 `ERROR_INCOMPATIBLE_DRIVER` → PhysX 回退软件 → 无 infer。已用 `/usr/share/vulkan/icd.d/nvidia_icd.json` 覆盖修复；新容器/重启后如再遇到，先查这个文件；
5. **gpu_utilizer**：评测期间已停（`pkill -f gpu_utilize[r].py`）；它不是根因，但会和 Isaac/训练抢资源，评测结束需要再起时自行恢复；
6. **p21 vs p20**：p21 用 refresh#2，p20 用 refresh#1；p20 的 13/16 vs 14/16 是 seeds 20–27 上的“打平”，不能拿来和 seeds 0–6 的 11/17 直接比。

---

## 6. 产物路径

| 类别 | 路径 |
|---|---|
| 续训脚本 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/run_p21_resume.sh` |
| p21 原 run | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/p21/ft300/` |
| 续训输出 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/p21/ft300_resume/` |
| 闭环评测 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/c2/p21_only_0_6/` |
| 失败现场（Vulkan 前） | `.../c2/p21_vs_c/_failed_physx_20261111_1252/` |
| dir_agree 三 WM | `comimic-c-wm6/docs/DIR_AGREE_3WM_20261111.md` + `.../p19/diag/dir_agree_3wm/` |
| refresh#2 WM | `.../p19/refresh2/wm_cl_candidate.pt`，sha **19163963** |
| C 基线 | `.../critic10/lora300_ddp/checkpoint_lora_best.pt`，sha **6942eab9** |
