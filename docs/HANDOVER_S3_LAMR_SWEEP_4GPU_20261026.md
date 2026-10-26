# 交接：S3 联合微调 λ_r 消融（λ_r = 0.1 / 1.0）在 4×4090 上跑

> 发起方：本机（2×4090）已跑掉 λ_r = 0 / 0.3 两格；**剩下 0.1 / 1.0 挪到你们的 4×4090 上跑**。
> 本文是自包含的：照着抄命令即可，不需要回头读别的文档。
> 代码版本：`git log --oneline -1` 应该是 `6e60eec` 或更新（S3 全部提交都在）。
> 产物目录约定：`runs/<RUN>/`，回传 `eval_curve.jsonl` + `summary.json` + `train_metrics.jsonl`。

---

## 1. 目标 & 产出

跑**同一个配置、只差 λ_r** 的两格（1000 步），并与本机的 λ_r=0 对照臂比较：

| 臂 | λ_r | 跑在哪 |
|---|---|---|
| `s3c6h_lam0` | 0（BC 对照，已在本机跑） | 本机 GPU0 |
| `s3c6h_lam0.3` | 0.3（已在本机跑） | 本机 GPU1 |
| **`s3c6h_lam0.1`** | **0.1** | **你们 4 卡机** |
| **`s3c6h_lam1.0`** | **1.0** | **你们 4 卡机** |

判读只看四个数：`L_fm(val)`、decode 后 `token_mse_decoded`/`hand_mse_decoded`、`hand_gt170_pred` vs 标签、
`imag_return_vla`（burn-in 起点，64 窗口）。**判据见 §8**。

---

## 2. 前提核对（先跑这三条，全过再开训）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic   # 若路径不同，见 §3
git log --oneline -1                       # 期望 6e60eec 或更新
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader   # 4 张 4090（≥24GB/卡）
```

必需文件（缺一个就别开）：

| 用途 | 路径 | 校验 |
|---|---|---|
| **执行器（C6h）** | `runs/S1_handover/WBC_A20000_wmvel_cmd.pt` | `md5sum` = **e0596acb98957d11931a7854bbe043d3**，52 MB |
| VLA 起点 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000/` | 目录，34 GB（含 processor/statistics） |
| 参考库（命令流） | `runs/isaac_motion/ha_cmd/packed.pt` | 86 MB，`meta.json` 里 `num_clips=699` |
| BC 标签 | `data/npz/ha_tokens/*.npz` | 699 个 npz + `convert_summary.json` |
| 视频/状态（VLA 输入） | 上面 npz 自带的 `g1_ego_view` mp4 路径 | 由 npz 内部字段指向 |
| SONIC encoder（burn-in 环内 token） | 自动解析；找不到就设 `SONIC_ENCODER_ONNX` | `third_party/SuperMimic-Reproduced/data/assets/model_encoder.onnx`（50 MB） |

C6h 的关键元数据（我的硬门禁会据此断言）：
```
iteration=20000  obs_mode=paper_obs93  wm_integration=explicit_dt0.02
provenance.token_window="hold"   robot_spec=True
provenance.library=.../bones_seed_plus_ha_cmd_x20.pt
```

---

## 3. 如果路径不一样

本仓库在共享 GPFS 上（`/inspire/hdd/.../comimic`）。**如果你们那台挂的是同一个路径，什么都不用拷**。
不是的话，按 §2 表格 rsync 这四样（+ 代码）即可，注意：
- **不要**拷 `runs/isaac_motion/ha_tokens/`（旧执行轨迹库，S3 不用）、`trackerlab-supermimic/logs/`（WBC 训练产物，几十 GB）；
- 脚本里 `COMIMIC=` 是**写死**的绝对路径：`scripts/launch_joint_ha_v2.sh:19`。路径不同就改这一行（或用 §5.2 的直接 CLI 版本，CLI 全部用相对/参数传入）。

```bash
# 从发起方机器（如果确实需要拷）
rsync -av --relative \
  ./runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
  ./runs/isaac_motion/ha_cmd/ \
  ./data/npz/ha_tokens/ \
  <目标机>:<目标repo>/
# VLA 起点 34 GB，建议直接从你们已有的 checkpoint-8000 复制/软链
```

---

## 4. 环境变量（三个都必须）

```bash
cd <repo>
export PYTHONPATH="$PWD:$PWD/Isaac-GR00T:$PWD/third_party/gr00t_deps:$PWD/third_party/gr00t_deps/_noflash${PYTHONPATH:+:$PYTHONPATH}"
export NO_ALBUMENTATIONS_UPDATE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export SUPERMIMIC_HA_ALIGN=1
export SUPERMIMIC_ROBOT_USD="$PWD/trackerlab-supermimic/data/assetslib/unitree/unitree_g1/usd/g1_29dof_rev_1_0.usd"
```
> `_noflash` 是"假 flash_attn"，让 transformers 回落 sdpa；`gr00t_deps` 提供 tyro 等依赖。
> **不需要 Isaac Sim / IsaacLab**：S3 只加载 WBC+WM 权重（纯 PyTorch），不需要仿真器。

---

## 5. 开训命令

### 5.1 用启动器（推荐）

```bash
# λ_r = 0.1
WBC_CKPT=$PWD/runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
VLM_CKPT=$PWD/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
RUN=s3c6h_lam0.1 GPU=0 STEPS=1000 BATCH=4 ACCUM=4 H=16 K=4 \
LAMBDA_R=0.1 EVAL_EVERY=200 EVAL_BATCHES=16 SAVE_EVERY=500 LOG_EVERY=20 \
  bash scripts/launch_joint_ha_v2.sh

# λ_r = 1.0（另一张卡并行）
WBC_CKPT=$PWD/runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
VLM_CKPT=$PWD/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
RUN=s3c6h_lam1.0 GPU=1 STEPS=1000 BATCH=4 ACCUM=4 H=16 K=4 \
LAMBDA_R=1.0 EVAL_EVERY=200 EVAL_BATCHES=16 SAVE_EVERY=500 LOG_EVERY=20 \
  bash scripts/launch_joint_ha_v2.sh
```

建议用 `setsid nohup ... &` 挂后台；两格可以同时起（各自一张卡，**不要**用 `torchrun`/多卡 DDP —— v2 trainer 是**单卡**实现，等效 batch 由 `--grad-accum` 给）。

### 5.2 不用启动器时的等价 CLI

```bash
python3 scripts/train_joint_v2.py \
  --wbc-ckpt runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
  --vlm-ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --motion data/npz/ha_tokens \
  --motion-packed runs/isaac_motion/ha_cmd/packed.pt \
  --device cuda:0 --out runs/s3c6h_lam0.1 \
  --steps 1000 --batch-size 4 --grad-accum 4 --horizon 16 --fm-steps 4 --burnin-steps 10 \
  --lambda-r 0.1 --lambda-bc 1.0 --fm-lr 1e-5 --fm-grad-backend pathwise \
  --eval-every 200 --eval-batches 16 --eval-out runs/s3c6h_lam0.1/eval_curve.jsonl \
  --save-every 500 --save-vla-every 500 --log-every 20
```
（`λ_r=1.0` 那格只改 `--lambda-r` / `--out` / `--device`。）

### 5.3 4 卡怎么排（可选）

只有 4 格消融，2 格在跑，剩下 2 卡**空着也没问题**。想加对照就补（每格约 20–50 min）：
```bash
# 跨机对照（强烈建议至少补一格 λ_r=0，消除"机器/驱动差异"的嫌疑）
RUN=s3c6h_lam0_4gpu GPU=2 LAMBDA_R=0   ... 同上
# 复跑本机的 0.3，确认两边一致
RUN=s3c6h_lam0.3_4gpu GPU=3 LAMBDA_R=0.3 ... 同上
```

---

## 6. 启动后 30 秒自检（三条 audit 行必须长这样）

```bash
grep -E "AUDIT\]\[token-frontend\] wbc_ckpt|AUDIT\]\[token-space\]|AUDIT\]\[v2\]" logs/s3c6h_lam0.1.log | head -3
```
期望：
```
[AUDIT][token-frontend] wbc_ckpt_window=hold wbc_lib=.../bones_seed_plus_ha_cmd_x20.pt requested_window=hold lib_kind=command source=online bc_token_space=label
[AUDIT][token-space] bc=label wbc=label mismatch=False port_dir=.../data/port_tokens/ha_tokens
[AUDIT][v2] trainable=head_only params=1,620,515,968 wbc_trainable=0 wm_trainable=0 H=16 K=4 lam_bc=1.0 lam_r=0.1 ... grad_accum=4 backend=pathwise head_eval=True
```
另外会出现（**预期，不是错误**）：
```
[data][warn] .../data/npz/ha_tokens: 699/699 个 clip 名与库不一致（例：HOI_double_desk__tw__ep1776268633456387），按**同序号**回退（长度已核对一致）
```
（`ha_cmd` 库的 clip 名是 raw ep 时间戳、npz 是顺序号；两库 clip 数与每段长度逐位相同，所以按序号回退是安全的。）

可选：先花 5 分钟跑五道门禁，确认这套环境能复现我的数字
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_joint_v2.py --gate-only --batch-size 4 --gate-batches 8 \
  --horizon 16 --fm-steps 4 --lambda-r 0.3 \
  --wbc-ckpt runs/S1_handover/WBC_A20000_wmvel_cmd.pt \
  --vlm-ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 \
  --out runs/s3_gates_wmvelcmd_4gpu
```
期望（我这边跑出来的）：codec 1.19e-07 PASS / freeze PASS（`pi` 逐张量相等、wbc/wm 0 梯度）/
burn-in `0.09128→0.08349` survival 0.9999 / **contrast +0.117（9.4%），t=7.82，win 91%** /
fm-bc `grad_ratio=0.149`。数值可有 ±5% 机器差异，但**正负号和 t 值量级必须一致**。

---

## 7. 产出与回传

每格产出（`runs/<RUN>/`）：

| 文件 | 内容 | 回传 |
|---|---|---|
| `eval_curve.jsonl` | 每 200 步一行（64 窗口）：`L_fm_val / token_mse_decoded / hand_mse_decoded / hand_gt170_pred / imag_return_vla / imag_return_gt / imag_survival_*` | ✅ 必须 |
| `summary.json` | 终点摘要（`freeze_gate`、`fm_steps`、`token_space_mismatch`、`final_eval`、`wall_s`） | ✅ 必须 |
| `train_metrics.jsonl` | 每步训练指标（含 `grad_ratio`，每 50 步出现） | ✅ 必须 |
| `args.json` / `v2_config.json` | 命令行指纹 / 配置快照 | ✅ 建议 |
| `vla_000500.pt` / `vla_001000.pt` / `vla_final.pt` | 各 6.5 GB（后续闭环要用） | 先别传，报路径 |
| `joint_000500.pt` / `joint_final.pt` | 各 54 MB（冻结门禁已过） | 可选 |

磁盘：每格 ≈ **14 GB**（3 个 `vla_*.pt` + 日志/评测）。两格 28 GB。
耗时：λ_r=0.1 ≈ 22 min；λ_r=1.0 ≈ 50–60 min（同机并行时各自不变，只会互相抢一点 CPU）。

---

## 8. 判读规则（三格一起看）

以**同一 64 窗口**的读数为准（`eval_curve.jsonl` 第 200/400/600/800/1000 行），与本机 λ_r=0 / 0.3 并排：

1. **不退化（硬条件）**：`token_mse_decoded` 与 λ_r=0 持平（±20% 内），`hand_mse_decoded` 持平，
   `hand_gt170_pred` 与标签（0.1138）差距不扩大（λ_r=0 约 0.096–0.102）。**手塌到 0 或 token 偏移涨 3× 以上就是失败**。
2. **有增益（软条件）**：`imag_return_vla` 比 λ_r=0 高、且逼近 `imag_return_gt`；
   `imag_survival_vla` 不降。
3. **训练稳定性**：`grad_ratio` 落在 **0.1–0.5**（λ_r=0.1/0.3/1.0 大致对应 0.05/0.15/0.5）。越界就按比例调 λ_r，
   不要动 lr/H/K。
4. `L_fm(val)` 只作"在学习、没发散"的诊断；**它不是门禁**（FM 损失对采样分布偏移不敏感）。

**最终判据仍是闭环**（`scripts/mini_closedloop_v2.sh`，需要 Isaac 容器）：F 臂相对 B 臂成功率不降 + 至少一项显著改善。
这一轮消融只用来**定主线要不要奖励项、λ_r 取多少**。

---

## 9. 常见异常

| 现象 | 原因 | 处置 |
|---|---|---|
| `[fail] WBC ckpt 的 token 窗口是 'future'…` | 拿错 ckpt（用了 port/future 时代的） | 换回 C6h；只想冒烟才加 `--allow-window-mismatch` |
| `[fail] 参考库 … 不是命令流库` | `--motion-packed` 指成 `ha_tokens` | 用 `runs/isaac_motion/ha_cmd/packed.pt` |
| `FileNotFoundError: …/data/npz/ha_tokens/<ep timestamp>.npz` | npz 目录与库名不匹配且文件数不同 | 确认 699 个 npz；`resolve_clip_files` 只在文件数相等时按序号回退 |
| CUDA OOM（<24 GB 卡） | 1.62B head + AdamW | `--batch-size 2 --grad-accum 8`，或 `--fm-grad-backend checkpointed`（显存 O(1)，慢 ~30%） |
| `codec/out_of_range` 长期 >0.3 | 采样跑到分布外 | 正常范围 2–11%；变大说明训练发散，回退上一个 ckpt |
| `loss/fm_bc` 变 NaN | 数值炸 | 降到 `--fm-lr 5e-6`，加 `--fm-clip 0.3` |
| `ModuleNotFoundError: tyro / isaaclab` | PYTHONPATH 没设 / 不需要 isaaclab | §4 的 PYTHONPATH；**不要**加载 Isaac |

---

## 10. 禁止事项

1. ❌ 不要 `git checkout` / `git clean` / `git stash`：这个仓库现在有别的 agent 在工作（S1/S2），
   只允许**新增** `runs/...` 和新增文件。
2. ❌ 不要改 token 口径：`--bc-token-space label`、`--wbc-token-space label`、`--motion-packed ha_cmd`、
   `--wbc-token-window hold`、`--fm-steps 4`、`--horizon 16`、`--burnin-steps 10` 全部保持默认。
3. ❌ 不要用 `--decode none`（那是旧 ckpt 的复现口径）；不要用 `data/port_tokens`。
4. ❌ 不要把不同 λ_r 的曲线放在不同窗口/不同 batch 上比（评测必须 `--eval-batches 16 --batch-size 4` = 64 窗口，seed 由代码固定）。
5. ❌ 不要为了让曲线好看去调 `--lambda-bc`（它是对照的一部分，固定 1.0）。

---

## 11. 本机已出的两格（供对照）

| 臂 | step 200 | step 400 | `grad_ratio` |
|---|---|---|---|
| `s3c6h_lam0`（λ_r=0） | `L_fm 0.0272`，`tok 0.00529`，`hand 0.00333`，`gt170 0.0961`，`ret 1.224/1.234` | 进行中 | —（BC-only 无梯度比） |
| `s3c6h_lam0.3` | — | 进行中 | 0.149（门禁时测） |

（这两格的完整曲线在 `runs/s3c6h_lam0/`、`runs/s3c6h_lam0.3/`，我会在它们跑完后把最终读数补进
`docs/S3_JOINT_V2_REPORT_20261026.md`。）

---

## 12. 一句话总结给你们

> 只要把 §4 的环境变量设好、`--lambda-r` 换成 0.1 / 1.0、其余全部默认，跑完把
> `eval_curve.jsonl + summary.json + train_metrics.jsonl` 发回来即可。
> 判据：**token/手部不退化 + `imag_return_vla` 上升 + `grad_ratio` 在 0.1–0.5**；最终结论仍要闭环。
