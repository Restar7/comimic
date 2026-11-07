# 主线集成报告（2026-11-07）

> 目标工作区：`/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic`（分支 `main`）
> 本文记录：合并了什么、冲突怎么解、验证了什么、**没验证什么**。

---

## 1. 最终状态

| 项 | 值 |
|---|---|
| 工作区 | `/inspire/.../wsc-workspace/comimic` |
| 分支 | `main` |
| 集成后 HEAD | `63d6e0a`（本文档提交后 → 见 `git log -1`） |
| Gate A | **PASS — 94 passed**（`runs/harness_04_final/GATE_A_TESTS.log`） |
| Gate B（真机 token 空间配对） | **NOT_RUN** |
| Gate C（SitSofa/VisionNavi 闭环） | **NOT_RUN** |

集成链路（`main` 上的提交顺序）：

```
dcdad61  wip(distill)  ← 本会话今晚的工作先落库（避免与 01/02 冲突时丢东西）
b74d0bf  = cherry-pick 9165043  会话01：Δtoken 加载/特征契约统一（35/64 + 关节序）
062e363  = cherry-pick 1fa87b0  会话02：Teacher 改为 shared-Δ on full chunk + 梯度探针
bc861b7  = cherry-pick 9c58c0d  会话02：DDP 指标全局聚合 + 结论文档
788d1ee  harness04：纳入会话03 诊断脚本 + schema v2 加载器接口测试
6526a99  harness04：标签采集口径对齐部署（VLA chunk / 逐帧 / discounted / 930 维 obs）
63d6e0a  harness04：折扣回报与主训练逐字对齐（补 steps=reward_horizon）+ 省原始历史开关
```

---

## 2. 冲突与解决（两处代码 + 一处文档）

| 文件 | 冲突性质 | 解决方式 |
|---|---|---|
| `scripts/ha_vla_latent64_server.py` | 我在会话 01 之前写过"按 state dict 反推 feat_dim + 按宽度二选一算特征"的**临时补丁**；会话 01 提供了 `diff_vla_wbc/delta_token_io.py` 的**唯一实现** | **取会话 01 的实现**，删掉我的临时补丁（加载走 `load_delta_token_from_checkpoint`，特征走 `delta_features_from_state64`，并把 `state_joint_order="ha_v31"` 传进去） |
| `scripts/probe_ft_landing.py` | 同上（同样的临时补丁） | 同上，改用 `delta_token_io` |
| `docs/HARNESS02_TEACHER_CONCLUSIONS_20261107.md` | 两个会话各写了一段"提交号/分支"的说明 | 保留**内容更全**的一侧（HEAD 版），删掉另一侧的重复行 |

**没有用** `--ours` / `--theirs` 整文件覆盖；所有冲突都是按语义手工解的。

---

## 3. 接口冲突（会话 01 加载器 × 会话 02 metadata）

会话 01 的加载器要"从 ckpt 读出输入契约"，会话 02 训练侧新增了 **schema v2 metadata**。
两条线各自都对，但**字段名/语义**没有测试保证——这正是历史上"训 A 部署 B"的来源。

处理：
1. 新增 `tests/test_delta_token_schema_v2.py`（5 项），把"训练侧真正写出的 v2 meta"当作输入，断言：
   - `feat_dim` / `feat_joint_order` 无歧义读出；
   - `feat_dim` 与首层宽度矛盾时 **fail-closed**（不是猜一个继续跑）；
   - 部署 `state64`（ha_v31）→ 特征（standard）的置换与 `joint_contract` 一致；
   - 无 v2 声明的 legacy ckpt 仍可按形状推断，并标记 schema=1。
2. `scripts/train_distill_offline.py` 改为**写 schema v2**（与 `V3Trainer.delta_token_meta()` 同键）。
3. 真实产物核对：`runs/distill5k_eta15.pt` 由 `legacy_inferred` → **`status = ok`**（`scripts/harness02_check_delta_ckpt.py`）。

---

## 4. 会话 03（BPTT 诊断）的纳入

会话 03 的产物在 `comimic-wt03` 里是**未提交文件**，本次按"只保留工具、不改生产默认"纳入：

| 文件 | 作用 |
|---|---|
| `scripts/diag_bptt_learnability.py` | 固定窗口 BPTT 可学性诊断（MLP vs Constant） |
| `scripts/harness03_env.sh` | 诊断用环境变量 |
| `scripts/harness03_report.py` / `harness03_token_stats.py` | 报告与 token 统计 |

**未**把任何诊断参数写进生产训练默认；**未**修改 `v3_step.py` 的生产路径来适配诊断。

---

## 5. Gate A 明细（已跑，PASS）

```
94 passed in 25.27s
tests/test_delta_token_io.py            # 35/64/v2 加载、前向对拍、cap 一致性
tests/test_deploy_audit.py              # provider 门禁（非 SONIC 必须拒绝）
tests/test_sonic_decoder.py             # ONNX ↔ Torch 对拍
tests/test_teacher_shared_delta.py      # Teacher 的 shared-Δ 求导一致性
tests/test_delta_token_schema_v2.py     # 训练侧 meta ↔ 部署侧加载（本次新增）
tests/test_deploy_obs_alignment.py      # 930 维部署 obs 布局逐位对齐（本次新增）
tests/test_joint_contract.py            # HA_V31 ↔ STANDARD
tests/test_quant.py
```

日志：`runs/harness_04_final/GATE_A_TESTS.log`；机器可读：`runs/harness_04_final/INTEGRATION_METRICS.json`

**注意**：在独立 worktree `comimic-wt04` 里跑同一套测试时有 3 项失败，原因是 worktree 没有
`third_party/HumanoidArena`（被 `.gitignore` 排除）。在**主工作区**跑全绿——这也是"最终验证必须在
ROOT 做"的原因。

---

## 6. Gate B / Gate C：NOT_RUN（明确不声称）

| Gate | 内容 | 状态 | 原因 |
|---|---|---|---|
| B | 真机 token 空间配对（Base vs Constant vs BPTT vs 蒸馏） | **NOT_RUN** | 需要 Isaac + 会话 03 的配对脚本按 task 配置；本次时间用在了口径修正与集成上 |
| C | SitSofa / VisionNavi 闭环（视频 + 逐步 dump + 服务端独立生命周期） | **NOT_RUN** | 同上；且上一轮闭环（`runs/ab_d15k`）**无视频、无逐步 dump、1 个 timeout + 1 个 `Connection refused`**，不构成证据 |

**不得**把 Gate B/C 的缺失读成"Δ 有效"或"Δ 无效"。

---

## 7. 仍未解决 / 需后续处理

1. **闭环 dump 的关节口径**：`runs/closedloop_r3/*/dump/steps.jsonl` 的 `q` 与运动库 ref 一步误差
   **0.415 rad**（加 `ha_v31→standard` 置换更差）。⇒ 任何用旧闭环录制数据做的漂移/误差结论**一律作废**。
2. **运行时参考库来源未确认**：`isaac_replay_online.sh` 的 `LIB` 默认 `ha_cmd/packed.pt`，
   但用 `frame` 索引两种库都 0% 命中（最近邻 ~1 rad）。
3. **root p/v 的 10 帧历史不在数据管线里**（npz 只有 `joint_pos/joint_vel/body_quat/state46`），
   所以"Δ 输入包含 root 位置/速度历史"这条在现有管线下做不到——不是漏存。
4. **工作树里有两个文件属于别的会话**（`docs/HARNESS03_BPTT_VERDICT_20261107.md`、
   `docs/BPTT_SCHEMES_COMPARISON_20261107.md`），本会话**未提交、未修改**。

---

## 8. 复现

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
R=$PWD; source scripts/deps_env.sh
export PYTHONPATH="$R:$R/trackerlab-supermimic/source/trackerLab:$R/trackerlab-supermimic/source/trackerTask"
python3 -m pytest -q tests/test_delta_token_io.py tests/test_deploy_audit.py \
    tests/test_sonic_decoder.py tests/test_teacher_shared_delta.py \
    tests/test_delta_token_schema_v2.py tests/test_deploy_obs_alignment.py \
    tests/test_joint_contract.py tests/test_quant.py
git log --oneline -8
git status --short
```
