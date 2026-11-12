# WM6 base6 监督报告：36/36 集 → WM1 候选 → 门禁 **PASS**

> 时间：2026-11-08 18:30（接手 16:41）｜执行：接手 Agent（容器 A）｜依据：`docs/HANDOVER_20261108_MASTER_A_WM6.md`
> 复现命令、sha、口径差异全部写在下面；所有数字都可用 `runs/wm_base6_v1/` 下的 json 复核。

---

## 0. 一句话结论

**`wm_base6_best`（WM0 + 60 min 长训的 s350 检查点）在主门禁上 6/6 全过**：
新分布 base val root e20 **8.26 → 4.27 cm（+48.3%）**；旧 A′ holdout **退化 +2.72%**（阈值 5%）；
`dir_agree` **0.979 ≥ 0.927**（首次真跑出该判据）。**未发布**：等人工放行（见 §7）。

---

## 1. 数据（P3）

| 项 | 结果 |
|---|---|
| 集数 | **36/36**（open_door/sit_sofa/vision_navi 由 A 采；pp_box/double_desk 由 B 采；**boxing 由 A 接手**） |
| tick 总数 | 75,216（dt=0.02 ⇒ 1504 s 有效仿真时间） |
| 逐集独立复核 | 18/18 A + 18/18 B 全过（15 字段、frame 连续、state 链 0.0、四元数、动作公式、done 回填、速度自洽） |
| 成功/超时（标注后） | open_door 4/2、sit_sofa 4/2、vision_navi 4/2、pp_box 2/4、double_desk 0/6、boxing 0/6（boxing 6 集全为 3000 步超时） |
| merge | train **2859 段/24 集**、val **426/6**、test **454/6**（每任务 4/1/1，窗口不跨集） |
| merged↔Raw 全量核对 | **3739/3739 段 0 坏段**（states/actions/next_states/state_final 逐位一致） |

A/B 两容器合同元数据逐项一致（contract、两个置换表、dq_ha/sc_ha、git_commit `38a2bf0`、decoder sha `c7241a12`、`policy_id=base_ft8000`）。

## 2. 训练（P4）

- 一次长训 600 步，`lr=1e-5`、`80/20`(A′/base)、`k=20`、`batch=256`、`window_batch=32`、每 50 步落检查点；`cuda:1`。
- **速度修复（关键）**：`MixedReplay.sample_windows` 每步重算 A′ 窗口池，实测 **13.1 s/次**（`contiguous_runs` 对 44 万行的 Python 循环）；
  A′ 真实窗口池只有 **567 个**（1.3 窗/clip）。加 `WM6_WINDOW_CACHE=1` 池化后 **0.3 ms/次**，
  单步 **10.6 s → 0.52 s（20×）**，600 步 ≈ 5.5 min。默认关闭，其他线零影响。
- 训练侧 val（内部 ssm）：single 0.962→0.39，kstep 11.2→5.55；it=180/210 各有一次坏 batch（loss 57 / 830，被 grad_clip 救回）。

## 3. 检查点选优（口径事故与修正）

| 口径 | 窗口来源 | 池 | WM0 | s350 | 结论 |
|---|---|---|---|---|---|
| 官方门禁 | `contiguous_runs` on holdout clips | 567 | 4.60 cm | 4.73 cm | **+2.72% 退化** ✅ |
| v1 选优器（**已弃用**） | lag=16 全量行（含训练 clip） | 398k | 4.67 cm | 6.24 cm | +33.6%，**误导** |
| v2 稠密 holdout | lag=16 × holdout clips | 40 万候选 | 4.67 cm | 7.10 cm | **+26.4%**（见 §6 警告） |

`select_wm6_checkpoint_v2.py` 同时报三列；选优约束用**官方 holdout 口径**（与门禁同规则），稠密列只作报告。
12 个检查点里旧官方口径全程 +1.6%~+5.9%（≈持平），新分布 s300/s350/s600 最好（4.27~4.60 cm）⇒ 取 **s350**。

## 4. 门禁（P5）——**PASS 6/6**

`gates/wm_base6_best_eval_with_probe/gate.tson`（`--gradient-probe-result` 口径）：

| 判据 | 值 | 阈值 | |
|---|---|---|---|
| finite / action_contract / no_leakage | ok | True | ✅ |
| new_e20_rel_improve | **+38.59%**（ci_lo +7.36%） | ≥5% 且 ci_lo>0 | ✅ |
| old_e20_rel_degrade | **+2.72%** | ≤5% | ✅ |
| dir_agree_new ≥ dir_agree_old | **0.9792 ≥ 0.9271** | ≥ | ✅ |

逐任务 val（6 集，e20 8.26→4.27 cm，全部单集）：vision_navi +60.0%、double_desk +48.9%、
sit_sofa +46.6%、boxing +45.6%、pp_box +39.8%、open_door +38.2%。

补充跑（`--val-fraction 0.9`，5 集）：new **+43.7%**（ci [+0.322,+0.487]）。
注意：该跑的旧 A′ 数字（+32%「改善」）**不可用**——val_fraction=0.9 时旧 val 会与训练 clip 重叠。

## 5. dir_agree 首次真跑（P5-3）

- 方法：A′ holdout（94/533 clips）× 96 窗口 × H=2 × δ=0.03 rad RMS。
  WM 侧用 **A′ 原样合同**（state STANDARD + action HA 归一化）反传得 `g = ∂J_WM/∂a`；
  精确换算成 Isaac 要的（state HA / action STANDARD 归一化）后跑真物理：
  `isaac_probe_action_rollout.py`（reset 写 root pose+vel+joint，`q_target=default_q+a*scale`，4×5 ms 子步）。
- 结果：**dir_agree_old = 0.927，dir_agree_new = 0.979**（去掉 |ΔJ_true| 过小的段后均为 0.989）。
- 幅度：ΔJ_true 均 +7.8e-4 vs ΔJ_WM 均 +1.1e-2 ⇒ **WM 把收益放大 ~14×**（方向对、幅度不可当真）。
- 已知限制：① 只能用 A′ 窗口（base6 tick 没有参考帧 `frames`，`bundle.reward` 无法求值）；
  ② `prev_action` 用窗口首动作近似；③ 只跑了 H=2、96 窗。
- 产物：`dir_agree/job_{old,new}.pt`、`roll_{old,new}.pt`、`gates/gradient_probe.json`。

## 6. 诚实清单（本次新发现）

1. **旧动力学其实退化了不少**：稠密 holdout 口径 +26.4%。官方口径只 +2.72% 是因为它的窗口池只有 567 个、
   偏慢速段（`contiguous_runs` 把 A′ 的 (step-major, env-minor) 布局切碎，平均段长 4.3 行）。
   ⇒ 若 WM1 要给 C 线当"可微裁判"，快速/敏捷动作段仍是弱项；**建议**：加 env-aware(lag=16) 窗口构造器 + 锚定/降 lr 再训一轮（现在 6 分钟一轮）。
2. 采样池化**不改变分布**（三档 max_windows 都得 567 窗），只是把重复劳动去掉；随机流不同。
3. 训练坏 batch（loss 57/830）来自新数据极端姿态；已由 grad_clip 拦住，但值得做一次异常段过滤。
4. 门禁 `--skip-gradient` 期间的两处数字（旧 2.72%、补充跑 32%）不能混用，见 §3/§4 口径说明。
5. 事故：18:03 曾在 bash 运行时原地改 `supervise_wm_base6_v3.sh` 导致脚本语法错退出（数据无损）；已改为"新文件+重启"。

## 7. 未做/待放行

| 项 | 状态 |
|---|---|
| 发布（`wm_candidates/active.json` → `wm_base6_best`） | **未做**，等指令 |
| 提升到 `runs/three_track_v7/wm_versions/`（C 线真正用的 active） | **未做**，需 P5-4/P5-5 后决定 |
| P5-4 C-V8 24 窗口困难案例 | 未跑（需 C 线 dump） |
| P5-5 真闭环 H20 | 未跑（需 Isaac 闭环 + `bptt_args` 的 `term_off/d_max` 修正） |
| base6 上的 dir_agree | 结构性不可算（tick 无参考帧）；需要任务奖励或补采参考 |

## 8. 复现

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03
# 数据审计/合并/逐位核对
bash scripts/run_wm_base6_pipeline.sh merge
python3 scripts/verify_merged_wm6.py --all
# 训练 600 步（池化+wb32）
WM6_WINDOW_CACHE=1 python3 scripts/train_wm_refresh_v7.py --version-store runs/wm_base6_v1/wm_candidates \
  --new-version wm_base6_001 --old-replay runs/a_prime_sonic_exc/all_transitions.pt \
  --new-replay runs/wm_base6_v1/merged/train --old-ratio 0.8 --new-ratio 0.2 --mode both --k 20 \
  --updates 600 --lr 1e-5 --batch 256 --window-batch 32 --eval-every 25 --save-every 50 \
  --save-dir runs/wm_base6_v1/train_ckpts/wm_base6_001 --device cuda:1 \
  --wbc-ckpt ../comimic/runs/S1_handover/WBC_A24000_wmvel_cmd.pt --packed-motion ../comimic/runs/isaac_motion/ha_cmd/packed.pt
# 选优（双口径）
python3 scripts/select_wm6_checkpoint_v2.py --ckpts runs/wm_base6_v1/train_ckpts/wm_base6_001 --device cuda:1
# dir_agree
python3 scripts/probe_wm6_dir_agree.py plan --wm runs/wm_base6_v1/wm_candidates/<ver>/wm.pt --job-out /tmp/job.pt
$ISAAC_VENV/bin/python -u scripts/isaac_probe_action_rollout.py --job /tmp/job.pt --out /tmp/roll.pt --device cuda:0 --headless
python3 scripts/probe_wm6_dir_agree.py score --job /tmp/job.pt --roll /tmp/roll.pt
# 门禁
python3 scripts/eval_wm_refresh_v7.py --version-store runs/wm_base6_v1/wm_candidates \
  --wm-old init_from_wm0 --wm-candidate wm_base6_best_candidate \
  --old-replay runs/a_prime_sonic_exc/all_transitions.pt --onpolicy-data runs/wm_base6_v1/merged/val \
  --old-ratio 0.8 --new-ratio 0.2 --horizons 1,5,10,20 --primary-h 20 --windows 128 --device cuda:1 \
  --gradient-probe-result runs/wm_base6_v1/supervision/gradient_probe.json \
  --out runs/wm_base6_v1/gates/final
```
