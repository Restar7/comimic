# P1b 交接文档：去掉 action_target + 加反常量项的联合微调（2×4090 DDP）

> 生成：2026-11-06。读者：拿到 **2 张 4090** 的执行者。
> 目标：跑 **P1b**（"纯奖励 + 动作信任域"，**不用** `action_target`），判它能不能让动作修正变得**状态相关**。
> 本文自包含：配置、命令、必须存在的代码改动、判据、坑，全在里面。

---

## 0. TL;DR（六句话）

1. **这一轮改两件事**：(a) 把 `action_target` **关掉**；(b) 补一个**显式的反常量项** `--var-weight 0.1`。
   ⚠️ 只做 (a) 是不够的：`HA_AT_CENTER` 写在 `action_target` 的代码块里，AT 一关它**跟着失效**，
   于是目标函数里对"常向量"变成**零成本**（v10 就是这样学到 0.398 = 97% 常向量的）。
2. **动机**：`action_target` 的目标是 `A0 + 0.15·ĝ`，其中 `ĝ` 的**均值分量占 40%** ⇒ "照抄它"天然把策略拉成一个**常向量**。实测：有 AT 的 v13 残差/均值 **0.10**，没 AT 的 v10 是 **0.52**。
3. **但去掉 AT 会失去幅度控制**（v10 冲到 2.65×η、真机方向 35%）⇒ 所以把 `act` 惩罚从 `w=0.2` 提到 **`w=0.5`**（`eps` 保持 0.30）。
4. **从 base 起，不要从 v13 续训**：`trust_center=base` 的锚点是 `lora_disabled()` 跑出来的**原始权重**，从 v13 起的话 act 刹车会立刻把 v13 的偏置推回去。
5. **两个手调旋钮**：`act/rmse` 落在 **0.05–0.15**（>0.15 加 `ACT_WEIGHT`，<0.05 减）；`action_var/ratio` 要从 **0.001 单调上升**（不动就加 `VAR_WEIGHT`，抖动就减）。
6. **胜负手**：落地探针的 **`残差/均值` 能否从 0.10 顶到 >0.5**。顶不上去 ⇒ 这条路（动作层）到头。

---

## 1. 这一臂在验证什么（为什么值得跑）

| 臂 | `action_target` | 落地 ‖Δa‖ | **残差/均值**（越大越"看状态"） | 真机方向 |
|---|---|---:|---:|---:|
| v10 | 无 | 0.398（2.65×η，失控） | **0.52** | 35% ❌ |
| v11 | 无 | 0.225（1.5×η） | — | 52% ❌ |
| v13 | 有 | 0.132（0.88×η） | **0.10** | 42% ❌ |

- **v10 证明了"没 AT 时动作确实更状态相关"（0.52）**，但它失控了；
- **v13 证明了"AT 能把幅度钉住"，但代价是几乎变成常向量（0.10）**；
- **P1b 想同时拿到两者**：幅度用 `act` 惩罚钉住，方向让奖励自己找。

---

## 2. 确切配置

```
# 起点（关键）
--real-transitions  runs/a_prime_sonic_exc/all_transitions.pt     # 529,131 行，exc
WM_CKPT             runs/a_prime_sonic_exc/wm_gain2.pt            # ★ 别用 round5 默认的 wm_exc.pt
TERM_OFF            1                                             # ★ 关软终止头（否则 survival≈0.87，回报被砍）
START_MAX_DRIFT     0.10        # → 实际保留 79%，起点偏差 p50 4.35 / p90 8.67 cm
INIT_VLA            <空>        # ★ 从 base 起，不加载 v13

# 目标（本轮的两处实质改动）
ACTION_TARGET_WEIGHT  0.0       # ★ 关掉 action_target
HA_AT_CENTER          0         # AT 关了它整块被跳过（v3_step.py:845），设 0 只为日志
ACT_PENALTY           l2
ACT_EPS               0.30      # ★ 保持 0.30（临时推导，见 §8）
ACT_WEIGHT            0.5       # ★ 0.2 → 0.5（唯一手调旋钮）
VAR_WEIGHT            0.1       # ★ 反常量项 −0.1·log(rms(残差)/rms(均值))；0=关

# 方差/自举
CRITIC_ON             1
CRITIC_SOURCE         imagined
--boot-one-sided                # rel := min(rel, 0)

# 视界/折扣（都不动）
H                     20        # = 部署重规划窗口
gamma                 0.99      # ⇒ critic 占 γ^20 = 81.8%
REWARD_HORIZON        0         # 满 20 步
TRUST_EPS / TRUST_WEIGHT    0.05 / 0.2
lambda_r / lambda_hand      5.0 / 0.05
LoRA rank 32（targets=attn1,ff）

# 预算
BATCH 16（全局）→ 每卡 8；ACCUM 3 → 有效 48
STEPS 300，SAVE_EVERY 50，EVAL_EVERY 50
```

**2 卡 DDP 的对应写法**：

```bash
NPROC=2  DDP=1  DDP_GPUS=0,1  BATCH=16  ACCUM=3
```
（`BATCH` 是**全局**值，launcher 会除以 `NPROC` 得到每卡 8；有效 batch = 16×3 = 48。）

---

## 3. 怎么跑

```bash
cd <comimic 根目录>

# 起训练（后台）
nohup bash scripts/run_p1_round_b.sh > runs/_p1b_outer.log 2>&1 &

# 前 10 步先确认三件事（见 §5）
sleep 300 && python3 scripts/watch_p1.py --run joint_p1b --tail 3

# 全程盯
python3 scripts/watch_p1.py --run joint_p1b            # 最后 12 条 + G1–G4 判定
tail -f logs/joint_p1b.log
```

**输出位置**：
- 日志 `logs/joint_p1b.log`
- ckpt `runs/joint_p1b/vla_0050.pt` … `vla_final.pt`、`args.json`、`v3_config.json`、`eval_curve.jsonl`

**预计时间**（2×4090，每卡 8 窗、ACCUM 3）：**约 1–1.5 小时 / 300 步**。单步会打印 `s_per_step`。

---

## 3.5 起训练前必做：2 步冒烟测试（5 分钟，能省一整轮）

```bash
STEPS=2 RUN=joint_p1b_smoke SAVE_EVERY=50 EVAL_EVERY=50 \
  bash scripts/run_p1_round_b.sh > runs/_p1b_smoke.log 2>&1 &
# 等 ~4 分钟，然后
python3 -c "import json;r=json.loads(open('runs/joint_p1b_smoke/train_metrics.jsonl').readline());print({k:r.get(k) for k in ['rollout/survival','act/rmse','action_var/ratio','critic/v_ev']})"
```

**三个必须对的数**：

| 检查 | 期望 | 不对的含义 |
|---|---|---|
| `rollout/survival` | **1.0** | ≠1.0 ⇒ `--term-off` 没进训练器，**回报被砍半**，整轮白跑 |
| `act/rmse` | **≈1e-6** | 明显 >0 ⇒ LoRA 不是零初始化 / 加载了 `INIT_VLA`（P1b 要从 base 起） |
| `action_var/ratio` | **0.001** | 出现 `nan` ⇒ 反常量项的形状/除零有问题，别开正式跑 |

> 今天这套冒烟抓到 3 个真 bug：`wm_ckpt` 用了 `wm_exc.pt`（错 WM）、`--term-off` 被命令替换吞掉（survival 0.875）、`VAR_WEIGHT` 字符串替换静默没匹配上。

---

## 4. 前置条件（缺一个就跑不起来）

| # | 需要什么 | 检查 |
|---|---|---|
| 1 | **`runs/a_prime_sonic_exc/all_transitions.pt`**（50 分片合并，529,131 行） | `python3 -c "import torch;d=torch.load('runs/a_prime_sonic_exc/all_transitions.pt',map_location='cpu',weights_only=False);print(len(d['states']))"` → 529131 |
| 2 | 底模 `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | 存在 |
| 3 | WM `runs/a_prime_sonic_exc/wm_gain2.pt` | 存在 |
| 4 | **代码改动**（见 §6）已在新机器上 | `python3 scripts/train_joint_v2.py --help \| grep boot-one-sided` |

如果 `all_transitions.pt` 没法拷过去，用这段重建：

```python
import torch, glob
files = sorted(glob.glob('runs/a_prime_sonic_exc/*_off*.pt'))
keys = ['states','actions','next_states','frames','clip_ids']
acc = {k: [] for k in keys}
for f in files:
    d = torch.load(f, map_location='cpu', weights_only=False)
    for k in keys: acc[k].append(d[k])
out = {k: torch.cat(v, 0) for k, v in acc.items()}
out['meta'] = {'source': 'runs/a_prime_sonic_exc (merged)'}
torch.save(out, 'runs/a_prime_sonic_exc/all_transitions.pt')
```

---

## 5. 跑起来先看这三个数（前 100 步）

| 检查 | 期望 | 不对怎么办 |
|---|---|---|
| `loss/action_target` | **不存在或恒为 0** | 不是 0 ⇒ `ACTION_TARGET_WEIGHT` 没生效（launcher 默认是 0，但 round5 可能没传），查 `args.json` |
| `critic/v_ev` | **> 0.90** | <0.9 ⇒ 自举项是噪声，**这轮作废**（见 §7-G3） |
| `act/rmse` | **0.05–0.15** | >0.15 ⇒ `ACT_WEIGHT=0.8` 重跑；<0.05 ⇒ 0.3 |
| `critic/v_adv_H_raw` | 不恒为正（>0 占比 <80%） | 恒正 ⇒ 在"奖励偏离"，检查 `--boot-one-sided` 是否真的加了 |

---

## 6. 必须存在的代码改动（这轮依赖的 patch 清单）

> 如果新机器是**同一个仓库的拷贝**，这些都已在内；如果是另一份 checkout，逐条核对。

| 文件 | 改动 | 为什么 |
|---|---|---|
| `scripts/train_joint_v2.py` | 新增 `--boot-one-sided`（`action="store_true"`），并在 `V3Config(...)` 里加 `boot_one_sided=bool(getattr(args,"boot_one_sided",False))` | 否则自举项只能"奖励偏离" |
| `scripts/run_joint_ft_round5.sh` | `CRITIC_ON=0` → `${CRITIC_ON:-0}`；`CRITIC_SOURCE` 可覆盖；起点 gate 改成校验 `EXTRA` 里实际的 `--real-transitions` 文件 | 原来 critic **写死关着**；gate 写死 leash |
| `diff_vla_wbc/v3_step.py` | 新增 `diverge/pos_t{5,10,20}`、`diverge/joint_t{5,10,20}`（策略轨迹 vs base 轨迹） | 下一轮用实测决定 `action_target_steps` |
| `scripts/launch_joint_ha_v2.sh` | `$( [[ "$TERM_OFF" == "1" ]] && echo --term-off \|\| true )` 改成显式变量 `TERM_OFF_ARG` | 实测这个命令替换**会被吞**：`term_off=1` 打出来了，但 `--term-off` 没进训练器 ⇒ `survival=0.87`、回报被砍半 |
| `diff_vla_wbc/v3_step.py` | 新增 `V3Config.var_weight`；在 act 惩罚后计算 `loss_var = −w·log(rms(残差)/rms(均值))`；日志加 `action_var/ratio` | **反常量项**，AT 关掉后唯一惩罚"固定方向"的东西 |
| `scripts/train_joint_v2.py` | 新增 `--var-weight`（float, 默认 0），传入 `V3Config(var_weight=...)` | — |
| `scripts/launch_joint_ha_v2.sh` | `VAR_WEIGHT` → `${VAR_WEIGHT:+--var-weight "$VAR_WEIGHT"}` | — |
| `scripts/run_p1_round_b.sh` | 本轮配置脚本 | — |
| `scripts/watch_p1.py` | 指标看板 + G1–G4 判定 | — |

---

## 7. 判据（先写死，不要事后改）

### G1–G4（训练侧，`watch_p1.py` 直接给）

| 门 | 指标 | 阈值 | 备注 |
|---|---|---|---|
| G1 | `pred_resid_ratio` / `loss_at` vs `tgt_resid_norm²` | `loss_at < 0.7×tgt_res²` | 训练侧"残差可学"的代理；**AT 关了之后 G1 不适用**，以 G5 为准 |
| G2 | `act/rmse` | **0.05–0.15** | 唯一手调旋钮 |
| G2b | `action_var/ratio` | **从 0.001 单调上升**，目标 > 0.5 | 训练中的"残差/均值"；**起始必为 0.001**（LoRA 零初始化 ⇒ 两条轨迹相同 ⇒ 0/1e-3） |
| G3 | `critic/v_ev` | **> 0.90** | 低于就作废本轮 |
| G4 | `critic/v_adv_H_raw > 0` 的占比 | **< 80%** | 防止"奖励偏离" |

### G5（落地探针，**必须做，20 分钟**）

```bash
# 同状态配对：base 头 vs P1b 头，24 请求 × 3 噪声，过部署同一个 ONNX decoder
python3 scripts/probe_ft_landing.py --dump runs/closedloop_r3/base_s0 \
    --arms base= p1b=runs/joint_p1b/vla_final.pt
```

| 指标 | 阈值 | 含义 |
|---|---|---|
| `‖Δa‖` | **0.10–0.20** | 幅度没失控、也没被钉死 |
| **`残差/均值`**（自己算：`Δa` 减去跨请求均值后的 RMS ÷ ‖均值‖） | **> 0.5** | **胜负手**：v13 是 0.10，v10 是 0.52 |
| `cos(Δa, ĝ)` | **> 0.3** | ⚠️ 这个探针目前**不可信**（见 §9-坑3），只能当趋势 |

**判定**：
- 残差/均值 **>0.5** 且 ‖Δa‖ 在区间内 ⇒ **进闭环**（8 集 OpenDoor 配对，≥6/8 更好才算赢）；
- 残差/均值 **仍 ≈0.1** ⇒ **动作层这条路到头**（说明不是 AT 的锅，是"逐状态不可学"），转交付口径。

---

## 8. 为什么 `eps=0.30, w=0.5`（不是 0.15）

l2 惩罚 `loss = w·(r/eps)² = (w/eps²)·r²`，平衡点 `r* ∝ eps²/w`。**eps 和 w 是简并的**。

用 **v10 当锚点**（唯一"AT 关掉"的实测臂：`eps=0.30, w=0.2 → r*=0.170`）：

| 配置 | `eps²/w` | 预测 `act_rms` | 判定 |
|---|---:|---:|---|
| v10（锚点） | 0.450 | 0.170 | 失控 |
| **eps=0.30, w=0.5** | **0.180** | **≈0.068** | ✅ 目标区 |
| eps=0.15, w=0.5 | 0.045 | ≈0.017 | ❌ 被钉死（<0.02 是 handover 坑#5） |

---

## 9. 坑（这一轮已经踩过的，别复发）

| # | 坑 | 表现 | 防线 |
|---|---|---|---|
| 1 | **`ACTION_TARGET_WEIGHT` 默认是 0**（`launch_joint_ha_v2.sh:162`） | 想开 AT 却没开；或者想关却没确认 | 跑完先 `grep action_target_weight runs/joint_p1b/args.json`，**必须看到 0.0** |
| 2 | **`GAMMA=0.95` 是死变量** | 以为在用 0.95，其实 0.99 | round5 里那个 export 没被 launcher 用；以 `args.json` 的 `gamma` 为准 |
| 3 | **方向探针口径是坏的** | `probe_ft_direction` 单步关节误差 **0.483 rad**，反序对照 cos 0.222 > 真序 0.160 | 只看 `cos` 的趋势，**不要拿它下结论**；真正的判据是 `残差/均值` |
| 4 | **起点分布别混** | 用 leash 窗口测 WM 误差会得到 2.11cm（真值 0.82） | 一切 WM 误差/SNR 都用 **exc** 窗口 |
| 5 | **`pkill -f <自己命令行里的词>`** | 把自己杀掉（SIGTERM） | 用 `pgrep -f "pat[t]ern"` 或按 PID 杀 |
| 6 | **`--sample-avg`** | 已判负（8 对 4/8、方差 2.45×） | 不要用 |
| 7 | **`term_off` 静默失效** | `[launch-v2] term_off=1` 打出来了，但 `rollout/survival=0.87`（应为 1.0） | **看 `rollout/survival` 这一个数**：≠1.0 就说明 `--term-off` 没进训练器 |
| 8 | **`wm_ckpt` 默认不是 `wm_gain2.pt`** | round5 默认 `$DIR/wm_exc.pt`，和我们验证的 WM 不是同一个 | 跑前 `grep wm_ckpt runs/<RUN>/args.json` |
| 9 | **`term_off` 静默失效** | `[launch-v2] term_off=1` 打出来了，但 `rollout/survival=0.87`（应为 1.0） | **只看一个数**：`rollout/survival` ≠1.0 就说明 `--term-off` 没进训练器 |
| 10 | **`wm_ckpt` 默认不是 `wm_gain2.pt`** | round5 默认 `$DIR/wm_exc.pt`，和我们验证过的 WM 不是一个 | 跑前 `grep wm_ckpt runs/<RUN>/args.json` |
| 11 | **批量替换静默失败** | 用 python 的 `str.replace` 改脚本，源码没匹配上却打印"成功" | **每次改完必须 `grep` 验证**：今天就这样漏掉了 `VAR_WEIGHT` 和把 `ACT_WEIGHT` 写成 0.3 |
| 12 | **单集视频不能当证据** | 同配置重跑，第 120 步就能差 30cm | 只看多集成功率/中位数 |

---

## 10. 这一轮之后怎么接

| 结果 | 下一步 |
|---|---|
| `action_var/ratio` **涨到 >0.5** 且落地探针残差/均值 >0.5 | 8 集 OpenDoor 配对；同时把 `diverge/pos_t*` 读出来，决定 `action_target_steps` 要不要缩 |
| 残差/均值 **<0.5**，但 `critic/v_ev` 高 | 说明"奖励能给方向，但逐状态学不出来" ⇒ 换**历史/记忆**（唯一不自带特权的破歧义手段），或改交付口径 |
| `critic/v_ev` **<0.9** | 先把 critic 修好（换 `critic_source=real` + `gae_targets_padded` 的 TD(λ)），再谈别的 |
| `act/rmse` 压不住（>0.15 且加 w 无效） | 说明奖励推力太强 ⇒ `lambda_r` 5→2，或改用 hinge 刹车 |

---

## 11. 文件索引

| 用途 | 路径 |
|---|---|
| **本轮启动脚本** | `scripts/run_p1_round_b.sh` |
| 启动器（下层） | `scripts/run_joint_ft_round5.sh` → `scripts/launch_joint_ha_v2.sh` |
| 训练主步（AT/act/critic 都在这里） | `diff_vla_wbc/v3_step.py` |
| critic | `diff_vla_wbc/token_critic.py`（`critic_update` / `gae_targets`） |
| 看板 + 门 | `scripts/watch_p1.py` |
| 落地探针 | `scripts/probe_ft_landing.py` |
| 合并后的起点集合 | `runs/a_prime_sonic_exc/all_transitions.pt` |
| WM 多步保真探针 | `scripts/probe_wm_multistep.py`（**必须带** `SUPERMIMIC_HA_ALIGN=1` 与 `SUPERMIMIC_ROBOT_USD`） |
| 设计/现状总文档 | `docs/JOINT_FT_DESIGN_STATUS_20261106.md` |

---

## 12. 一句话总结

> **这一轮不是"再加点训练量"，而是回答一个二元问题：把人为构造的动作目标拿掉之后，奖励信号能不能自己产生"随状态变化"的修正。**
> 能（残差/均值 >0.5）⇒ 这条路有救；不能 ⇒ 动作层到头，转交付口径或换输入。
