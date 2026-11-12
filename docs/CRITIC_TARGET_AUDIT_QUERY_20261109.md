# 外部评审提问：我们的 Critic Target 到底属于方案 A 还是方案 B？

> 提出人：C 线执行 Agent ｜ 2026-11-09 ｜ 想请人判断"这样做对不对、要不要改"
> 一句话结论：**在线更新是方案 B；离线预训练是第三种（截断 MC）；但 B 的"自举根"扎在 WM 想象里，
> 且 `K>H` 门槛在我们实现中已名不副实。**

---

## 0. 先给结论表

| 项 | 我们实际是 | 原本想要的 | 证据 |
|---|---|---|---|
| 在线 critic 目标 | **方案 B**：H=10 想象短轨迹 + TD(λ) + 终点 `V̄(y_10)` 自举 | 方案 B | §2 代码 |
| 2026-11-09 之前 | **方案 A**：TD(λ) 取到 `K_ROLL=20` 步（标签里带 WM 的 10→20 步累积预测） | — | §5 事故记录 |
| 离线预训练 | **第三种**：10 步**截断 MC**（无自举，尾端=0） | 应与在线目标一致 | §3 代码 |
| 自举落点 | **想象状态** `s_10` + 由想象标签训出的 `V̄` | 真实执行到的 `s_10` | §4 |
| `bootstrap_identifiable(K=20,H=10)` | 字面满足 | 语义不满足（见 §4） | `imagined_value_v5.py:290` |

---

## 1. 系统与口径（供评审对齐）

```
图像+语言+state46 →[FT8000 frozen]→ token[40,64]
   →[SONIC release, 冻结]→ q_target(29) @50Hz
   → LoRA16(16.89M) 只改 token
   →[WM6 8.82M 冻结, 可微] → H=10 步想象 → 奖励 J10
   + 在线 TokenCritic / ImaginedValueV5 → 尾部价值 β·γ^H·survival_H·(V(s_H) − V(ref_H))
```

* 训练器：`scripts/train_sonic_lora16_ft8000_bptt.py`（本 worktree）
* 世界模型：WM6（`wm_base6_best_candidate/wm.pt`，sha `8edb59cf…`），**冻结**
* 超参：`H=10`（可微 horizon）、`K=20`（一次执行块）、`γ=0.99`、`λ=0.95`、`EMA=0.995`
* actor 侧：`β=0.3` + 健康度门控（最近一次 v_ev < 0.30 ⇒ β 归零）
* critic：`ImaginedValueV5`，`in_dim=1086`，hidden 512/depth 2 → 819,713 参数
  特征 = `s71 | ref71 | (s−ref)71 | xi33 | hist10(58×10) | plan Z_{t..t+3}(4×64) | extra4`

---

## 2. 在线 critic 目标（**方案 B**）

`train_sonic_lora16_ft8000_bptt.py:1424-1457`

```python
with torch.no_grad():
    _seqk = _frames_seq(batch["frames"], Bc, K_ROLL)
    _, exk = d03.window_returns(env.bundle, {**_batchk, "tokens": tok_lora.detach()},
                                K_ROLL, critic_gamma, want_actions=True)   # ← WM 想象 K=20 步
    rok = exk["rollout"]
    ...
    _ys   = torch.stack([_CRIT_ADAPTER.build(rok, _cbk, _t, K=K_ROLL, T=_Tk)
                         for _t in range(H + 1)], 0)              # t = 0..10
    v_bar = torch.stack([critic_tgt(_ys[_t]) for _t in range(H + 1)], 0)   # V̄(y_0..y_10)
    rs    = torch.stack([rok.rewards[t].detach()          for t in range(H)], 0)  # r_0..r_9
    ds    = torch.stack([1.0 - rok.survival[t + 1].detach() for t in range(H)], 0)
    G     = td_lambda_targets(rs, ds, v_bar, gamma=critic_gamma, lam=critic_lam)
```

`td_lambda_targets`（`diff_vla_wbc/imagined_value_v5.py:220`）的定义：

```
G_t = r_t + γ·m_t·[(1−λ)·sg(V̄(y_{t+1})) + λ·G_{t+1}],   t = 0..H−1
末端： G_{H} = sg(V̄(y_H))        （未终止时自举；真终止时填 0）
```

⇒ **H=10 步的想象短轨迹 + TD(λ) + 终点价值自举。不是"把 10~20 步的 MC 累加当标签"。**
`K_ROLL=20` 的 rollout 现在只用来取前 11 个状态（`t=0..10`），后 10 步计算完直接丢弃。

---

## 3. 离线预训练目标（**第三种：截断 MC，无自举**）

`_collect_critic_feats`（`train_sonic_lora16_ft8000_bptt.py:510-537`）

```python
_, ex = d03.window_returns(env.bundle, {**b, "tokens": tok}, H, GAMMA, want_actions=True)
ro = ex["rollout"]
G = mc_returns(ro.rewards[:H].detach(), None, gamma=GAMMA)      # end_bootstrap=None ⇒ 尾端=0
```

`mc_returns`（`imagined_value_v5.py:274`）注释原文：**"不依赖自身 bootstrap 的 MC return-to-go（仅用于校验 v_ev）"**。
`end_bootstrap=None` ⇒ `G_{H-1} = r_{H-1}`，**没有 γ^H·V̄(s_H) 项**。

⇒ 预训练在拟合"10 步截断想象回报"，在线在拟合"带自举的 TD 目标"。
**同一个 V 被两套目标依次训练，`v_ev`（截断 MC 口径）与在线 `critic_v_loss`（TD 口径）不可比。**
（注：B 线的 `pretrain_critic.py` 也是默认 `--target mc`，所以两边都这样；但这是否合理需要评审。）

---

## 4. 我们这版 B 的两处"名不副实"

### 4.1 自举落点是想象状态

`V̄(y_10)` 里的 `s_10` 是 WM 滚出来的；`V̄` 本身是历史上由**想象标签**训出来的。
所以偏差仍通过 bootstrap 链传播（只是不再以"20 步 MC 累加"的形式一次性注入）。

### 4.2 `K > H` 这条门槛失去原本语义

`imagined_value_v5.py:290`：

```python
def bootstrap_identifiable(K: int, H: int) -> bool:
    """K == H（或更小）⇒ 尾部区间为 0 ⇒ V(y_H) 不可识别。"""
    return int(K) > int(H)
```

原本的语义应是：**actor 只优化前 H 步，但实际执行到 K 步 ⇒ 到 `s_H` 时还有 `K−H` 步真实后续，
`V(s_H)` 才有真实数据可依**。而我们所有 `s_t`（含 `s_H`）都是想象的，`K−H=10` 步补偿**完全没用上**（被丢弃）。
⇒ 字面满足 `K=20>H=10`，设计意图未满足。

---

## 5. 事故记录：方案 A 确实存在过（2026-11-09 已修）

修复前同一段代码：

```python
rs = ... for t in range(K_ROLL)      # 20 步
ds = ... for t in range(K_ROLL)
v_bar = ... for t in range(K_ROLL+1)
G = td_lambda_targets(rs, ds, v_bar)
```

表现（C3-v5 第一次训练，it=50）：

```
val 窗口:  v_ev=-5.9467  corr=+0.5194  mae=0.6636（常数基线 0.2282）
           v_mean=1.1026  vs  target_mean=0.4465      ← 系统性高 2.4 倍
在线批:    v_mean=0.5333  vs  target_mean=0.9773      ← 训练目标是 20 步回报
```

⇒ 标签是 20 步回报（0.98），考核/自举是 10 步回报（0.45），尺度错 2.4 倍。
改成 H 步后同一 it=50：`v_ev=-1.18`，`v_mean=0.76 vs target=0.45`。

---

## 6. 现状数据（评审可能需要判断"该不该换真标签"）

### 6.1 预训练结果（想象分布）

```
数据：eval_set 的 train split（256 窗，按窗口 80/20 切 dev，dev 不参与梯度）
样本：n=2560（256 窗 × 10 步），target mean=+0.4556 std=0.2619
结果：600 epoch → dev v_ev=+0.9828  corr=+0.991  mae=0.0191（常数基线 0.2191）
```

⚠️ **这个 0.983 是"拟合 WM6 想象回报"的 v_ev，不是真物理 v_ev。**
理由：WM 近确定性 ⇒ 给定 `(s_t, plan token)`，未来 10 步想象回报近似确定函数 ⇒ v_ev 天然可以很高。
B 线同口径（`imagine_with_tokens`）得 0.6259（`critic_v5_b3.json`：dev v_ev 0.6259、corr 0.804、mae 0.131 vs 0.229）。

### 6.2 手上有多少**真实 Isaac** 数据

| 数据 | 规模 | 字段 | 能支撑什么 |
|---|---|---|---|
| base6（6 任务真实闭环采集） | **36 集 / 75,216 tick** | `states_before/after(71)`、`actions_model(29)`、`q_target`、`q_applied`、`frame_count`、`done` | 状态/动作/裁剪对照；**不能算 tracking 标签** |
| C2 配对闭环（进行中） | 1 对（OpenDoor seed0，各 192 tick） | 同上，两臂 | 策略级 A/B 对照（同上限制） |
| A′ 语料（`a_prime_sonic_exc/all_transitions.pt`） | 502,636 行 / 1369 段 | `states/actions/next_states/frames(全局库行号)/clip_ids` + **Reference** | 唯一能直接算六分量 tracking 奖励的真实数据 |

> ## ★ 勘误（2026-11-09，评审指出后核实）
>
> **上一版这里写"base6 可以立刻做真标签 critic"是错的。** base6/C2 的 tick **没有合法 Reference**：
> 交接 §3.2 已写明"base6 tick 无 Reference ⇒ 不能算 tracking 判据"；
> 本次又核实了更硬的一条：记录器里的 `frame_count` 是 provider 的**局部 tick 计数器**
> （`action_provider_sonic.py:2981,3599` 初始化为 0、逐 tick 自增），**不是** motion library 的全局帧号。
>
> 连带发现一个已发生的 bug：`scripts/c_wm6/c2_reward.py` 拿 `frame_count` 直接索引
> `bundle.motion.ref`，因此它算出的 C2 `J10/J20`、`e_root_*` 等**含参考的项全部无效**
> （不依赖参考的 `joint_limit`/`action_rate` 仍有效）。该脚本需修或标注作废。
>
> ⇒ 真实 tracking 标签目前**只能**来自 A′ 语料，或做**带 Reference 记录**的新 Isaac 采集。

### 6.3 已知的"方向不可靠"证据

* P5-4 随机窗复评（H=10，真实 Isaac 回放）：符号一致率 **WM0 55.6%/50.0%，WM6 44.4%/45.7%**；
  配对 McNemar **p=0.18**（不能断言显著更差，但也没有解决"抛硬币"）。
* dir_agree（A′ 局部扰动）：H≤10 约 0.97–0.99，但那是"局部梯度方向"，不是"策略级动作块的收益排序"。
* B 线 β 实验：β=1 把 val ΔJ 从 +0.00914 打到 +0.00235，动作幅度 6.8→15–17 mrad；
  `‖g_βV‖/‖g_J10‖=2.69`、`cos=0.085`。

---

## 7. 想请评审直接回答的问题

1. **方案 B 的判定对不对？** 我们认定当前在线实现属于 B（H=10 TD(λ)+自举），而不是 A（20 步 MC）。
   如果是 B，那"Critic 只是蒸馏 WM 长程偏差"这个批评是否成立？成立到什么程度？
2. **自举落点在想象状态**（而不是 `K>H` 所保证的真实后续）——这是否是必须修的问题？
   如果是，最小修法是：(a) 目标轨迹改用真实 tick（base6 75,216 tick）；(b) 把 H 降到"执行块内确有真实数据"的范围；
   还是 (c) 其他？
3. **预训练用截断 MC、在线用 TD(λ)+自举**，两套目标是否必须统一？统一成哪一个？
4. 若最终目标是"让 critic 给出真实长期价值"，在**没有大规模真机后 10 步数据**的前提下，
   以下哪条是可接受的路线，或者都不行：
   - 真标签 critic（base6 真实 tick 的 MC return-to-go，off-policy 但改动小）；
   - 混合：真实 tick 打底 + 想象轨迹做数据增强（并报告两者的 v_ev 差）；
   - 保持想象标签，但**只用它做方差缩减/正则**，不用它决定收益方向。
5. **`K>H` 这条门槛我们是否应该保留？** 在当前实现下它没有提供任何真实数据，
   继续用它做 fail-closed 检查是否会给出虚假的安全感？

---

## 8. 附：复现命令

```bash
# 预训练（critic only，~10 min）
python3 scripts/train_sonic_lora16_ft8000_bptt.py ... \
  --critic-kind v5 --critic-enabled 1 \
  --critic-pretrain-epochs 600 --critic-pretrain-batch 256 --critic-pretrain-dev-frac 0.2

# 正式训练（2 卡 DDP，280 步，β=0.3 + 门控 + 热启动）
bash runs/dual_actor_wm6/c_lora/run_c3_final.sh

# critic 的 v_ev（每 50 步，val 96 窗，截断 MC 口径）
grep '"kind": "v_ev"' runs/dual_actor_wm6/c_lora/h10_critic_final/seed0/critic.jsonl
```


---

## 9. 评审后修订（2026-11-09 第二轮）

### 9.1 已确认并已修的代码问题

| 问题 | 位置 | 状态 |
|---|---|---|
| actor 侧的 critic 特征 `z` 用了未 detach 的 plan token ⇒ 价值梯度可以**绕过 WM6→s10** 直接推 token | `train_sonic_lora16_ft8000_bptt.py:1285` | **已修**：`z=tok_lora.detach()` |
| 预训练（截断 MC、尾端=0）与在线（TD(λ)+自举）**两套价值定义** | `_collect_critic_feats:529` vs `:1457` | 待统一（见 9.2） |
| `c2_reward.py` 用局部 tick 计数当参考帧号 | `scripts/c_wm6/c2_reward.py` | 待修/标注作废 |

### 9.2 建议的统一价值定义（待两边确认）

> **`V(s_10)` 定义为"执行块后半段的折现回报"**，而不是"无限视界的持续价值"：
>
> ```
> V(s_t) ≈ E[ Σ_{j=0}^{H-1} γ^j · r_{t+j} ]        （t = 10，即 K=20 块的后半段）
> ```
>
> 这样三条路径**同一个目标**、且都不需要自举：
>
> | 用途 | 目标 |
> |---|---|
> | 预训练 | `G_t = Σ_{j<10} γ^j r_{t+j}^{WM}`（t 取 10..19 —— 正好是被丢弃的那 10 步） |
> | 在线 critic | 同上（同一函数、同一窗口） |
> | actor 自举 | `J_ac = J10 + β·γ^10·surv·(V(s_10) − V(ref_10))` |
> | `v_ev` | 同一目标算 `1 − MSE/Var`，于是 v_ev 与训练目标**同口径** |
>
> 好处：① 预训练/在线/验证三者一致；② 去掉自举链（不再自我折叠）；③ `K>H` 那 10 步终于真的被用上。
> 代价：尾部标签仍是 WM 想象（这是模型式 RL 的正常做法），因此 §7 的第 4 问仍然要回答。
