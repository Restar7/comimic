# 交接单：容器 B —— p25 全流程（新数据 refresh → critic → 联合微调）+ 阶段 2 正式残差

> 生成：2026-11-12 08:40 ｜ 交接方：容器 A（本机） ｜ 接收方：**容器 B**
> 配套文档：`docs/P25_WM_REFRESH_ON_REPEAT_PLAN_20261112.md`、`docs/P26_RESIDUAL_THREE_STAGE_PLAN_20261112.md`
> 一句话：容器 B 负责**把正式 VLA 训出来**（p25）并**在它上面训正式残差**；
> 容器 A 同时在本机做**阶段 0 残差快速判定**，不占你的卡。

---

## 0. 30 秒速览

| 项 | 值 |
|---|---|
| 你的任务 | ① 构建转移对 → ② WM refresh #3 → ③ critic 预训 → ④ 联合微调 300 步 → ⑤ 选优 → ⑥（等补丁）正式残差 |
| 与 p21 的差异 | **只有 WM**（新数据刷出来、A/B 选优的 v3）+ critic 重训；**不动奖励、不动 std、不加上下文噪声** |
| LoRA 起点 | **c_base**：`runs/dual_actor_wm6/c_lora/critic10/lora300_ddp/checkpoint_lora_best.pt`（sha `6942eab9`） |
| 判据 | G1 WM 门 → G2 critic `v_ev≥0.30` → **G3 `dJ/J_base > 0.6%`** → （阶段 2）残差 `e_body_pos` −15% |
| 总耗时 | 约 **3 小时**（不含阶段 2 的评测） |
| GPU | 2 卡（DDP），全程占用 |

---

## 1. 职责边界（很重要，别越界）

| 文件/目录 | 归谁 | 说明 |
|---|---|---|
| `scripts/c_wm6/build_cl_pairs_v3.py` | **容器 B 新建** | 从 `build_cl_pairs.py` 复制改常量 |
| `scripts/c_wm6/p25_refresh3_cl.py` | **容器 B 新建** | 从 `p21_refresh2_cl.py` 复制改常量 |
| `runs/dual_actor_wm6/c_lora/p25/**` | **容器 B 独占** | 所有产出写这里 |
| `diff_vla_wbc/rollout.py` | **容器 A 负责改**（阶段 0 需要） | **容器 B 不要动**，等 A 通知“补丁已就位” |
| `scripts/c_wm6/train_residual.py` | **容器 A 新建** | 阶段 2 直接调用，不要自己另写一份 |
| `runs/dual_actor_wm6/c_lora/p21_seed012_*` | 只读 | 新闭环数据，用来构建转移对 |

> **并发约定**：容器 A 会改 `rollout.py`（加 `residual_fn` 参数，默认 None，不影响现有行为）。
> 容器 B 在阶段 1 期间**不碰这个文件**；阶段 2 开始前先 `git diff --stat diff_vla_wbc/rollout.py`
> 确认补丁在，再跑残差。

---

## 2. 前置检查（不满足别开工）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
git rev-parse --short HEAD                       # 期望 4513b42
grep -c "EPISODE_SEED_BASE" scripts/c_wm6/c2_driver.py        # 期望 3
grep -c 'path == "/reseed"'   scripts/ha_vla_latent64_server.py  # 期望 1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader  # 期望两卡空闲
df -h . | tail -1                                # 剩余 >200G
```

**数据前置**：新闭环在 `runs/dual_actor_wm6/c_lora/c2/` 下：

```
p21_seed012_r0_5/  p21_seed012_r6_11/  p21_seed012_r12_17/  p21_recheck_r0_5/
```

> 只采**已完成的集**：构建脚本要过滤 `result.json` 存在的 episode 目录
> （有的集可能只写了一半 ticks）。下面 §3 的脚本已内置这个过滤。

---

## 3. 步骤 1：构建闭环转移对（约 10 min，不需要 GPU）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
cp scripts/c_wm6/build_cl_pairs.py scripts/c_wm6/build_cl_pairs_v3.py
```

改 `build_cl_pairs_v3.py` 的三处常量：

```python
OUT  = WT / 'runs/dual_actor_wm6/c_lora/p25/refresh3'
RUNS = [
    # 新：目标分布（repeat 协议，seeds 0-2）
    'p21_seed012_r0_5', 'p21_seed012_r6_11', 'p21_seed012_r12_17', 'p21_recheck_r0_5',
    # 旧：防止过拟合到单一任务
    'collect_tok', 'ab2way_s35_50', 's7_12_container1', 's7_12_container2', 'ab2way_0_7',
]
```

并在扫描 `ticks_*.pt` 的循环里**只收已完成的 episode**（关键，防止半集污染）：

```python
ep_dir = f.parents[1]                     # .../<task>_seed<k>/try1_.../
if not (ep_dir / ".../result.json").exists() and not list(ep_dir.glob('*/result.json')):
    n_skip += 1; continue
```

跑：

```bash
set +u; source scripts/harness03_env.sh
python3 scripts/c_wm6/build_cl_pairs_v3.py
```

**验收**（`p25/refresh3/cl_pairs_meta.json`）：

| 项 | 阈值 |
|---|---|
| train + val 条数 | ≥ **35 万** |
| 闭环 ep 数 | ≥ **250** |
| val episode 占比 | ≈ 20%（`uid % 5 == 0`） |

---

## 4. 步骤 2：WM refresh #3 —— **双 init A/B**（各 10 min）

### 4.1 为什么要 A/B

| init | 是什么 | 好处 | 风险 |
|---|---|---|---|
| **A：`wm_base6`** | C_base 训练时用的 WM（动作敏感度仅 0.98，录制误差 5.38） | 干净、无旧闭环偏置、与 C 的训练分布一致 | 3000 步里前一半都在补“动作敏感度”，留给适应新分布的预算少 |
| **B：`refresh2/wm_cl_candidate.pt`** | p21 用的 WM（动作敏感度 2.77、闭环误差 0.25） | 起点已“动作敏感 + 闭环适应”，微调效率高 | 它在**旧闭环数据**上刷的，可能带旧任务偏置 |

**两个都刷（一共 20 分钟），用 G1 选优，只对胜者跑下游。** 不要凭直觉二选一。

### 4.2 准备脚本

```bash
cp scripts/c_wm6/p21_refresh2_cl.py scripts/c_wm6/p25_refresh3_cl.py
```

改三处常量（其余不动）：

```python
INIT = None                                   # ← 下面用 --init 覆盖
OUT  = WT / 'runs/dual_actor_wm6/c_lora/p25/refresh3'
DEV  = 'cuda:0'
```

若脚本不支持 `--init`，就在命令行前用环境变量：

```bash
P25_INIT=<path> python3 scripts/c_wm6/p25_refresh3_cl.py ...
```

（脚本里把 `INIT` 那行改成 `INIT = Path(os.environ.get('P25_INIT', str(默认值)))`。）

### 4.3 两次 refresh

**A：从 wm_base6（C_base 的 WM）出发**

```bash
export P25_INIT=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-wt03/runs/wm_base6_v1/wm_candidates/wm_base6_best_candidate/wm.pt
python3 scripts/c_wm6/p25_refresh3_cl.py --steps 3000 --lr 1e-5 --cl-frac 0.3 \
  --out runs/dual_actor_wm6/c_lora/p25/refresh3_from_base6
```

**B：从 refresh2 出发**

```bash
export P25_INIT=runs/dual_actor_wm6/c_lora/p19/refresh2/wm_cl_candidate.pt
python3 scripts/c_wm6/p25_refresh3_cl.py --steps 3000 --lr 1e-5 --cl-frac 0.3 \
  --out runs/dual_actor_wm6/c_lora/p25/refresh3_from_refresh2
```

**loss 一行不改**：`ssm_wm_loss + 2.0 × increment_loss`，`--cl-frac 0.3`（闭环 30% / A' 70%）。

> 为什么 0.3 而不是 0.5：联合微调的 `--eval-set` / `--motion` 窗口全部来自 A'，
> critic 和 J 都建在 A' 上；refresh#2 用 0.5 时录制侧误差 **0.1416 → 0.2221（退化 1.57×）**。

### 4.4 G1 对比（都在**新闭环 val** 上，四项）

| # | 指标 | 阈值 | 选优方向 |
|---|---|---|---|
| 1 | 零动作误差比 | ≥ 1.15 | 越高越好 |
| 2 | 打乱动作比 | ≥ 1.15 | 越高越好 |
| 3 | 闭环单步误差 | 退化 ≤ 20%（vs refresh2 的 0.2515） | 越低越好 |
| 4 | 闭环单步误差 vs refresh2 | **至少降 15%** | 越低越好 |
| 5 | 录制侧单步误差 | 退化 ≤ 30%（vs refresh2 的 0.2221） | 保护 A' |

**选优规则（写死，别事后改）**：

1. 先筛掉 1/2 不过的；
2. 在剩下的里，**按“闭环单步误差”最低者胜**；
3. 若两者差距 < 5%，**选录制侧误差更低的那一个**（更稳，保护下游 critic）；
4. 记录两个 WM 的 sha256，写进报告。

```bash
for f in runs/dual_actor_wm6/c_lora/p25/refresh3_from_*/wm_cl_candidate.pt; do
  echo "$f $(sha256sum $f | cut -c1-8)"; done
```

### 4.5 门禁不过时的分支

| 情况 | 处理 |
|---|---|
| A、B 都不过 1/2 | `--steps 4000` 重刷两版各一次；仍不过 → **停，报告** |
| 都过 1/2，但只有 4 不过（没适应目标分布） | 用 `--cl-frac 0.5` 各刷一版（写 `refresh3_*_cl50/`），四个 WM 一起比；仍不过 → 说明闭环信号不足，停下报告 |
| 只有 5 不过（A' 被挤坏） | 降 `--cl-frac` 到 0.2 重刷；或直接选录制侧误差更低的那版 |

---

## 5. 步骤 3：用新 WM 重训 v5 critic（约 10 min）

从 `runs/dual_actor_wm6/c_lora/run_p23_std_critic.sh` 复制成 `run_p25.sh`，**删掉 4 个 std 参数**，改：

```bash
OUT="$C/p25/ft300"
--wm-checkpoint "<G1 胜出的那个 WM>"
--strict-wm-hash <上一步的 sha8>
--critic-pretrain-out "$OUT/critic_pretrain.pt"
```

只跑阶段 1（critic 预训）：

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=2 --master_port=29593 \
  scripts/train_sonic_lora16_ft8000_bptt.py "${COMMON[@]}" --critic-enabled 1 \
  --critic-pretrain-epochs 20 --critic-pretrain-batch 256 \
  --critic-pretrain-out "$OUT/critic_pretrain.pt"
```

**G2 门禁**：日志里 `v_ev(val) ≥ 0.30`（p23 那次是 0.930）。不过就重训一次；仍不过 → 关 critic 跑并标注。

---

## 6. 步骤 4：联合微调 300 步（约 40 min）

同一个 `run_p25.sh`，阶段 2：

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=2 --master_port=29594 \
  scripts/train_sonic_lora16_ft8000_bptt.py "${COMMON[@]}" \
  --critic-enabled 1 --critic-warm-from "$OUT/critic_pretrain.pt"
```

`COMMON` 必须与 p21 逐项一致，只允许这三处不同：

```bash
--init-lora      "$C/critic10/lora300_ddp/checkpoint_lora_best.pt"    # c_base
--wm-checkpoint  "<G1 胜出的那个 WM>"
--strict-wm-hash <sha8>
```

其余照抄 p21：`--updates 300 --lr-total-updates 300 --warmup 20 --lr 1e-4 --min-lr 1e-6
--batch 16 --accum 1 --save-every 25 --eval-every 50 --lambda-lim 0.00490
--beta 0.975 --beta-warmup-updates 100 --beta-gate none --critic-grad-budget 0.20
--critic-updates 4 --critic-batch 2048 --critic-lr 3e-4 --c-recipe c1_lora_proj --seed 0 --ddp 1`

---

## 7. 步骤 5：选优 + G3 判据

训练自动按 val 选 best。**必须报相对量**（不同 WM 的 J 尺度不同，绝对 dJ 不可比）：

```
dJ_rel = dJ / J_base
```

| 模型 | dJ_rel |
|---|---|
| p21 | 0.38% |
| p22（噪声） | 0.25% |
| p23（锐化） | 0.24–0.32% |
| **p25 目标** | **> 0.6%** |

**G3 不过（≤0.5%）→ 停止流程，把结果报回来**，不要自己加实验。

---
我们
## 8. 步骤 6：阶段 2 —— 在 p25 的 VLA 上训**动作残差**

> 目标：冻结 p25 的 VLA，训一个 5 万参数的小 MLP，把**跟踪漂移**从 ~1.04 cm 压下去。
> 残差不改变任务意图（硬限幅 5%），部署时可单独开关，**不稀释 VLA 的贡献**。

### 8.1 前置：补丁与脚本（容器 A 提供，不要自己改）

| 文件 | 提供方 | 检查命令 |
|---|---|---|
| `diff_vla_wbc/rollout.py` 的 `residual_fn` 参数（默认 None） | 容器 A | `grep -n "residual_fn" diff_vla_wbc/rollout.py` |
| `scripts/c_wm6/train_residual.py` | 容器 A | `ls -l scripts/c_wm6/train_residual.py` |

> 两个都在共享盘上。**没就位就先跑 §9 的监控等通知**，不要自己改 `rollout.py`。

### 8.2 模型结构（已定，不要改）

```python
class ActionResidual(nn.Module):
    """输入 state(71) + token(64) + prev_action(29) = 164 维 → 输出 Δa(29)"""
    def __init__(self, d_state=71, d_token=64, d_act=29, hidden=256, max_res=0.05):
        super().__init__()
        self.max_res = max_res
        self.net = nn.Sequential(
            nn.Linear(d_state + d_token + d_act, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_act),
        )
        nn.init.zeros_(self.net[-1].weight)      # ★ 初始 Δa = 0
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, token, a_prev):
        return self.max_res * torch.tanh(self.net(torch.cat([state, token, a_prev], -1)))
```

| 设计点 | 值 | 理由 |
|---|---|---|
| 输入 | state71 + token64 + prev_action29 | 控制层反馈律；**不接图像** |
| 输出 | Δa 29 维，`0.05 × tanh` | 硬限幅 5%，改不了任务意图 |
| 末层 | 零初始化 | 第一步等价于原策略 |
| 参数 | ~5 万 | 部署 <0.1 ms/帧 |

### 8.3 训练循环（脚本内已实现，这里只说明口径）

```python
vla.eval(); wbc.eval(); wm.eval()                      # 全冻
res = ActionResidual().to(dev)
opt = torch.optim.Adam(res.parameters(), lr=args.lr)
for step in range(args.steps):
    batch = provider.sample(args.batch, seed=...)      # eval_set 的 train 窗口
    with torch.no_grad():
        tok = sampler.sample(batch["frames"], seed=..., grad=False)   # 冻结 VLA 的 token
    J, ex = window_returns(bundle, {**batch, "tokens": tok}, H, GAMMA,
                           want_actions=True, residual_fn=res)        # ★ 注入残差
    a_res = ex["residual"]                                             # (H,B,29)
    loss = (-J.mean()
            + 1e-3 * (a_res ** 2).mean()                               # 幅度惩罚
            + 1e-2 * (a_res[1:] - a_res[:-1]).square().mean())          # 平滑惩罚
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
```

### 8.4 跑

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh
python3 scripts/c_wm6/train_residual.py \
  --vla-ckpt  runs/dual_actor_wm6/c_lora/p25/ft300/checkpoint_lora_best.pt \
  --wm-ckpt   <G1 胜出的那个 WM> \
  --strict-wm-hash <sha8> \
  --out       runs/dual_actor_wm6/c_lora/p25/residual \
  --steps 300 --batch 32 --lr 1e-3 --max-res 0.05
```

参数表：

| 参数 | 值 | 说明 |
|---|---|---|
| `--steps` | 300 | 只训 5 万参数，约 10–15 min |
| `--batch` | 32 | 窗口数 |
| `--lr` | 1e-3 | Adam |
| `--max-res` | 0.05 | 残差上限（可调到 0.02 / 0.10 做敏感性） |

### 8.5 阶段 2 验收

| 指标 | 阈值 | 不过怎么办 |
|---|---|---|
| `e_body_pos`（训练窗口） | **≥15% 下降**（1.04 → ≤0.88 cm） | 先加 `--max-res 0.10` 重训；仍不过 → 报告“动作层没有空间” |
| `‖Δa‖` 均值 | ≤ **0.5 ×** `max_res` | 顶到上限说明在硬撑 → 提高幅度惩罚（改脚本常数需找容器 A） |
| 越限率 | ≤ p21 的 6.8% + 2pp | 加平滑惩罚或降 `max-res` |

### 8.6 阶段 3：真仿真配对（**必做，别只看 WM 指标**）

WM 里的改进可能是“利用 WM 误差”，必须在真 Isaac 闭环里验证。

```bash
# 残差臂：30 格（OpenDoor + Boxing，seed0-2 × repeat0-4）
# 注意：需要 train_residual.py 导出的 residual.pt 被闭环脚本加载
#       加载方式见 docs/P26_RESIDUAL_THREE_STAGE_PLAN_20261112.md §3.5
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 C2_SERVER_SEED=20261109 \
RESIDUAL_CKPT=runs/dual_actor_wm6/c_lora/p25/residual/residual.pt \
python3 scripts/c_wm6/c2_driver.py --tasks OpenDoor --seeds 0 1 2 --repeat-start 0 --repeats 5 \
  --arms c --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/p25_resid_repeat/card0 \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p25/ft300/checkpoint_lora_best.pt --lora-tag p25
```

对照臂：**同一 VLA、不挂残差**，同 30 格（`--lora-tag p25`，`RESIDUAL_CKPT` 留空）。

判据：逐格配对，残差臂净胜 **≥6 格**，且真仿真的 `e_body_pos` 也下降 ≥10%。

### 8.7 报告里必须写清的三件事

1. **残差是在 WM 里有效还是真仿真也有效**（两个数分开报）；
2. **残差是否顶到限幅**（`‖Δa‖/max_res`）；
3. **交付建议**：带残差（指标好）还是不带（纯 VLA）。

---

## 9. 监控（贴给容器 A 的三行）

```bash
# refresh 进度
tail -20 runs/dual_actor_wm6/c_lora/p25/refresh3_from_*/*.log 2>/dev/null
# 训练进度
grep -E "^\[diag03\] \[C\] it=|\[eval\] it=" runs/dual_actor_wm6/c_lora/p25/*.log | tail -5
# 残差进度
grep -E "step|e_body_pos|res_norm" runs/dual_actor_wm6/c_lora/p25/residual/*.log | tail -5
```

---

## 10. 交付物清单

- [ ] `p25/refresh3/` 的 `cl_pairs.pt` + `cl_pairs_meta.json`
- [ ] `p25/refresh3_from_base6/` 与 `p25/refresh3_from_refresh2/` 两版 WM + **sha256**
- [ ] **G1 A/B 对比表**（四项）+ 选优结论
- [ ] G1 三项指标（零动作比 / 打乱比 / 单步误差）与 refresh2 的对照
- [ ] `p25/ft300/{critic_pretrain.pt, checkpoint_lora_best.pt, checkpoint_lora_best.json}`
- [ ] `val dJ`、`test dJ`、`J_base`、**`dJ_rel`**
- [ ] （阶段 2）`p25/residual/residual.pt` + 训练曲线 + `e_body_pos` 前后
- [ ] （阶段 3）`p25_resid_repeat/` 30 集 + 与「同 VLA 无残差」的逐格配对 + 真仿真 `e_body_pos`
- [ ] 一句话结论：**新 WM 是否让 `dJ_rel` 超过 0.6%；残差是否把误差降 ≥15%（WM 与真仿真分开报）**

---

## 11. 风险与止损

| 风险 | 表现 | 处理 |
|---|---|---|
| 新数据把 WM 带偏 | G1 零动作比 < 1.15 | `--cl-frac 0.35` 重刷一次；仍不过 → 停 |
| G2 critic 学不好 | `v_ev < 0.3` | 重训一次；仍不过 → 关 critic 并标注 |
| **G3 不过（最可能）** | `dJ_rel ≤ 0.5%` | **停**，把 val/test 曲线发回来，不要加实验 |
| 残差在 WM 有效、真仿真无效 | 阶段 2 指标好但闭环不动 | 报告里单独标注“疑似 WM 利用”，等容器 A 的真仿真验证 |
| 两容器抢磁盘/带宽 | 训练变慢 | 各写各的目录；不要同时跑 GPFS 密集的 tick 扫描 |

---

## 12. 一句话总结

**两步走**：

1. **先做 §3–§7（约 1 小时）**：把 `dJ_rel` 跑出来，**>0.6% 再继续**；否则停下报告。
2. **再做 §8（约 1.5 小时）**：在 p25 的 VLA 上训残差 → **必须做 §8.6 的真仿真配对**，否则不知道是不是在利用 WM 误差。

阶段 0（残差能不能降漂移）由容器 A 在本机并行验证，结论会第一时间同步给你；
如果阶段 0 就判定“残差降不了漂移”，§8 直接取消，省下 1.5 小时。
