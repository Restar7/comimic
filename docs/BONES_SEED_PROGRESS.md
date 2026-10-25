# BONES-SEED 训练进度可视化 / 查看方式

> 当前在跑：`trackerlab-supermimic/logs/bones_seed_2x8192_compile`
> （2 卡 × 8192 env = 16384 环境，32000 迭代，torch.compile 开，BONES-SEED 全量）

---

## 1. 三条命令（复制即用）

先设两个变量（后面都用它们）：

```bash
export REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic
export RUN=$REPO/logs/bones_seed_2x8192_compile
export PY=$REPO/../third_party/trackerlab_env/bin/python
```

### ① 一眼看进度（markdown 面板 + 进度条 + ETA）

```bash
$PY $REPO/scripts/rls/supermimic/progress.py --run_dir $RUN
```

输出长这样（自动算「最近 20 行的中位 it/s」，不会像单行那样跳）：

```
[████████░░░░░░░░░░░░░░░░░░░░░░░░]  25.0%  iteration 8000 / 32000
elapsed 3h58m   ETA 11h52m   speed 0.56 it/s (last 0.57)

| metric | now | best |
| simulator reward / step (max 2.0) | 0.712 | 0.731 |
| normalised episode return       | 0.604 | 0.618 |
| terminated fraction (lower better) | 0.41 | — |
...
```

### ② 实时刷新（每 60 秒重画，Ctrl-C 退出）

```bash
$PY $REPO/scripts/rls/supermimic/progress.py --run_dir $RUN --watch 60
```

### ③ 出曲线图（PNG，4 张子图：sim_rew / ep_ret / losses / it/s）

```bash
$PY $REPO/scripts/rls/supermimic/progress.py --run_dir $RUN --plot $RUN/progress.png
```

也可以顺手把面板落成 markdown 文件（方便贴给别人 / 存证）：

```bash
$PY $REPO/scripts/rls/supermimic/progress.py --run_dir $RUN \
    --markdown_out $REPO/../docs/BONES_SEED_PROGRESS_SNAPSHOT.md --plot $RUN/progress.png
```

### ④ 最原始的两种看法（不依赖脚本）

```bash
tail -F $RUN/../bones_seed_2x8192_compile.launcher.log | grep --line-buffered '^it '
column -s, -t $RUN/train_log.csv | tail -20                  # 全字段 CSV
ls -lh $RUN/ckpt_*.pt                                        # 已存 checkpoint（每 2000 迭代）
nvidia-smi                                                   # 每卡应稳定在 ~26 GB
```

### ⑤ 只看「每一轮更新」= 每 50 迭代一行（**最常用**）

**从任意目录都能跑，直接整行复制**（注意：脚本参数是**真实路径**，不是带尖括号的占位符；
shell 里 `<xxx>` 会被当成重定向，报 `syntax error near unexpected token newline`）：

```bash
LINES=10 bash /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/scripts/rls/supermimic/watch_updates.sh
```

想指定别的 run，把**目录名**作为第一个参数（同样不要尖括号）：

```bash
bash /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/scripts/rls/supermimic/watch_updates.sh \
  /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/logs/bones_seed_2x8192_compile
```

如果想用短的相对路径，先正确 `export`（新开 shell 必须重设，`$REPO` 为空时 `cd $REPO` 会回到 home，
于是就出现 `No such file or directory`）：

```bash
export REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic
cd "$REPO" && bash scripts/rls/supermimic/watch_updates.sh
```

它做的事就是 `tail -F logs/<run 名>.launcher.log | grep '^it '`（把 `<run 名>` 换成真实目录名），
只读、Ctrl-C 退出、不影响训练。等价的一行命令：

```bash
tail -F /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic/logs/bones_seed_2x8192_compile.launcher.log \
  | grep --line-buffered '^it '
```

> ⚠️ **launcher 日志会滞后于 CSV**：`torchrun` 会缓冲子进程输出，所以「日志里还没出现的新行」
> 其实已经进了 `train_log.csv`。要看**最实时**的数字，读 CSV（`progress.py` 就是读 CSV 的）。
> 两者内容一致，只是刷新时机不同。

**一行里的字段**（`--log_every` 默认 50，即每 50 迭代一行）：

| 字段 | 含义 | 期望方向 |
|---|---|---|
| `it N` | 已完成迭代数 | ↑ |
| `wm` / `e8` | 世界模型 loss / 8 步预测误差 | ↓ |
| `pi` / `v` | 策略 loss / critic loss（`--wm_only_iters 500` 前恒为 0） | 收敛 |
| `imag_r` / `surv` | 想象 rollout 的 reward / 存活率 | ↑ |
| `sim_r` | 真仿真逐步 tracking reward（满分 2.0） | ↑ |
| `ep_len` / `ep_ret` | 回合长度 / 归一化回合回报 | ↑ |
| `term` | 结束回合里被终止的比例 | ↓ |
| `it/s` | 迭代速度 | ↑ |
| `peak` | PyTorch 峰值显存 | 稳定 |

**本机真实样本**（`logs/bones_seed_2x8192_compile`，wm-only 热身阶段）：

```
it  50 | wm 0.7415 e8 1.2831 | pi 0.000 v 0.0000 | sim_r 0.445 ep_len 32.1 ep_ret 0.450 term 0.82 | 0.28 it/s | peak 13.1 GiB
it 100 | wm 0.2944 e8 0.4672 | pi 0.000 v 0.0000 | sim_r 0.440 ep_len 36.5 ep_ret 0.439 term 0.83 | 1.27 it/s | peak 13.1 GiB
it 200 | wm 0.1927 e8 0.2940 | pi 0.000 v 0.0000 | sim_r 0.440 ep_len 38.4 ep_ret 0.439 term 0.83 | 1.29 it/s | peak 13.1 GiB
it 300 | wm 0.1577 e8 0.2388 | pi 0.000 v 0.0000 | sim_r 0.442 ep_len 38.6 ep_ret 0.438 term 0.83 | 1.30 it/s | peak 13.1 GiB
it 450 | wm 0.1300 e8 0.1957 | pi 0.000 v 0.0000 | sim_r 0.440 ep_len 39.0 ep_ret 0.438 term 0.83 | 1.24 it/s | peak 13.1 GiB
```

读法：`wm` 0.74 → 0.13、`e8` 1.28 → 0.196 说明**世界模型在收敛**（这就是前 500 迭代唯一在学的东西）；
`sim_r` 还停在 0.44、`term 0.83` 是因为策略还没开始更新（`pi = 0`）。

---

## 2. 指标怎么读（哪些数字算"在学"）

| 指标 | 含义 | 变好的方向 | 参考 |
|---|---|---|---|
| `sim_rew` | 真仿真里的逐步 tracking reward（满分 2.0/步） | ↑ | 训练早期 0.44 左右（还没学） |
| `ep_ret_norm` | 归一化 episode return | ↑ | 上游参考 2000 迭代 0.5416 |
| `term_frac` | 结束的 episode 里被终止的比例 | ↓ | 越低说明没摔/没跑飞 |
| `ep_len` | episode 长度 | ↑ | 上限由 clip 长度决定 |
| `wm_loss` / `wm_err8` | 世界模型拟合误差 | ↓ | 前 500 迭代只有它在学（wm-only 热身） |
| `pi_loss` / `v_loss` | 策略 / critic 损失 | 收敛即可 | 500 迭代后才开始动 |
| `it_per_s` | 迭代速度 | ↑ | 编译期那几行偏低是正常的（见 §3） |
| `peak_allocated_gib` | PyTorch 峰值显存 | 稳定 | 8192 env 实测 **25.5 GiB** / 48 GB |

**可判读的里程碑（对齐上游同款实现）**

| 迭代 | 上游参考（LAFAN1 40 条） | 我们（BONES-SEED 129,943 条）该怎么看 |
|---|---|---|
| 2,000 | normalized return **0.5416**，12–40 条失败 | 数据多 3000 倍，同样迭代数会更低，属正常 |
| 32,000 | **0.8695**，0 条失败 | 本轮终点；到 32k 看 `ep_ret_norm` / `term_frac` 趋势 |

> 注意：上游是 1 卡 8096 环境、40 条 clip；我们是 **2 卡 × 8192 = 16384 环境**、13 万条 clip。
> 绝对数值不可直接比，看**趋势**（`sim_rew` 单调抬升、`term_frac` 下降）才有效。

---

## 3. 三个容易误判的地方

1. **前 50 行左右 it/s 偏低**：`torch.compile` 是首次编译（5 个热函数 × 2 个进程），
   编译完 it/s 才升到稳态；`progress.py` 用的是最近若干行的**中位数**，会自动把这段稀释掉，
   但刚开始几十分钟你仍会看到 ETA 偏大。
2. **前 500 迭代 `pi_loss = 0.000`**：`--wm_only_iters 500` 是设计如此，只有世界模型在学。
3. **`ep_len` 在 30–40 徘徊不涨**：早期策略随机、tracking 奖励低，episode 很快终止；
   这是 `term_frac 0.82` 的来源，等 `sim_rew` 过 0.6 之后再关注 `ep_len`。

---

## 4. 进程在哪 / 停了怎么办

```bash
# 看是不是还活着（launcher 是独立 session，PPID=1）
ps -eo pid,ppid,sid,etime,args | grep train_bones_seed | grep -v grep
```

万一挂了：从最近的 `last.pt` 续训（关键参数必须和原 run 一致）。
下面用**绝对路径**，新开 shell 也能直接跑：

```bash
REPO=/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic/trackerlab-supermimic
PY=$REPO/../third_party/trackerlab_env/bin/python
RUN=$REPO/logs/bones_seed_2x8192_compile

$PY $REPO/scripts/rls/supermimic/train.py \
  --task g1 --motion_packed $REPO/data/bones_seed/bones_seed_compact.pt \
  --num_envs 8192 --iterations 32000 --interface motion \
  --goal_frames 10 --hist 10 --horizon 32 --utd 0.4 --lr 2e-4 \
  --compact_library --compile --shard_motion --shard_mode interleave \
  --amp bf16 --kit_args=--allow-root \
  --resume $RUN/last.pt --log_dir $REPO/logs/bones_seed_2x8192_resume
```

`--resume` 要求 `last.pt`（含 optimizer 状态），`ckpt_*.pt` 是导出用的轻量快照。
