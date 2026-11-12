# p16 闭环评测交接（2 卡 4090 · 六任务 × 6 seeds）

> 生成于 2026-11-10 ｜ 目标机器：**2 × RTX 4090（同目录结构）** ｜ 只跑推理，不训练
> 一句话：用 **p16 的 LoRA 适配器**在 Isaac 里跑 `OpenDoor / SitSofa / Boxing / VisionNavi / PPBox / DoubleDesk` 各 **seeds 0–5**，产出与交付物同口径的成功率表。

---

## 0. 30 秒速览

| 事项 | 内容 |
|---|---|
| 适配器 | `comimic-c-wm6/runs/dual_actor_wm6/c_lora/p16/margin_only300/checkpoint_lora_best.pt` |
| sha256 前 16 | `ad87d44432874b40`（完整值用 §1 命令核对） |
| 期望 LoRA meta | `update=100`、`dJ_val≈0.0045253`、`physical.lim=0.0306794`、440 tensors / 218 层 / rank16 / alpha32 |
| p16 是什么 | **交付物配方 + 软限位 margin（λ_margin=0.067918, κ=0.05）**，λ_lim 保持旧值 0.0306794 |
| 已测结论 | VN **3/6、0 摔**（交付物 3/6、摔 2）；OpenDoor seeds0/1 **2/2**（交付物同 2/2） |
| 本交接要补的 | 六任务全量 6 seeds（尤其 OpenDoor seeds 2–5、Boxing、SitSofa、PPBox、DoubleDesk） |
| 规模/耗时 | 36 集；两卡并行各 18 集，**约 1.5–2.5 小时**（超时集 ~8–10 min，成功集 2–6 min） |
| 判据 | 逐任务对比 §6 基线；**重点看六任务成功率不降 + VN 摔倒仍为 0** |

---

## 1. 资产与前置检查

在目标机器上依次确认（路径按"同目录"假设；不一致就改 §2 顶部三个变量）：

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6

# 1) 适配器在位、哈希对得上（必须一致，否则不是 p16）
ls -la runs/dual_actor_wm6/c_lora/p16/margin_only300/checkpoint_lora_best.pt
sha256sum runs/dual_actor_wm6/c_lora/p16/margin_only300/checkpoint_lora_best.pt | cut -c1-16
#   → 期望 ad87d44432874b40

# 2) meta 自检（update/dJ_val/λ_lim 三个都要对）
python3 -c "
import torch;d=torch.load('runs/dual_actor_wm6/c_lora/p16/margin_only300/checkpoint_lora_best.pt',map_location='cpu',weights_only=False)
m=d.get('meta',{});print('update',m.get('update'),'dJ',m.get('dJ_val'),'phys',m.get('physical'),'tensors',d.get('n_tensors'))"
#   → update 100 / dJ 0.0045252875 / phys {'lim': 0.0306794,...}

# 3) 其它必需资产
ls -d /inspire/.../comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000      # VLA 底模
ls -d /inspire/.../comimic/groot_repro/nvidia/Cosmos-Reason2-2B-weights             # VLM
ls -d /inspire/.../comimic-wt03/runs/temporal_ac_bptt_v5/eval_set                  # 任务窗口/元数据
ls    /inspire/.../comimic-wt03/runs/three_track_v6/physical/limits_standard29.npz  # 关节限位
ls    /inspire/.../comimic/scripts/ha_isaac_closedloop.sh                           # Isaac 闭环入口
ls -d /inspire/.../comimic/third_party/isaacsim_env/bin                             # Isaac python
ls -d /inspire/.../comimic/third_party/HumanoidArena                                # HA 场景
```

> ⚠️ `comimic/scripts/ha_isaac_closedloop.sh` 里 `COMIMIC=` 是**写死的绝对路径**。换机器第一件事就是改它；否则 `cd` 到不存在的目录直接失败。

---

## 2. 一次性准备（每台机器只做一次）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u
source scripts/harness03_env.sh          # 提供 trackerLab/trackerTask 的 PYTHONPATH（server 用）
export C2_SERVER_SEED=20261109           # ★ 固定采样种子，保证同 seed 可复现
```

**冒烟测试（必做，约 5 分钟）**：先单卡跑 1 个任务 1 个 seed，确认整条链能起来。

```bash
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
python3 scripts/c_wm6/c2_driver.py --tasks Boxing --seeds 2 --arms c --max-steps 3000 \
  --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/p16_smoke \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p16/margin_only300/checkpoint_lora_best.pt \
  --lora-tag p16
```

冒烟通过的三条标志：

1. 日志里出现 **`[server][lora16] loaded ... tensors=440 ... meta={'update': 100, ...}`**（若没有这行，driver 会 fail-closed 拒绝评测）；
2. 日志出现 `共 N 个 (task,seed) × 1 臂 = N 集`；
3. 结束时出现 `rc=0 ... success=True/False`，且 `.../result.json` 落盘。

（参考：Boxing seed2 在 p16 和交付物上都是成功，约 159–169 步。若这里超时，先别急着跑全量，见 §7。）

---

## 3. 正式跑：两卡两 driver，六任务拆两半

**分工（按历史耗时平衡过）**

| 卡 | 端口 | 任务 | seeds |
|---|---|---|---|
| 0 | 18080 | `DoubleDesk Boxing VisionNavi` | 0–5 |
| 1 | 18081 | `OpenDoor SitSofa PPBox` | 0–5 |

**卡 0**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18080 C2_GPU_C=0 C2_ISAAC_GPU=0 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks DoubleDesk Boxing VisionNavi --seeds 0 1 2 3 4 5 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/p16_6task \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p16/margin_only300/checkpoint_lora_best.pt \
  --lora-tag p16 > runs/dual_actor_wm6/c_lora/p16_6task_A.log 2>&1 &
```

**卡 1**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh; export C2_SERVER_SEED=20261109
C2_SERVERS=c C2_PORT_C=18081 C2_GPU_C=1 C2_ISAAC_GPU=1 \
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks OpenDoor SitSofa PPBox --seeds 0 1 2 3 4 5 --arms c \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/p16_6task_B \
  --lora-ckpt runs/dual_actor_wm6/c_lora/p16/margin_only300/checkpoint_lora_best.pt \
  --lora-tag p16 > runs/dual_actor_wm6/c_lora/p16_6task_B.log 2>&1 &
```

> 两个 `--out` 故意不同，方便两 driver 并行写；汇总时把两个根一起交给聚合脚本（§5）。
> `--record-video 0` 是最快档；要留视频证据就改 `2`，代价是**单集慢约 2 倍**。

---

## 4. 监控

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
# 进度（每个 rc= 是一集结束）
grep -hE "rc=0" runs/dual_actor_wm6/c_lora/p16_6task_{A,B}.log | tail -6
# 计数
for f in runs/dual_actor_wm6/c_lora/p16_6task_{A,B}.log; do echo -n "$f "; grep -c "rc=0" $f; done
# 卡上都有谁（每卡应只有一个 server + 一个 Isaac worker）
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

每集耗时参考：成功 2–6 min，超时（3000 步）8–10 min。**36 集预计 1.5–2.5 h**。

---

## 5. 汇总

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
python3 scripts/c_wm6/agg_c2_episodes.py --roots p16_6task_A p16_6task_B --arm c --fall all
```

输出形态（seed 去重，一个 seed 成功过即算成功）：

```
| task | succ | fall | timeout | sim_err | per-seed |
| OpenDoor | x/6 | ... |
```

---

## 6. 基线对照（交付物 `lora300_ddp`，seeds 0–5，可直接复用，不必重跑）

| 任务 | 交付物 | 备注 |
|---|---|---|
| OpenDoor | **6/6**（1013/1192/2214/548/772/2515） | p16 已测 seeds0/1 = ✓494/✓1559 |
| SitSofa | **5/6**（s5 超时；✓365/✓811/✓452/✓485/✓2416） | p16 未测 |
| Boxing | **4/6**（s2✓169 s3✓594 s4✓191 s5✓418；s0/s1 超时） | p16 未测 |
| VisionNavi | **3/6，摔 2**（s2 摔1832、s3 摔1006；✓s1 675/s4 1648/s5 923） | **p16 = 3/6，摔 0** |
| PPBox | **2/6**（s1✓616 s4✓683；s0/s2/s3/s5 超时、s0 还有一次 sim_err） | p16 未测 |
| DoubleDesk | **0/6**（6 集全超时，base 也是 0） | 该任务全栈都过不了，属已知 |
| 合计 | **20/36** | |

**判读口径**

- 逐任务允许 ≤1 个 seed 的波动；
- **重点**：OpenDoor ≥5/6、SitSofa ≥4/6 不能塌；VN 若维持 3/6 且**摔倒仍为 0**，则 p16 = "成功率不降、安全更好"的可用升级；
- 若 VN 摔倒回到 2 次，说明防摔来自别处，p16 的卖点不成立。

---

## 7. 已知坑（都踩过）

| # | 现象 | 处理 |
|---|---|---|
| 1 | 启动就报 `c.log 里没有 '[server][lora16] loaded'` | 适配器路径错/没加载；先核对 §1 的 sha 与 meta |
| 2 | `FileNotFoundError: 'ss'` | 该镜像缺 `iproute2`；本仓库的 `c2_driver.py` 已加兜底（老版本需 `apt install iproute2` 或打同样的补丁） |
| 3 | 端口被占 → 报 `cd: too many arguments` 或直接退出 | 换端口（18080/18081 之外），或 `pkill -f "c2_driver[.]py"` 清干净（注意方括号防自匹配） |
| 4 | 两卡各跑一个 driver 时变慢 2 倍 | 正常（Isaac 物理抢 CPU）；想快就减少并发、或接受更长墙钟 |
| 5 | 4090 只有 24 GB | 每个 driver = VLA server(~13.4 GB) + Isaac(~4–8 GB) ≈ 18–21 GB，**勉强够**；别在卡上跑别的东西，先 `nvidia-smi` 看占用 |
| 6 | 任务全超时 | 先看是不是"慢速游走不推进"（已知失败形态）；再查 VLA/Isaac 版本与 `ha_isaac_closedloop.sh` 的 `COMIMIC` 路径 |
| 7 | 想用 `c2_reward.py` 的 J10/J20 | **不可引用**（`frame_count` 是局部计数器）；只用 `success / episode_steps / failure_reason` |
| 8 | 磁盘 | 每集约 190 个 `ticks_*.pt`；36 集请预留 **≥20 GB** 余量 |

---

## 8. 交付什么

1. `runs/dual_actor_wm6/c_lora/c2/p16_6task_{A,B}/` 两个目录（含 `jobs.jsonl`、每集 `result.json`、`ticks/`）；
2. §5 的聚合表（六任务 × 成功率/摔倒/超时）；
3. 一句话结论：**p16 相对交付物是否逐任务不降、VN 摔倒是否仍为 0**；
4. 有异常集请附 `episode.log` 尾部 20 行。
