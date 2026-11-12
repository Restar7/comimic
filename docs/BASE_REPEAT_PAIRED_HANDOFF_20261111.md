# base 臂 × repeat 协议配对评测交接（OpenDoor + Boxing，seeds 0/1/2，repeat 0–4）

> 生成：2026-11-11 23:10 ｜ 执行者：base 线 ｜ 目的：给 p21 的 repeat 协议成绩找**同协议对照**
> 一句话：p21 现在报 59%（90/152），但**从来没有 base 在这个协议上的数据**，所以这个数高还是低无法判断。本单跑 30 集 base，逐格配对。

---

## 0. 30 秒速览

| 项 | 值 |
|---|---|
| 评测对象 | 官方 FT8000 `comimic/runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`，**不挂 LoRA** |
| 臂 | `--arms base`（driver 里 base 臂不会追加 `--lora16-ckpt`） |
| 任务 | card0 = `OpenDoor`，card1 = `Boxing` |
| seeds | **0 1 2** |
| repeat | **0 1 2 3 4**（`--repeat-start 0 --repeats 5`） |
| 规模 | 2 任务 × 3 seed × 5 repeat = **30 集**（每卡 15 集） |
| 成本 | 每集 3–13 min（失败集跑满 3000 步），单卡 15 集 ≈ **2–3 小时** |
| 输出 | `runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4/{card0,card1}` |
| 主判据 | 30 对逐格 McNemar 精确检验（见 §2，**先写死**） |

---

## 1. 为什么跑这一单

| 数据 | 成绩 | 能不能和 p21 比 |
|---|---|---|
| p21 在 repeat 协议（本批） | **90/152 = 59%**（重跑替换后；原始 82/152 = 54%） | — |
| base 在 `/reseed` 协议（0–50） | 114/228 = 50.0% | ❌ 场景不同 |
| C 在 `/reseed` 协议（0–50） | 112/247 = 45.3% | ❌ 场景不同 |

两套协议的 `episode_seed` 来源不同（`/reseed` 用顺序种子，repeat 用 sha256 派生），场景分布不一样，**跨协议比较必然被批次效应污染**（p21 自己在 0–50 扫里就是高 seed 94%、低 seed 54%）。

⇒ 唯一干净的对照：**base 跑同一批 repeat 格**。

---

## 2. 预注册判据（跑之前先定，跑完不许改）

1. **每格只跑 1 次**，取**首次尝试**（`try1`）；`success` 以 `result.json` 的 `success` 字段为准。
2. **禁止**：只重跑失败格、取“最好一次”、事后挑格。若某格因环境异常（进程被杀/OOM）没有产出，整批重跑该格并在报告里标注 `rerun`，但报告中**同时保留两次结果**。
3. 主判据：30 对（task, seed, repeat）里 base vs p21 的 **McNemar 精确检验**（双侧）。
4. **采样量要提前说清**：30 对只能看出 **≥30 个百分点**的差。若 p 不显著，结论就写“该协议下分不出”，**不许**改口成“base 更好/更差”。
5. 分任务（OpenDoor 15 对 / Boxing 15 对）只作描述，不做显著性结论。

---

## 3. 目标格 + p21 参考值（这 30 格必须一模一样）

格 = `(task, seed, repeat_idx)`。p21 的值取“重跑替换后”的口径（`p21_recheck_r0_5` 覆盖 `p21_seed012_*`）。

> 逐格明细（含 p21 步数、失败原因、来源 original/recheck）另见 `docs/BASE_REPEAT_REF_P21_30CELLS_20261111.csv`；最后一列 `base_result` 是空的，跑完直接填，填完即为配对表。

### card0 — OpenDoor（p21 参考 **11/15**）

| seed | r0 | r1 | r2 | r3 | r4 | 小计 |
|---|---|---|---|---|---|---|
| 0 | ❌ | ✅ | ❌ | ✅ | ✅ | 3/5 |
| 1 | ✅ | ❌ | ✅ | ✅ | ❌ | 3/5 |
| 2 | ✅ | ✅ | ✅ | ✅ | ✅ | 5/5 |

### card1 — Boxing（p21 参考 **10/15**）

| seed | r0 | r1 | r2 | r3 | r4 | 小计 |
|---|---|---|---|---|---|---|
| 0 | ✅ | ✅ | ❌ | ❌ | ✅ | 3/5 |
| 1 | ❌ | ✅ | ✅ | ✅ | ✅ | 4/5 |
| 2 | ✅ | ❌ | ✅ | ❌ | ✅ | 3/5 |

> p21 合计 **21/30 = 70%**。base 要能明显低于这个数才算“p21 有优势”。

---

## 4. 前置检查（不满足就别开跑）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
git rev-parse --short HEAD                      # 期望 4513b42
grep -c "EPISODE_SEED_BASE" scripts/c_wm6/c2_driver.py     # 期望 3
grep -c 'path == "/reseed"'  scripts/ha_vla_latent64_server.py  # 期望 1
ss -ltnp | grep -E ":1808[0-9]"                 # 期望为空（端口没被占）
```

**base 臂的正确性**（关键）：

- driver 的 `Servers._cmd()` 只在 `arm == "c"` 时追加 `--lora16-ckpt`；
- 因此 `--arms base` 时 server 命令行里**不应出现** `--lora16-ckpt`，日志里也**不应出现** `[server][lora16] loaded`；
- `run_meta.json` 里 `arms` 必须是 `["base"]`。注意 `lora_sha` 字段仍会记录默认的 `h10_critic`（b01e423f），**那是没被加载的默认值，不要拿它当过滤条件**。

---

## 5. 冒烟（先跑 1 集，300 步）

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh
export C2_SERVER_SEED=20261109
export C2_SERVERS=base C2_PORT_BASE=18080 C2_GPU_BASE=0 C2_ISAAC_GPU=0

python3 scripts/c_wm6/c2_driver.py --tasks OpenDoor --seeds 0 --repeat-start 0 --repeats 1 \
  --arms base --max-steps 300 --workers 1 --episode-timeout 900 --record-video 0 \
  --out runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4/_smoke
```

通过标志：① server 日志**没有** `[server][lora16] loaded`；② 有 `reseed` 行；③ `rc=0` + `result.json`。
通过后删掉 `_smoke` 目录再开正式跑。

---

## 6. 正式跑（两卡并行）

**card0（GPU0，OpenDoor）**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh
export C2_SERVER_SEED=20261109
export C2_SERVERS=base C2_PORT_BASE=18080 C2_GPU_BASE=0 C2_ISAAC_GPU=0
OUT=runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4
mkdir -p "$OUT"
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks OpenDoor --seeds 0 1 2 --repeat-start 0 --repeats 5 --arms base \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out "$OUT/card0" > "$OUT/card0.log" 2>&1 &
```

**card1（GPU1，Boxing）**

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
set +u; source scripts/harness03_env.sh
export C2_SERVER_SEED=20261109
export C2_SERVERS=base C2_PORT_BASE=18081 C2_GPU_BASE=1 C2_ISAAC_GPU=1
OUT=runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4
mkdir -p "$OUT"
nohup python3 scripts/c_wm6/c2_driver.py \
  --tasks Boxing --seeds 0 1 2 --repeat-start 0 --repeats 5 --arms base \
  --max-steps 3000 --workers 1 --episode-timeout 3600 --record-video 0 \
  --out "$OUT/card1" > "$OUT/card1.log" 2>&1 &
```

> 如果这台机器只有一张卡：把两条命令串行，card1 的 `C2_PORT_BASE` 改回 18080 且 `C2_GPU_BASE=0`。

---

## 7. 监控

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic-c-wm6
OUT=runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4
tail -f $OUT/card0.log $OUT/card1.log
find $OUT -name result.json | wc -l          # 期望 30
grep -c "rc=0" $OUT/card0.log $OUT/card1.log # 期望各 15
```

**每集落地后必须抽查 `episode_seed`**（保证和 p21 是同一格同一种子）：

```bash
python3 - <<'EOF'
import glob, json, hashlib
def derive(task, seed, rep):
    return int.from_bytes(hashlib.sha256(f"{task}|{int(seed)}|{int(rep)}".encode()).digest()[-4:], 'big') & 0x7FFFFFFF
bad = 0
for f in glob.glob('runs/dual_actor_wm6/c_lora/c2/base_repeat_s012_r0_4/**/result.json', recursive=True):
    r = json.load(open(f))
    if r.get('episode_seed') != derive(r['task'], r['seed'], r['repeat_idx']):
        bad += 1; print('MISMATCH', f)
print('mismatch =', bad)
EOF
```

---

## 8. 跑完怎么汇总（直接可用）

```bash
python3 - <<'EOF'
import glob, json
from math import comb
BASE = 'runs/dual_actor_wm6/c_lora/c2'

def load(dirs):
    out = {}
    for d in dirs:
        for f in glob.glob(f'{BASE}/{d}/**/result.json', recursive=True):
            r = json.load(open(f))
            if r.get('max_steps') != 3000: continue
            out[(r['model_label'], r['seed'], r['repeat_idx'])] = bool(r['success'])
    return out

p21 = {}
for d in ['p21_seed012_r0_5', 'p21_seed012_r6_11', 'p21_seed012_r12_17']:
    p21.update(load([d]))
p21.update(load(['p21_recheck_r0_5']))          # 重跑替换
base = load(['base_repeat_s012_r0_4'])

def mcnemar(w, l):
    n = w + l
    if n == 0: return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

for t in ['OpenDoor', 'Boxing', None]:
    ks = [k for k in p21 if k in base and (t is None or k[0] == t)]
    a = sum(p21[k] for k in ks); b = sum(base[k] for k in ks)
    w = sum(1 for k in ks if p21[k] and not base[k])
    l = sum(1 for k in ks if base[k] and not p21[k])
    print(f"{t or '合计':10s} n={len(ks):3d} p21={a}/{len(ks)} base={b}/{len(ks)} "
          f"p21-only={w} base-only={l} McNemar p={mcnemar(w, l):.3f}")
EOF
```

报告中必须同时给出：每格 4 种组合（都对/都错/只有 p21 对/只有 base 对）、两个分任务数字、以及**采样量限制**（30 对看得出 ≥30pp）。

---

## 9. 风险与告警

1. **容器/机器的批次效应是真的**：p21 那批里，容器0 的 17 个失败格重跑后 **8 格翻面（全是 ❌→✅）**。所以 base 这 30 集**必须在开机后确认机器干净**（无他人任务、`nvidia-smi` 只有本栈进程），并且**同一台机器跑完 30 集**，不要中途换机。
2. **别重跑失败格**：这会引入单边选择偏差（p21 那批已经吃过一次亏）。
3. **场景一致性**：`episode_seed` 必须逐集核对（§7 的脚本），不一致就说明 repeat 补丁没生效，整批作废。
4. **不要用 `lora_sha` 过滤 base**：base 的 `run_meta` 仍记录默认 LoRA 路径，它没被加载；用 `arm == "base"` 过滤。
5. **`--record-video 0`**：开着会拖慢并占盘，对判据没有价值。

---

## 10. 交付物清单

- [ ] `base_repeat_s012_r0_4/card0/`、`card1/` 各 15 个 `result.json`（共 30）
- [ ] 两份 `run_meta.json`（`arms=["base"]`、`max_steps=3000`）
- [ ] 两份 server 日志（确认**没有** `[lora16] loaded`）
- [ ] `episode_seed` 核对结果（mismatch = 0）
- [ ] §8 的配对汇总表（p21 vs base，含 McNemar）
- [ ] 一句话结论：**该协议下 p21 与 base 是否有可分辨的差异；若否，明确写“分不出”**

---

## 附：常见坑速查

| 现象 | 原因 | 处理 |
|---|---|---|
| server 日志出现 `[server][lora16] loaded` | 误用了 `--arms c` 或 `C2_SERVERS` 带了 c | 检查 `--arms base` 且 `C2_SERVERS=base` |
| 端口占用起不来 | 上一批 driver 没退干净 | `pkill -f c2_driver` 后重开，或换端口 |
| 某格 `episode_seed` 和公式不符 | repeat 补丁没打 / 用的不是本 worktree | 回 §4，整批作废 |
| 集数只有 29/30 | 有一集进程被杀没写 result | 补跑该格并标注 `rerun`，报告里保留原始记录 |
| 失败集特别多 | 正常：该任务失败就是 3000 步跑满，10–13 min | 不要中途 kill，等它自己结束 |
