# base × repeat 扩展 · Phase A 结果（5 任务 r0–5）

> 生成：2026-11-12 06:55 ｜ 执行：base 线 ｜ 上游：`docs/BASE_REPEAT_PAIRED_HANDOFF_20261111.md`、`docs/BASE_REPEAT_PAIRED_RESULT_20261112.md`
> **一句话结论：5 任务 × seeds 0/1/2 × repeat 0–5 的 base 网格已补齐（90 格，0 缺失）。可配对的 72 格里 base 38/72 vs p21 36/72，McNemar p=0.804 —— 仍然分不出。PPBox 没有 p21 参考，base 5/18 = 27.8% 只能作基线。**

---

## 0. 速览

| 项 | 值 |
|---|---|
| 本阶段新增 | **60 集**（SitSofa/VisionNavi/PPBox × seeds 0/1/2 × r0–5 = 54；OpenDoor/Boxing × r5 = 6） |
| 累计网格 | **90 格**（5 任务 × 3 seeds × r0–5），0 缺失、0 重复、全部 `attempt=1`、`rc=0` |
| base r0–5 总成绩 | **43/90 = 47.8%** |
| 可配对格 | 72（OD/BX/SS/VN 各 18）；PPBox 18 格无 p21 |
| 配对结果（预注册口径） | base **38/72** vs p21 **36/72**，都对 29 / 都错 27 / 只 p21 对 7 / 只 base 对 9，**McNemar p=0.804** |
| 敏感性（纳入 recheck2） | p21 44/72 vs base 38/72，只 p21 对 13 / 只 base 对 7，p=0.263（仍不显著） |
| 校验 | `episode_seed` **90/90** 一致；15 + 2 份 `run_meta` 均 `arms=["base"]`、`max_steps=3000`；server 日志 `[lora16] loaded` = **0** |
| 本阶段耗时 | 01:23:17 → 06:41:13（**5h18m**，两卡并行）；单集中位 11.2 min（成功 5.1 / 失败 12.4） |
| Phase B | 已自动开始（06:41:13），180 集（5 任务 × 3 seeds × r6–17），预计 19:00–21:00 |

---

## 1. 5 任务 r0–5 的 base 成绩（90 格）

| 任务 | 成功/格数 | 成功率 | Wilson 95% CI | p21 有参考？ |
|---|---|---|---|---|
| OpenDoor | 13/18 | 72.2% | [49.1%, 87.5%] | ✅ 18/18 |
| Boxing | 13/18 | 72.2% | [49.1%, 87.5%] | ✅ 18/18 |
| VisionNavi | 7/18 | 38.9% | [20.3%, 61.4%] | ✅ 18/18 |
| SitSofa | 5/18 | 27.8% | [12.5%, 50.9%] | ✅ 18/18 |
| PPBox | 5/18 | 27.8% | [12.5%, 50.9%] | ❌ 无 |
| **合计** | **43/90** | **47.8%** | — | 72 格可配对 |

> 本阶段新增的 60 集里成功 20 集：PPBox 5/18、SitSofa 5/18、VisionNavi 7/18、OpenDoor r5 2/3、Boxing r5 1/3。

---

## 2. 配对结果（72 格，p21 预注册口径）

**p21 口径**：`p21_seed012_*` 为原始首跑；`p21_recheck_r0_5` 存在时覆盖（第一轮重跑替换，即 handoff 预注册口径）。**注意：`p21_recheck_r0_5` 只覆盖 OpenDoor/Boxing**，所以 SitSofa/VisionNavi 用的是原始首跑；`p21_recheck2_r0_5` 不在预注册口径内（见 §3）。

| 任务 | n | p21 | base | 都对 | 都错 | 只有 p21 对 | 只有 base 对 | McNemar p |
|---|---|---|---|---|---|---|---|---|
| OpenDoor | 18 | 14/18 | 13/18 | 11 | 2 | 3 | 2 | 1.000 |
| Boxing | 18 | 11/18 | 13/18 | 10 | 4 | 1 | 3 | 0.625 |
| SitSofa | 18 | 5/18 | 5/18 | 3 | 11 | 2 | 2 | 1.000 |
| VisionNavi | 18 | 6/18 | 7/18 | 5 | 10 | 1 | 2 | 1.000 |
| **合计** | **72** | **36/72** | **38/72** | **29** | **27** | **7** | **9** | **0.804** |

- 合计差 +2.8pp（base 高），不一致 16 对（7 vs 9）⇒ **分不出**。
- 采样量：72 对只够看 ≥20pp 量级的差；本次差 2.8pp，p=0.80。

---

## 3. 敏感性：p21 的第二轮重跑（`p21_recheck2_r0_5`）

`p21_recheck2_r0_5` 是**另一批"只重跑失败格"**的结果（不在 handoff 预注册口径内，且在我们 11-11 夜间跑 base 时仍在写盘）。它覆盖 SitSofa/VisionNavi/Boxing 的部分失败格：

| 口径 | p21 | base | 只有 p21 对 | 只有 base 对 | McNemar p |
|---|---|---|---|---|---|
| **预注册（主判据）** | 36/72 | 38/72 | 7 | 9 | **0.804** |
| 纳入 recheck2 | 44/72 | 38/72 | 13 | 7 | 0.263 |

- 两轮重跑共给 p21 增加 **8 格**（SitSofa +4、VisionNavi +3、Boxing +1），全部 ❌→✅。
- 这就是 handoff §2.2 禁止的"只重跑失败格"偏差；**主判据不能用它**，但必须把它写进报告：**若给 p21 这个红利，方向变成 p21 略高但不显著（p=0.263）**。
- 结论对口径选择是稳健的：两种口径都 **p>0.05，分不出**。

---

## 4. 校验（fail-closed）

| 检查 | 结果 |
|---|---|
| 集数 | 90/90 格（本阶段 60/60），0 缺失、0 重复 |
| 首跑一次 | 60/60 `attempt=1`；60/60 `rc=0` |
| `episode_seed` | **90/90** 与 sha256 公式一致（含 PPBox 的 `…Wholedoby` 命名） |
| `arms` | 15 份本阶段 `run_meta` 全部 `["base"]` |
| `max_steps` | 全部 3000 |
| LoRA | 15 份 base server 日志 `[server][lora16] loaded` = **0** |
| 运行状态 | Phase B 已在跑（两卡各 12 集/job） |

---

## 5. 覆盖缺口（Phase B 会用到）

| 任务 | p21 参考范围 | 无 p21 的格（Phase B 中） |
|---|---|---|
| OpenDoor | r0–17（r12–17 缺 4 格） | 4 格 |
| Boxing | r0–17 全 | — |
| SitSofa | **只到 r11** | r12–17（18 格） |
| VisionNavi | **只到 r13** | r14–17（16 格） |
| PPBox | **完全没有** | 全部 54 格 |

⇒ Phase B 的 180 格里，**106 格可配对**、74 格只能作 base 基线。报告会把两类分开，不会把无参考的格混进 McNemar。

---

## 6. 观察

1. **base 的强项/弱项很清楚**：OpenDoor/Boxing 72%、VisionNavi 39%、SitSofa/PPBox 28%。这和 `/reseed` 历史一致（OD 60%、BX 76%、VN 46%、SS 44%、PP 25%），没有异常。
2. **p21 唯一看起来更好的地方全靠 recheck2**：SitSofa 严格口径 5=5，纳入 recheck2 才 9/18；VisionNavi 6 vs 7（base 略高）。没有一处是 p21 用"首跑对首跑"赢下的。
3. **噪声底依然是大问题**：本阶段 60 集里 40 个失败全部跑满 3000 步（12.4 min/集），而 p21 的重跑能在同一格翻面 8 次——**单格结果的可重复性远低于我们想分辨的效应量**。这也再次支持"先量噪声底、再做方法实验"的路线。

---

## 7. 文件

| 文件 | 说明 |
|---|---|
| `runs/dual_actor_wm6/c_lora/c2/base_repeat_ext/phase_A_r0_5/` | 本阶段 60 集（15 个 job 目录，各含 `run_meta.json`、`driver.log`、`servers/base.log`） |
| `runs/dual_actor_wm6/c_lora/c2/base_repeat_ext/supervisor.log` / `progress.json` | 调度与进度 |
| `docs/BASE_REPEAT_EXT_R0_5_20261112.csv` | 90 格机读表（`p21`/`p21_steps`/`base_result`/`base_steps`；PPBox 的 p21 列留空） |
| `scripts/c_wm6/agg_base_repeat_full.py` | 全网格配对汇总（可复跑） |
| `scripts/c_wm6/base_repeat_ext_supervisor.py` | 断点续跑调度器 |

**Phase B 进行中**：5 任务 × seeds 0/1/2 × r6–17 = 180 集，预计 19:00–21:00 完成；完成后出全网格（270 格）汇总与最终报告。
