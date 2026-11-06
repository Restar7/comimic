# GRU 版世界模型：接管审查 + 精度消融（2026-10-31）

> 问题：**给 WM 加一层 GRU 记忆，能不能把 k 步预测误差压下去？**
> 结论（先说）：**不能。** 在**同一把裁判**（134 条固定留出、动作语义正确、k=1/5/10/20）上，
> GRU 的**位置误差全面变差**（k=5 −44%、k=10 −43%、k=20 −44%），姿态也变差（−13%），
> 只有 29 维关节 RMSE 在 k=20 略好（−9%）。奖励核吃的是位置/姿态（位置 σ=0.2 m、tilt、yaw）
> ⇒ 对下游没有正面意义。而且"把隐状态用真实历史填满"之后它**更差**（见 §6）⇒
> **记忆里没有可用信息**，不是"接错线"。
>
> 关联：`docs/HANDOVER_WM_20261031.md`（§7 提出 GRU 假设）、`docs/ABLATION_SUITE.md`。

---

## 1. 接管时发现的实现问题（4 个真 bug + 2 个结构隐患）

上一轮把 GRU 实现交给了子代理，交接文档写"仅实现+自测+dry-run，未启动训练"。
实际接管时（16:35~17:00）逐条核过，发现：

| # | 问题 | 后果 | 修法 |
|---|---|---|---|
| 1 | `GRUWorldModel` 定义在 `scripts/finetune_wm_velocity_policy.py` 里，而**评估**走 `diff_vla_wbc.isaac_joint.load_isaac_bundle`（固定造 `WorldModel`） | 训练出来的是 GRU、**裁判读不了**：`load_state_dict(ck["wm"], strict=True)` 直接 `Unexpected key(s) gru.*` | GRU/构建逻辑提到共享模块 `diff_vla_wbc/wm_arch.py`；`load_isaac_bundle` 按 ckpt 的 `wm_arch` 键/权重键名自动造模型 |
| 2 | `bundle` 加载后 WM 被 `.eval()`，而 cuDNN 的 RNN **反向只允许 training 模式** | 第一次冒烟就 `RuntimeError: cudnn RNN backward can only be called in training mode` —— **GRU 根本训不了** | 训练循环每轮 `wm.train()`（自愈式，即使某个 eval 把它留在 eval 模式也不会静默跑偏） |
| 3 | 隐状态只在训练循环里显式传递；`imagine_with_tokens` / 闭环 rollout 调 `bundle.step` **不经过** `begin_unroll` | 想象/闭环时 GRU 每步都用 `h=0` ⇒ "训练带记忆、部署不带记忆"，静默不一致 | `IsaacJointBundle.step` 用 `step_idx` 当**生命周期信号**：`==1` 清零、`>1` 延续、`None` 无状态（旧调用方行为不变） |
| 4 | 续训 GRU ckpt 用 `strict=False` 加载 bundle | pi 缺键也会被静默吞掉（"没有报错但错了"） | 统一走 `load_isaac_bundle`（它现在自己判 arch），strict 恢复 |
| 5 | 结构代码两处（脚本 + 将来可能的评估侧） | 迟早分叉 | 单一来源 `diff_vla_wbc/wm_arch.py` |
| 6 | 复用 MLP 的 `in_norm/out_norm` 时只搬 `mean/var`、**没搬 `count`** | `count=0` 会让下一次 `update()` 用当前批统计把标定好的归一化冲掉（现在没人调 update，属于埋雷） | 连 `count` 一起搬（`count` 已到 freeze 阈值） |

**审查还确认了两个"配方事实"（不是 bug，但决定了 GRU 必输）**：

1. **阶段1/2 都是 K=1 ⇒ 循环矩阵 `weight_hh` 梯度严格为 0**。已在
   `scripts/test_wm_arch.py` 里写成断言：h≡0 时 `dL/dW_hh = 0`。也就是说 d3 配方的 7500 步里，
   **前 6000 步对 GRU 的记忆完全无效**——跑 6 千步还是 6 万步都一样。
2. **d3 自己的阶段 2b 就是负收益**：留出 val20 `0.1432(起点) → 0.1421(第100步，最好) → 0.1676(第1400步)`，
   早停只能取第 100 步 ⇒ 现有 20 步自回归目标与"20 步精度"是**错位**的。GRU 同样如此
   （`0.1614 → 0.1712 → 0.2040`），所以"最新的 ckpt"恰恰是最差的。

---

## 2. 改动清单

| 文件 | 改动 |
|---|---|
| `diff_vla_wbc/wm_arch.py`（新） | `GRUWorldModel`（单层 GRU 133→1024 + 头 1024→2048→2048→70）、`build_wm_arch`、`detect_wm_arch`、`load_wm_weights`、`report_wm_arch`、`wm_param_count` |
| `diff_vla_wbc/isaac_joint.py` | `self.wm` 按 ckpt 的 arch 构造；`step()` 按 `step_idx` 管理隐状态；`save_checkpoint` 写 `wm_arch`（仅非 mlp） |
| `scripts/finetune_wm_velocity_policy.py` | 删除重复定义改为 import 共享模块；`--arch {mlp,gru}`、`--clip-norm`、lr warmup；训练循环每轮 `wm.train()`；保存 `wm_arch` |
| `scripts/s3_pipeline.sh` | `WM_ARCH`（默认 mlp）、`WM_CLIP_NORM`，三阶段都透传 `--arch` |
| `scripts/eval_wm_multistep.py` | `--warmup/--warmup-mode {teacher,free}`（**记忆诊断**：先用真实历史把隐状态填满再量；对无状态模型逐位等价于 warmup=0） |
| `scripts/test_wm_arch.py`（扩展） | 见 §3 |
| `scripts/run_wm_gru_ablation.sh`（新） | WM 三阶段消融驱动（`K1/K2/ARCH/GPU/ROUND` 可配） |
| `scripts/compare_wm_eval.py`（新） | 多个评测 json → 一张 markdown 对比表（含 Δ%） |

---

## 3. 验证清单（`python3 scripts/test_wm_arch.py`，纯 CPU，全 PASS）

| 项 | 断言内容 |
|---|---|
| 参数量 | mlp **8,804,352**（与 `wm_d3.pt` 的 2D 矩阵口径一致）；gru **9,989,120（+13.46%）**，超出 ±20% 直接 `SystemExit` |
| 形状 | 两结构输出同一套 6-tuple；GRU 隐状态 `(1,B,1024)` |
| 记忆接线 | 换一个隐状态 ⇒ 输出必须变（防"假 GRU"） |
| **积分等价** | 同一个 70 维残差 `y` ⇒ MLP 与 GRU 的下一状态**逐位相同**（explicit 与 residual 两套都测）——防止"抄错一个转置" |
| **BPTT** | K=3 展开后 backward，梯度必须回到第 1 步动作；对照（每步清零 h）必须**零梯度** |
| **隐状态生命周期** | `bundle.step(step_idx=1)` 清零 / `>1` 延续 / `None` 无状态；MLP 路径对 `step_idx` 完全不敏感（逐位） |
| **ckpt 往返** | GRU 存档经 `load_wm_weights` strict 读回**逐位相等**；MLP 存档不会被误判成 GRU |
| 配方事实 | K=1 时 `weight_hh.grad` 严格为 0 |

**MLP 回归门**：改完所有代码后，用同一命令重跑 `wm_d3.pt` 的全池留出评测，
与改动前的 json 逐字段对比：**最大绝对差 = 0.0（逐位一致）** ⇒ 基线没被碰。

**裁判交叉验证**：训练内 `val_multistep`（7 条 SitSofa）与 `eval_wm_multistep`（134 条全池）
在 GRU 上给出同量级数字（0.1614 vs 0.16~0.17），说明两把尺子没有错位。

---

## 4. 实验设计（唯一变量是 arch）

数据/超参照抄 d3（`runs/S3_pipeline/d3/4a_wm_p1.log`、`4b_wm_p2.log`、`4_wm_refresh.log`）：

| 阶段 | 数据 | K | steps | batch | lr |
|---|---|---|---|---|---|
| 1 | 纯 A 线（WBC-only 真物理，3504 条） | 1 | 3000 | 256 | 1e-4 |
| 2 | A:B = 707:822 条（整批 128:128） | 1 | 3000 | 256 | 3e-5 |
| 2b | 同上，K=20 自回归 | 20 | 1500 | 128 | 1e-5 |

训练数据 = `runs/wm_real/traj_d3_train`（822 条 B 轨迹，与 d3 同一批）；早停留出 =
`runs/wm_real/traj_d3_val`（7 条 SitSofa）；**终局判分 = `runs/wm_real/traj_wmuse2_val`（134 条，从未参与任何训练/选点）**。

跑了三条线：
- `d4gru`：**同配方**（K=1,1,20）+ 2b 双档（`k20`=lr1e-5/1500 步；`mem`=lr3e-5/3000 步）
- `mem4`：**记忆友好配方**（阶段1/2 就用 **K=4**，让 `weight_hh` 从第一阶段起就有梯度）+ 2b
- `d3_k20last`：MLP 的 2b **跑满 1500 步不早停**（回答"用最新的会怎样"）

---

## 5. 终局结果（`eval_wm_multistep`，134 条留出、warmup=0、动作语义正确）

| ckpt | 结构/配方 | k=1 root | k=5 root | k=10 root | k=20 root | k=20 joint | k=20 rot |
|---|---|---|---|---|---|---|---|
| `wm_d3.pt` | MLP，2b 按关节早停（现状） | 0.1 cm | 0.9 | 2.4 | 6.5 cm | 0.144 | 13.9° |
| `wm_d3_p2.pt` | MLP，阶段2 产物 | 0.1 | 0.9 | 2.6 | 7.3 | 0.141 | 16.4° |
| **`wm_d3_k20last.pt`** | **MLP，2b 跑满 1500 步** | 0.1 | **0.8** | **2.0** | **5.7** | 0.158 | **12.4°** |
| `wm_d4gru_p2.pt` | GRU，阶段2（K=1 训不到记忆） | 0.1 | 1.4 | 4.2 | 11.2 | 0.133 | 17.2° |
| `wm_d4gru_mem.pt` | GRU，2b lr3e-5 / 早停@100 | 0.1 | 1.2 | 3.5 | 9.4 | 0.131 | 15.8° |
| `wm_d4gru_k20.pt` | GRU，2b lr1e-5 / 跑满 | 0.1 | 1.2 | 3.1 | 8.1 | 0.173 | 19.2° |
| `wm_mem4_p2.pt` | GRU，阶段1/2 改 K=4 | 0.2 | 1.1 | 3.1 | 8.8 | 0.224 | 23.0° |
| `wm_mem4_k20.pt` | GRU，K=4 + 2b 跑满 | 0.1 | 1.1 | 2.8 | 7.3 | 0.185 | 14.1° |

**best-vs-best（按 k=20 位置误差挑各自最好的）**

| | root k=20 | joint k=20 | rot k=20 |
|---|---|---|---|
| MLP `wm_d3_k20last` | **5.7 cm** | **0.158** | **12.4°** |
| GRU `wm_mem4_k20` | 7.3 cm（**+28%**） | 0.185（+17%） | 14.1°（+14%） |

⇒ **三个轴全输**，且 k=1 也不占优（GRU 最好 0.008 rad / 0.1 cm = MLP 相同，K=4 配方反而退化到 0.020）。
按交接文档的判据（k=1 ≤0.2 cm **且** k=20 root ≤3 cm）：MLP 5.7 cm、GRU 7.3 cm，**都不过**；
GRU 相对 MLP 是**净退步**，不是改进。

### 5.1 附带发现（对项目更有用）：现有 2b 早停指标把更好的 ckpt 扔了

d3 的 2b 在第 100 步就早停（当时 val20 关节 0.1421 最好），但**同一条训练轨迹**跑到 1500 步时，
位置误差从 **6.5 → 5.7 cm（−12%）**、姿态 **13.9° → 12.4°（−11%）**，代价是关节 0.144 → 0.158（+10%）。
原因是早停用的是**关节 RMSE**，而奖励/验收吃的是**位置与姿态**。两个指标在 2b 里是**反向**的：
关节单调变差、位置/姿态单调变好。
⇒ 建议：早停指标换成 `root + λ·rot`（或与奖励同口径），**不改任何超参就能白拿 −12% 位置误差**。
（这是 WM 侧可直接落地的改进；是否对闭环有正收益必须走阶段7 的真机/仿真验证，不能只看想象。）

---

## 6. 决定性诊断：把隐状态用真实历史填满（`--warmup 10 --warmup-mode teacher`）

做法：从窗口起点往前 10 帧开始，**喂真实状态**（只为填隐状态，不累积漂移），然后从真实起点量 k 步误差。
对 MLP 是**逐位零变化**（无状态）；对 GRU 是"记忆有没有用"的直接检验。

**MLP 在 warmup=0 与 warmup=10 下逐位一致**（起点集合也已固定成同一批窗口）⇒ 对照干净。
**GRU 反而变差**（k=20 位置 9.4→9.7 cm、k=10 3.5→3.6、k=5 1.2→1.3）⇒
**隐状态里没有可用信息**，多给真实历史只会多注入一点误差。

---

## 7. 归因与下一步

1. GRU 的**容量没变**（头 1024→2048→2048→70，只比 MLP 多 13.5% 参数），所以差异化只能来自记忆；
   而 §6 证明记忆无信息，§5 显示位置/姿态变差 ⇒ **"一阶无记忆导致长视界漂移"这个假设被否掉**。
   （注意 §5 里 GRU 的两个"最好"档是**不同配方**：`d4gru` 的 k=1 好但 k=20 差、`mem4` 的 k=20 好但 k=1 差，
   两者都不如 MLP 的 `wm_d3_k20last`，所以不存在"再调一下就能赢"的那一档。）
2. 真正的瓶颈仍是**数据规模/覆盖**与**目标函数错位**（K=20 自回归训练让留出 20 步误差**变差**，
   两个结构都一样）。按 `HANDOVER_WM_20261031.md` §8 的决策规则，应转
   **路线 A（数据放大：A 线 4900 万行）** 或 **路线 C（H=2 + critic 用真实回报）**，
   而不是继续给 GRU 加步数/加宽。
3. 一个**必须写明的让步**：MLP 基线是从**预训练过的** v6 WM 起跑（6000 步微调），GRU 是本轮从零重建
   （只复用输出层与归一化统计）。所以严格说这条对比不是"等算力"。但即便如此，GRU 也**没有**
   在位置/姿态上追平，且记忆诊断（§6）与预训练无关 —— 该结论不受此让步影响。

---

## 8. 复现命令

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic
# 同配方（K=1,1,20）
ROUND=d4gru ARCH=gru GPU=0 bash scripts/run_wm_gru_ablation.sh stage12
ROUND=d4gru ARCH=gru GPU=0 bash scripts/run_wm_gru_ablation.sh 2b k20 1500 1e-5
ROUND=d4gru ARCH=gru GPU=1 bash scripts/run_wm_gru_ablation.sh 2b mem 3000 3e-5
# 记忆友好配方（阶段1/2 用 K=4）
ROUND=mem4 ARCH=gru GPU=0 K1=4 K2=4 bash scripts/run_wm_gru_ablation.sh stage12
ROUND=mem4 ARCH=gru GPU=0 bash scripts/run_wm_gru_ablation.sh 2b k20 1500 1e-5
# 终局评测（134 条固定留出）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/eval_wm_multistep.py \
  --ckpt runs/S1_handover/wm_d3.pt runs/S1_handover/wm_d4gru_mem.pt \
  --dump "runs/wm_real/traj_wmuse2_val/*.pt" --horizon 20 --stride 20 --device cuda:0 \
  --out runs/S1_handover/eval_wm_gru_mem.json
# 记忆诊断
... --warmup 10 --warmup-mode teacher --out .../eval_wm_gru_warm10b.json
# 汇总成表
python3 scripts/compare_wm_eval.py --baseline wm_d3.pt runs/S1_handover/eval_wm_*.json
```
