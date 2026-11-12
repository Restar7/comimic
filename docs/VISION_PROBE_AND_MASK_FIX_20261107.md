# 视觉信息探针 + mask 修复（2026-11-07）

> 回答两件事：① `train_distill_v3.py` 的 mask 缺没有真的污染过结论？
> ② 视觉特征里有没有 Δ头 拿不到的额外信息（= LoRA-on-action-head 的信息优势成立吗）？

---

## 1. mask 修复（`scripts/train_distill_v3.py`）

### 改了什么

* 新增 `--mask-frames`（默认 **20**，0 = 旧行为）。
* 逐帧模式下：损失只在**前 K 帧**上算；`cos` 也只在有效帧上平均；
  同时打印**旧口径**（全 40 帧平均）以便对照。
* ckpt meta 里记录该值（`mask_frames`）。

### 实测（`--per-frame --use-chunk all --use-history flat930`，6 epoch 冒烟）

```
[ep   5] loss=0.01813  held-out cos=+0.412  train cos=+0.422  (旧口径 +0.206)
```

**新口径 +0.412 / 旧口径 +0.206 = 正好 2×**，证实了"40 帧里 20 帧标签精确为 0 ⇒ 旧 metric 被稀释一半"。

### ⚠️ 但要说清：**这个 bug 没有污染任何已报出的数字**

查了两处真正产出过结论的脚本：

| 脚本 | 是否 mask | 产出 |
|---|---|---|
| `scripts/train_distill_temporal.py`（5 臂 B1–B5） | ✅ `K_EXEC=20` + `cos_masked` | 0.165/0.176/0.299/0.371/0.337 —— **口径正确** |
| `scripts/pretrain_v5_supervised.py`（v5 初始化） | ✅ `mask[:, :20]=1` | 0.397 —— **口径正确** |
| `scripts/train_distill_v3.py` **逐帧路径** | ❌ 之前缺 | **从未产出过结论数字**（S/R 两臂都是 `per_frame=false`，mask 不适用） |

⇒ 结论：**5 臂架构排序（B4 0.371 最好）仍然有效**，不需要作废；
修复只是让 `train_distill_v3.py` 的逐帧路径今后不再出错。

---

## 2. 视觉信息探针（`scripts/probe_vision_information.py`）

### 问题

`token_chunk` 本来就是 VLA 从画面推出来的 ⇒ 视觉信息**已经被压进 token**。
真正要问的是：**DiT 直接吃到的 2048 维 `backbone_features` 里，还有多少 token 里没有的？**
（这是"把 Δ 换成 action-head LoRA"唯一的信息学理由。）

### 做法

* 特征取自**部署同一条路径**：`processed_batch` → `vlm.prepare_input` → `vlm.backbone`
  → `action_head._encode_features(...).backbone_features` → 均值池化成 2048 维。
* 样本：n=1998 条标签，按 4 个族分层（open_door / sit_sofa / vision_navi / double_desk 各 333）。
* 留出：**族内留出 30% 的示教运动**（= 部署真实情形），cos 只在非零标签帧上算。
* 对照：把视觉特征**按行打乱**后拼上（同容量、零信息），用来吸收"多加维度就会涨一点"的容量效应。

### 结果

| 特征 | 平均 cos（4 族） |
|---|---|
| 现有（token+obs930+state64） | **0.212** |
| 仅视觉（原始 2048） | 0.060 |
| 现有 + 视觉 PCA-16 | 0.216（打乱对照 0.210）⇒ **Δ +0.005** |
| 现有 + 视觉 PCA-32 | 0.216（打乱对照 0.209）⇒ **Δ +0.007** |
| 现有 + 视觉 PCA-64 | 0.209（打乱对照 0.202）⇒ **Δ +0.007** |

### 结论

> **单帧、均值池化的视觉特征里，没有 Δ头 拿不到的额外方向信息** ——
> 增量 +0.005~0.007，与"打乱后的同容量对照"完全重叠。

⚠️ **口径边界（必须写清）**：
1. 只用了**当前单帧**的 ego view；若信息在**帧间**（运动方向/目标接近速度），本探针看不到。
2. 视觉特征被**均值池化**成 2048 维，丢掉了空间位置；LoRA 的 cross-attn 可以看具体 token。
3. n=1998（每族 ~500）⇒ 这里的 base 0.212 是**容量受限**的数（全量 38,400 时是 0.341），
   所以"绝对水平"不能引用，只有**与打乱对照的差**有意义。

---

## 3. 由此得到的判断

1. **"Δ head 缺信息 ⇒ 加视觉就能变好"这个假设，在单帧口径下不成立。**
   视觉信息已经在 token 里了；LoRA 不会因为"多看了一眼画面"而获得新信息。
2. LoRA 真正会改变的是**策略分布**（同样的画面 → 不同的 token），而不是"Δ 的信息量"。
   而"在我们的数据上再适配一轮 action head"这件事，已经用**全参微调**做过
   （`checkpoint-8000` 的 `tune_diffusion_model=true`，就是 61% 的 base），
   再叠一轮是 v13（30.8%）—— 负收益。
3. 还剩唯一没验证的环节：**Gate B（token 空间的方向是否上升）**。
   这条不依赖视觉，也不依赖 H，是整条 Δtoken 链路的前提。

## 4. 复现

```bash
# 特征抽取（GPU，约 11 min / 2000 条）
PYTHONPATH="$PWD:$PWD/Isaac-GR00T:$PWD/third_party/py312_shims:$PWD/third_party/gr00t_deps:$PWD/third_party/gr00t_deps/_noflash" \
  python3 scripts/probe_vision_information.py extract --n 2000 --batch 16 --out runs/vision_probe
# 对比（含打乱对照）
python3 scripts/probe_vision_information.py probe --dir runs/vision_probe
```

产物：`runs/vision_probe/vision_feats.npz`（1998×2048）、`runs/vision_probe/vision_probe.json`、
日志 `runs/vision_probe_extract.log`、`runs/vision_probe/probe2.log`。
