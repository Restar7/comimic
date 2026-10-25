# SONIC decoder 找到了，而且是 release 真身（2026-10-25 验证）

> 更正 `docs/HANDOVER_LATEST_FINDINGS_FOR_4090_20261025.md` §1.1 里那句
> 「官方解码器权重（`model_decoder.onnx`）**不在本工作区**（release 目录只有 yaml）→ 所以我们用记录蒸馏替代」。
> 权重一直在，只是找错了地方。**以后优先用真权重，`runs/sonic_decoder_final.pt` 只作离线对照。**

---

## 1. 权重在哪（三处，只有一处是对的）

| 位置 | 内容 | 判定 |
|---|---|---|
| `/inspire/.../wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/` | `model_decoder.onnx` 39 MB、`model_encoder.onnx` 50 MB、`policy_model_decoder.trt` 41 MB | ✅ **release 真身，HA 录制用的就是它** |
| 同上 `.../policy/low_latency/` | decoder 150 MB、encoder 46 MB、TRT 150 MB | ⚠️ 另一个模型（low-latency 变体），**残差 5°** |
| HF cache `$HF_HOME/hub/models--nvidia--GEAR-SONIC/snapshots/9c0ff…/low_latency/` | 只有 low_latency 三件套 + `planner_sonic.onnx` | ⚠️ **没有 release**；`ha_isaac_closedloop.sh` 原来默认找的就是这里 |
| `comimic/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/` | 只有 3 个 `observation_config*.yaml` | ❌ 仓库内嵌副本不含权重（这就是"没找到"的来源） |
| `wsc-workspace/SONIC_DATA/deployment/` | 没有 `gear_sonic_deploy/policy/` | ❌ 同上 |

ONNX 契约（实测 `onnxruntime` 加载）：

```
release      model_encoder.onnx  obs_dict [1,1762] -> encoded_tokens [1,64]
release      model_decoder.onnx  obs_dict [1, 994] -> action         [1,29]
low_latency  model_encoder.onnx  obs_dict [1,1247] -> encoded_tokens [1,64]
low_latency  model_decoder.onnx  obs_dict [1, 994] -> action         [1,29]
```

HumanoidArena 的 `action_provider_sonic.py` 按 **1762** 维拼 encoder 输入、按 **994** 维拼 decoder 输入
（`_latest_encoder_input = np.zeros((1762,))`），所以它本来就是按 **release** 写的。

---

## 2. 怎么验的：拿 HA 自己的录制数据对拍（可复现）

HA raw npz 里逐帧留了 `decoder_obs (T,994)` 和 `decoder_target_action (T,29)`，还有
`meta_decoder_path = …/policy/release/model_decoder.onnx`。把 `decoder_obs` 直接灌进两个 ONNX：

```python
q = DEFAULT_ANGLES + decoder_out * ACTION_SCALE      # groot_repro/g1_sim.py
```

再对每一关节做一次「`target_j = a_j * q_j + b_j`」的最小二乘，看残差：

| 权重 | 文件数 | 每关节残差 MAE | 最差单关节 |
|---|---|---|---|
| `release/model_decoder.onnx` | 8（4 任务） | **0.00001°** | 0.0000° |
| `low_latency/model_decoder.onnx` | 8（4 任务） | 5.0° | 19.5° |

结论：**release decoder 与 HA 录制逐帧吻合到浮点噪声级**（1e-5 度），
`b_j` 恰好是 `DEFAULT_ANGLES` 的一个排列、`a_j` 是 1（少数关节是 pair 的 scale 比），
说明 994 维布局、`q-默认角`、`isaaclab↔mujoco` 关节顺序三处都对了。low_latency 是**另一个模型**，不是口径标定问题。

复现脚本（一次性）：

```bash
cd $COMIMIC && python3 - <<'PY'
import numpy as np, glob, onnxruntime as ort, sys; sys.path.insert(0,".")
from groot_repro.g1_sim import ACTION_SCALE
f=sorted(glob.glob("data/ha_raw/HOI_double_desk/*/*.npz"))[0]
d=np.load(f,allow_pickle=True); obs=d["decoder_obs"].astype(np.float32); tgt=d["decoder_target_action"].astype(np.float32)
for tag in ("release","low_latency"):
    s=ort.InferenceSession(f"/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/{tag}/model_decoder.onnx",providers=["CPUExecutionProvider"])
    out=np.concatenate([s.run(None,{s.get_inputs()[0].name:obs[i:i+1]})[0] for i in range(min(len(obs),300))])*ACTION_SCALE
    e=[np.abs(tgt[:len(out),j]-np.polyval(np.polyfit(out[:,j],tgt[:len(out),j],1),out[:,j])).mean() for j in range(29)]
    print(tag, "residual MAE = %.5f deg" % np.degrees(np.mean(e)))
PY
```

---

## 3. 顺手修掉的一个口径 bug：闭环脚本原来用的是 low_latency

`scripts/ha_isaac_closedloop.sh` 原来默认

```bash
SONIC_DIR=$(ls -d "$HF_HOME"/hub/models--nvidia--GEAR-SONIC/snapshots/*/low_latency | head -1)
```

即 2026-10-24 那 6 条 Isaac 闭环**全部**用 low_latency decoder 执行
（日志可查：`runs/ha_isaac_batch_20261024_141023/runs/*.log` → `…/low_latency/model_decoder.onnx`），
而我们的 VLA 是在 release token 空间上微调的（HA `encoder_latent` 出自 release encoder）。
已改为**优先 release、找不到才退回 low_latency 并打 WARN**。

> 只是"少一个口径错误"，不等于闭环就站得住；10-24 六条仍然全 fall（35–61 步），
> 那批产物在 `runs/ha_arena_videos_sonic/`。

---

## 4. 当前容器能不能出闭环视频

**能**，这条链路 2026-10-24 就在本容器跑通过，产物是 `runs/ha_arena_videos_sonic/*.mp4`：

| 件 | 位置 / 状态 |
|---|---|
| Isaac Sim 5.0 | `third_party/isaacsim_env`（5.0.0 + IsaacLab `third_party/isaaclab_220` release/2.2.0） |
| 驱动 | 570.124.06（5.0 兼容；595 会段错误，见 `HANDOVER_ISAAC_CLOSEDLOOP.md`） |
| 资产 | `third_party/HumanoidArena/isaaclab_twist2_g1/assets` 4.8 GB |
| VLA ckpt | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000`（`runs/OFFICIAL_CKPT.json` 锁定的官方档） |
| decoder | `wsc-workspace/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx` |
| onnxruntime | 只有 CPU EP（decoder 2.5 ms/帧，够用），`--device cuda:0` 只影响 Isaac |

跑法（官方 decoder 线，必须显式打开 SONIC，项目默认是 SuperMimic）：

```bash
cd $COMIMIC
# 1) VLA 服务（另一张卡或等训练结束）
python3 scripts/ha_vla_latent64_server.py --ckpt runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000 --port 18080 &
# 2) Isaac 闭环（出 mp4）
HA_ALLOW_SONIC=1 TASK=DoubleDesk MAX_STEPS=300 bash scripts/ha_isaac_closedloop.sh
# 产物：$OUT_ROOT/DoubleDesk__seed0__repeat0__<stamp>/{videos_success,videos_failure}/*.mp4
```

**现在的阻塞不是 decoder，是显存**：本容器只有 1 张 4090（49 GB），此刻
`scripts/train_joint.py`（PID 68125）占了 39.7 GB，只剩 ~9 GB —— Isaac 渲染 + 3B VLA 放不下。
训练停掉/换卡后即可直接出片。

---

## 5. 两条路各自需要什么（避免再混）

| | 官方微调线（基线） | 联合微调线（主线） |
|---|---|---|
| 执行器 | **SONIC release decoder**（994→29） | **SuperMimic WBC**（token 条件化，我们训的） |
| 接线 | `--sonic_decoder_path` + `HA_ALLOW_SONIC=1` → `SonicActionProvider` | 默认 `supermimic_wholebody` → `SuperMimicActionProvider` |
| 需要 decoder？ | ✅ 需要（已就位、已验证） | ❌ 不需要 |
| 现状 | 能出视频，0/6 成功（全 fall） | 在训：`--motion data/npz/ha_tokens`，PID 68125 |
| 训练产物 | `runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000` | `runs/joint_ha_a1/…`（见 `HANDOVER_MAINLINE_AND_ABLATIONS_20261025.md`） |
