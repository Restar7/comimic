# patch: SuperMimic provider 支持 `npy` token 回放（免相机/免 HTTP）

> 目的：本机 Isaac 渲染器不可用（`Graphics plugins not available`），而 `http` 模式必须传
> `front_rgb`。加一个 `npy` 模式，直接回放预先算好的 token 流，让 **GT / ORACLE / VLA 三臂
> 只差 token**（手部、起点、种子全同），并报真实路径长照。
>
> `third_party/HumanoidArena/` 被 gitignore，所以改动记录在本文件。若 GPFS 上文件被回滚，
> 按下面 diff 重新打一遍即可。

## 涉及文件

1. `third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_supermimic.py`
2. `third_party/HumanoidArena/isaaclab_twist2_g1/script/eval_scripts/sonic/sim_eval_vla.py`

## 1. provider：新增 `npy` token 源

### 1.1 `_make_token_source()` 末尾（`zeros` 分支之后）

```python
        if mode == "npy":
            path = getattr(self.args, "sm_token_npy", None) or _os.environ.get("SM_TOKEN_NPY", "")
            if not path:
                raise SystemExit("--sm_token_npy is required when --sm_token_source npy")
            arr = np.load(str(path)).astype(np.float32)
            if arr.ndim != 2:
                raise SystemExit(f"npy token stream must be (T,64), got {arr.shape}")
            hp = getattr(self.args, "sm_hands_npy", None) or _os.environ.get("SM_HANDS_NPY", "")
            hands = np.load(str(hp)).astype(np.float32) if hp else None
            print(f"[SuperMimic] npy token source: {path} tokens={arr.shape} "
                  f"hands={None if hands is None else hands.shape} (no camera, no HTTP)", flush=True)
            return ("npy", (arr, hands))
        raise SystemExit(f"unknown --sm_token_source {mode!r} (http|zeros|npy)")
```

### 1.2 `_next_latent()` 开头（`zeros` 分支之后）

```python
        if kind == "npy":
            return self._next_latent_replay(token_dim)
```

### 1.3 新增方法（放在 `_next_latent` 之前）

```python
    def _next_latent_replay(self, token_dim: int) -> torch.Tensor:
        """回放模式：按部署节奏（每 HA_VLA_EXEC_STEPS 步重规划，默认 20）推进光标。"""
        arr, hands = self.token_source[1]
        exec_steps = int(_os.environ.get("HA_VLA_EXEC_STEPS", "20") or 20)
        if not self._chunk:
            cur = int(getattr(self, "_np_cursor", 0))
            rows = arr[cur: cur + exec_steps]
            if rows.shape[0] < exec_steps:                     # clip 末尾：重复最后一帧
                rows = np.concatenate(
                    [rows, np.repeat(arr[-1:], exec_steps - rows.shape[0], axis=0)], axis=0)
            self._chunk = [torch.as_tensor(r[:token_dim], dtype=torch.float32, device=self.device)
                           for r in rows]
            if self._hand_enabled and hands is not None:
                hrows = hands[cur: cur + exec_steps]
                if hrows.shape[0] < exec_steps:
                    hrows = np.concatenate(
                        [hrows, np.repeat(hands[-1:], exec_steps - hrows.shape[0], axis=0)], axis=0)
                self._hand_queue = [np.asarray(r, dtype=np.float32) for r in hrows]
            self._np_cursor = cur + exec_steps
            self._chunks_fetched += 1
            self._chunk_popped = 0
        token = self._chunk.pop(0)
        self._chunk_popped = getattr(self, "_chunk_popped", 0) + 1
        return token.unsqueeze(0).expand(self.num_envs, -1)
```

### 1.4 `on_env_reset()` 里加一行

```python
        self._np_cursor = 0
```

## 2. `sim_eval_vla.py`：新增两个参数 + 放开 choices

```python
    parser.add_argument("--sm_token_source", type=str,
                        default=os.environ.get("SM_TOKEN_SOURCE", "http"),
                        choices=["http", "zeros", "npy"],
                        help="npy = 回放一段预先算好的 (T,64) token 流（不需要相机/渲染器）")
    parser.add_argument("--sm_token_npy", type=str, default=os.environ.get("SM_TOKEN_NPY", ""),
                        help="--sm_token_source npy 时读的 token 流 (.npy, (T,64) float32)")
    parser.add_argument("--sm_hands_npy", type=str, default=os.environ.get("SM_HANDS_NPY", ""),
                        help="同上，手部流 (.npy, (T,14))；三臂必须用同一份")
```

## 3. 校验（改完必须跑）

```bash
python3 -c "import ast;[ast.parse(open(f).read()) for f in (
 'third_party/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_supermimic.py',
 'third_party/HumanoidArena/isaaclab_twist2_g1/script/eval_scripts/sonic/sim_eval_vla.py')];print('OK')"
```

## 4. 口径约定（不可变）

| 项 | 值 | 理由 |
|---|---|---|
| `HA_VLA_EXEC_STEPS` | **20**（npy 模式默认）| 部署 receding horizon（`action_provider_sonic.py` 同值）|
| chunk 长度 | 40（服务端）| 实际执行前 20 |
| 手部 | **三臂同一份**（录制 `g1_hand_joints_14`）| 否则比的是手不是 token |
| token 维度 | 64（`--sm_token_dim`）| FSQ 冻结 encoder 输出 |
| 起点帧 | **三臂同帧**（由 seed 决定）| 唯一的变量是 token |

## 5. 已知限制

* 回放是**开环**的 token 流（不随观测变化）——这是为"三臂可比"付出的代价；
  在线环（VLA 实时推理）需要渲染器，见 `docs/S3_REAL_JUDGE_PLAN_20261027.md §4.G`。
* 光标到尾时**重复最后一帧**而不是报错，方便短 clip；日志里 `npy token source` 行会打印总长度，
  跑完请核对 `episode_steps ≤ T`。

---

## 6. 追加补丁（2026-10-27）：状态落盘 `SM_DUMP_STATES`

目的：给"真物理窗口评测 / on-policy 转移"提供逐帧状态，**奖励仍由
`scripts/compute_collect_reward.py` 用训练同一个 `bundle.reward` 补算**（口径不重写）。

* `action_provider_supermimic.py` 新增 `_record_state_dump()` / `_flush_state_dump()`；
* `get_action()` 末尾调用 record（在 `_step_physics` 之后）；`on_env_reset()` 里 flush 上一集；
* 环境变量 `SM_DUMP_STATES=<前缀>` 打开，产物 `<前缀>_ep000.pt` 含
  `states (N,71) asset 序` + `actions (N,29) canonical 序` + `frames` + `joint_names`；
* 布局：`[p(3), quat_wxyz(4), q(29), v(3), w(3), qd(29)]`。

**为什么不做在 `sim_eval_vla.py` 里**：那个文件 1000+ 行、且被 pin 契约锁哈希；
改 provider 的改动面更小、失败也不会影响评测（全部包在 try/except 里）。
