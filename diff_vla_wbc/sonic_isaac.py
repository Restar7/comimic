"""SONIC release decoder 作为 **Isaac 联合微调 bundle** 的执行器（2026-11-04）

为什么要这个文件
----------------
`diff_vla_wbc/sonic_mjlab.py` 已经把 SONIC 执行器接到了 **mjlab** bundle 上，但联合微调
（`v3_step` / `rollout`）实际用的是 **Isaac/trackerlab** bundle（`isaac_joint.py`），它的
``act`` 是 ``self.policy(obs, cond)`` = **SuperMimic WBC 策略**。

而我们的采集与部署都已经是 **SONIC release decoder**（2026-11-04 起）。如果微调还在用
SuperMimic 把 token 变成动作，那就是"训 A 部 B"：优化出来的 token 对部署没有意义。

本文件只做一层**薄适配**（与 `SonicMjlabBundle` 同构）：

* 复用 `SonicExecutor`（994 观测 → 可微 ONNX decoder → 29 维动作，token 侧有梯度）；
* 把 Isaac bundle 的 ``act`` 换成 SONIC decoder，其余**全部代理**（WM / 奖励 / FK / 参考库）。

三个口径必须对齐（都有实测依据）
--------------------------------
1. **观测**：Isaac obs 是 93 维/帧（`q−q0 | qd | ang_vel_b | gravity | a_prev`），
   可选 99 维（末尾 6 维 ref-relative 朝向，解码器不用）⇒ 每帧取前 93 维再展平。
2. **q 块**：SONIC decoder 要 `q − SONIC_DEFAULT_POS`（录制里逐位验证过，max|Δ|=0），
   而 Isaac obs 给的是 `q − bundle.default_joint_pos`（HA 口径 −0.2/0.42/−0.23 …）。
   两者差一个常数 ⇒ 构造时把 bundle 的 `default_joint_pos` 换成 **SONIC 默认角（STANDARD 序）**。
   `default_xi` 的编码器偏置为 0（`xi[:,1:30]=0`），所以 q 块就是 `q − SONIC_DEFAULT`。
3. **动作序**：`SonicExecutor.act` 返回 STANDARD 序（与 mjlab 一致）。
   ⇒ WM 的训练数据也必须用 **STANDARD 序动作**（脚本 `scripts/a_prime_actions_to_std.py`
   负责把 A' 的 SONIC/HA 序动作换算过去）。否则 WM 学到的动作通道与 rollout 用的不是一套。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .sonic_mjlab import DEFAULT_DECODER_ONNX, SonicExecutor, ha_to_std
from .compat import g1_default_joint_pos


class SonicIsaacBundle:
    """把任意 Isaac 联合微调 bundle 的执行器换成 SONIC decoder，其余能力全部代理。

    用法::

        base = load_isaac_bundle(wbc_ckpt, ...)      # 提供 WM / 奖励 / FK / 参考库
        bundle = SonicIsaacBundle(base, onnx_path=..., token_quant="none")
        # 之后 v3_step / rollout / train_v2 一行都不用改
    """

    def __init__(self, base: Any, *, onnx_path: str | Path | None = None,
                 token_quant: str = "none", device: str | torch.device | None = None,
                 set_default_to_sonic: bool = True):
        self.base = base
        dev = device or getattr(base, "device", "cpu")
        self.executor = SonicExecutor(onnx_path or DEFAULT_DECODER_ONNX,
                                      token_quant=token_quant, device=dev)
        self.obs_dim = int(getattr(base, "obs_dim", 0) or 0)
        self.hist = int(getattr(base, "hist", 10) or 10)
        # ---- 口径 2：把 bundle 的默认关节角换成 SONIC（STANDARD 序），让 obs 的 q 块
        #      正好等于 decoder 要的 `q − SONIC_DEFAULT_POS`。
        self._sonic_default_std = g1_default_joint_pos()
        self._orig_default = None
        if set_default_to_sonic:
            self._orig_default = self._swap_default(self._sonic_default_std)
        # ---- 口径 3（2026-11-05）：奖励里 `action_rate` 的 `(a−a_prev)*action_scale`。
        #   MDP 的 reward 有两处用动作：
        #     * `joint_limit` —— 用的是 **状态里的 q**（见 supermimic.py:219），**不吃动作**；
        #     * `action_rate` —— 用 `(a−a_prev)*action_scale` 把归一化动作差换成弧度。
        #   而我们的动作是 **SONIC 单位**（`q_target = SONIC_DEFAULT + a*SONIC_SCALE`），
        #   MDP 里存的 scale 是 **HA 口径**（两套不是置换关系，diff 最大 0.68）。
        #   ⇒ 需要把动作换算成 HA 口径，才能让 `Δq_target` 对得上物理量：
        #        a_ha = a_sonic * SONIC_SCALE / HA_SCALE        （default 相互抵消）
        #   影响面：`action_rate` 一项，实测在语料/闭环都为 0.0（动作变化很慢），故实际为 0 影响；
        #   但留着它就成了"下次踩坑的种子"，所以显式修掉。
        #   ★ 2026-11-06：SONIC 尺度不再读 `/tmp/sonic_scale.npy`（那是手工产物，
        #   文件不在就静默降级 ⇒ 单位换算会悄悄失效）。改为从 `joint_contract` 的置换现算，
        #   见 `diff_vla_wbc.sonic_mjlab.action_scale_sonic`。
        try:
            from .sonic_mjlab import action_scale_sonic  # noqa: PLC0415

            _sd_std = self._sonic_default_std
            _ss_sonic = action_scale_sonic().to(dtype=torch.float32, device=_sd_std.device)
            # ★ 2026-11-05 深夜：`self.base.spec` 在不同 bundle 里分别是 dict / RobotSpec 对象。
            #   以前直接 `.get()`（对象没有）⇒ 被 try/except 吞掉、`_action_conv=None`、
            #   单位换算**静默失效**（§4.2 的那个"已修"在训练侧其实没生效）。两种都支持。
            _spec = self.base.spec
            _sa = (_spec.get("action_scale") if hasattr(_spec, "get")
                   else getattr(_spec, "action_scale", None))
            if _sa is None:
                raise RuntimeError(f"spec 里没有 action_scale（type={type(_spec).__name__}）")
            _ha_scale = torch.as_tensor(
                __import__("numpy").asarray(_sa, dtype="float32"),
                dtype=torch.float32, device=_sd_std.device)
            self._action_conv = (_sd_std, _ss_sonic, _sd_std, _ha_scale)
        except Exception as _e:  # noqa: BLE001
            raise RuntimeError(
                f"[sonic-isaac][FATAL] 动作口径换算（SONIC↔HA）装不起来：{_e}；"
                "这一项缺失会让 reward 的 action_rate 用错尺度 —— 拒绝静默降级"
            ) from _e
        self.spec_extra = {
            "executor": "sonic_release_decoder",
            "decoder_onnx": self.executor.onnx_path,
            "token_quant": self.executor.token_quant,
            "obs_dim": self.obs_dim,
            "default_pose": "SONIC_DEFAULT_POS (STANDARD order)",
            "action_order": "STANDARD_29",
        }

    # ------------------------------------------------------------------ helpers
    # ★ 2026-11-04 关键口径：SONIC decoder 要的 q 块是 `q − SONIC_DEFAULT_POS`
    #   （录制的 `decoder_obs` 逐位验证过），而 Isaac 的 `mdp.obs()` 给出的是
    #   `q − self.default_q`（HA 口径：knee 0.420 / ankle −0.230，与 SONIC 的
    #   knee 0.669 / ankle −0.363 差一个常数）。**必须把 mdp.default_q 换成 SONIC 的**，
    #   否则 decoder 的输入整体偏移一个常数向量（token→动作就错了）。
    #   注意只换 `mdp.default_q`（观测用），不动 `bundle.default_joint_pos`
    #   （那是动作换算用的，FT 里动作由 executor 直接给，不需要换算）。
    _DEFAULT_ATTRS = ("default_q", "default_joint_pos")

    def _targets(self):
        out = []
        for name in ("mdp", "heads"):
            obj = getattr(self.base, name, None)
            if obj is None:
                continue
            for attr in SonicIsaacBundle._DEFAULT_ATTRS:
                if hasattr(obj, attr):
                    out.append((obj, attr))
        return out

    def _swap_default(self, new_default: torch.Tensor):
        olds = []
        for obj, attr in self._targets():
            old = getattr(obj, attr)
            olds.append((obj, attr, old))
            try:
                setattr(obj, attr, new_default.to(old.device).to(old.dtype))
            except Exception:  # noqa: BLE001
                pass
        if not olds:
            raise RuntimeError("[sonic-isaac] bundle 里找不到 default_q/default_joint_pos，"
                               "无法对齐 obs 的 q 块")
        print(f"[sonic-isaac] 默认关节角换成 SONIC 口径："
              f"{[(type(o).__name__, a) for o, a, _ in olds]}", flush=True)
        return olds

    def restore_default(self) -> None:
        for obj, attr, old in (self._orig_default or []):
            setattr(obj, attr, old)

    def _to_obs93(self, obs_hist_flat: torch.Tensor) -> torch.Tensor:
        """(B, hist*dim) → (B, hist*93)：每帧取前 93 维（丢掉 ref-relative 的 6 维）。"""
        d = self.obs_dim or (obs_hist_flat.shape[-1] // max(1, self.hist))
        if int(d) == 93:
            return obs_hist_flat
        b = obs_hist_flat.shape[0]
        x = obs_hist_flat.view(b, self.hist, int(d))
        if int(d) < 93:
            raise RuntimeError(f"[sonic-isaac] obs 每帧只有 {int(d)} 维（SONIC 需要 93）")
        return x[..., :93].reshape(b, self.hist * 93).contiguous()

    # ------------------------------------------------------------------ 执行器
    def act(self, obs_hist_flat: torch.Tensor, cond: torch.Tensor, ref: Any | None = None):
        """token (B,64) + 观测历史 → 动作 (B,29)，**STANDARD 序**，对 token 可微。"""
        obs93 = self._to_obs93(obs_hist_flat)
        return self.executor.act(cond, obs93)

    def step(self, state, action, xi, step_idx: int | None = None, **kw):
        """★ 2026-11-05（**真 bug 修复**）：喂给 WM 的动作必须先换回 HA/SONIC 序。

        为什么：WM（`wm_gain2` 等）是拿 **A' 语料的动作**训的，而语料的 `actions` 是
        decoder 原始输出 = **SONIC/HA 序**（实测 `mean|Δ|=0.000000` 对齐）。
        但 `act()` 返回的是 **STANDARD 序**（`SonicExecutor.act` 里 `index_select(ha_to_std)`），
        于是微调时 WM 收到的动作是**同一个张量被置换过**的版本（置换幅度 ≈1.24，
        而动作自身量级只有 0.945）⇒ WM 的"动作通道"完全错位。

        后果（实测）：想象里的漂移能稳定降 30%，真机一点不动 —— 因为优化器在
        **换序后的错误通道**上找方向。所有 2026-11-05 之前的联合微调结论因此作废。

        为什么在**这里**换、而不是"全局统一到 SONIC 序"：状态的关节序天生是 STANDARD
        （语料 / VLA 观测 / FK / 逐关节限位全按它），MDP 内部是
        `q_target = default_q + a*action_scale`。只把 default/scale 换成 SONIC 会让
        **q(STANDARD) 与 default(SONIC) 对不上**，撕裂 MDP。所以正确做法是在这个唯一
        边界上做显式适配：**WM 拿 SONIC 序、MDP 拿 STANDARD 序**。
        """
        act_wm = action
        if bool(getattr(self, "wm_wants_sonic_order", True)):
            act_wm = action.index_select(-1, self.executor._std_to_ha.to(action.device))
        fn = self.base.step
        try:
            return fn(state, act_wm, xi, step_idx=step_idx) if step_idx is not None \
                else fn(state, act_wm, xi)
        except TypeError:
            return fn(state, act_wm, xi)

    def reward(self, state, action, prev_action, reference, bodies=None):
        """★ 2026-11-05（**第二处口径债**）：奖励里的动作项必须换算到 MDP 的 HA 口径。

        为什么：`SonicExecutor.act` 返回的动作是 **SONIC 口径**
        （`q_target = SONIC_DEFAULT_POS + a * G1_ACTION_SCALE_ISAACLAB`，SONIC 序）；
        而 MDP 内部是 **HA/SuperMimic 口径**（`q_target = HA_default_q + a * HA_action_scale`，
        STANDARD 序）。**这两套常量不是置换关系**（实测 default 差 0.27 rad、
        scale 差 0.68），所以同一个动作向量在两边的物理含义不同。

        影响面（实测）：奖励主项（根/body 位置·朝向·速度、q、qd）**只从状态算**，不受影响；
        受影响的是 `joint_limit` 与 `action_rate` 两个小项（我们在语料上实测 0.0002 / 0.0）。

        换算：先换序（SONIC→STANDARD），再换单位：
            a_ha = (SONIC_DEFAULT + a_sonic*SONIC_SCALE − HA_default) / HA_scale
        这样 MDP 算出的 `q_target` 与 decoder 的物理目标**逐位一致**。
        """
        a_std = action
        if bool(getattr(self, "wm_wants_sonic_order", True)):
            # 到这里 action 已经是 STANDARD 序（act() 的输出）；先还原成 SONIC 口径的量
            a_sonic_units_std = action
        else:
            a_sonic_units_std = action
        if getattr(self, "_action_conv", None) is None:
            return self.base.reward(state, action, prev_action, reference, bodies)
        # ★ 2026-11-05 深夜：四个换算张量在构造时可能落在 CPU（默认姿态表），而动作在 cuda:X
        #   ⇒ 运行时按**动作的 device/dtype** 搬一次（以前 `_action_conv=None` 把这段整个跳过，
        #   所以这个 device 问题从没暴露过）。缓存避免每步都搬。
        s0, s1, d0, d1 = self._action_conv
        _dev, _dt = a_sonic_units_std.device, a_sonic_units_std.dtype
        _key = (_dev, _dt)
        if getattr(self, "_action_conv_key", None) != _key:
            self._action_conv_dev = tuple(x.to(device=_dev, dtype=_dt) for x in (s0, s1, d0, d1))
            self._action_conv_key = _key
        s0, s1, d0, d1 = self._action_conv_dev
        a_ha = (s0 + a_sonic_units_std * s1 - d0) / d1
        p_ha = prev_action
        if p_ha is not None:
            p_ha = (s0 + p_ha * s1 - d0) / d1
        return self.base.reward(state, a_ha, p_ha if p_ha is not None else a_ha, reference, bodies)

    # ------------------------------------------------------------------ 代理
    #: 只留在 wrapper 自己身上的属性；其余**赋值也要转发到 base**。
    #  ★ 2026-11-04 真 bug：`train_joint_v2` 在包了执行器之后还会写一堆属性
    #    （`bundle.reward_mode` / `huber_err_d_*` / `tail_weight` / `root_ori_mode` …），
    #    而 `bundle.reward()` 是 `__getattr__` 代理到 base、由 **base** 读这些属性。
    #    只做 `__getattr__` 不做 `__setattr__` ⇒ 赋值全落在 wrapper 上、base 看不见
    #    ⇒ DDP 子进程直接 exit 1（没有 traceback）。这里按名转发。
    _OWN_ATTRS = frozenset({
        "base", "executor", "obs_dim", "hist", "spec_extra",
        "_sonic_default_std", "_orig_default",
    })

    def __setattr__(self, name: str, value: Any) -> None:
        if name in SonicIsaacBundle._OWN_ATTRS or name.startswith("_") \
                or name in ("base", "executor"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.base, name, value)

    def __getattr__(self, name: str) -> Any:   # only when not found on self
        return getattr(self.base, name)

    def __getitem__(self, key: str) -> Any:
        return self.base[key]

    @property
    def spec(self) -> dict:
        sp = dict(getattr(self.base, "spec", {}))
        sp.update(self.spec_extra)
        return sp

    def save_checkpoint(self, path, **extra) -> None:
        extra = dict(extra)
        extra.setdefault("executor_meta", {"decoder_onnx": self.executor.onnx_path,
                                           "token_quant": self.executor.token_quant})
        return self.base.save_checkpoint(path, **extra)


__all__ = ["SonicIsaacBundle", "ha_to_std"]
