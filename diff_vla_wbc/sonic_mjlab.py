"""SONIC release decoder as the WBC executor of the mjlab joint-finetune bundle.

为什么要这个文件
----------------
2026-11-04 起，WBC **不再重训 SuperMimic**，改用官方 **SONIC release decoder**（冻结）。
但联合微调的主干（`v3_step` / `rollout` / `train_v2` / `mjlab_joint`）是围绕
``MjlabJointBundle`` 写的：它只通过 ``act(obs_hist_flat, cond) -> action`` 和
``step(state, action)`` 访问"执行器 + 动力学"。所以这里只做**一层薄适配**：

* ``SonicExecutor``   : 93×10 观测 → SONIC 994 → decoder → 29 维动作（可微，token 侧有梯度）；
* ``SonicMjlabBundle``: 把 ``MjlabJointBundle`` 的 ``act`` 换成 SONIC，其余**全部代理**。

两个空间必须逐项对齐（已核对，见 §"口径"）：

===========================  ====================  ==========================
项                            mjlab / SuperMimic    SONIC release decoder
===========================  ====================  ==========================
默认关节位                    `g1_default_joint_pos` = `SONIC_DEFAULT_POS`
                              (-0.312 / 0.669 / -0.363 / 0.6 / ±0.2)
动作尺度                      `0.25·effort/stiffness`  **同一个公式**
kp / kd                       `stiffness` / `damping`  同一张电机表（并联关节 ×2）
关节序                        `STANDARD_29`         `HA_V31_29`（左右交错）
===========================  ====================  ==========================

⇒ **动作空间完全同底**，换执行器只需要一次关节序置换（``joint_contract.permutation``）。
这正是 `docs/MUJOCO_SONIC_JOINTFT_PLAN_20261104.md` §0 表里那一行"同底"的依据。

观测口径（两侧的硬事实）
------------------------
mjlab 93 维/帧（term-major 历史，每项 10 帧连续）::

    [q-q0(29) | qd(29) | base_ang_vel_b(3) | projected_gravity(3) | a_prev(29)]

SONIC 994::

    [token(64) | ang_vel 10×3 | q-q0 10×29 | qd 10×29 | last_action 10×29 | gravity 10×3]

⇒ 通道要重排（``supermimic_obs_to_blocks``），q/qd/last_action 三块要换序（STANDARD → HA_V31），
q 块用的就是 ``q - q0``（**不要再加回 default**），这与
`docs/DEBUG_WBC_DECODER_20261103.md` §6/§8 的逐位结论一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .compat import g1_default_joint_pos
from .sonic_decoder import (
    HIST,
    OBS_DIM as SONIC_OBS_DIM,
    SonicDecoderTorch,
    build_obs_994,
    supermimic_obs_to_blocks,
)
from .mjlab_joint import OBS_DIM, TERM_DIMS

DEFAULT_DECODER_ONNX = Path(
    "/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/"
    "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
)

__all__ = [
    "DEFAULT_DECODER_ONNX",
    "SonicExecutor",
    "SonicMjlabBundle",
    "action_scale_sonic",
    "obs930_to_obs93",
    "std_to_ha",
    "ha_to_std",
]


# ---------------------------------------------------------------------------- 顺序
def std_to_ha() -> list[int]:
    """``v_ha = v_std[std_to_ha()]``（STANDARD_29 → HA_V31_29）。"""
    import joint_contract as jc

    return jc.permutation(src="standard", dst="ha_v31")


def ha_to_std() -> list[int]:
    """``v_std = v_ha[ha_to_std()]``（HA_V31_29 → STANDARD_29）。"""
    import joint_contract as jc

    return jc.permutation(src="ha_v31", dst="standard")


def action_scale_sonic() -> "torch.Tensor":
    """**SONIC 序**的动作尺度（rad / 归一化动作单位），29 维。

    为什么需要它（2026-11-05 口径 bug 4.2）：动作是 SONIC 单位
    （``q_target = SONIC_DEFAULT_POS + a × SONIC_SCALE``），而 MDP 的 ``action_rate``
    用 ``(a − a_prev) × action_scale``（HA 口径），两套表不是逐项相等。

    真值来源：SONIC 的 ``G1_ACTION_SCALE_ISAACLAB`` 与 HA 的 ``G1_ACTION_SCALE``
    **是同一张电机表**（`0.25·effort/stiffness`），差别只在关节序 ——
    实测 ``SONIC_SCALE == G1_ACTION_SCALE[std_to_ha()]`` 逐位为 0 误差。

    以前这里读 `/tmp/sonic_scale.npy`（一个 /tmp 里的手工产物），文件不在就静默降级
    ⇒ 换机器/清 /tmp 后单位换算会**悄悄失效**。现在从 `joint_contract` 的置换现算，
    没有外部文件依赖。
    """
    from .compat import G1_ACTION_SCALE  # noqa: PLC0415

    perm = torch.as_tensor(std_to_ha(), dtype=torch.long)
    return G1_ACTION_SCALE.cpu().float()[perm]


# ---------------------------------------------------------------------------- 观测
def obs930_to_obs93(obs_hist_flat: torch.Tensor) -> torch.Tensor:
    """mjlab 的 term-major ``(B, 930)`` → 逐帧 ``(B, 10, 93)``。

    ``flatten_history_term_major`` 是**按项**拼的（每项 10 帧连续），所以不能直接
    ``view(B, 10, 93)`` —— 那样会把第 10 帧的关节角当成第 1 帧的关节速度。
    """
    if obs_hist_flat.shape[-1] != HIST * OBS_DIM:
        raise ValueError(
            f"obs_hist_flat 必须是 (B,{HIST * OBS_DIM})，得到 {tuple(obs_hist_flat.shape)}"
        )
    b = obs_hist_flat.shape[0]
    out = obs_hist_flat.new_zeros(b, HIST, OBS_DIM)
    src = 0
    dst = 0
    for width in TERM_DIMS:
        out[..., dst:dst + width] = obs_hist_flat[:, src:src + HIST * width].reshape(b, HIST, width)
        src += HIST * width
        dst += width
    return out


def _quantize(token: torch.Tensor, mode: str) -> torch.Tensor:
    """FSQ 格点化（``none`` / ``ste`` / ``hard``），与两侧已接的开关同口径。"""
    if mode == "none":
        return token
    q = (token * 16.0).round().clamp(-16, 15) / 16.0
    if mode == "hard":
        return q
    if mode == "ste":
        return token + (q - token).detach()
    raise ValueError(f"unknown token_quant mode: {mode}")


# ---------------------------------------------------------------------------- 执行器
class SonicExecutor(torch.nn.Module):
    """``token + 930 维观测历史`` → ``29 维动作``（与 mjlab 动作空间同底，STANDARD 序）。

    ``act`` 保持对 ``token`` 可微（decoder 参数冻结），这是联合微调反传的唯一通路。
    """

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        *,
        dtype: torch.dtype = torch.float32,
        token_quant: str = "none",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.onnx_path = str(onnx_path or DEFAULT_DECODER_ONNX)
        self.decoder = SonicDecoderTorch.from_onnx(self.onnx_path, dtype=dtype)
        self.decoder.freeze()
        # ``_GraphDecoder.consts`` 是普通 dict（不是 buffer），``.to()`` 不会搬它们；
        # 不搬的话 CUDA 上 matmul 会报 "mat2 is on cpu"。参数（initializer）走 ParameterDict 会自动搬。
        for k, v in list(self.decoder.graph.consts.items()):
            if torch.is_tensor(v):
                self.decoder.graph.consts[k] = v.to(device)
        self.decoder.to(device)          # ParameterDict（initializer 权重）走 nn.Module.to
        self._device = torch.device(device)
        self.token_quant = token_quant
        self._ha_to_std = torch.tensor(ha_to_std(), dtype=torch.long, device=device)
        self._std_to_ha = torch.tensor(std_to_ha(), dtype=torch.long, device=device)

    def to_device(self, device) -> "SonicExecutor":
        """把 decoder 的参数与常量（``consts`` 不是 buffer，得手动搬）一起搬到 ``device``。"""
        device = torch.device(device)
        for k, v in list(self.decoder.graph.consts.items()):
            if torch.is_tensor(v):
                self.decoder.graph.consts[k] = v.to(device)
        self.decoder.to(device)
        self._ha_to_std = self._ha_to_std.to(device)
        self._std_to_ha = self._std_to_ha.to(device)
        self._device = device
        return self

    # ---------------------------------------------------------------- forward
    def obs994(self, obs_hist_flat: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        """mjlab 930 → SONIC 994（token 只占位，真正可微的 token 由 :meth:`act` 拼）。"""
        obs93 = obs930_to_obs93(obs_hist_flat)
        blocks = supermimic_obs_to_blocks(obs93)  # q 已是 q - q0（bias=0）
        idx = self._std_to_ha.to(obs_hist_flat.device)
        return build_obs_994(
            token,
            blocks["ang_vel"],
            blocks["joint_pos"].index_select(-1, idx),
            blocks["joint_vel"].index_select(-1, idx),
            blocks["last_action"].index_select(-1, idx),
            blocks["gravity"],
        )

    def act(self, token: torch.Tensor, obs_hist_flat: torch.Tensor) -> torch.Tensor:
        """``token (B,64)`` + ``obs (B,930)`` → ``action (B,29)``（STANDARD 序）。"""
        if token.shape[-1] != 64:
            raise ValueError(f"token 最后一维必须是 64，得到 {tuple(token.shape)}")
        tok = _quantize(token, self.token_quant)
        obs = self.obs994(obs_hist_flat, torch.zeros_like(tok))
        if obs.device != self._device:
            self.to_device(obs.device)      # 跟随观测（cuda vs cuda:0 这种等价写法不要报错）
        a_ha = self.decoder.act(tok.to(obs.dtype).to(obs.device), obs)
        return a_ha.index_select(-1, self._ha_to_std.to(a_ha.device))

    def raw_action_to_ha(self, action: torch.Tensor) -> torch.Tensor:
        """mjlab/SuperMimic 动作（STANDARD 序）→ SONIC 原始动作（HA_V31 序）。"""
        return action.index_select(-1, self._std_to_ha.to(action.device))


# ---------------------------------------------------------------------------- bundle
class SonicMjlabBundle:
    """把任意 ``MjlabJointBundle`` 的执行器换成 SONIC，其余能力全部代理。

    用法（`train_joint_mjlab.py` 里只需换构造）::

        base = MjlabJointBundle.from_checkpoint(wbc_ckpt, ...)   # 提供 WM / 奖励 / 参考库
        bundle = SonicMjlabBundle(base, onnx_path=..., token_quant="none")
        # 之后 v3_step / rollout / train_v2 一行都不用改
    """

    def __init__(self, base: Any, *, onnx_path: str | Path | None = None,
                 token_quant: str = "none", device: str | torch.device | None = None):
        self.base = base
        dev = device or getattr(base, "device", "cpu")
        self.executor = SonicExecutor(onnx_path, token_quant=token_quant, device=dev)
        self.default_joint_pos = g1_default_joint_pos()
        #: 口径自检：动作空间必须同底（不同底就必须先做换算，不允许静默）
        base_scale = getattr(getattr(base, "assets", None), "action_scale", None)
        if base_scale is not None:
            from .compat import G1_ACTION_SCALE  # noqa: PLC0415

            delta = (base_scale.detach().cpu().float() - G1_ACTION_SCALE.cpu().float()).abs().max()
            if float(delta) > 1e-5:
                raise ValueError(
                    f"动作尺度不一致（max|Δ|={float(delta):.3g}）：SONIC 与 mjlab bundle 不同底，"
                    "必须先实现 q_target 换算再继续"
                )

    # ---------------------------------------------------------------- 执行器
    def act(self, obs_hist_flat: torch.Tensor, cond: torch.Tensor, ref: Any | None = None):
        return self.executor.act(cond, obs_hist_flat)

    def step(self, state, action, xi, step_idx: int | None = None, **kw):
        """``rollout`` 会传 ``step_idx``（Isaac 侧有相位相关项），mjlab 的 WM 不用 ⇒ 吃掉。"""
        base_step = self.base.step
        try:
            return base_step(state, action, xi, step_idx=step_idx) if step_idx is not None else base_step(state, action, xi)
        except TypeError:
            return base_step(state, action, xi)

    # ---------------------------------------------------------------- 代理
    def __getattr__(self, name: str) -> Any:  # only called when not found on self
        return getattr(self.base, name)

    def __getitem__(self, key: str) -> Any:
        return self.base[key]

    # ---------------------------------------------------------------- 记录口径
    @property
    def spec(self) -> dict:
        spec = dict(getattr(self.base, "spec", {}))
        spec["executor"] = "sonic_release_decoder"
        spec["decoder_onnx"] = self.executor.onnx_path
        spec["token_quant"] = self.executor.token_quant
        spec["joint_order"] = "STANDARD_29 (bundle) / HA_V31_29 (decoder)"
        return spec

    def save_checkpoint(self, path, **extra) -> None:
        extra = dict(extra)
        extra.setdefault("executor_meta", {"decoder_onnx": self.executor.onnx_path,
                                           "token_quant": self.executor.token_quant})
        return self.base.save_checkpoint(path, **extra)
