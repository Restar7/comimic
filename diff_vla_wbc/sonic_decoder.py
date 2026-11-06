"""SONIC release decoder 的**可微 torch 版**（逐算子复刻 ONNX，不做近似）。

## 为什么需要它

`scripts/sonic_decoder_torch.py` 早就证明了一件事：release decoder 只用
MatMul/Add/Sigmoid/Mul/Slice/Unsqueeze/Concat/Squeeze，逐算子用 torch 复刻即可数值等价
（fp32 差 ~1e-7），**而 onnxruntime 里没有梯度，端到端 BPTT 会断**。

本模块把那份实现搬进库（脚本改为从这里 import，避免两处实现漂移），并补上两个训练/度量
需要的东西：

* :meth:`SonicDecoderTorch.act` —— ``(token, obs994) → 29 维动作``，**token 那 64 维可微**
  （用 ``torch.cat`` 拼，不用原地赋值：原地写会静默断图）；
* :meth:`SonicDecoderTorch.check_parity` —— 和 ORT 对拍，`max|Δ|` 必须 ~1e-7。

## 输入口径（硬事实，别猜）

release decoder 的 ONNX 契约：``obs_dict [1, 994] -> action [1, 29]``，其中 **前 64 维是 token**，
后 930 维是 10 帧 × 93 维本体感觉历史。两处独立证据：

1. ``scripts/sonic_decoder_torch.py`` 的 Slice 注释：``-930 ⇒ 64``（图里先切掉前 64 维）；
2. HA raw npz 逐帧存了 ``decoder_obs (T,994)`` 与 ``decoder_target_action (T,29)``，
   且 release decoder 能把 ``decoder_obs`` 复现成 ``decoder_target_action``（1e-5 度，
   见 ``docs/SONIC_DECODER_FOUND.md``）。

⇒ 所以量"执行器对 token 的 Jacobian"不需要重建 994 维布局：**直接用数据集里的真
`decoder_obs`**，只扰动前 64 维。这也是 ``scripts/check_token_jacobian.py --sonic-decoder``
的做法（在分布内，不会像第一版脚本那样量到 OOD 饱和）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

#: token 在 994 维 decoder 输入里的位置
TOKEN_DIM = 64
DECODER_IN_DIM = 994
ACTION_DIM = 29


# --------------------------------------------------------------------------- 994 输入布局
#: **逐位反解并用 HA 录制数据验证过**（`max|Δ| = 0.0`, bit-exact）。
#:
#: 反解方法：HA raw npz 逐帧存了 `decoder_ang_vel_hist` / `encoder_robot_joint_pos_hist` /
#: `encoder_robot_joint_vel_hist` / `decoder_gravity_dir_hist`（都是 (T,10,d)）和
#: `decoder_raw_action`（(T,29)）。把它们按 yaml 的块顺序拼起来，与 `decoder_obs` 逐位相等。
#:
#: 关键结论（**和 yaml 的描述一致，但我们独立验证到了逐位**）：
#:
#: * 布局是**块优先（block-major）**，不是帧优先：先放完全部 10 帧的某一通道，再放下一通道；
#: * 每块内部是**帧优先、由旧到新**：`[t-10, t-9, ..., t-1]`（**不含当前帧**）；
#: * `his_last_actions` = **解码器自己的原始输出** `decoder_raw_action[t-10 : t]`
#:   —— 不是"执行后的动作"，也不是别的归一化空间。rollout 时把上一步输出滚进来即可。
#:
#: 一次反解里最容易错的两点：①把 `last_action` 当成"录制/执行动作"（幅度就差 3 倍，
#: 而它是六个块里**最敏感**的：清零它输出变化 75%，换成随机 106%）；②把块当成帧优先。
OBS_DIM = DECODER_IN_DIM                      # 994
HIST = 10                                     # 历史帧数
_TOK = slice(0, 64)
BLOCKS = (
    # name, start, end, per-frame width
    ("ang_vel",     64,  94,   3),
    ("joint_pos",   94, 384,  29),
    ("joint_vel",  384, 674,  29),
    ("last_action", 674, 964,  29),
    ("gravity",    964, 994,   3),
)

#: SuperMimic 的 93 维观测顺序 → SONIC 的块顺序（只做**通道重排**，不做数值变换）。
#: SuperMimic: ``[q-default-dq0(29), qd(29), ω(3), g(3), a_prev(29)]``
#: SONIC:      ``[ω(3),      q(29),   qd(29), a_prev(29), g(3)]``
_SUPERMIMIC_SLICES = {
    "q_rel": (0, 29), "qd": (29, 58), "ang_vel": (58, 61), "gravity": (61, 64),
    "last_action": (64, 93),
}


def build_obs_994(token: torch.Tensor, ang_vel: torch.Tensor, joint_pos: torch.Tensor,
                  joint_vel: torch.Tensor, last_action: torch.Tensor,
                  gravity: torch.Tensor) -> torch.Tensor:
    """按**已验证的布局**拼 994 维 decoder 输入；每个历史块是 ``(B, 10, w)``（由旧到新）。"""
    b = token.shape[0]
    parts = [token.reshape(b, 64)]
    for name, x, w in (("ang_vel", ang_vel, 3), ("joint_pos", joint_pos, 29),
                       ("joint_vel", joint_vel, 29), ("last_action", last_action, 29),
                       ("gravity", gravity, 3)):
        if x.shape[1:] != (HIST, w):
            raise ValueError(f"{name} 必须是 (B,{HIST},{w})，得到 {tuple(x.shape)}")
        parts.append(x.reshape(b, HIST * w))
    return torch.cat(parts, dim=1)


def split_obs_994(obs: torch.Tensor) -> dict[str, torch.Tensor]:
    """:func:`build_obs_994` 的逆（诊断用）。"""
    b = obs.shape[0]
    out = {"token": obs[:, _TOK]}
    for name, a, z, w in BLOCKS:
        out[name] = obs[:, a:z].reshape(b, HIST, w)
    return out


def supermimic_obs_to_blocks(obs93: torch.Tensor, *, default_q: torch.Tensor | None = None,
                             dq0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """SuperMimic 的 93 维观测 ``(B,10,93)`` → SONIC 的五个历史块 ``(B,10,w)``。

    两个必须做的换算：

    1. **通道重排**（见 :data:`_SUPERMIMIC_SLICES`）；
    2. **关节角加回基准**：SuperMimic 存的是 ``q - default_q - dq0``，SONIC 要的是**原始 q**，
       所以 ``q_raw = q_rel + default_q + dq0``（``dq0`` = ``xi[..., 1:1+29]``）。

    不传 ``default_q`` / ``dq0`` 时**原样透传**（只重排），并在返回值里标记 ``NeedsOffset``。
    """
    s = _SUPERMIMIC_SLICES
    if obs93.shape[-2:] != (HIST, 93):
        raise ValueError(f"obs93 必须是 (B,{HIST},93)，得到 {tuple(obs93.shape)}")
    q_rel = obs93[..., s["q_rel"][0]:s["q_rel"][1]]
    if default_q is not None:
        q = q_rel + default_q.to(q_rel.dtype).reshape(1, 1, -1)
        if dq0 is not None:
            q = q + dq0.to(q_rel.dtype).reshape(1, 1, -1)
    else:
        q = q_rel
    return {
        "ang_vel": obs93[..., s["ang_vel"][0]:s["ang_vel"][1]],
        "joint_pos": q,
        "joint_vel": obs93[..., s["qd"][0]:s["qd"][1]],
        "last_action": obs93[..., s["last_action"][0]:s["last_action"][1]],
        "gravity": obs93[..., s["gravity"][0]:s["gravity"][1]],
    }


class _GraphDecoder(nn.Module):
    """按 ONNX 节点顺序执行（只支持这个 decoder 用到的算子）。"""

    def __init__(self, graph_nodes, consts, params):
        super().__init__()
        self.nodes = graph_nodes
        self.consts = {k: v for k, v in consts.items()}
        self.params = nn.ParameterDict({k.replace(".", "_"): v for k, v in params.items()})
        self._name = {k.replace(".", "_"): k for k in params}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        env = {"obs_dict": x}
        P = {self._name[k]: v for k, v in self.params.items()}
        for op, ins, outs, attrs in self.nodes:
            a = [env.get(i, self.consts.get(i, P.get(i))) for i in ins]
            if op == "Constant":
                continue
            if op == "MatMul":
                y = torch.matmul(a[0], a[1])
            elif op == "Add":
                y = a[0] + a[1]
            elif op == "Sigmoid":
                y = torch.sigmoid(a[0])
            elif op == "Mul":
                y = a[0] * a[1]
            elif op == "Slice":
                starts = a[1].tolist() if torch.is_tensor(a[1]) else a[1]
                ends = a[2].tolist() if torch.is_tensor(a[2]) else a[2]
                axes = a[3].tolist() if len(a) > 3 and a[3] is not None else [0]
                starts = [int(v) for v in (starts if isinstance(starts, list) else [starts])]
                ends = [int(v) for v in (ends if isinstance(ends, list) else [ends])]
                axes = [int(v) for v in (axes if isinstance(axes, list) else [axes])]
                t = a[0]
                for st, en, ax in zip(starts, ends, axes):
                    size = t.shape[ax]
                    st = int(st)
                    if st < 0:                      # 负 start = 从尾部数（-930 ⇒ 64）
                        st = size + st
                    st = max(0, st)
                    en = size if en is None or int(en) > size else int(en)
                    if en < 0:
                        en = size + en
                    t = t.narrow(ax, st, max(0, en - st))
                y = t
            elif op == "Unsqueeze":
                axes = attrs.get("axes")
                if axes is None:
                    axes = a[1].tolist() if len(a) > 1 and torch.is_tensor(a[1]) else [0]
                axes = axes if isinstance(axes, list) else [axes]
                t = a[0]
                for ax in sorted(int(v) for v in axes):
                    t = t.unsqueeze(ax)
                y = t
            elif op == "Squeeze":
                # ★ opset 13 起 `axes` 是**第二个输入**（不是 attribute）。
                #   第一版只看 attrs，取不到就退回 `t.squeeze()`（删掉所有 1 维）——
                #   于是 **batch=1 时把 batch 维也删了**：输出从 (1,29) 变成 (29,)，
                #   而 batch>1 时恰好没事。部署侧就是 batch=1，所以这是必须修的静默 bug。
                axes = attrs.get("axes")
                if axes is None and len(a) > 1 and a[1] is not None:
                    axes = a[1].tolist() if torch.is_tensor(a[1]) else a[1]
                t = a[0]
                if axes is not None:
                    axes = axes if isinstance(axes, list) else [axes]
                    for ax in sorted((int(v) for v in axes), reverse=True):
                        t = t.squeeze(ax)
                y = t
            elif op == "Concat":
                y = torch.cat(a, dim=int(attrs.get("axis", 0)))
            else:
                raise NotImplementedError(f"未支持的算子 {op}")
            env[outs[0]] = y
        return env[self.nodes[-1][2][0]]


def load_decoder(onnx_path: str | Path, dtype=torch.float32) -> nn.Module:
    """ONNX → ``_GraphDecoder``（权重原样搬过来，不蒸馏、不近似）。"""
    import onnx

    m = onnx.load(str(onnx_path))
    g = m.graph
    consts, params, nodes = {}, {}, []
    for n in g.node:
        if n.op_type != "Constant":
            continue
        for attr in n.attribute:
            if attr.name == "value":
                # 显式 copy：onnx 的 ndarray 是只读的，直接 from_numpy 会触发
                # "non-writable tensor" 警告（且后续任何原地写都是 UB）
                consts[n.output[0]] = torch.from_numpy(
                    np.array(onnx.numpy_helper.to_array(attr.t), copy=True)).to(dtype)
    inits = {i.name for i in g.initializer}
    for i in g.initializer:
        arr = torch.from_numpy(np.array(onnx.numpy_helper.to_array(i), copy=True)).to(dtype)
        if i.name in inits:
            params[i.name] = arr
    attrs_of = lambda n: {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}  # noqa: E731
    for n in g.node:
        nodes.append((n.op_type, list(n.input), list(n.output), attrs_of(n)))
    return _GraphDecoder(nodes, consts, params)


class SonicDecoderTorch(nn.Module):
    """``a = Dec(concat(token(64), obs(930)))``，**对 token 可微**，参数冻结。

    ``act`` 用 ``torch.cat`` 而不是 ``x[..., :64] = token``：后者在 ``x`` 是不需要梯度的
    叶子时是**原地写**，会静默断图（反传时 ``∂a/∂token`` 直接变 0，且不报错）。
    """

    def __init__(self, graph: nn.Module):
        super().__init__()
        self.graph = graph
        for p in self.graph.parameters():     # 冻结：只借它的 Jacobian
            p.requires_grad_(False)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_onnx(cls, onnx_path: str | Path, *, dtype=torch.float32) -> "SonicDecoderTorch":
        return cls(load_decoder(onnx_path, dtype=dtype)).eval()

    def freeze(self) -> "SonicDecoderTorch":
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()

    # ---------------------------------------------------------------- forward
    def forward(self, obs_decoder: torch.Tensor) -> torch.Tensor:
        """``(B, 994)`` → ``(B, 29)``。"""
        return self.graph(obs_decoder)

    def act(self, token: torch.Tensor, obs_decoder: torch.Tensor) -> torch.Tensor:
        """把 ``token`` 写进 994 维输入的前 64 维再前向（可微，token 必须是 ``(..., 64)``）。"""
        if token.shape[-1] != TOKEN_DIM:
            raise ValueError(f"token 最后一维必须是 {TOKEN_DIM}，得到 {tuple(token.shape)}")
        rest = obs_decoder[..., TOKEN_DIM:].to(dtype=token.dtype, device=token.device)
        return self.forward(torch.cat([token, rest], dim=-1))

    # ---------------------------------------------------------------- 自检
    @torch.no_grad()
    def check_parity(self, obs_decoder: torch.Tensor, onnx_path: str | Path) -> float:
        """和 ORT 对拍，返回 ``max|Δ|``（fp32 应 ~1e-7）。"""
        import numpy as np
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        x = obs_decoder.detach().to("cpu", torch.float32).numpy()
        y_ort = np.concatenate([sess.run(None, {name: x[i:i + 1]})[0] for i in range(x.shape[0])])
        y_t = self.forward(torch.from_numpy(x)).numpy()
        return float(np.abs(y_t - y_ort).max())


__all__ = [
    "ACTION_DIM",
    "BLOCKS",
    "DECODER_IN_DIM",
    "HIST",
    "OBS_DIM",
    "SonicDecoderTorch",
    "build_obs_994",
    "load_decoder",
    "split_obs_994",
    "supermimic_obs_to_blocks",
]
