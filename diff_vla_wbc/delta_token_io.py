"""Δtoken checkpoint 的**唯一**加载 / 特征构造契约（训练侧保存、部署侧加载共用）。

为什么要这个模块
----------------
`DeltaTokenNet` 的输入宽度有两套历史口径，而 checkpoint 里的
``delta_token_meta`` 一直**没有** ``feat_dim``（见 ``v3_step.py::save_vla``）：

===============  ====================================  ==================
feat_dim         特征                                   出现时期
===============  ====================================  ==================
35                ``rot6d(6) + dof_vel(29)``             2026-11-06 之前
64                ``state64 = rot6d(6)+dof_pos(29)+dof_vel(29)``  第 2 刀
===============  ====================================  ==================

部署端如果默认按 35 建网，64 维 checkpoint 会在 ``load_state_dict`` 上
**尺寸不匹配直接报错**（`strict=False` 不会吞掉 shape mismatch），但报错信息是
``size mismatch for net.0.weight``，看不出"这是 feat 口径问题"；
若反过来（64 建网 + 35 checkpoint）同理。

更隐蔽的一条：**特征里 29 维关节块的顺序**。训练侧的 ``batch.start_state`` 来自
``runs/isaac_motion/ha_cmd/packed.pt``，它的 q/qd 是 **MJCF/STANDARD_29** 序
（``scripts/build_ha_command_lib.py`` 把 encoder 命令流按 ``argsort(mjcf_to_isaaclab_perm)``
转回 MJCF 序；数据侧独立复核：packed 的 q 与 ``data/ha_raw`` 的
``vla_state_dof_pos_29`` 在 25/29 维上按该置换对齐，相关系数最高 0.99）。
而部署端 ``state64[6:35] / [35:64]`` 是 **HA_V31_29**（= isaaclab/asset 序，
与 ``robot_qpos_before_decimation`` 逐位相等）。

⇒ 部署侧喂给 Δ 的 29 维关节块必须按 ``ha_v31 → standard`` 置换，否则训练和部署
看到的不是同一个特征（35 维口径 29/35 维错序，64 维口径 58/64 维错序）。

本模块把这件事收敛到一处：**推断 → 严格校验 → 构造特征**，任何不一致当场 FAIL，
绝不静默退回 35 维或退回底模（那会把配置错误伪装成"Δ 没效果"）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .delta_token import FEAT_DIM, STATE64_DIM, TOKEN_DIM, DeltaTokenNet, apply_norm_cap

#: 训练侧关节序（packed motion library / trackerlab / MuJoCo MJCF）。
TRAIN_JOINT_ORDER = "standard"
#: 部署侧 ``state64`` 关节序（HumanoidArena asset / isaaclab）。
DEPLOY_JOINT_ORDER = "ha_v31"

#: ``feat_dim`` → 特征块布局（**这是契约，不是提示**）。
FEAT_LAYOUTS = {
    int(FEAT_DIM): "rot6d+dof_vel29",      # 35
    int(STATE64_DIM): "state64",           # 64
}

CTAIL_KEYS = ("delta_token_state", "delta_token_meta")


class DeltaTokenContractError(RuntimeError):
    """Δtoken checkpoint 与部署口径不一致（必须让调用方 FAIL，不许降级）。"""


# --------------------------------------------------------------------------- spec
@dataclass(frozen=True)
class DeltaTokenSpec:
    """一份 Δtoken checkpoint 的完整输入契约 + 每个字段的来源（可审计）。"""

    token_dim: int
    feat_dim: int
    hidden: int
    depth: int
    init_std: float
    cap: float
    feat_joint_order: str          # 特征里关节块的顺序
    state_joint_order: str         # 入站 state64 关节块的顺序
    layer_shapes: tuple[tuple[int, ...], ...]
    sources: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- 派生属性 ----
    @property
    def feat_layout(self) -> str:
        return FEAT_LAYOUTS[int(self.feat_dim)]

    @property
    def n_params(self) -> int:
        return sum(int(np.prod(s)) for s in self.layer_shapes) + sum(
            int(self.layer_shapes[i][0]) for i in range(len(self.layer_shapes)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_dim": int(self.token_dim),
            "feat_dim": int(self.feat_dim),
            "feat_layout": self.feat_layout,
            "hidden": int(self.hidden),
            "depth": int(self.depth),
            "init_std": float(self.init_std),
            "cap": float(self.cap),
            "feat_joint_order": self.feat_joint_order,
            "state_joint_order": self.state_joint_order,
            "layer_shapes": [list(s) for s in self.layer_shapes],
            "sources": dict(self.sources),
        }


# --------------------------------------------------------------------------- 推断
def _q(name: str) -> str:
    return str(name).replace(".", "_")


def _linear_layers(sd: Mapping[str, Any]) -> list[str]:
    """按 ``net.<idx>.weight`` 排序返回 Linear 权重名（``depth+1`` 层）。"""
    names = [k for k in sd if k.startswith("net.") and k.endswith(".weight")]
    if not names:
        raise DeltaTokenContractError(
            "[delta-token] state dict 里没有 net.*.weight —— 这不是 DeltaTokenNet 的权重")

    def _idx(k: str) -> int:
        try:
            return int(k.split(".")[1])
        except (IndexError, ValueError) as exc:  # pragma: no cover - 防御
            raise DeltaTokenContractError(f"[delta-token] 无法解析层号：{k!r}") from exc

    names.sort(key=_idx)
    return names


def infer_delta_token_spec(
    sd: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
    *,
    state_joint_order: str = DEPLOY_JOINT_ORDER,
    feat_joint_order: str | None = None,
) -> DeltaTokenSpec:
    """从权重形状 + （可选）metadata 推断契约；**任何冲突都 FAIL**。

    规则（顺序固定，先 metadata、后形状，冲突即报错）::

        token_dim  : meta.token_dim 或 64（TOKEN_DIM 契约）
        feat_dim   : meta.feat_dim  或 in_dim − token_dim
        hidden     : meta.hidden    或 net.0.weight.shape[0]
        depth      : meta.depth     或 Linear 层数 − 1
    """
    meta = dict(meta or {})
    src: dict[str, str] = {}

    weights = _linear_layers(sd)
    shapes = [tuple(int(v) for v in sd[k].shape) for k in weights]
    if len(shapes) < 2:
        raise DeltaTokenContractError(
            f"[delta-token] 只有 {len(shapes)} 个 Linear 层，DeltaTokenNet 至少 2 层")

    in_dim = shapes[0][1]
    hidden_w = shapes[0][0]
    hidden_meta = meta.get("hidden")
    if hidden_meta is not None and int(hidden_meta) != int(hidden_w):
        raise DeltaTokenContractError(
            f"[delta-token] metadata.hidden={int(hidden_meta)} 与权重 net.0.weight[0]="
            f"{int(hidden_w)} 不一致 —— metadata 与 state dict 不是同一份 checkpoint")
    hidden = int(hidden_w)
    src["hidden"] = "meta+weights" if hidden_meta is not None else "weights"

    depth_inferred = len(shapes) - 1
    depth_meta = meta.get("depth")
    if depth_meta is not None and int(depth_meta) != depth_inferred:
        raise DeltaTokenContractError(
            f"[delta-token] metadata.depth={int(depth_meta)} 与权重层数 "
            f"{depth_inferred} 不一致 —— metadata 与 state dict 不是同一份 checkpoint")
    depth = depth_inferred
    src["depth"] = "meta+weights" if depth_meta is not None else "weights"

    token_dim_meta = meta.get("token_dim")
    if token_dim_meta is not None:
        token_dim = int(token_dim_meta)
        src["token_dim"] = "meta"
    else:
        token_dim = int(TOKEN_DIM)
        src["token_dim"] = "default(TOKEN_DIM=64)"
    if token_dim != int(TOKEN_DIM):
        raise DeltaTokenContractError(
            f"[delta-token] token_dim={token_dim} ≠ 部署契约 {int(TOKEN_DIM)}"
            "（服务端永远是 64 维 token chunk）")

    feat_inferred = int(in_dim) - int(token_dim)
    feat_meta = meta.get("feat_dim")
    if feat_meta is not None:
        feat_dim = int(feat_meta)
        if feat_dim != feat_inferred:
            raise DeltaTokenContractError(
                f"[delta-token] metadata.feat_dim={feat_dim} 与权重推断 "
                f"{feat_inferred}（in_dim {in_dim} − token {token_dim}）不一致")
        src["feat_dim"] = "meta+weights"
    else:
        feat_dim = feat_inferred
        src["feat_dim"] = "weights(in_dim−token_dim)"
    if feat_dim not in FEAT_LAYOUTS:
        raise DeltaTokenContractError(
            f"[delta-token] 推断出的 feat_dim={feat_dim} 不是已知口径 {sorted(FEAT_LAYOUTS)}"
            f"（in_dim={in_dim}, token_dim={token_dim}）")

    # 结构：每层 (out, in) 必须首尾相接；最后一层输出 token_dim
    for i, shp in enumerate(shapes):
        if len(shp) != 2:
            raise DeltaTokenContractError(f"[delta-token] 第 {i} 层权重不是二维：{shp}")
        if i and shp[1] != shapes[i - 1][0]:
            raise DeltaTokenContractError(
                f"[delta-token] 第 {i} 层 in={shp[1]} 与上一层 out={shapes[i - 1][0]} 不连")
    if shapes[-1][0] != token_dim:
        raise DeltaTokenContractError(
            f"[delta-token] 输出宽度 {shapes[-1][0]} ≠ token_dim {token_dim}")
    if shapes[0][1] != token_dim + feat_dim:
        raise DeltaTokenContractError(
            f"[delta-token] 首层输入宽度 {shapes[0][1]} ≠ token {token_dim} + feat {feat_dim}")

    # 特征块关节序：metadata（会话 02 会写）> 训练侧历史约定（packed 库 = MJCF 序）
    fj = (feat_joint_order or meta.get("feat_joint_order")
          or meta.get("feat_order") or meta.get("joint_order"))
    if fj is not None:
        src["feat_joint_order"] = "meta/arg"
        fj = str(fj)
    else:
        src["feat_joint_order"] = f"default(training convention={TRAIN_JOINT_ORDER})"
        fj = TRAIN_JOINT_ORDER
    if fj not in ("standard", "ha_v31"):
        raise DeltaTokenContractError(f"[delta-token] 未知 feat_joint_order={fj!r}")
    sj = str(meta.get("state_joint_order") or state_joint_order)
    src["state_joint_order"] = ("meta" if meta.get("state_joint_order") else "arg/default")
    if sj not in ("standard", "ha_v31"):
        raise DeltaTokenContractError(f"[delta-token] 未知 state_joint_order={sj!r}")

    cap = float(meta.get("cap", 0.0) or 0.0)
    src["cap"] = "meta" if "cap" in meta else "default(0)"
    init_std = float(meta.get("init_std", 0.0) or 0.0)
    src["init_std"] = "meta" if "init_std" in meta else "default(0, 不影响推理)"
    if not np.isfinite(cap) or cap < 0:
        raise DeltaTokenContractError(f"[delta-token] cap={cap} 非法（必须 ≥0）")

    return DeltaTokenSpec(
        token_dim=int(token_dim), feat_dim=int(feat_dim), hidden=hidden, depth=depth,
        init_std=init_std, cap=cap,
        feat_joint_order=fj, state_joint_order=sj,
        layer_shapes=tuple(shapes), sources=src, meta=meta,
    )


# --------------------------------------------------------------------------- 构造
def build_delta_token_net(spec: DeltaTokenSpec, *, device="cpu", dtype=None):
    """按 spec 建网并**严格**装载权重（多余/缺失/形状不符一律 FAIL）。

    注意 ``DeltaTokenNet.__init__`` 会做末层初始化（``init_std``），对推理无影响；
    这里传 ``init_std=0`` 以免给"零初始化"假象——权重马上被 checkpoint 覆盖。
    """
    import torch  # noqa: PLC0415

    net = DeltaTokenNet(
        token_dim=int(spec.token_dim), feat_dim=int(spec.feat_dim),
        hidden=int(spec.hidden), depth=int(spec.depth), init_std=0.0,
    )
    return net.to(device=device, dtype=(dtype or torch.float32)).eval()


def load_delta_token_state(
    sd: Mapping[str, Any],
    meta: Mapping[str, Any] | None,
    *,
    device="cpu",
    state_joint_order: str = DEPLOY_JOINT_ORDER,
    feat_joint_order: str | None = None,
    dtype=None,
):
    """``(state_dict, meta) → (net, spec)``；权重严格装载，shape 不符直接抛。"""
    import torch  # noqa: PLC0415

    spec = infer_delta_token_spec(
        sd, meta, state_joint_order=state_joint_order, feat_joint_order=feat_joint_order)
    net = build_delta_token_net(spec, device=device, dtype=dtype)
    own = net.state_dict()
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    shape_bad = [k for k in own if k in sd and tuple(own[k].shape) != tuple(sd[k].shape)]
    if missing or unexpected or shape_bad:
        raise DeltaTokenContractError(
            "[delta-token] 权重结构与推断出的契约不符："
            f"missing={missing[:4]} unexpected={unexpected[:4]} "
            f"shape_mismatch={[(k, tuple(own[k].shape), tuple(sd[k].shape)) for k in shape_bad[:4]]}"
            f"（spec={spec.as_dict()}）")
    net.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()}, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, spec


def load_delta_token_from_checkpoint(
    obj: Any,
    *,
    device="cpu",
    state_joint_order: str = DEPLOY_JOINT_ORDER,
    feat_joint_order: str | None = None,
) -> tuple[Any | None, DeltaTokenSpec | None]:
    """从 ``vla_*.pt`` 的 dict 里取 Δ（缺失 ⇒ ``(None, None)``，调用方须显式记录）。"""
    if not isinstance(obj, dict) or obj.get("delta_token_state") is None:
        return None, None
    return load_delta_token_state(
        obj["delta_token_state"], obj.get("delta_token_meta") or {},
        device=device, state_joint_order=state_joint_order,
        feat_joint_order=feat_joint_order,
    )


# --------------------------------------------------------------------------- 特征
def reorder_joints(x: np.ndarray, src: str, dst: str) -> np.ndarray:
    """1-D 关节块按 ``joint_contract`` 换序（``src == dst`` 时原样返回）。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if src == dst:
        return x
    import joint_contract as jc  # noqa: PLC0415

    perm = np.asarray(jc.permutation(src=src, dst=dst), dtype=np.int64)
    if x.size != perm.size:
        raise DeltaTokenContractError(
            f"[delta-token] 关节块宽度 {x.size} ≠ 置换宽度 {perm.size}（{src}→{dst}）")
    return x[perm]


def delta_features_from_state64(state64: Any, spec: DeltaTokenSpec) -> np.ndarray:
    """部署端 ``state64`` → Δ 网络的特征（**已按 spec 换序**），形状 ``(feat_dim,)``。

    ``state64`` = ``[root_rot6d(6) | dof_pos(29) | dof_vel(29)]``，顺序由
    ``spec.state_joint_order`` 声明（部署默认 ``ha_v31``）。
    """
    s = np.asarray(state64, dtype=np.float32).reshape(-1)
    if s.size < 64:
        raise DeltaTokenContractError(
            f"[delta-token] state64 宽度 {s.size} < 64，算不出偏差特征")
    rot6 = s[0:6]
    q = reorder_joints(s[6:35], spec.state_joint_order, spec.feat_joint_order)
    qd = reorder_joints(s[35:64], spec.state_joint_order, spec.feat_joint_order)
    if int(spec.feat_dim) == int(STATE64_DIM):
        feat = np.concatenate([rot6, q, qd]).astype(np.float32)
    elif int(spec.feat_dim) == int(FEAT_DIM):
        feat = np.concatenate([rot6, qd]).astype(np.float32)
    else:  # pragma: no cover - spec 已在推断期校验
        raise DeltaTokenContractError(f"[delta-token] feat_dim={spec.feat_dim} 未知")
    if feat.size != int(spec.feat_dim):
        raise DeltaTokenContractError(
            f"[delta-token] 特征宽度 {feat.size} ≠ feat_dim {spec.feat_dim}")
    return feat


def delta_from_token_chunk(token0: Any, feat: Any, net, spec: DeltaTokenSpec) -> np.ndarray:
    """``(token0(64), feat)`` → ``Δtoken(64)``（含与训练同一个 ``apply_norm_cap``）。"""
    import torch  # noqa: PLC0415

    dev = next(net.parameters()).device
    t = torch.as_tensor(np.asarray(token0, np.float32).reshape(1, -1), device=dev)
    f = torch.as_tensor(np.asarray(feat, np.float32).reshape(1, -1), device=dev)
    if int(t.shape[-1]) != int(spec.token_dim):
        raise DeltaTokenContractError(
            f"[delta-token] token 宽度 {int(t.shape[-1])} ≠ token_dim {spec.token_dim}")
    with torch.no_grad():
        d = apply_norm_cap(net(t, f), float(spec.cap))
    return d[0].detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- 杂项
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ckpt_sha256_of_tensor_dict(sd: Mapping[str, Any]) -> str:
    """state dict 的稳定指纹（逐键名 + float64 字节），写进运行时日志用于对账。"""
    import torch  # noqa: PLC0415

    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(np.ascontiguousarray(
            torch.as_tensor(sd[k]).detach().cpu().to(torch.float64).numpy()).tobytes())
    return h.hexdigest()


__all__ = [
    "CTAIL_KEYS",
    "DEPLOY_JOINT_ORDER",
    "DeltaTokenContractError",
    "DeltaTokenSpec",
    "FEAT_LAYOUTS",
    "TRAIN_JOINT_ORDER",
    "build_delta_token_net",
    "ckpt_sha256_of_tensor_dict",
    "delta_features_from_state64",
    "delta_from_token_chunk",
    "infer_delta_token_spec",
    "load_delta_token_from_checkpoint",
    "load_delta_token_state",
    "reorder_joints",
    "sha256_file",
]
