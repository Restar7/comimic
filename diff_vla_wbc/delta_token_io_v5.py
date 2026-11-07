"""v5 Δtoken 部署契约：``per-frame Δ`` + ``hist10`` + ``T/K/H`` 时间轴（**新增，不改 v4**）。

为什么另开一份而不是改 `delta_token_io.py`
----------------------------------------
会话 01 在改 `delta_token_io.py` / 服务端；v5 主线只要求**最小追加**（规范 §2.4），
所以这里提供**新符号**，与 v4 的 ``DeltaTokenSpec`` / ``delta_from_token_chunk`` 并存：

============================  =========================================
v4（共享 Δ）                   v5（逐帧 Δ）
============================  =========================================
``delta_from_token_chunk``    ``delta_from_chunk_full_v5`` → ``(T,64)``
广播到整段                     ``t ≥ K`` **恒 0**，服务端按帧加
无 history                     ``hist10 (10,58)`` **必需**，缺即抛
无时间轴                       ``T/K/H`` 进 spec，加载时逐项校验
============================  =========================================

关节序口径（规范 §1.3 / §4）
---------------------------
训练侧 ``hist10`` 的 ``q/qd`` 与 ``state64`` 的 ``dof_pos/dof_vel`` 是**同一套关节序**
（HA_V31，逐位实测 ``max|Δ|=0``），只差一个 ``−default`` 常数偏移。所以 v5 的 spec 用
``state_joint_order`` / ``hist_joint_order`` **显式声明**两侧口径：一致时不做任何换序；
不一致时用 `joint_contract.permutation` 换序，并且**只换 29 维关节块**。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .delta_token_io import (
    DeltaTokenContractError,
    ckpt_sha256_of_tensor_dict,
    reorder_joints,
    sha256_file,
)
from .delta_token_temporal_v5 import HIST_DIM, TemporalActorCfg, TemporalDeltaActorV5

#: v5 **训练侧**的关节序 = `standard`（= MJCF / SuperMimic MDP 序）。
#: 证据（四处独立一致）：① `MDP.obs` 直接算 `q − default_q`（`supermimic.py:171-181`），
#: 而 `bundle.default_joint_pos[:6]=[-0.2,0,0,0.42,-0.23,0]` 的 knee 在 index 3（= STANDARD 序）；
#: ② `SonicIsaacBundle.step` 注释："状态的关节序天生是 STANDARD（语料/VLA 观测/FK 全按它）"；
#: ③ `SonicExecutor.obs994` 对 q/qd/last_action 做 `index_select(std_to_ha)` 才喂 SONIC decoder
#: ⇒ 进 decoder 之前那块 q 是 STANDARD；
#: ④ `delta_token_io`（v4）把训练侧约定写成 `standard`。
TRAIN_STATE_JOINT_ORDER = "standard"
TRAIN_HIST_JOINT_ORDER = "standard"
#: **部署侧**的关节序：客户端 `state64[6:35]/[35:64]` 与 decoder 的 hist 块都是 **HA_V31**
#: （isaaclab/asset 序）。所以部署契约必须把收到的东西换到 `standard` 再喂网络。
DEPLOY_STATE_JOINT_ORDER = "ha_v31"
DEPLOY_HIST_JOINT_ORDER = "ha_v31"
#: 部署侧必须声明"history 来自连续控制 tick"，否则不许加载带 history 的模型
DEPLOY_HIST_CLOCK = "control_tick_50hz"
KNOWN_JOINT_ORDERS = ("standard", "ha_v31")
KNOWN_ARCH = "temporal_v5"


def joint_permutation(src: str, dst: str) -> np.ndarray:
    """``x_dst = x_src[perm]`` 的置换（对最后一维做 fancy index）。

    ``src == dst`` ⇒ 恒等置换。**注意**：v4 的 `reorder_joints` 只吃 1-D 向量；
    v5 的 hist10 是 ``(10,29)``，所以统一用这个置换在最后一维上换序。
    """
    if str(src) not in KNOWN_JOINT_ORDERS or str(dst) not in KNOWN_JOINT_ORDERS:
        raise DeltaTokenContractError(
            f"[delta-token-v5] 未知关节序 {src!r} → {dst!r}（只认 {KNOWN_JOINT_ORDERS}）")
    if str(src) == str(dst):
        return np.arange(29, dtype=np.int64)
    import joint_contract as jc  # noqa: PLC0415
    perm = np.asarray(jc.permutation(src=str(src), dst=str(dst)), dtype=np.int64)
    if perm.size != 29:
        raise DeltaTokenContractError(
            f"[delta-token-v5] 关节置换宽度 {perm.size} != 29（{src}→{dst}）")
    return perm


# --------------------------------------------------------------------------- spec
@dataclass(frozen=True)
class DeltaTokenSpecV5:
    """v5 checkpoint 的**完整输入契约** + 每个字段的来源（可审计）。"""

    arch: str = KNOWN_ARCH
    tok_dim: int = 64
    d_model: int = 128
    n_heads: int = 4
    ffn: int = 256
    n_blocks: int = 1
    hist_frames: int = 10
    hist_dim: int = HIST_DIM
    state_dim: int = 64
    pe_pos_dim: int = 16
    hidden: int = 1024
    depth: int = 2
    init_std: float = 4e-4
    cap: float = 0.08
    use_history: bool = True
    out_mode: str = "per_frame"
    T: int = 40
    K: int = 20
    H: int = 20
    # 这两个字段声明的是**入站（部署）**口径；内部一律换到 `TRAIN_*`（standard）再算
    state_joint_order: str = DEPLOY_STATE_JOINT_ORDER
    hist_joint_order: str = DEPLOY_HIST_JOINT_ORDER
    hist_clock: str = DEPLOY_HIST_CLOCK
    sources: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- 校验（构造即校验，失败不允许"带着错配置跑"）----
    def __post_init__(self) -> None:
        if str(self.arch) != KNOWN_ARCH:
            raise DeltaTokenContractError(
                f"[delta-token-v5] 未知 arch={self.arch!r}（只认 {KNOWN_ARCH!r}）")
        if str(self.out_mode) != "per_frame":
            raise DeltaTokenContractError(
                f"[delta-token-v5] out_mode={self.out_mode!r} 非法（v5 只支持 per_frame）")
        for name, val in (("tok_dim", self.tok_dim), ("d_model", self.d_model),
                          ("state_dim", self.state_dim), ("T", self.T), ("K", self.K)):
            if int(val) <= 0:
                raise DeltaTokenContractError(f"[delta-token-v5] {name}={val} 必须 > 0")
        if int(self.K) > int(self.T):
            raise DeltaTokenContractError(f"[delta-token-v5] K={self.K} > T={self.T}")
        if int(self.H) <= 0:
            raise DeltaTokenContractError(f"[delta-token-v5] H={self.H} 必须 > 0")
        if float(self.cap) < 0:
            raise DeltaTokenContractError(f"[delta-token-v5] cap={self.cap} 必须 ≥ 0")
        if bool(self.use_history):
            if int(self.hist_frames) != 10:
                raise DeltaTokenContractError(
                    f"[delta-token-v5] use_history=True 要求 hist_frames=10，得到 {self.hist_frames}")
            if int(self.hist_dim) != HIST_DIM:
                raise DeltaTokenContractError(
                    f"[delta-token-v5] hist_dim 必须是 {HIST_DIM}（q29+qd29），得到 {self.hist_dim}")
            if str(self.hist_clock) != DEPLOY_HIST_CLOCK:
                raise DeltaTokenContractError(
                    f"[delta-token-v5] hist_clock={self.hist_clock!r} 非法（必须 "
                    f"{DEPLOY_HIST_CLOCK!r}：chunk 级快照不许冒充控制 tick 历史）")
        if str(self.state_joint_order) not in KNOWN_JOINT_ORDERS:
            raise DeltaTokenContractError(
                f"[delta-token-v5] 未知 state_joint_order={self.state_joint_order!r}")
        if str(self.hist_joint_order) not in KNOWN_JOINT_ORDERS:
            raise DeltaTokenContractError(
                f"[delta-token-v5] 未知 hist_joint_order={self.hist_joint_order!r}")

    # ---- 派生 ----
    @property
    def d_in(self) -> int:
        return (int(self.tok_dim) + int(self.d_model)
                + (int(self.d_model) if bool(self.use_history) else 0)
                + int(self.state_dim) + int(self.pe_pos_dim))

    @property
    def n_params(self) -> int:
        """与 `TemporalDeltaActorV5.n_params` 一致（用于 ckpt 对账）。"""
        p = int(self.tok_dim) * int(self.d_model) + int(self.d_model)          # token_proj
        p += int(self.T) * int(self.d_model)                                    # pe_chunk
        d = int(self.d_model)
        # TransformerEncoderLayer(pre-LN, batch_first): self-attn in_proj(3d,d)+out(d,d)+bias(3d+d)
        p += 3 * d * d + 3 * d + d * d + d
        p += 2 * d + 2 * d                                                      # norm1 / norm2
        p += 2 * d * int(self.ffn) + int(self.ffn) + d                          # linear1 / linear2
        p += 2 * d                                                              # chunk_norm
        if bool(self.use_history):
            h = int(self.hist_dim)                                              # GRU 输入 = 58
            p += 3 * d * h + 3 * d + 3 * d * d + 3 * d                          # GRU(58→128)
        p += int(self.T) * int(self.pe_pos_dim)                                 # pe_pos
        d_in = self.d_in
        p += d_in * int(self.hidden) + int(self.hidden)                         # head.0
        for _ in range(max(0, int(self.depth) - 1)):
            p += int(self.hidden) * int(self.hidden) + int(self.hidden)
        p += int(self.hidden) * int(self.tok_dim) + int(self.tok_dim)           # last
        return int(p)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arch": str(self.arch), "tok_dim": int(self.tok_dim),
            "d_model": int(self.d_model), "n_heads": int(self.n_heads),
            "ffn": int(self.ffn), "n_blocks": int(self.n_blocks),
            "hist_frames": int(self.hist_frames), "hist_dim": int(self.hist_dim),
            "state_dim": int(self.state_dim), "pe_pos_dim": int(self.pe_pos_dim),
            "hidden": int(self.hidden), "depth": int(self.depth),
            "init_std": float(self.init_std), "cap": float(self.cap),
            "use_history": bool(self.use_history), "out_mode": str(self.out_mode),
            "T": int(self.T), "K": int(self.K), "H": int(self.H),
            "d_in": int(self.d_in), "n_params": int(self.n_params),
            "state_joint_order": str(self.state_joint_order),
            "hist_joint_order": str(self.hist_joint_order),
            "hist_clock": str(self.hist_clock),
            "sources": dict(self.sources),
        }

    def actor_cfg(self) -> TemporalActorCfg:
        return TemporalActorCfg(
            tok_dim=int(self.tok_dim), d_model=int(self.d_model), n_heads=int(self.n_heads),
            ffn=int(self.ffn), n_blocks=int(self.n_blocks), hist_frames=int(self.hist_frames),
            hist_dim=int(self.hist_dim), state_dim=int(self.state_dim),
            pe_pos_dim=int(self.pe_pos_dim), hidden=int(self.hidden), depth=int(self.depth),
            init_std=float(self.init_std), cap=float(self.cap), chunk_len=int(self.T),
            use_history=bool(self.use_history),
        )


# --------------------------------------------------------------------------- 推断
def _layer_groups(sd: Mapping[str, Any], prefix: str) -> list[int]:
    seen: set[int] = set()
    for k in sd:
        if k.startswith(prefix + "."):
            tail = k[len(prefix) + 1:]
            head = tail.split(".", 1)[0]
            if head.isdigit():
                seen.add(int(head))
    return sorted(seen)


def infer_delta_token_spec_v5(
    sd: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
    *,
    state_joint_order: str | None = None,
    hist_joint_order: str | None = None,
) -> DeltaTokenSpecV5:
    """从 ``state_dict`` + ``meta`` 推断**并严格校验** v5 契约。

    ``T/K/H`` 这类**不在权重形状里**的量必须由 ``meta`` 给出（训练脚本写入 checkpoint）；
    缺失即抛 —— 部署端不允许猜时间轴。
    """
    meta = dict(meta or {})
    arch = str(meta.get("arch", KNOWN_ARCH))
    if arch != KNOWN_ARCH:
        raise DeltaTokenContractError(f"[delta-token-v5] meta.arch={arch!r} 不是 {KNOWN_ARCH!r}")

    def _w(name: str) -> tuple[int, ...]:
        if name not in sd:
            raise DeltaTokenContractError(
                f"[delta-token-v5] state dict 缺少 {name!r} —— 这不是 TemporalDeltaActorV5 的权重")
        return tuple(int(x) for x in np.asarray(sd[name].detach().cpu() if hasattr(sd[name], "detach")
                                                else sd[name]).shape)

    if "pe_chunk" not in sd:
        raise DeltaTokenContractError("[delta-token-v5] 缺少 pe_chunk（无法确定 T / d_model）")
    T, d_model = _w("pe_chunk")
    T_pos, pe_pos_dim = _w("pe_pos")
    if T_pos != T:
        raise DeltaTokenContractError(f"[delta-token-v5] pe_pos 的 T={T_pos} != pe_chunk 的 T={T}")
    d_model_tp, tok_dim = _w("token_proj.weight")
    if d_model_tp != d_model:
        raise DeltaTokenContractError(
            f"[delta-token-v5] token_proj 输出 {d_model_tp} != d_model {d_model}")
    state_dim = int(meta.get("state_dim", 64))
    if "chunk_encoder.layers.0.linear1.weight" not in sd:
        raise DeltaTokenContractError("[delta-token-v5] 缺少 chunk_encoder.layers.0.*")
    ffn, _ = _w("chunk_encoder.layers.0.linear1.weight")
    n_blocks = len(_layer_groups(sd, "chunk_encoder.layers"))
    use_history = "hist_gru.weight_ih_l0" in sd
    hist_dim = int(_w("hist_gru.weight_ih_l0")[1]) if use_history else int(meta.get("hist_dim", 0) or 0)
    hist_frames = int(meta.get("hist_frames", 10))
    # head：0/2/4... 交替 Linear/SiLU
    head_lin = sorted([k for k in sd if k.startswith("head.") and k.endswith(".weight")],
                      key=lambda s: int(s.split(".")[1]))
    if len(head_lin) < 2:
        raise DeltaTokenContractError(f"[delta-token-v5] head 层数过少：{head_lin}")
    hidden, d_in = _w(head_lin[0])
    depth = len(head_lin) - 1
    out_dim, hidden_last = _w(head_lin[-1])
    if out_dim != tok_dim:
        raise DeltaTokenContractError(
            f"[delta-token-v5] head 输出 {out_dim} != tok_dim {tok_dim}")
    if hidden_last != hidden and depth > 0:
        raise DeltaTokenContractError(
            f"[delta-token-v5] head 末层输入 {hidden_last} != hidden {hidden}")

    for key in ("T", "K", "H"):
        if key not in meta:
            raise DeltaTokenContractError(
                f"[delta-token-v5] meta 缺 {key!r}：时间轴合同必须由训练侧显式写入 checkpoint")
    if int(meta["T"]) != T:
        raise DeltaTokenContractError(
            f"[delta-token-v5] meta.T={int(meta['T'])} 与权重 PE 表 T={T} 不一致")

    cap = float(meta.get("cap", 0.0))
    init_std = float(meta.get("init_std", 0.0))
    spec = DeltaTokenSpecV5(
        arch=arch, tok_dim=int(tok_dim), d_model=int(d_model),
        n_heads=int(meta.get("n_heads", 4)), ffn=int(ffn), n_blocks=int(n_blocks),
        hist_frames=int(hist_frames), hist_dim=int(hist_dim), state_dim=int(state_dim),
        pe_pos_dim=int(pe_pos_dim), hidden=int(hidden), depth=int(depth),
        init_std=init_std, cap=cap, use_history=bool(use_history),
        out_mode=str(meta.get("out_mode", "per_frame")),
        T=int(T), K=int(meta["K"]), H=int(meta["H"]),
        state_joint_order=str(state_joint_order or meta.get("state_joint_order")
                               or DEPLOY_STATE_JOINT_ORDER),
        hist_joint_order=str(hist_joint_order or meta.get("hist_joint_order")
                             or DEPLOY_HIST_JOINT_ORDER),
        hist_clock=str(meta.get("hist_clock", DEPLOY_HIST_CLOCK)),
        sources={"arch": "meta", "T": "pe_chunk", "d_model": "pe_chunk",
                 "tok_dim": "token_proj", "ffn": "chunk_encoder", "n_blocks": "chunk_encoder",
                 "hidden": "head", "depth": "head", "use_history": "hist_gru",
                 "K/H": "meta", "cap": "meta"},
        meta=meta,
    )
    if int(spec.d_in) != int(d_in):
        raise DeltaTokenContractError(
            f"[delta-token-v5] 推断 d_in={spec.d_in} 与 head 首层输入 {d_in} 不一致")
    return spec


# --------------------------------------------------------------------------- 构造 / 装载
def build_temporal_actor_v5(spec: DeltaTokenSpecV5, *, device="cpu", dtype=None):
    """按 spec 建网（**不装权重**；装权重用 :func:`load_temporal_actor_v5`）。"""
    import torch  # noqa: PLC0415

    net = TemporalDeltaActorV5(spec.actor_cfg())
    return net.to(device=device, dtype=(dtype or torch.float32)).eval()


def load_temporal_actor_v5(
    sd: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
    *,
    device="cpu",
    dtype=None,
    state_joint_order: str | None = None,
    hist_joint_order: str | None = None,
):
    """``(state_dict, meta) → (net, spec)``；结构/形状/时间轴不符一律抛。"""
    import torch  # noqa: PLC0415

    spec = infer_delta_token_spec_v5(
        sd, meta, state_joint_order=state_joint_order, hist_joint_order=hist_joint_order)
    net = build_temporal_actor_v5(spec, device=device, dtype=dtype)
    own = net.state_dict()
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    shape_bad = [k for k in own if k in sd and tuple(own[k].shape) != tuple(np.shape(sd[k]))]
    if missing or unexpected or shape_bad:
        raise DeltaTokenContractError(
            "[delta-token-v5] 权重结构与契约不符："
            f"missing={missing[:4]} unexpected={unexpected[:4]} "
            f"shape_mismatch={[(k, tuple(own[k].shape), tuple(np.shape(sd[k]))) for k in shape_bad[:4]]}")
    net.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()}, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, spec


def save_temporal_actor_v5(path: str | Path, actor, spec: DeltaTokenSpecV5,
                           extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """写 checkpoint：``delta_token_v5_state`` + ``delta_token_v5_spec``（含 T/K/H）。"""
    import torch  # noqa: PLC0415

    if int(actor.n_params) != int(spec.n_params):
        raise DeltaTokenContractError(
            f"[delta-token-v5] 实际参数 {int(actor.n_params)} != spec 预期 {int(spec.n_params)}")
    sd = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
    payload = {
        "delta_token_v5_state": sd,
        "delta_token_v5_spec": spec.as_dict(),
        "delta_token_v5_meta": dict(spec.meta),
        "actor_params": int(actor.n_params),
        "ckpt_sha256": ckpt_sha256_of_tensor_dict(sd),
    }
    if extra:
        payload.update({str(k): v for k, v in dict(extra).items()})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(p))
    return payload


def load_delta_token_v5_from_checkpoint(obj: Any, *, device="cpu",
                                        state_joint_order: str | None = None,
                                        hist_joint_order: str | None = None):
    """从任意 v5 checkpoint dict 取 ``(net, spec)``；不是 v5 ⇒ ``(None, None)``。"""
    if not isinstance(obj, dict) or obj.get("delta_token_v5_state") is None:
        return None, None
    spec_d = dict(obj.get("delta_token_v5_spec") or {})
    meta = dict(obj.get("delta_token_v5_meta") or {})
    meta = {**meta, **{k: spec_d[k] for k in spec_d if k not in ("sources",)}}
    return load_temporal_actor_v5(
        obj["delta_token_v5_state"], meta, device=device,
        state_joint_order=state_joint_order, hist_joint_order=hist_joint_order)


# --------------------------------------------------------------------------- 时间轴
def token_index(step: int, T: int) -> int:
    """``ℓ(k)``：WM 步 ``k`` → token 位置。**``ℓ(k)=k``（k<T）**，``k≥T`` 钳到 ``T−1``。

    与 `rollout.py:366-367` 的 ``token_at(step) = tokens[:, min(step, T-1)]`` 逐字一致。
    """
    if int(T) <= 0:
        raise DeltaTokenContractError(f"[delta-token-v5] T={T} 非法")
    return int(min(max(0, int(step)), int(T) - 1))


def exec_token_count(K: int, T: int) -> int:
    """``K = min(H, T)`` 的有效输出位置数（规范 §1.2 / §9）。"""
    return int(min(int(K), int(T)))


# --------------------------------------------------------------------------- 特征
def _q_qd_blocks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size != 58:
        raise DeltaTokenContractError(f"[delta-token-v5] hist10 单帧必须是 58 维，得到 {x.size}")
    return x[:29], x[29:]


def _to_float32(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.dtype != np.float32:  # pragma: no cover - numpy 已转换
        raise DeltaTokenContractError(f"[delta-token-v5] {name} 必须是 float32")
    return arr


def hist10_from_blocks(
    q_block: Any, qd_block: Any, spec: DeltaTokenSpecV5, *,
    default_joint_pos: Any = None,
) -> np.ndarray:
    """部署侧两块历史 ``(10,29)``（**已减 default / 未减由 ``default_joint_pos`` 决定**）
    → ``hist10 (10,58) = [q | qd]``，并换序到 ``spec.hist_joint_order``。

    ``q_block`` / ``qd_block`` 是**入站顺序**（= ``spec.hist_joint_order``，部署默认 HA_V31）；
    输出一律换到训练顺序（standard）。``q_block`` 必须是 ``q − default``；
    若传的是绝对角，请给 ``default_joint_pos``（**standard 序**，= `g1_default_joint_pos()`），
    并在换序后相减（fail-closed：宽度必须是 29）。
    """
    q = np.asarray(q_block, dtype=np.float32)
    qd = np.asarray(qd_block, dtype=np.float32)
    if q.shape != (spec.hist_frames, 29) or qd.shape != (spec.hist_frames, 29):
        raise DeltaTokenContractError(
            f"[delta-token-v5] 两块历史必须是 ({spec.hist_frames},29)，"
            f"得到 {q.shape} / {qd.shape}")
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise DeltaTokenContractError("[delta-token-v5] hist10 出现非有限值")
    out = np.concatenate([q, qd], axis=-1).astype(np.float32)
    out = _hist_to_train_order(out, spec)          # 入站顺序 → standard
    if default_joint_pos is not None:
        dq = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
        if dq.size != 29:
            raise DeltaTokenContractError(
                f"[delta-token-v5] default_joint_pos 必须是 29 维，得到 {dq.size}")
        # default 必须是 **standard 序**（`compat.g1_default_joint_pos()`）；已在训练序上相减
        out = out.copy()
        out[:, :29] = out[:, :29] - dq[None, :]
    return out


def hist10_from_hist64(
    hist64: Any, spec: DeltaTokenSpecV5, *, default_joint_pos: Any = None,
) -> np.ndarray:
    """部署侧 ``state_hist_10 (10,64) = [rot6d|q|qd]`` → ``(10,58) = [q−default | qd]``。

    ``default_joint_pos`` 缺省时读 ``spec.meta['hist_default_joint_pos']``（**obs/decoder 用的
    mjlab `KNEES_BENT` 默认角**，= `compat.g1_default_joint_pos()`），再回落到
    ``spec.meta['default_joint_pos']``（老 ckpt 兼容）。

    ★ **两个 default 不是同一个东西**（本机实测，容易踩）：
      * ``hist`` 的 q 块用的是 **obs/decoder 的 default** = `g1_default_joint_pos()`
        （`[-0.312, 0, 0, 0.669, -0.363, …]`）；`supermimic_obs_to_blocks` 的 q 就是
        ``q − g1_default_joint_pos()``，SONIC decoder 的 994 q 块也是它；
      * ``q_target = default + scale·a`` 里的 default 是 **MDP/SONIC** 的
        `bundle.default_joint_pos`（`[-0.2, 0, 0, 0.42, -0.23, …]`），**只用于动作换算**。
    两者差 ≈0.27 rad，混用会让 hist10 整体偏移一个常数（静默错误）。训练侧把两个都写进
    ckpt meta（`hist_default_joint_pos` / `default_joint_pos`）。
    """
    h = np.asarray(hist64, dtype=np.float32)
    if h.shape != (spec.hist_frames, 64):
        raise DeltaTokenContractError(
            f"[delta-token-v5] state_hist_10 必须是 ({spec.hist_frames},64)，得到 {h.shape}")
    if default_joint_pos is None:
        default_joint_pos = spec.meta.get("hist_default_joint_pos")
    if default_joint_pos is None:
        default_joint_pos = spec.meta.get("default_joint_pos")   # 老 ckpt 兼容
    if default_joint_pos is None:
        raise DeltaTokenContractError(
            "[delta-token-v5] 缺 hist_default_joint_pos（ckpt meta 里没有、调用方也没给）")
    dq = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
    if dq.size != 29:
        raise DeltaTokenContractError(f"[delta-token-v5] default_joint_pos 必须是 29 维，得到 {dq.size}")
    # ① 先把 q/qd 从入站顺序换到训练顺序（standard），② 再减 standard 序的 default
    perm = joint_permutation(str(spec.state_joint_order), TRAIN_STATE_JOINT_ORDER)
    q_std = np.asarray(h[:, 6:35], np.float32)[:, perm]
    qd_std = np.asarray(h[:, 35:64], np.float32)[:, perm]
    return np.ascontiguousarray(
        np.concatenate([q_std - dq[None, :], qd_std], axis=-1), dtype=np.float32)


def _hist_to_train_order(hist10: np.ndarray, spec: DeltaTokenSpecV5) -> np.ndarray:
    """把历史从 `spec.hist_joint_order` 换到**训练顺序**（v5 训练侧固定 standard）。"""
    src, dst = str(spec.hist_joint_order), TRAIN_HIST_JOINT_ORDER
    arr = np.ascontiguousarray(hist10, dtype=np.float32)
    if src == dst:
        return arr
    perm = joint_permutation(src, dst)
    q = arr[:, :29][:, perm]
    qd = arr[:, 29:][:, perm]
    return np.ascontiguousarray(np.concatenate([q, qd], axis=-1), dtype=np.float32)


def delta_hist_from_history10(hist10: Any, spec: DeltaTokenSpecV5) -> np.ndarray:
    """训练/部署共用的 ``hist10`` 校验门：形状 / dtype / 关节序不符即抛（规范 §4.7）。"""
    arr = np.asarray(hist10)
    if arr.shape != (spec.hist_frames, spec.hist_dim):
        raise DeltaTokenContractError(
            f"[delta-token-v5] hist10 必须是 ({spec.hist_frames},{spec.hist_dim})，得到 {arr.shape}")
    if arr.dtype != np.float32:
        raise DeltaTokenContractError(
            f"[delta-token-v5] hist10 必须是 float32（部署请求里的 float64 不自动降级），得到 {arr.dtype}")
    if not np.isfinite(arr).all():
        raise DeltaTokenContractError("[delta-token-v5] hist10 出现非有限值")
    return _hist_to_train_order(arr, spec)


def _state64_to_train_order(state64: Any, spec: DeltaTokenSpecV5) -> np.ndarray:
    s = np.asarray(state64, dtype=np.float32).reshape(-1)
    if s.size != 64:
        raise DeltaTokenContractError(f"[delta-token-v5] state64 必须是 64 维，得到 {s.size}")
    if not np.isfinite(s).all():
        raise DeltaTokenContractError("[delta-token-v5] state64 出现非有限值")
    src, dst = str(spec.state_joint_order), TRAIN_STATE_JOINT_ORDER
    if src == dst:
        return s
    perm = joint_permutation(src, dst)
    return np.ascontiguousarray(
        np.concatenate([s[0:6], s[6:35][perm], s[35:64][perm]]), dtype=np.float32)


# --------------------------------------------------------------------------- 整段 Δ
def delta_from_chunk_full_v5(
    Z: Any, state64: Any, hist10: Any, net, spec: DeltaTokenSpecV5, *,
    cap: float | None = None, dtype: str = "float32",
) -> np.ndarray:
    """``(Z(T,64), state64(64), hist10(10,58))`` → ``Δ (T,64)``，含 mask + 软上限。

    fail-closed（缺一即抛，绝不返回全 0 糊过去）：
      1. ``Z.shape == (T,64)`` 且 ``T == spec.T``；
      2. ``hist10`` 形状/dtype 正确（``use_history=True`` 时必需）；
      3. ``state64`` 宽度 64，关节序按 ``spec`` 换到训练顺序；
      4. 输出 ``(T,64)``，``t ≥ K`` **逐位为 0**，全张量有限。
    """
    import torch  # noqa: PLC0415

    if hasattr(net, "use_history") and bool(net.use_history) != bool(spec.use_history):
        raise DeltaTokenContractError(
            f"[delta-token-v5] 网络 use_history={bool(net.use_history)} 与 spec="
            f"{bool(spec.use_history)} 不一致（拒绝静默退化）")
    z = _to_float32(Z, "Z")
    if z.shape != (int(spec.T), int(spec.tok_dim)):
        raise DeltaTokenContractError(
            f"[delta-token-v5] Z 必须是 ({spec.T},{spec.tok_dim})，得到 {z.shape}")
    s = _state64_to_train_order(state64, spec)
    if bool(spec.use_history):
        if hist10 is None:
            raise DeltaTokenContractError("[delta-token-v5] use_history=True 但 hist10=None")
        h = delta_hist_from_history10(hist10, spec)
    else:
        h = None
    dev = next(net.parameters()).device
    t_z = torch.as_tensor(z, device=dev)[None]
    t_s = torch.as_tensor(s, device=dev)[None]
    t_h = None if h is None else torch.as_tensor(h, device=dev)[None]
    mask = torch.zeros(1, int(spec.T), device=dev)
    mask[:, : int(spec.K)] = 1.0
    with torch.no_grad():
        d = net(t_z, t_s, t_h, mask, cap=float(spec.cap if cap is None else cap))[0]
    out = d.detach().cpu().numpy().astype(np.float32)
    if out.shape != (int(spec.T), int(spec.tok_dim)):
        raise DeltaTokenContractError(
            f"[delta-token-v5] 输出形状 {out.shape} != ({spec.T},{spec.tok_dim})")
    if not np.isfinite(out).all():
        raise DeltaTokenContractError("[delta-token-v5] Δ 出现非有限值")
    tail = out[int(spec.K):]
    if tail.size and float(np.abs(tail).max()) != 0.0:
        raise DeltaTokenContractError(
            f"[delta-token-v5] t≥K 的位置必须恒 0，实测 max|Δ|={float(np.abs(tail).max()):.3e}")
    if float(spec.cap) > 0:
        n = np.linalg.norm(out, axis=-1)
        if float(n.max()) > float(spec.cap) + 1e-6:
            raise DeltaTokenContractError(
                f"[delta-token-v5] ‖Δ‖={float(n.max()):.4f} 超过 cap={spec.cap}")
    return out


def apply_delta_per_frame(chunk64: Any, delta: Any, spec: DeltaTokenSpecV5) -> np.ndarray:
    """部署施加点：``chunk64[:K] += Δ[:K]``，``chunk64[K:]`` **不动**（禁止广播）。"""
    c = np.asarray(chunk64, dtype=np.float32)
    d = np.asarray(delta, dtype=np.float32)
    if c.shape != (int(spec.T), int(spec.tok_dim)):
        raise DeltaTokenContractError(
            f"[delta-token-v5] chunk64 必须是 ({spec.T},{spec.tok_dim})，得到 {c.shape}")
    if d.shape != c.shape:
        raise DeltaTokenContractError(f"[delta-token-v5] Δ 形状 {d.shape} != chunk {c.shape}")
    if float(np.abs(d[int(spec.K):]).max(initial=0.0)) != 0.0:
        raise DeltaTokenContractError("[delta-token-v5] Δ 在 t≥K 上非零（契约违反）")
    out = c.copy()
    out[: int(spec.K)] = out[: int(spec.K)] + d[: int(spec.K)]
    return out


def spec_v5_from_actor(actor, meta: Mapping[str, Any] | None = None,
                       **kwargs: Any) -> DeltaTokenSpecV5:
    """训练侧：从 actor（+ meta）造 spec，保证保存的契约与权重一致。"""
    cfg = actor.cfg
    base = dict(
        tok_dim=int(cfg.tok_dim), d_model=int(cfg.d_model), n_heads=int(cfg.n_heads),
        ffn=int(cfg.ffn), n_blocks=int(cfg.n_blocks), hist_frames=int(cfg.hist_frames),
        hist_dim=int(cfg.hist_dim), state_dim=int(cfg.state_dim),
        pe_pos_dim=int(cfg.pe_pos_dim), hidden=int(cfg.hidden), depth=int(cfg.depth),
        init_std=float(cfg.init_std), cap=float(cfg.cap), use_history=bool(cfg.use_history),
        T=int(cfg.chunk_len),
    )
    base.update({k: v for k, v in kwargs.items() if v is not None})
    return DeltaTokenSpecV5(meta=dict(meta or {}), **base)


def with_meta(spec: DeltaTokenSpecV5, **meta: Any) -> DeltaTokenSpecV5:
    """返回一份替换了 ``meta`` 的 spec（frozen dataclass 的便捷包装）。"""
    return replace(spec, meta=dict(spec.meta, **meta))


__all__ = [
    "DEPLOY_HIST_CLOCK",
    "DeltaTokenSpecV5",
    "KNOWN_ARCH",
    "TRAIN_HIST_JOINT_ORDER",
    "TRAIN_STATE_JOINT_ORDER",
    "apply_delta_per_frame",
    "build_temporal_actor_v5",
    "ckpt_sha256_of_tensor_dict",
    "delta_from_chunk_full_v5",
    "delta_hist_from_history10",
    "hist10_from_blocks",
    "hist10_from_hist64",
    "joint_permutation",
    "infer_delta_token_spec_v5",
    "load_delta_token_v5_from_checkpoint",
    "load_temporal_actor_v5",
    "save_temporal_actor_v5",
    "sha256_file",
    "exec_token_count",
    "spec_v5_from_actor",
    "token_index",
    "with_meta",
]
