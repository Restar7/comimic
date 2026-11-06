"""WM 结构开关：``--arch {mlp,gru}``（单份实现，训练和评估共用）。

为什么必须单独成模块（2026-11-01 接管审查）：
    第一版把 ``GRUWorldModel`` 写在 ``scripts/finetune_wm_velocity_policy.py`` 里，于是
    **只有训练脚本认识 GRU**；``scripts/eval_wm_multistep.py`` 走的是
    ``diff_vla_wbc.isaac_joint.load_isaac_bundle``，那里固定 ``WorldModel(...)`` 再
    ``load_state_dict(ck["wm"])`` ⇒ 拿 GRU 的 ckpt 去评估会直接 `Unexpected key(s) gru.*`。
    也就是说：训练能出 ckpt，**裁判却读不了**。所以结构代码放在这里，两条路径 import 同一份。

口径（与 ``wm_d3.pt`` 对齐，逐项可核对）：
    * 每步输入 ``invariant_features(S) || a || xi`` = 71 + 29 + 33 = **133** 维；
    * 输出 70 维帧局部残差 + 同样的 ``in_norm/out_norm``；
    * ``explicit_integration`` 下 dt 积分逐行一致（见 :meth:`GRUWorldModel._integrate`）；
    * MLP 版 = 133 → 2048 → 2048 → 2048 → 70（8,804,352 个 2D 矩阵参数）。

GRU 版（用户定稿）：单层 ``GRU(133 → 1024)`` + 头 ``1024 → 2048 → 2048 → 70``
（9,989,120，**+13.46%**）。隐状态 ``h (1,B,1024)`` 跨 K 步传递，``loss.backward()``
穿过全部 K 步（BPTT）——这就是"加记忆"的全部内容；头部宽度与 MLP 同级，所以任何
提升只能归因于记忆，而不是容量。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_TL = ROOT / "trackerlab-supermimic/source/trackerLab"
if str(_TL) not in sys.path:            # 允许独立 import（不自带这个路径的调用方也能用）
    sys.path.insert(0, str(_TL))

from trackerLab.algorithms.supermimic.networks import (  # noqa: E402
    RunningNorm,
    WorldModel,
    mlp,
)
from trackerLab.kinematics.supermimic.rotations import so3_exp, yaw_mat  # noqa: E402
from trackerLab.tracker_env.mdp.supermimic import NUM_J  # noqa: E402

#: MLP 基线参数量 = ``runs/S1_handover/wm_d3.pt`` 的 2D 矩阵权重和
#: ``net.0 (2048,133) + net.3 (2048,2048) + net.6 (2048,2048) + net.9 (70,2048)``
WM_MLP_BASELINE_PARAMS = 8_804_352
#: gru 版的隐层宽度与头形状：hidden -> 2048 -> 2048 -> 70
WM_GRU_HIDDEN = 1024
WM_GRU_HEAD = (2048, 2048)
#: |gru − mlp| / mlp 的上限（h=1024 时实测 +13.46%）
WM_PARAM_TOL = 0.20
#: ckpt 里可能出现的结构标记键
WM_ARCH_KEYS = ("wm_arch",)


def wm_param_count(module: torch.nn.Module) -> int:
    """矩阵参数量（``ndim >= 2``）。

    口径必须和任务里给的 8,804,352 一致：那是 ``net.0/3/6/9`` 的**权重矩阵**之和，
    不含 bias、LayerNorm 的 1 维 weight/bias，也不含 RunningNorm 的 buffer。
    """
    return int(sum(p.numel() for p in module.parameters() if p.ndim >= 2))


class GRUWorldModel(WorldModel):
    """GRU 版世界模型：单层 ``GRU(133 -> 1024)`` + 与 MLP 同形状/同激活的头。

    与 MLP 版**完全相同**的部分：
      * 每步输入 ``invariant_features(S) || a || xi``（71 + 29 + 33 = 133 维）；
      * 输出 70 维帧局部残差，``in_norm/out_norm`` 归一化口径一致；
      * ``explicit_integration`` 下的 dt 积分（``p += dt*v1 + dp``、``R = R@exp(dt*w_b + dθ)``、
        ``q += dt*qd1 + dq``）—— 与 ``WorldModel.step`` 的下半段逐行一致
        （``scripts/test_wm_arch.py`` 里有"同一个 y ⇒ 同一个状态更新"的等价性断言）。

    不同点（这就是加它的目的）：``step_h(S, a, xi, h)`` 显式接收并返回隐状态
    ``h (1,B,1024)``，多步展开时由调用方跨 K 步传递 ⇒ 梯度穿过全部 K 步（BPTT），
    模型能利用历史而不是只看单帧。

    ``step(S, a, xi)`` 保持与 MLP 版同签名（``IsaacJointBundle.step`` / rollout 直接调用）：
    无粘性隐状态时等价于 ``h = 0``（无记忆，与 MLP 行为一致）；需要"想象/验证期间也带记忆"时用
    ``begin_unroll(B)`` / ``end_unroll()`` 打开粘性隐状态（``IsaacJointBundle.step`` 会按
    ``step_idx`` 自动开合，见那里的注释）。
    """

    def __init__(self, num_joints: int = NUM_J, hidden: int = WM_GRU_HIDDEN,
                 head_hidden=WM_GRU_HEAD):
        # 故意不调 WorldModel.__init__：它会建 MLP 的 ``net``（我们不需要）。
        # 只借用它的 ``inputs`` / ``targets`` / ``state_error``（纯函数，无实例状态）。
        torch.nn.Module.__init__(self)
        self.arch = "gru"
        self.num_joints = int(num_joints)
        self.hidden = int(hidden)
        self.in_dim = 13 + 2 * self.num_joints + self.num_joints + (self.num_joints + 4)  # 133
        self.out_dim = 12 + 2 * self.num_joints                                          # 70
        self.gru = torch.nn.GRU(self.in_dim, self.hidden, num_layers=1)
        # 与 WorldModel 的 ``mlp(..., layer_norm=True, act=nn.SiLU)`` 同一套层/顺序/激活
        self.head = mlp(self.hidden, list(head_hidden), self.out_dim, act=torch.nn.SiLU,
                        layer_norm=True)
        torch.nn.init.zeros_(self.head[-1].weight)   # 残差从 0 起步（与 MLP 版一致）
        torch.nn.init.zeros_(self.head[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        # 结构开关：由脚本按 CLI/ckpt 覆盖（与 MLP 版同样的属性名，下游读法不变）
        self.explicit_integration = True
        self.dt = 0.02
        self._roll_h = None

    # ------------------------------------------------------------------ 隐状态
    def init_hidden(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        p = next(self.parameters())
        return torch.zeros(1, int(batch), self.hidden,
                           device=p.device if device is None else device,
                           dtype=p.dtype if dtype is None else dtype)

    def begin_unroll(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        """打开粘性隐状态：之后的 ``step`` 会在内部延续记忆（验证/想象用）。"""
        self._roll_h = self.init_hidden(batch, device, dtype)
        return self._roll_h

    def end_unroll(self) -> None:
        self._roll_h = None

    def step_h(self, S, a, xi_n, h):
        """一步前向 + 返回**新的**隐状态；调用方负责跨步传递（BPTT 走这条）。"""
        x = self.in_norm(self.inputs(S, a, xi_n)).unsqueeze(0)      # (1,B,133)
        y, h = self.gru(x, h)
        y = self.head(y.squeeze(0))                                 # (B,70)
        return self._integrate(S, y), h

    def step(self, S, a, xi_n):
        """与 MLP 版同签名的一步。无粘性隐状态时等价于 h=0（无记忆）。

        ★ batch 变了就丢弃旧隐状态：这是**防串味**的硬保护 —— 上一批 rollout 留下的
        ``_roll_h`` 若形状对得上，会静默地喂进这一批，结果是"没有报错但数字全错"。
        """
        b = int(S[0].shape[0])
        h = self._roll_h
        sticky = h is not None
        if h is None or int(h.shape[1]) != b:
            h = self.init_hidden(b, S[0].device, S[0].dtype)
            sticky = False
        out, h = self.step_h(S, a, xi_n, h)
        if sticky:
            self._roll_h = h
        return out

    # ------------------------------------------------------------------ 积分（与 MLP 版逐行一致）
    def _integrate(self, S, y):
        p, R, v, w, q, qd = S
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        if not self.explicit_integration:
            p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
            R1 = R @ so3_exp(dth)
            return (p1, R1, v1, w1, q + dq, qd + dqd)
        # 半隐式显式积分：位置/朝向/关节由 **新速度** 推进，网络输出是残差。
        qd1 = qd + dqd
        dt = self.dt
        p1 = p + dt * v1 + (H @ dp.unsqueeze(-1)).squeeze(-1)
        w1_b = (R.transpose(-1, -2) @ w1.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dt * w1_b + dth)
        return (p1, R1, v1, w1, q + dt * qd1 + dq, qd1)


def build_wm_arch(arch: str, num_joints: int = NUM_J) -> torch.nn.Module:
    """按 ``--arch`` 造 WM（训练脚本、评估 loader、自检脚本都走这个入口）。"""
    arch = str(arch).lower()
    if arch == "mlp":
        wm = WorldModel(num_joints=int(num_joints))
        wm.arch = "mlp"
        return wm
    if arch == "gru":
        return GRUWorldModel(num_joints=int(num_joints))
    raise SystemExit(f"[wm-arch] 未知 arch={arch!r}；只支持 mlp / gru")


def detect_wm_arch(ck: dict) -> str:
    """从 ckpt 判断 WM 结构：显式 ``wm_arch`` → 权重键 → 默认 ``mlp``。"""
    tag = str(ck.get("wm_arch") or "").strip().lower()
    if tag in ("mlp", "gru"):
        return tag
    if is_gru_state_dict(ck.get("wm") or {}):
        return "gru"
    return "mlp"


def report_wm_arch(arch: str, wm: torch.nn.Module) -> int:
    """打印 ``[wm-arch] arch=<x> params=<n>``，gru 额外对比 MLP 基线并断言 ≤ 20%。"""
    n = wm_param_count(wm)
    print(f"[wm-arch] arch={arch} params={n}", flush=True)
    if arch == "gru":
        rel = (n - WM_MLP_BASELINE_PARAMS) / float(WM_MLP_BASELINE_PARAMS)
        print(f"[wm-arch] mlp baseline params={WM_MLP_BASELINE_PARAMS} "
              f"diff={n - WM_MLP_BASELINE_PARAMS:+d} rel={rel:+.4f} ({100.0 * rel:+.2f}%)",
              flush=True)
        if abs(rel) > WM_PARAM_TOL:
            raise SystemExit(
                f"[wm-arch] gru params={n} 与 mlp 基线 {WM_MLP_BASELINE_PARAMS} 相差 "
                f"{rel:+.2%}，超过 ±{WM_PARAM_TOL:.0%} 上限 —— 不要静默改变模型容量")
    return n


def is_gru_state_dict(state: dict) -> bool:
    return any(str(k).startswith("gru.") for k in state)


def load_wm_weights(wm: torch.nn.Module, arch: str, ckpt: Path) -> torch.nn.Module:
    """``--arch gru`` 的加载：能整份载入就整份载入，否则从零初始化 + 形状匹配的复用。

    两类 ckpt：
      * ``--arch gru`` 自己保存的（``wm`` 里是 ``gru.*/head.*``，或 ``ckpt['wm_arch']=='gru'``）
        ⇒ **strict 整份载入**（可续训 / 可评估）；
      * MLP ckpt（``net.0/3/6/9``）⇒ GRU 从零初始化，只复用**形状相同**的部分：
        最终层 ``net.9.weight/bias (70,2048)`` → ``head.6``，以及 ``in_norm/out_norm``
        的统计量（输入 133 / 输出 70 是同一个空间，属于数据统计而不是学到的权重）。
        其余（GRU 三个门、head.0、head.3、LayerNorm）全部随机初始化，并**打印清楚**。
    """
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    src = ck.get("wm") or {}
    if not src:
        raise SystemExit(f"[wm-load] {ckpt} 里没有 'wm' 子字典，无法初始化 --arch {arch}")
    if arch != "gru":
        raise SystemExit(f"[wm-load] arch={arch} 不需要 load_wm_weights（只有 gru 从零重建）")
    if is_gru_state_dict(src) or str(ck.get("wm_arch") or "").lower() == "gru":
        sd = wm.state_dict()
        missing = [k for k in sd if k not in src]
        unexpected = [k for k in src if k not in sd]
        if missing or unexpected:
            raise SystemExit(
                f"[wm-load] gru ckpt 与当前 GRU 结构不匹配：missing={missing[:8]} "
                f"unexpected={unexpected[:8]}（请检查 hidden/head 尺寸）")
        wm.load_state_dict({k: v for k, v in src.items()}, strict=True)
        print(f"[wm-load] ckpt={ckpt} wm_arch=gru：整份载入 GRU 权重 "
              f"（{len(sd)} 个张量，strict=True）", flush=True)
        return wm
    # ---- MLP ckpt → GRU ----
    sd = dict(wm.state_dict())
    new, reused = {}, []
    for s, t in (("net.9.weight", "head.6.weight"), ("net.9.bias", "head.6.bias"),
                 ("in_norm.mean", "in_norm.mean"), ("in_norm.var", "in_norm.var"),
                 ("out_norm.mean", "out_norm.mean"), ("out_norm.var", "out_norm.var")):
        v = src.get(s)
        if torch.is_tensor(v) and tuple(v.shape) == tuple(sd[t].shape):
            new[t] = v.detach().float().clone()
            reused.append(f"{s}->{t}")
    # ★ RunningNorm 的 ``count`` 必须一起搬：count 已到 max_count 时 update() 直接返回，
    #   即"冻结统计"。若只搬 mean/var 不搬 count（=0），第一次 update() 会用**这一批**
    #   的统计把标定好的归一化冲掉。
    for s, t in (("in_norm.count", "in_norm.count"), ("out_norm.count", "out_norm.count")):
        v = src.get(s)
        if torch.is_tensor(v):
            new[t] = v.detach().float().clone().reshape(sd[t].shape)
            reused.append(f"{s}->{t}")
    sd.update(new)
    wm.load_state_dict(sd, strict=True)
    print(f"[wm-load] ckpt={ckpt} 里是 MLP 的 wm（net.*）：--arch gru 从零初始化", flush=True)
    print(f"[wm-load] reused（形状匹配）: {', '.join(reused) if reused else '无'}", flush=True)
    _fresh = [k for k in sd if k not in new and not k.endswith("count")]
    print(f"[wm-load] FRESH（随机初始化，不来自 ckpt）: {', '.join(_fresh)}", flush=True)
    if not reused:
        print("[wm-load][warn] 没有任何张量被复用（ckpt 结构对不上）—— GRU 完全从头训",
              flush=True)
    return wm


__all__ = [
    "GRUWorldModel",
    "WM_MLP_BASELINE_PARAMS",
    "WM_GRU_HIDDEN",
    "WM_GRU_HEAD",
    "WM_PARAM_TOL",
    "build_wm_arch",
    "detect_wm_arch",
    "is_gru_state_dict",
    "load_wm_weights",
    "report_wm_arch",
    "wm_param_count",
]
