"""Adapter: run our joint fine-tune on the **Isaac Lab / PhysX** SuperMimic checkpoint.

Source of truth for the architecture is ``trackerlab-supermimic`` (not the mjlab replica):

* ``Policy``   — ``mlp(64 + 10*99 -> [2048,1024,512,256] -> 29)``, act=ELU, **no** LayerNorm,
  with a ``RunningNorm(99)`` on the observation; token interface has **no encoder**.
  (``source/trackerLab/trackerLab/algorithms/supermimic/networks.py``)
* ``WorldModel`` — ``in_dim = (13+2*29) + 29 + (29+4) = 133``, ``out_dim = 12+2*29 = 70``,
  ``mlp(133 -> [2048,2048,2048] -> 70)`` with ``act=SiLU`` and **LayerNorm after every hidden
  layer**, plus ``in_norm(133)`` / ``out_norm(70)`` running normalisers.  Integration in the
  yaw frame with ``so3_exp``:
  ``p1 = p + H dp; R1 = R so3_exp(dθ); v1 = v + R dv; w1 = w + R dw; q1 = q + dq; qd1 = qd + dqd``

* Observations (99-d per frame): ``[q - q0 - dq0(29), qd(29), Rᵀw(3), g_b(3), a_prev(29),
  rot6d(Rᵀ R_ref)(6)]`` — the last 6 dims are the reference-relative root orientation, which is
  **computed**, not recorded.

Everything here is additive: the mjlab path stays untouched, nothing overwrites the source
checkpoint.  Parity against the reference implementation is asserted by
``tests/test_trackerlab_parity.py`` (max |Δ| < 1e-4), and the training entry point
(``train_joint.py`` / losses / rollout) is unchanged — the adapter only swaps the policy and the
world model behind the same ``act() / step() / reward()`` interface.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

NUM_J = 29
HIST = 10
OBS93 = 3 * NUM_J + 6          # 93
OBS99 = OBS93 + 6              # 99  (relative_ori99)
WM_IN = (13 + 2 * NUM_J) + NUM_J + (NUM_J + 4)   # 133
WM_OUT = 12 + 2 * NUM_J                          # 70


# --------------------------------------------------------------------------- primitives
class RunningNorm(nn.Module):
    """Running mean/std normaliser, matching trackerlab's ``RunningNorm``.

    Inference only: the statistics come from the checkpoint (``mean``/``var``); nothing is
    updated here, because our joint fine-tune never trains these statistics.
    """

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + self.eps)

    def std(self) -> torch.Tensor:
        return self.var.sqrt()

    def load_from(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        self.mean.copy_(mean.reshape_as(self.mean))
        self.var.copy_(var.reshape_as(self.var))


def mlp(inp: int, hidden, out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def mat_to_6d(m: torch.Tensor) -> torch.Tensor:
    """Row-major 6D rotation representation (``rotation_6d_layout: row`` in HA's protocol)."""
    return torch.cat([m[..., 0, :], m[..., 1, :]], dim=-1)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    m = torch.empty(*q.shape[:-1], 3, 3, dtype=q.dtype, device=q.device)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    """Yaw-only component of a rotation matrix (used for the frame-local position deltas)."""
    x, y = R[..., 0, 0], R[..., 1, 0]
    theta = torch.atan2(y, x)
    c, s = torch.cos(theta), torch.sin(theta)
    H = torch.zeros_like(R)
    H[..., 0, 0], H[..., 0, 1] = c, -s
    H[..., 1, 0], H[..., 1, 1] = s, c
    H[..., 2, 2] = 1.0
    return H


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues: axis-angle -> rotation matrix."""
    theta = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / theta
    K = torch.zeros(*w.shape[:-1], 3, 3, dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    s, c = torch.sin(theta).unsqueeze(-1), torch.cos(theta).unsqueeze(-1)
    return I + s * K + (1 - c) * (K @ K)


# --------------------------------------------------------------------------- policy / WM
class TLPolicy(nn.Module):
    """Token-conditioned action decoder of the Isaac SuperMimic ``Policy`` (encoder removed)."""

    def __init__(self, obs_dim: int = OBS99, token_dim: int = 64, hist: int = HIST,
                 num_joints: int = NUM_J, hidden=(2048, 1024, 512, 256)):
        super().__init__()
        self.obs_dim, self.token_dim, self.hist = obs_dim, token_dim, hist
        self.dec = mlp(token_dim + obs_dim * hist, list(hidden), num_joints, act=nn.ELU)
        self.obs_norm = RunningNorm(obs_dim)

    def forward(self, obs, token, interface: str | None = None) -> torch.Tensor:
        """``obs`` 可以是 (B,hist,99) 或拍平的 (B,hist*99)；``token`` (B,64) -> (B,29)。

        ``interface`` 只是为了让调用方（``bundle.act`` 会传 ``interface="token"``）签名匹配，
        这里不需要分支：token 接口下 encoder 已被移除。
        """
        if obs.ndim == 2:
            obs = obs.reshape(obs.shape[0], self.hist, self.obs_dim)
        o = self.obs_norm(obs).reshape(obs.shape[0], -1)
        return self.dec(torch.cat([token, o], dim=-1))

    # -- checkpoint I/O ----------------------------------------------------
    def load_state(self, sd: dict) -> None:
        self.dec.load_state_dict({k[len("dec."):]: v for k, v in sd.items() if k.startswith("dec.")})
        self.obs_norm.load_from(sd["obs_norm.mean"], sd["obs_norm.var"])


class TLWorldModel(nn.Module):
    """Deterministic dynamics ``s_{t+1} = f(s_t, a_t, xi)`` of the Isaac SuperMimic."""

    def __init__(self, hidden=(2048, 2048, 2048), num_joints: int = NUM_J):
        super().__init__()
        self.num_joints = num_joints
        self.in_dim, self.out_dim = WM_IN, WM_OUT
        self.net = mlp(self.in_dim, list(hidden), self.out_dim, act=nn.SiLU, layer_norm=True)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)

    # state is a flat 71-vector: [p(3), Rmat(9), v(3), w(3), q(29), qd(29)] would be 76;
    # our flat layout is [p(3), quat(4), v(3), w(3), q(29), qd(29)] = 71.  The adapter converts.
    @staticmethod
    def split_state(s: torch.Tensor):
        """我们的扁平 71 维布局（见 supermimic/world_model/heads.py:split_state）：
        [root_pos(0:3), root_quat wxyz(3:7), joint_pos(7:36), lin_vel_w(36:39),
         **ang_vel_b**(39:42), joint_vel(42:71)]。
        ⚠️ trackerlab 的状态里 w 是**世界系**角速度（其特征用 Rᵀw），我们是体坐标系，
        所以跨实现时必须做一次换系：w_world = R · w_body。
        """
        p = s[..., 0:3]
        R = quat_to_mat(s[..., 3:7])
        q = s[..., 7:36]
        v = s[..., 36:39]
        w_b = s[..., 39:42]
        qd = s[..., 42:71]
        return p, R, v, w_b, q, qd

    @staticmethod
    def merge_state(p, R, v, w, q, qd) -> torch.Tensor:
        # quat from R (w-first), matching ``quat_to_mat``
        w4 = torch.stack(
            [
                torch.sqrt(torch.clamp(1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], min=1e-12)) / 2,
                (R[..., 2, 1] - R[..., 1, 2]) / 4,
                (R[..., 0, 2] - R[..., 2, 0]) / 4,
                (R[..., 1, 0] - R[..., 0, 1]) / 4,
            ],
            dim=-1,
        )
        w4 = w4 / w4.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        # keep the sign continuous with the previous quaternion is the caller's job
        return torch.cat([p, w4, v, w, q, qd], dim=-1)

    def invariant_features(self, s: torch.Tensor) -> torch.Tensor:
        """71-d features，**逐项等于** trackerlab ``mdp.invariant_features``：
        [p_z(1), mat_to_6d(HᵀR)(6), Rᵀv(3), Rᵀw(3), q(29), qd(29)]。
        我们自己 spec.canonical 的顺序与之相同，但额外除以 in_scale、且把 w 当体坐标系；
        这里按 trackerlab 口径写（w 用 Rᵀ 旋转），归一化交给 in_norm。
        """
        p, R, v, w_b, q, qd = self.split_state(s)
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        w_w = (R @ w_b.unsqueeze(-1)).squeeze(-1)      # 体系 -> 世界系（跨实现换系）
        return torch.cat(
            [
                p[..., 2:3],
                mat_to_6d(H.transpose(-1, -2) @ R),
                (Rt @ v.unsqueeze(-1)).squeeze(-1),
                (Rt @ w_w.unsqueeze(-1)).squeeze(-1),
                q,
                qd,
            ],
            dim=-1,
        )

    def step(self, s: torch.Tensor, action: torch.Tensor, xi_n: torch.Tensor) -> torch.Tensor:
        p, R, v, w, q, qd = self.split_state(s)
        x = torch.cat([self.invariant_features(s), action, xi_n], dim=-1)
        y = self.net(self.in_norm(x))
        d = y * self.out_norm.std() + self.out_norm.mean
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        H = yaw_mat(R)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return self.merge_state(p1, R1, v1, w1, q + dq, qd + dqd)

    def load_state(self, sd: dict) -> None:
        keep = {k: v for k, v in sd.items() if k.startswith("net.")}
        self.net.load_state_dict({k[len("net."):]: v for k, v in keep.items()})
        self.in_norm.load_from(sd["in_norm.mean"], sd["in_norm.var"])
        self.out_norm.load_from(sd["out_norm.mean"], sd["out_norm.var"])


def load_trackerlab(path: str | Path, *, num_joints: int = NUM_J):
    """Load ``{pi, wm, args, ...}`` and return (policy, world_model, meta)."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    args = ck.get("args") or {}
    obs_mode = args.get("obs_mode", "relative_ori99")
    obs_dim = OBS99 if obs_mode in ("relative_ori99", "relative_orientation") else OBS93
    pi = TLPolicy(obs_dim=obs_dim, token_dim=64, hist=int(args.get("hist", HIST)),
                  num_joints=num_joints)
    pi.load_state(ck["pi"])
    wm = TLWorldModel(num_joints=num_joints)
    wm.load_state(ck["wm"])
    return pi, wm, {"iteration": ck.get("iteration"), "args": args, "obs_dim": obs_dim}


__all__ = [
    "RunningNorm", "TLPolicy", "TLWorldModel", "load_trackerlab",
    "mat_to_6d", "quat_to_mat", "yaw_mat", "so3_exp", "OBS93", "OBS99", "WM_IN", "WM_OUT",
]


# --------------------------------------------------------------------------- bundle
class TrackerLabBundle:
    """把 Isaac 版 policy/WM 接到我们 rollout 期望的接口上（``act/step/reward/observation``）。

    资产、FK、解析奖励、x 均匀分布（``xi``）都复用我们既有实现 —— 它们不含可学权重，
    只是几何/奖励计算，所以不存在"混用旧底模权重"的问题。
    """

    def __init__(self, policy: TLPolicy, wm: TLWorldModel, assets, *, device="cuda",
                 default_joint_pos: torch.Tensor | None = None, action_scale: float = 1.0,
                 term_head: nn.Module | None = None):
        self.policy = policy.to(device).eval()
        self.wm = wm.to(device).eval()
        self.assets = assets
        self.device = torch.device(device)
        self.default_joint_pos = default_joint_pos
        self.action_scale = float(action_scale)
        self.term_head = term_head

    # ---- 前向片段 ----
    def act(self, obs_hist_flat: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.policy(obs_hist_flat, goal, interface="token")

    def step(self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return self.wm.step(state, action, xi)

    def reward(self, state, action, prev_action, reference, bodies=None):
        return self.assets.heads.reward(state, action, prev_action, reference, bodies)

    def observation(self, state, prev_action, xi, reference=None) -> torch.Tensor:
        """93 维本体观测 + 6 维 reference-relative 朝向 = 99 维（Isaac 口径）。"""
        base = analytic_observation_93(state, prev_action, xi, self.default_joint_pos)
        R = quat_to_mat(state[..., 3:7])
        if reference is not None:
            R_ref = reference  # (B,3,3) 已由调用方给出
            rel = mat_to_6d(R.transpose(-1, -2) @ R_ref)
        else:
            rel = torch.zeros(*state.shape[:-1], 6, dtype=state.dtype, device=state.device)
        return torch.cat([base, rel], dim=-1)

    def termination_logits(self, state, reference):
        if self.term_head is None:
            return torch.zeros(state.shape[0], device=state.device)
        return self.term_head(state, reference)


def analytic_observation_93(state, prev_action, xi, default_joint_pos) -> torch.Tensor:
    """与 ``supermimic.world_model.heads.analytic_observation`` 完全一致的 93 维（本文件自带，
    避免 adapter 依赖 mjlab 包）。"""
    p = state[..., 0:3]
    q = state[..., 3:7]
    qj = state[..., 7:36]
    qd = state[..., 42:71]
    R = quat_to_mat(q)
    bias = xi[..., 1:30]
    g_b = -R[..., 2, :]
    w_b = state[..., 39:42]
    re = (R.transpose(-1, -2) @ w_b.unsqueeze(-1)).squeeze(-1)
    return torch.cat([qj + bias - default_joint_pos, qd, re, g_b, prev_action], dim=-1)
