"""Isaac-Lab-native bundle for the joint finetune (no MuJoCo/mjlab code paths).

Rule of this module
-------------------
The WBC policy and the world model were trained in **Isaac Lab / PhysX** by
``trackerlab-supermimic``.  Everything that consumes or supervises them must therefore come
from that same implementation:

===========================================  ==========================================================
piece                                        source
===========================================  ==========================================================
policy / world model weights                 ``trackerLab.algorithms.supermimic.networks`` (``ckpt['pi'/'wm']``)
observation (99-dim, ``relative_ori99``)     ``trackerLab.tracker_env.mdp.supermimic.MDP.obs``
reward (paper Table A.2, ×dt)                ``MDP.reward`` — same config the run logged (``ckpt['reward_cfg']``)
termination / survival                       ``MDP.soft_termination``
differentiable FK (14 tracked bodies)        ``trackerLab.kinematics.supermimic.g1.G1Kinematics`` (trackerlab MJCF)
joint limits / action scale / default pose   ``trackerTask.supermimic.robots.get_robot('g1')``
reference motion + velocities                ``trackerLab.motion_buffer.motion_lib.CompactMotionLib``
===========================================  ==========================================================

Nothing is taken from ``third_party/SuperMimic-Reproduced`` (the mjlab/MuJoCo port).  The legacy
bridge ``diff_vla_wbc.trackerlab_bundle`` did mix the two (Isaac weights + mjlab analytic heads)
and is not used by the training entry point any more.

State convention
----------------
Our flat state (what ``data.py`` / the rollout / the loss carry) is::

    s(71) = [root_pos(3), root_quat_wxyz(4), q(29), root_lin_vel_w(3), root_ang_vel_w(3), qd(29)]

i.e. trackerlab's state tuple ``(p, R, v, w, q, qd)`` with the joint block moved in front of the
velocities — the layout ``compute_joint_loss`` and the analytic-loss helpers already assume.
``root_ang_vel_w`` is the **world** frame value, exactly like ``CompactMotionLib.ref`` and the
Isaac env, so the world model and the MDP consume it without any extra conversion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
_TL = ROOT / "trackerlab-supermimic" / "source"
for _p in (_TL / "trackerLab", _TL / "trackerTask"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from trackerLab.kinematics.supermimic.rotations import (  # noqa: E402
    mat_to_6d,
    mat_to_quat,
    quat_to_mat,
)
from trackerLab.tracker_env.mdp.supermimic import (  # noqa: E402
    MAX_STEP_REWARD,
    REWARD_TERMS,
    RewardCfg,
    normalize_xi,
)

NUM_JOINTS = 29
ACTION_DIM = 29
XI_DIM = 33
STATE_DIM = 71
HISTORY = 10


def _huber_on_error(d: torch.Tensor, delta: float, d_max: float = 0.0) -> torch.Tensor:
    """2026-11-03：对**误差距离** d（米/弧度）做 Huber，远端线性。

    注意：``e_*`` 是平方误差，调用前先 ``sqrt``；不要对平方误差直接用 Huber
    （那会把 delta 变成 m²，量级差一次方）。

    ★ 2026-11-04 **奖励死区**（`d_max > 0` 时生效）：
    把 d 夹到 ``d_max`` 之后再算 Huber ⇒ 超过死区后惩罚**恒为常数、梯度为 0**。
    为什么需要（用户口径）：VLA 的状态里**没有 root 位置**（state46 = 29 关节 +
    14 手 + 3 维重力），所以大范围的位置漂移它根本**看不见**，硬用"贴参考"的梯度
    把它拉回来等于让它拟合噪声。小漂移（姿态级，0.01–0.05 rad）该拉，大漂移
    （root 平移 >10–30 cm）不该硬拉 —— 那就是这条死区。
    """
    dd = float(delta)
    if dd <= 0:
        return d
    dm = float(d_max or 0.0)
    if dm > 0.0:
        d = torch.clamp(d, max=dm)
    return torch.where(d <= dd, 0.5 * d * d / dd, d - 0.5 * dd)


OBS_TERM_DIMS_93 = (29, 29, 3, 3, 29)
OBS_TERM_DIMS_99 = OBS_TERM_DIMS_93 + (6,)
MAX_STEP_REWARD = float(MAX_STEP_REWARD)

#: our flat-state field order (see module docstring)
FLAT_LAYOUT = ("p", "quat", "q", "v", "w", "qd")
#: ``CompactMotionLib`` canonical row order
CANONICAL_LAYOUT = ("p", "quat", "v", "w", "q", "qd")
_SIZES = {"p": 3, "quat": 4, "v": 3, "w": 3, "q": NUM_JOINTS, "qd": NUM_JOINTS}


def default_xi(batch: int, device) -> torch.Tensor:
    """Nominal sim context: friction 0.75, zero encoder bias, zero CoM offset."""
    xi = torch.zeros(batch, XI_DIM, device=device)
    xi[:, 0] = 0.75
    return xi


def term_dims_for(obs_mode: str) -> tuple[int, ...]:
    return OBS_TERM_DIMS_99 if obs_mode.startswith("relative") else OBS_TERM_DIMS_93


def flatten_term_major(hist: torch.Tensor, term_dims: tuple[int, ...]) -> torch.Tensor:
    """``(H, B, W)`` oldest→newest → ``(B, H*W)`` term-major (the policy's flat obs order)."""
    if hist.shape[-1] != sum(term_dims):
        raise ValueError(f"history width {hist.shape[-1]} != sum(term_dims) {sum(term_dims)}")
    chunks = []
    off = 0
    for dim in term_dims:
        chunks.append(hist[:, :, off : off + dim].permute(1, 0, 2).reshape(hist.shape[1], -1))
        off += dim
    return torch.cat(chunks, dim=-1)


def split_flat_obs(flat: torch.Tensor, hist: int, term_dims: tuple[int, ...]) -> torch.Tensor:
    """``(B, H*W)`` term-major → ``(B, H, W)``."""
    b = flat.shape[0]
    out, off = [], 0
    for dim in term_dims:
        out.append(flat[:, off : off + dim * hist].reshape(b, hist, dim))
        off += dim * hist
    return torch.cat(out, dim=-1)


def flat_state_tuple(flat: torch.Tensor):
    """Our flat state ``(…, 71)`` → trackerlab state tuple ``(p, R, v, w, q, qd)``."""
    if int(flat.shape[-1]) != STATE_DIM:
        raise ValueError(f"expected an {STATE_DIM}-d flat state, got {tuple(flat.shape)}")
    return (
        flat[..., 0:3],
        quat_to_mat(flat[..., 3:7]),
        flat[..., 36:39],
        flat[..., 39:42],
        flat[..., 7:36],
        flat[..., 42:71],
    )


def canonical_to_flat(rows: torch.Tensor) -> torch.Tensor:
    """``CompactMotionLib`` canonical rows ``(N, [p,quat,v,w,q,qd])`` → our flat ``(N, 71)``."""
    p, quat = rows[..., 0:3], rows[..., 3:7]
    v, w = rows[..., 7:10], rows[..., 10:13]
    q, qd = rows[..., 13 : 13 + NUM_JOINTS], rows[..., 13 + NUM_JOINTS :]
    return torch.cat((p, quat, q, v, w, qd), dim=-1)


def canonical_to_state(rows: torch.Tensor):
    """Canonical rows → trackerlab state tuple."""
    p, quat = rows[..., 0:3], rows[..., 3:7]
    v, w = rows[..., 7:10], rows[..., 10:13]
    q, qd = rows[..., 13 : 13 + NUM_JOINTS], rows[..., 13 + NUM_JOINTS :]
    return p, quat_to_mat(quat), v, w, q, qd


@dataclass
class FrameRef:
    """Reference of one transition: just the motion-library frame index.

    ``MDP.reward`` / ``MDP.obs`` / ``MDP.soft_termination`` index the motion library by frame,
    so nothing else has to be carried around (and nothing can drift out of sync).
    """

    f: torch.Tensor


@dataclass
class IsaacPolicyCfg:
    """``asdict``-able provenance record (``JointTrainer.save`` round-trip)."""

    implementation: str = "trackerlab"
    obs_dim: int = 99
    obs_mode: str = "relative_ori99"
    token_dim: int = 64
    hist: int = HISTORY
    hidden: tuple = (2048, 1024, 512, 256)
    source_checkpoint: str = ""
    source_iteration: int = -1


@dataclass
class IsaacWorldModelConfig:
    implementation: str = "trackerlab"
    hidden: tuple = (2048, 2048, 2048)
    num_joints: int = NUM_JOINTS
    in_norm: bool = True
    out_norm: bool = True
    source_checkpoint: str = ""
    source_iteration: int = -1


def wm_err_floor(num_joints: int, device) -> torch.Tensor:
    """Per-component floors on the WM error normalisation (trackerlab ``runner.py``)."""
    return torch.cat(
        (
            torch.full((3,), 3e-3),
            torch.full((3,), 5e-3),
            torch.full((3,), 2e-2),
            torch.full((3,), 5e-2),
            torch.full((num_joints,), 3e-3),
            torch.full((num_joints,), 5e-2),
        )
    ).to(device)


def trackerlab_window_loss(self, states, a_exec, xi, done, refs):
    """k-step autoregressive WM loss, mirroring ``trackerlab/.../runner.py``.

    ``states`` (k+1, B, 71) **our flat layout**, ``a_exec`` (k, B, 29), ``xi`` (B, 33),
    ``done`` (k, B) unused (trackerlab's world model has no termination head — the MDP's
    ``soft_termination`` is analytic), ``refs[j]`` unused.
    """
    k = int(a_exec.shape[0])
    S = flat_state_tuple(states[0])
    floor = wm_err_floor(int(self.num_joints), a_exec.device)
    std = torch.maximum(self.out_norm.std(), floor)
    total = states.new_zeros(())
    nse_first = nse_last = None
    for j in range(k):
        S = self.step(S, a_exec[j], normalize_xi(xi))
        err = (self.state_error(S, flat_state_tuple(states[j + 1])) / std) ** 2  # (B, 70)
        nse = err.mean(-1).mean()
        total = total + nse
        if j == 0:
            nse_first = nse
        nse_last = nse
    loss = total / k
    return loss, {
        "state_loss": loss.detach(),
        "nse_1step": nse_first.detach(),
        "nse_kstep": nse_last.detach(),
        "horizon": a_exec.new_tensor(float(k)),
    }


class IsaacJointBundle:
    """Drop-in bundle (same surface as ``SuperMimicBundle``) built purely from trackerlab."""

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        packed_motion: str | Path,
        device: str | torch.device = "cuda",
        obs_mode: str | None = None,
        hist: int | None = None,
        token_dim: int = 64,
        reward_cfg: dict | None = None,
        strict: bool = True,
    ) -> None:
        from trackerLab.algorithms.supermimic.networks import Policy, WorldModel
        from trackerLab.kinematics.supermimic.g1 import G1Kinematics
        from trackerLab.motion_buffer.motion_lib.compact import CompactMotionLib
        from trackerLab.tracker_env.mdp.supermimic import MDP
        from trackerTask.supermimic.robots import get_robot

        self.device = torch.device(device)
        self.source_checkpoint = str(ckpt_path)
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "pi" not in ck or "wm" not in ck:
            raise SystemExit(
                f"{ckpt_path}: expected an Isaac-side checkpoint with 'pi' and 'wm' "
                f"(got {sorted(ck)})"
            )
        args = ck.get("args") or {}
        self.obs_mode = obs_mode or args.get("obs_mode") or "relative_ori99"
        self.hist = int(hist or args.get("hist", HISTORY) or HISTORY)
        self.token_dim = int(token_dim)
        self.source_iteration = int(ck.get("iteration", -1))
        self.term_dims = term_dims_for(self.obs_mode)
        self.obs_width = sum(self.term_dims)

        # ---- robot spec / differentiable FK (trackerlab's own) --------------------
        self.spec = get_robot("g1")
        self.kin = G1Kinematics(self.device)
        self.default_joint_pos = torch.tensor(self.spec.default_q, device=self.device, dtype=torch.float32)
        # ---- 参考关节目标 <-> 动作 的换算尺度（口径 bug 修复，2026-10-25）-------------
        # ``env.py`` 的口径是 ``q_target = default_q + xi + a * action_scale``，所以代码里一直用
        # ``a = (q_ref - default_q - bias) / action_scale``。但拿被测 ckpt 实测（
        # ``scripts/probe_action_decomp.py``）表明：spec 的 per-joint 0.0745–0.5476 与被测 run
        # 的训练口径 **不一致** —— 它把参考动作 / ``a_prev`` 放大了 4–8 倍，参考动作 rollout 的
        # reward 从 +0.069 掉到 −0.129、末步状态误差 1.31；``a = Δq``（即 scale=1）才能复现参考跟踪
        # （reward +0.0686 / 末步 0.047），录制 token 的 state_term 也从 20.6 降到 6.5。
        # 默认用 ones；``COMIMIC_ACTION_SCALE=spec`` 保留旧行为，仅用于 A/B 对照。
        _scale_mode = os.environ.get("COMIMIC_ACTION_SCALE", "one").strip().lower()
        if _scale_mode == "spec":
            self.action_scale = torch.tensor(self.spec.action_scale, device=self.device, dtype=torch.float32)
        else:
            self.action_scale = torch.ones(NUM_JOINTS, device=self.device, dtype=torch.float32)
        self.action_scale_mode = _scale_mode
        # 奖励里的 ``(a - a_prev) * action_scale`` 是另一件事：它把动作差换算成弧度，用来让
        # 平滑项与关节量纲无关。实测（同脚本）**换算用 ones、奖励仍用 spec** 才是被测 run 的
        # 口径：参考动作能拿到 +0.0686 / 末步 0.047（用 ones 做奖励则只有 −0.015 / 10.7）。
        _rw_mode = os.environ.get("COMIMIC_REWARD_ACTION_SCALE", "spec").strip().lower()
        self.reward_action_scale = (
            torch.tensor(self.spec.action_scale, device=self.device, dtype=torch.float32)
            if _rw_mode == "spec"
            else torch.ones(NUM_JOINTS, device=self.device, dtype=torch.float32)
        )

        # ---- S0.1 / J1：执行器 spec 审计 + 与 ckpt 里的 robot_spec 对齐 -----------------
        # WBC 训练用 HA_ALIGN=1（knee 0.420 / ankle −0.230 / elbow 0.870，scale 0.3789/0.75）。
        # 联合训练若不设这个环境变量会退回 BeyondMimic（0.669/−0.363/0.600，0.3950/0.5475），
        # WM 想象里的 (q − default_q) 与 a_prev 换算会整体偏移 —— 必须断言，不能静默。
        import numpy as _np  # noqa: PLC0415

        _dq = _np.asarray(self.spec.default_q, dtype="float32")
        _sc = _np.asarray(self.spec.action_scale, dtype="float32")
        print(
            f"[AUDIT][spec] SUPERMIMIC_HA_ALIGN={os.environ.get('SUPERMIMIC_HA_ALIGN')} "
            f"default_q[knee/ankle/elbow]={_dq[3]:.3f}/{_dq[4]:.3f}/{_dq[18]:.3f} "
            f"spec_scale mean/max={_sc.mean():.4f}/{_sc.max():.4f} "
            f"action_scale_mode={getattr(self, 'action_scale_mode', '?')}",
            flush=True,
        )
        _rs = ck.get("robot_spec")
        if _rs is not None:
            if not _np.allclose(_np.asarray(_rs["default_q"], dtype="float32"), _dq, atol=1e-6):
                raise SystemExit(
                    "[spec] default_q 与 WBC 训练不一致：\n"
                    f"  ckpt : {_np.asarray(_rs['default_q'], dtype='float32')[:8]}\n"
                    f"  现在 : {_dq[:8]}\n"
                    "  修法：export SUPERMIMIC_HA_ALIGN=1 + SUPERMIMIC_ROBOT_USD=<训练时的 USD>"
                )
            if not _np.allclose(_np.asarray(_rs["action_scale"], dtype="float32"), _sc, atol=1e-6):
                raise SystemExit("[spec] action_scale 与 WBC 训练不一致（见 AUDIT 行）")
            print("[AUDIT][spec] robot_spec 对齐 PASS", flush=True)
        elif os.environ.get("COMIMIC_ALLOW_NO_SPEC") != "1":
            raise SystemExit(
                "[spec] 该 WBC ckpt 没有 robot_spec；先跑 scripts/stamp_robot_spec.py 生成 _spec.pt，"
                "或设 COMIMIC_ALLOW_NO_SPEC=1 临时放行（结果需标注）"
            )

        # ---- reference motion (Isaac-side packed library, frame-aligned with the npz)
        self.motion = CompactMotionLib.from_packed(
            str(packed_motion), self.kin, str(self.device), layout="canonical"
        )
        _attach_clip_view(self.motion)

        # ---- analytic MDP: obs / reward / termination, exactly as the Isaac run -----
        cfg_kwargs = dict(ck.get("reward_cfg") or {})
        if reward_cfg:
            cfg_kwargs.update(reward_cfg)
        self.mdp = MDP(
            self.kin, self.motion, self.default_joint_pos,
            cfg=RewardCfg(**cfg_kwargs), obs_mode=self.obs_mode, action_scale=self.reward_action_scale,
        )

        # ---- learned layers ------------------------------------------------------
        obs_dim = self.obs_width
        self.policy = Policy(obs_dim, self.token_dim, hist=self.hist,
                             num_joints=NUM_JOINTS, token_dim=self.token_dim)
        # ★ 2026-11-01：WM 结构由 ckpt 决定（``wm_arch`` 键 / 权重键名），不在这里写死 MLP。
        #   否则训练出的 GRU ckpt（``gru.*/head.*``）在评估侧 load 会直接 Unexpected key(s)，
        #   更糟的是有人用 strict=False 把它"糊"过去 —— 那样等于拿一个随机初始化的 WM 当裁判。
        from diff_vla_wbc.wm_arch import build_wm_arch, detect_wm_arch  # noqa: PLC0415

        self.wm_arch = detect_wm_arch(ck)
        self.wm = build_wm_arch(self.wm_arch, num_joints=NUM_JOINTS)
        self.policy.load_state_dict(ck["pi"], strict=strict)
        self.wm.load_state_dict(ck["wm"], strict=strict)
        print(f"[bundle] wm arch = {self.wm_arch}", flush=True)
        # 积分结构（2026-10-25）：官方 ckpt 是在"无 dt 耦合"的结构下训的，必须保持旧语义；
        # 校准过的 ckpt 带 ``wm_integration="explicit_dt0.02"`` → 打开显式物理积分
        # （位置/朝向/关节由 dt×速度推进，速度通道因此被位置误差隐式监督）。
        # 细节见 patches/wm_explicit_integration.md。
        _integ = str(ck.get("wm_integration") or "").strip().lower()
        if _integ.startswith("explicit"):
            self.wm.explicit_integration = True
            if "dt" in _integ:
                try:
                    self.wm.dt = float(_integ.split("dt", 1)[1])
                except ValueError:
                    pass
        else:
            self.wm.explicit_integration = False
        self.wm_integration = ("explicit_dt%g" % self.wm.dt) if self.wm.explicit_integration else "residual"
        # 审计行：joint 训练日志里必须能看到这一行，才能确认速度通道用的是哪一种积分结构
        print(f"[bundle] world-model structure = {self.wm_integration} "
              f"(ckpt wm_integration={ck.get('wm_integration')!r})", flush=True)
        self.policy.to(self.device).eval()
        self.wm.to(self.device).eval()
        # the token-conditioned policy exposes ``token_dim``; every other consumer reads it
        self.policy.token_dim = self.token_dim
        if not hasattr(self.wm, "window_loss"):
            from types import MethodType

            self.wm.window_loss = MethodType(trackerlab_window_loss, self.wm)

        self.goal_dim = self.token_dim
        self.policy_cfg = IsaacPolicyCfg(
            obs_dim=obs_dim, obs_mode=self.obs_mode, token_dim=self.token_dim, hist=self.hist,
            source_checkpoint=str(ckpt_path), source_iteration=self.source_iteration,
        )
        self.world_model_cfg = IsaacWorldModelConfig(
            source_checkpoint=str(ckpt_path), source_iteration=self.source_iteration,
        )

    # ------------------------------------------------------------------ plumbing
    @property
    def num_joints(self) -> int:
        return NUM_JOINTS

    @property
    def library(self):
        return self.motion

    def to(self, device):
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.wm.to(self.device)
        return self

    def train(self):
        self.policy.train()
        return self

    def eval(self):
        self.policy.eval()
        self.wm.eval()
        return self

    def save_checkpoint(self, path, **extra) -> None:
        """Write a trackerlab-format ``{pi, wm, ...}`` checkpoint (same key names as the source)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pi": {k: v.detach().cpu() for k, v in self.policy.state_dict().items()},
                "wm": {k: v.detach().cpu() for k, v in self.wm.state_dict().items()},
                "iteration": int(extra.pop("iteration", self.source_iteration)),
                "args": {
                    "obs_mode": self.obs_mode,
                    "hist": self.hist,
                    "token_dim": self.token_dim,
                    "source_checkpoint": self.source_checkpoint,
                },
                "reward_cfg": asdict(self.mdp.cfg),
                # 结构开关必须一起存：否则下游从 joint_*.pt 加载时会静默退回残差结构
                # （权重按显式积分训的，读成残差结构 = 静默跑偏）。
                "wm_integration": str(getattr(self, "wm_integration", "residual")),
                # WM 结构（只有非 mlp 才写，保持旧 ckpt 的键集合不变）
                **({"wm_arch": str(self.wm_arch)} if str(getattr(self, "wm_arch", "mlp")) != "mlp" else {}),
                **extra,
            },
            path,
        )

    # ------------------------------------------------------------------ learned layer
    def act(self, obs_hist_flat: torch.Tensor, cond: torch.Tensor, ref: Any | None = None) -> torch.Tensor:
        obs = split_flat_obs(obs_hist_flat, self.hist, self.term_dims)
        return self.policy(obs, cond)

    def _w_calib_alpha(self, step_idx):
        """逐视界的 root 角速度幅度标定 α_k（可选，默认关闭 = 1.0）。

        为什么需要：L2 训练出的确定性 WM 会回归条件均值 ⇒ 角速度幅度系统性偏大
        （实测 in-dist k=5 `root_w_ratio` **1.40**，即想象转速比真实快 40%）。
        奖励里的 `yaw_rate` 项吃这个量，幅度偏 40% 就会把策略往"少转"推。

        为什么不能"每步乘一个常数"：那会指数复利（α=1.10 时 k=20 放大 6.7 倍）。
        所以用一张**按视界**的标定表 α_k，由 ``scripts/fit_wm_w_calib.py`` 在**独立标定集**上拟合，
        通过环境变量 ``HA_WM_W_CALIB=<json>`` 挂载；未设置时完全不动（向后兼容）。
        """
        if step_idx is None:
            return None
        if not hasattr(self, "_wcal"):
            self._wcal = None
            _p = os.environ.get("HA_WM_W_CALIB", "")
            if _p and Path(_p).exists():
                try:
                    _d = json.loads(Path(_p).read_text())
                    _a = _d.get("gamma_w") or _d.get("alpha_w") or []
                    self._wcal = [float(x) for x in _a] if _a else None
                    if self._wcal:
                        print(f"[wcalib] 已挂载 {_p}: α_1={self._wcal[1]:.3f} "
                              f"α_5={self._wcal[min(5, len(self._wcal)-1)]:.3f} "
                              f"α_H={self._wcal[-1]:.3f}", flush=True)
                except Exception as _e:  # noqa: BLE001
                    print(f"[wcalib][warn] 读取 {_p} 失败: {_e}", flush=True)
        if not self._wcal:
            return None
        k = min(int(step_idx), len(self._wcal) - 1)
        return self._wcal[k]

    def step(self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor,
             step_idx: int | None = None) -> torch.Tensor:
        """One world-model step; ``state`` is in our flat layout ``(B, 71)``。

        ``step_idx``（从 1 开始）用于可选的 root 角速度幅度标定（见 ``_w_calib_alpha``）。

        ★ 2026-11-01（GRU 版）：``step_idx`` 同时是**隐状态的生命周期信号**——
          * ``step_idx == 1`` ⇒ 新 rollout 开始，把隐状态清零（和训练窗口的 h=0 一致）；
          * ``step_idx > 1``  ⇒ 延续上一步的隐状态（记忆）；
          * ``step_idx is None`` ⇒ 单步调用，无记忆（与 MLP 行为完全一致，防串味）。
        这样 ``imagine_with_tokens`` / ``eval_wm_multistep`` / 闭环 rollout 全都自动带记忆，
        不需要每个调用点记得手动开合；而任何**不传** step_idx 的旧代码行为不变。
        """
        if str(getattr(self.wm, "arch", "mlp")) == "gru":
            if step_idx is None:
                self.wm.end_unroll()
            elif int(step_idx) <= 1:
                self.wm.begin_unroll(int(state.shape[0]), device=state.device,
                                     dtype=state.dtype)
        S = flat_state_tuple(state)
        p1, R1, v1, w1, q1, qd1 = self.wm.step(S, action, normalize_xi(xi))
        _al = self._w_calib_alpha(step_idx)
        if _al is not None and abs(_al - 1.0) > 1e-9:
            w1 = w1 * float(_al)
        return torch.cat((p1, mat_to_quat(R1), q1, v1, w1, qd1), dim=-1)

    # ------------------------------------------------------------------ analytic layer
    def robot_bodies(self, state: torch.Tensor, bodies=None):
        """The Isaac reward recomputes FK internally from the state, so nothing is cached here.

        Returning ``None`` is deliberate: ``rollout`` forwards this to :meth:`reward`, which
        calls ``mdp.reward`` → ``kin.forward`` exactly once.
        """
        return None

    def observation(self, next_state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor,
                    ref: FrameRef | None = None) -> torch.Tensor:
        if ref is None:
            raise ValueError("the Isaac observation needs a FrameRef (relative_ori99 uses the "
                             "reference root rotation of the same frame)")
        return self.mdp.obs(flat_state_tuple(next_state), action, xi, ref.f)

    def reward(self, state, action, prev_action, reference, bodies=None):
        r, terms = self.mdp.reward(flat_state_tuple(state), reference.f, action, prev_action, terms=True)
        out = {k: v for k, v in terms.items()}

        # ---- 2026-10-25 新增（默认关闭，SM/CLI 开关控制）-------------------------
        # reward_mode="linear"：把 6 个 exp(-e/std^2) 跟踪核换成线性的 -(e/std^2)，
        # 避免"误差小的时候核已经饱和到 1 -> reward 对 token 不敏感"（0.083/0.10）。
        # ★ posture(tilt/height) 惩罚已按用户口径**删除**（见下）。
        _mode = str(getattr(self, "reward_mode", "saturating")).lower()
        _pw = 0.0        # 保留变量仅为兼容旧调用方；恒为 0，永不启用
        # reward_vel_scale（2026-10-26）：WM 的**速度通道**最不准（refresh 后 qd_corr≈0.64，
        # 位置/姿态是 7e-4）。速度项贡献给优化器的基本就是"模型误差方向"，可设 0 精确扣掉。
        # 注意：这里从 saturating 核里**精确减去**那两项（e_* 是原始误差，核 = w·exp(-e/std²)·dt），
        # 所以 _vs 在两种 reward_mode 下语义一致，且不会改评估口径以外的任何项。
        _vs = float(getattr(self, "reward_vel_scale", 1.0))
        if _mode in ("linear", "huber", "huber_err") or _pw > 0:
            c = self.mdp.cfg
            dt = 0.02  # CONTROL_DT；与 mdp.reward 内部的乘数一致
            if _mode == "linear":
                r_lin = -(
                    c.w_body_pos * terms["e_body_pos"] / (c.body_pos_std ** 2)
                    + c.w_body_ori * terms["e_body_ori"] / (c.body_ori_std ** 2)
                    + _vs * c.w_body_linvel * terms["e_body_linvel"] / (c.body_linvel_std ** 2)
                    + _vs * c.w_body_angvel * terms["e_body_angvel"] / (c.body_angvel_std ** 2)
                    + c.w_root_pos * terms["e_root_pos"] / (c.root_pos_std ** 2)
                    + c.w_root_ori * terms["e_root_ori"] / (c.root_ori_std ** 2)
                ) * dt
                r = r_lin + (r - terms["track"] * dt)
            elif _mode == "huber":
                # 2026-11-02：cm 级 Huber 主项（用实测漂移定尺度）
                #   h(e) = 0.5 e^2/δ (e<=δ) | e-0.5δ (e>δ)，e 先按 cap 截断
                #   根位置项用 Huber，其余项沿用线性口径；δ/cap 由 HUBER_DELTA/HUBER_CAP 控制
                _hd = float(getattr(self, "huber_delta", 0.02))
                _hc = float(getattr(self, "huber_cap", 0.15))
                _er = terms["e_root_pos"].clamp(max=_hc)
                _h = torch.where(_er <= _hd, 0.5 * _er * _er / _hd, _er - 0.5 * _hd)
                r_hub = -(
                    c.w_body_pos * terms["e_body_pos"] / (c.body_pos_std ** 2)
                    + c.w_body_ori * terms["e_body_ori"] / (c.body_ori_std ** 2)
                    + _vs * c.w_body_linvel * terms["e_body_linvel"] / (c.body_linvel_std ** 2)
                    + _vs * c.w_body_angvel * terms["e_body_angvel"] / (c.body_angvel_std ** 2)
                    + c.w_root_pos * _h / (c.root_pos_std ** 2)
                    + c.w_root_ori * terms["e_root_ori"] / (c.root_ori_std ** 2)
                ) * dt
                r = r_hub + (r - terms["track"] * dt)
                out["huber_root"] = _h.detach()
            elif _mode == "huber_err":
                # 2026-11-03：对**误差距离** d=sqrt(e) 做 Huber（远端线性）
                #   目标 mm 级；大误差梯度=1（不随误差增长），小误差二次。
                #   速度项不在此分支（新线 REWARD_VEL_SCALE=0）。
                _dp = float(getattr(self, "huber_err_delta_pos", 0.05))
                _do = float(getattr(self, "huber_err_delta_ori", 0.05))
                _d_bp = torch.sqrt(terms["e_body_pos"].clamp_min(0.0))
                _d_bo = torch.sqrt(terms["e_body_ori"].clamp_min(0.0))
                _d_rp = torch.sqrt(terms["e_root_pos"].clamp_min(0.0))
                _d_ro = torch.sqrt(terms["e_root_ori"].clamp_min(0.0))
                _dmx = float(getattr(self, "huber_err_d_max", 0.0) or 0.0)
                r_he = -(
                    c.w_body_pos * _huber_on_error(_d_bp, _dp, _dmx)
                    + c.w_body_ori * _huber_on_error(_d_bo, _do, _dmx)
                    + c.w_root_pos * _huber_on_error(_d_rp, _dp, _dmx)
                    + c.w_root_ori * _huber_on_error(_d_ro, _do, _dmx)
                ) * dt
                # ★★ 2026-11-05 真缺口：`huber_err` 分支**完全没有速度项**（旧注释写着
                #   "速度项不在此分支"）⇒ 奖励只看得见"偏了多少"，看不见"走得多快"。
                #   而 base 的症结恰恰是**平均速度只有录制的 1/2**（0.050 vs 0.102 m/s）：
                #   这个亏欠在 H=20（0.4 s）里只积累 2.1 cm（≈ WM 自身误差），速度却是**即时**可见的。
                #   ⇒ 打开 `REWARD_VEL_SCALE`（默认 0，行为不变）即可给"走快些"供梯度，
                #   而且**不需要拉长视界**（因此不会与部署 2.5 Hz / 每 20 步重规划的口径冲突）。
                if _vs != 0.0:
                    _dv = float(getattr(self, "huber_err_delta_vel", 0.05))
                    _d_bv = torch.sqrt(terms["e_body_linvel"].clamp_min(0.0))
                    _d_bw = torch.sqrt(terms["e_body_angvel"].clamp_min(0.0))
                    r_vel = -(
                        c.w_body_linvel * _huber_on_error(_d_bv, _dv, _dmx)
                        + c.w_body_angvel * _huber_on_error(_d_bw, _dv, _dmx)
                    ) * dt
                    r_he = r_he + _vs * r_vel
                    out["huber_err_linvel"] = _huber_on_error(_d_bv, _dv).detach()
                    out["huber_err_angvel"] = _huber_on_error(_d_bw, _dv).detach()
                r = r_he + (r - terms["track"] * dt)
                out["huber_err_root"] = _huber_on_error(_d_rp, _dp).detach()
            # ★ 2026-10-30 用户口径：**tilt / height 惩罚一律不要**（这里原来有一段
            #   `-w*(relu(tilt-0.5)^2 + relu(0.6-height)^2)` 的 posture 惩罚，已删除）。
            #   理由（用户侧实测）：它惩罚的是"绝对歪/绝对低"，会与参考动作本身的倾身/下蹲打架，
            #   而且 `acos` 在完美跟踪处导数发散（NaN 源）。摔倒这件事由
            #   `mdp.soft_termination` 的终止头和 root_pos/tilt 的**相对跟踪**尾部承担。
        if _vs != 1.0 and _mode == "saturating":
            c = self.mdp.cfg
            _sub = (c.w_body_linvel * torch.exp(-terms["e_body_linvel"] / c.body_linvel_std ** 2)
                    + c.w_body_angvel * torch.exp(-terms["e_body_angvel"] / c.body_angvel_std ** 2)) * 0.02
            r = r - (1.0 - _vs) * _sub
            out["vel_dropped"] = _sub.detach()

        # ---- 2026-10-30：对标 mjlab FULL_ARCH §6.4/§6.5 的奖励形状 ----------------
        # (a) root_ori_mode="split"：把单块 root 朝向拆成 tilt / yaw / yaw_rate。
        #     原因：SONIC token 按定义 yaw-相对，官方 VLA 路径没有 anchor 输入 ⇒ 绝对 yaw
        #     只能从相机猜（token→yaw R²=0.226）。实测 ori 误差 10.24° 里约 7.5° 是 yaw
        #     —— **架构上拿不到**，满权重喂给优化器只会制造一个不可学的常数项。
        #     拆分后按可观测性配权重：tilt 1.0 / yaw 0.2 / yaw_rate 0.3（合计 1.5）。
        # (b) tail_weight>0：加 hinge 尾部（位置 0.35 m / 倾斜 20°，cap 见 reward_shape）。
        #     原因：exp 核两端梯度都趋 0（>0.4 m 的 21.2% 时刻只提供 0.22% 的位置梯度）
        #     ⇒ 纯 exp 目标会"拿尾部换中位"：中位变好、p90 翻 2.4 倍。
        _orimode = str(getattr(self, "root_ori_mode", "mono")).lower()
        _lam_tail = float(getattr(self, "tail_weight", 0.0) or 0.0)
        if _orimode != "mono" or _lam_tail > 0.0:
            from .reward_shape import ROOT_POS_CAP, ROOT_POS_THR, hinge_tail, ori_split_reward  # noqa: PLC0415

            _dt = 0.02
            _ref = self.ref_state_at(reference.f)
            if _orimode == "split":
                c = self.mdp.cfg
                _old_ori = c.w_root_ori * torch.exp(-terms["e_root_ori"] / c.root_ori_std ** 2) * _dt
                # ★ 2026-10-31：yaw_rate 权重允许**按轮覆盖**（bundle.yawrate_weight）。
                #   为什么可以"gate 之后再决定"：奖励只在 stage7(joint) 的**想象 rollout** 里
                #   使用，而 gate 在 stage6 ⇒ 时间上完全来得及（见 s3_pipeline 的 joint 前置判定）。
                _wyr = getattr(self, "yawrate_weight", None)
                # ★ 2026-10-31（AB6）：远场 hinge 的 λ 也可按轮覆盖（0 = 关掉远场补偿）。
                _yl = getattr(self, "yaw_hinge_lambda", None)
                _kw = {}
                if _wyr is not None:
                    _kw["w_yawrate"] = float(_wyr)
                if _yl is not None:
                    _kw["hinge_lambda"] = float(_yl)
                _r_ori, _oterms = ori_split_reward(state, _ref, **_kw)
                r = r - _old_ori + _r_ori * _dt
                out.update(_oterms)
                out["root_ori_removed"] = _old_ori.detach()
            elif _orimode == "split_huber":
                # 2026-11-03：split 的 Huber 版（tilt 1.0 / yaw 0.2 / yaw_rate 0.3）
                #   只支持 reward_mode=huber_err（_old_ori 必须与 mono 项一致）
                from .reward_shape import tilt_yaw_yawrate_err  # noqa: PLC0415
                if _mode != "huber_err":
                    raise ValueError("root_ori_mode=split_huber 只支持 reward_mode=huber_err")
                c = self.mdp.cfg
                _do = float(getattr(self, "huber_err_delta_ori", 0.05))
                _e = tilt_yaw_yawrate_err(state, _ref)
                _d_tilt = torch.sqrt(_e["tilt2"].clamp_min(0.0))
                _d_yaw = _e["dyaw"].abs()
                _d_yawrate = torch.sqrt(_e["yawrate2"].clamp_min(0.0))
                _w_t, _w_y, _w_r = 1.0, 0.2, 0.3
                _dmx_o = float(getattr(self, "huber_err_d_max", 0.0) or 0.0)
                _r_ori = -(_w_t * _huber_on_error(_d_tilt, _do, _dmx_o)
                           + _w_y * _huber_on_error(_d_yaw, _do, _dmx_o)
                           + _w_r * _huber_on_error(_d_yawrate, _do, _dmx_o))
                _old_ori = c.w_root_ori * _huber_on_error(
                    torch.sqrt(terms["e_root_ori"].clamp_min(0.0)), _do, _dmx_o) * _dt
                r = r - _old_ori + _r_ori * _dt
                out["root_ori_removed"] = _old_ori.detach()
                out["ori_split_huber"] = _r_ori.detach()
            if _lam_tail > 0.0:
                # ★ 2026-10-31（e2）：`clamp(max=cap)` 越过上限后惩罚是常数、梯度恰为 0，
                #   等于把"烂到没边"的远场整段扔掉（B 线 52% 的时刻在 0.5 m 以外）。
                #   tail_cap_slope>0 时越过 cap 仍保留该斜率；0 保持旧行为（与 d1–d2b 对拍）。
                _slope = float(getattr(self, "tail_cap_slope", 0.0) or 0.0)
                # ★ 2026-11-01（E=1.0 标定）：根位置 hinge 的阈值/上限必须能按轮覆盖；
                #   旧版写死 reward_shape.ROOT_POS_THR/CAP，5 轮共用一个 e_h，课程无法联动。
                _tpt = getattr(self, "tail_pos_thr", None)
                _tpc = getattr(self, "tail_pos_cap", None)
                _tail, _tterms = hinge_tail(
                    state, _ref,
                    pos_thr=float(_tpt) if _tpt is not None else ROOT_POS_THR,
                    pos_cap=float(_tpc) if _tpc is not None else ROOT_POS_CAP,
                    cap_slope=_slope,
                )
                r = r - _lam_tail * _tail * _dt
                out.update(_tterms)
        out["total"] = r
        return out

    def termination_logits(self, state, reference):
        """``logit(p)`` of the Isaac soft-termination probability (``rollout`` sigmoids it back)."""
        p = self.mdp.soft_termination(flat_state_tuple(state), reference.f)
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)

    def flatten_history(self, history: torch.Tensor) -> torch.Tensor:
        return flatten_term_major(history, self.term_dims)

    # ------------------------------------------------------------------ backend-agnostic hooks
    # 下面 4 个方法是给 `burnin` / `data` 用的**后端无关**接口（mjlab 版同名实现见
    # `diff_vla_wbc.mjlab_joint`）。加它们是为了让 burnin 不再写死 trackerlab 的库布局。
    def default_xi(self, batch: int, device) -> torch.Tensor:
        return default_xi(batch, device)

    def clip_start_of(self, frames: torch.Tensor) -> torch.Tensor:
        motion = self.motion
        return motion.clip_start[motion.clip_id[frames]].to(frames.device)

    def ref_state_at(self, frames: torch.Tensor) -> torch.Tensor:
        """库帧 → 扁平 71 状态（trackerlab canonical 布局）。v3 的 ``_ref_states`` 用它。"""
        motion = self.motion
        return canonical_to_flat(motion.ref[frames.to(motion.ref.device)].float())

    def ref_joint_pos(self, frames: torch.Tensor) -> torch.Tensor:
        """库里的 29 维关节位置（trackerlab canonical 布局里从 13 开始）。"""
        motion = self.motion
        return motion.ref[frames.to(motion.ref.device), 13 : 13 + NUM_JOINTS].float()

    def obs_from_states(self, states_flat: torch.Tensor, actions: torch.Tensor,
                        xi: torch.Tensor, frames: torch.Tensor | None = None) -> torch.Tensor:
        """扁平 71 状态 + 动作 + xi → 每帧观测（99 维）；``frames`` 供 relative_ori99 用。"""
        if frames is None:
            raise ValueError("the Isaac observation needs `frames` (relative_ori99 reads the ref)")
        return self.mdp.obs(flat_state_tuple(states_flat), actions, xi, frames)

    # ------------------------------------------------------------------ data helpers
    def reference_start(self, library, frames: torch.Tensor, *, history: int = HISTORY, xi=None):
        """Observation history of the ``history`` frames ending at ``frames``.

        Built from the reference motion itself (the same recipe the mjlab stack used, and what
        the Isaac replay buffer holds at episode start: robot tracking the reference), with
        ``a_prev`` implied by the reference joint targets — the env applies
        ``q_target = default_q + xi_bias + a * action_scale`` (``tracker_env/supermimic/env.py``),
        so ``a = (q_ref - default_q - bias) / action_scale`` is the action that would have
        produced it.
        """
        motion = self.motion
        b = int(frames.shape[0])
        device = frames.device
        offsets = torch.arange(history, device=device) - (history - 1)
        clip_first = motion.clip_start[motion.clip_id[frames]]
        hf = (frames[:, None] + offsets[None, :]).clamp(min=clip_first[:, None])   # (B, H)
        xi = default_xi(b, device) if xi is None else xi
        rows = motion.ref[hf.reshape(-1).to(motion.ref.device)].float().reshape(b, history, -1)
        q_ref = rows[..., 13 : 13 + NUM_JOINTS]
        bias = xi[..., 1 : 1 + NUM_JOINTS]
        a_ref = (q_ref - self.default_joint_pos - bias[:, None]) / self.action_scale
        a_prev = torch.cat((torch.zeros_like(a_ref[:, :1]), a_ref[:, :-1]), dim=1)
        xi_hist = xi[:, None].expand(b, history, XI_DIM)
        rows_flat = rows.reshape(-1, rows.shape[-1])
        obs = self.mdp.obs(canonical_to_state(rows_flat), a_prev.reshape(-1, ACTION_DIM),
                           xi_hist.reshape(-1, XI_DIM), hf.reshape(-1)).reshape(b, history, -1)
        return {
            "state": canonical_to_flat(rows[:, -1]),
            "hist_states": canonical_to_flat(rows),          # (B, H, 71)
            "hist_actions": a_prev,
            "observations": obs.permute(1, 0, 2).contiguous(),  # (H, B, W) — rollout order
            "obs_hist": flatten_term_major(obs.permute(1, 0, 2).contiguous(), self.term_dims),
            "prev_action": a_prev[:, -1],
            "xi": xi,
            "frames": frames,
        }

    def teacher_states(self, library, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        motion = self.motion
        last = motion.clip_end[motion.clip_id[frames]]
        offs = torch.arange(1, horizon + 1, device=frames.device)
        idx = torch.minimum(frames[:, None] + offs[None, :], last[:, None])
        rows = motion.ref[idx.reshape(-1).to(motion.ref.device)].float().reshape(
            frames.shape[0], horizon, -1)
        return canonical_to_flat(rows)

    def teacher_references(self, library, frames: torch.Tensor, horizon: int) -> list[FrameRef]:
        motion = self.motion
        last = motion.clip_end[motion.clip_id[frames]]
        offs = torch.arange(1, horizon + 1, device=frames.device)
        idx = torch.minimum(frames[:, None] + offs[None, :], last[:, None])
        return [FrameRef(idx[:, t].contiguous()) for t in range(horizon)]

    def teacher_goal(self, library, frames: torch.Tensor):
        """Unused on the token route (the VLA token is the conditioning); kept for API parity."""
        raise NotImplementedError(
            "the token-conditioned Isaac policy has no 680-d goal input; "
            "MotionBatchProvider skips teacher_goal on the token route"
        )

    # ------------------------------------------------------------------ WM windows
    def sample_windows(self, *, batch_size: int = 8, window: int = 8, seed: int | None = None):
        """Reference transition windows for the world-model fit (no simulator needed).

        Mirrors ``data.sample_reference_windows`` but reads the Isaac motion library, so the
        transitions the WM is fitted on have the same convention as its Isaac training data.
        """
        motion = self.motion
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        starts = motion.clip_start.detach().cpu()
        lengths = motion.clip_len.detach().cpu()
        hi = starts + lengths - 1 - window          # last start whose window stays in the clip
        ok = hi >= starts
        if not bool(ok.any()):
            raise ValueError(f"no clip is longer than window+1={window + 1} frames")
        starts, hi = starts[ok], hi[ok]
        counts = (hi - starts + 1).to(torch.float32)
        pick = torch.multinomial(counts, batch_size, replacement=True, generator=g)
        offs_in_clip = (torch.rand(batch_size, generator=g) * counts[pick]).long()
        frames = (starts[pick] + offs_in_clip).to(self.device)
        idx = frames[:, None] + torch.arange(window + 1, device=self.device)[None, :]
        rows = motion.ref[idx.reshape(-1).to(motion.ref.device)].float().reshape(
            batch_size, window + 1, -1)
        states = canonical_to_flat(rows)                       # (B, k+1, 71)
        q = rows[..., 13 : 13 + NUM_JOINTS]
        actions = (q[:, 1:] - self.default_joint_pos) / self.action_scale
        return {
            "states": states.permute(1, 0, 2).contiguous(),    # (k+1, B, 71)
            "actions": actions.permute(1, 0, 2).contiguous(),  # (k, B, 29)
            "xi": default_xi(batch_size, self.device),
            "done": torch.zeros(window, batch_size, device=self.device),
            "refs": [FrameRef(idx[:, t].contiguous()) for t in range(window)],
            "frames": frames,
        }


def _attach_clip_view(motion) -> None:
    """Expose ``motion.clips`` (start/length/name) so ``MotionBatchProvider`` can index hand labels."""
    if getattr(motion, "clips", None) is not None:
        return
    from types import SimpleNamespace

    motion.clips = [
        SimpleNamespace(start=int(s), length=int(n), name=str(name))
        for s, n, name in zip(motion.clip_start.tolist(), motion.clip_len.tolist(), motion.names)
    ]


# ---------------------------------------------------------------------- loading helpers
DEFAULT_PACKED_ROOT = ROOT / "runs" / "isaac_motion"
#: 本项目**永久**只使用纯 HA 数据（700 条）；mixed / ft318 被硬性拒绝。
DEFAULT_MOTION_SET = "ha_tokens"
FORBIDDEN_DATA_TOKENS = ("mixed", "ft318")


def forbid_legacy_dataset(paths, what: str = "dataset") -> None:
    """Hard stop on anything that is not pure HA (mixed / ft318 are permanently banned)."""
    for p in ([paths] if isinstance(paths, (str, Path)) else list(paths)):
        text = str(p).lower()
        bad = [tok for tok in FORBIDDEN_DATA_TOKENS if tok in text]
        if bad:
            raise SystemExit(
                f"{what} {p}: 含禁用数据集标记 {bad}；本工程永久只走纯 HA 数据 "
                f"(data/npz/{DEFAULT_MOTION_SET})，mixed / ft318 一律不得使用。"
            )


def is_isaac_checkpoint(ckpt: str | Path) -> bool:
    """True for a trackerlab/Isaac ``ckpt_*.pt`` (has ``pi`` + ``wm`` and no ``policy``)."""
    if isinstance(ckpt, dict):
        return "pi" in ckpt and "wm" in ckpt and "policy" not in ckpt
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    return "pi" in ck and "wm" in ck and "policy" not in ck


def ensure_packed_motion(npz: list[Path], *, out_dir: Path | None = None) -> Path:
    """Return a packed Isaac motion library for ``npz``, building it if missing or stale."""
    forbid_legacy_dataset(npz, "motion")
    if out_dir is None:
        # name the cache after the dataset directory so a second (allowed) set cannot silently
        # reuse another set's packed tensor
        first = next((p for p in npz if Path(p).is_dir()), None)
        out_dir = DEFAULT_PACKED_ROOT / (Path(first).name if first is not None else DEFAULT_MOTION_SET)
    out_dir = Path(out_dir)
    packed = out_dir / "packed.pt"
    meta = out_dir / "meta.json"
    want_clips = sum(1 for p in npz for _ in ([p] if p.suffix == ".npz" else sorted(p.glob("*.npz"))))
    if packed.exists() and meta.exists():
        import json

        info = json.loads(meta.read_text(encoding="utf-8"))
        if int(info.get("num_clips", -1)) == want_clips and info.get("ref_layout") == "canonical":
            return packed
    import subprocess

    cmd = [sys.executable, str(ROOT / "scripts" / "build_isaac_motion_packed.py"),
           "--npz", *[str(p) for p in npz], "--out", str(packed)]
    print(f"[isaac] building packed motion library: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return packed


def load_isaac_bundle(
    ckpt: str | Path,
    *,
    device: str | torch.device = "cuda",
    npz: list[Path] | None = None,
    packed_motion: str | Path | None = None,
    packed_dir: Path | None = None,
    obs_mode: str | None = None,
    hist: int | None = None,
    token_dim: int = 64,
    reward_cfg: dict | None = None,
) -> IsaacJointBundle:
    """Load the Isaac bundle (building/caching the packed motion library when needed)."""
    if packed_motion is None:
        if npz is None:
            raise ValueError("pass either packed_motion=... or npz=[...] to locate the motion data")
        packed_motion = ensure_packed_motion(list(npz), out_dir=packed_dir)
    print(
        f"[bundle] Isaac/trackerlab ckpt {ckpt} -> full Isaac stack "
        f"(policy+WM+obs+reward+termination+FK+motion); packed motion {packed_motion}",
        flush=True,
    )
    return IsaacJointBundle(
        ckpt, packed_motion=packed_motion, device=device, obs_mode=obs_mode, hist=hist,
        token_dim=token_dim, reward_cfg=reward_cfg,
    )


__all__ = [
    "IsaacJointBundle",
    "FrameRef",
    "IsaacPolicyCfg",
    "IsaacWorldModelConfig",
    "load_isaac_bundle",
    "ensure_packed_motion",
    "is_isaac_checkpoint",
    "default_xi",
    "flatten_term_major",
    "split_flat_obs",
    "flat_state_tuple",
    "canonical_to_flat",
    "canonical_to_state",
    "term_dims_for",
    "MAX_STEP_REWARD",
    "REWARD_TERMS",
    "STATE_DIM",
]
