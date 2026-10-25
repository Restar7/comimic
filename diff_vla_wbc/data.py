"""Motion data plumbing and batch providers for joint training.

Two paths are supported:

* ``npz`` clips in the upstream ``MotionLibrary`` format (LAFAN1-style, full
  body states).  ``prepare_motion_npz`` converts the SONIC-style per-body CSV
  clips shipped in this workspace into that format using the G1 MJCF and
  differentiable FK.
* packed BONES-SEED compact ``.pt`` libraries produced by the upstream
  ``scripts/data/build_library.py``.  Those can be loaded directly with
  ``load_motion_library(..., compact=True)``.

The batch provider is intentionally agnostic about where FM tokens come from:
pass a real ``Gr00tTokenSource`` for end-to-end training, or random/offline
tokens for gradient smoke tests.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

# Data plumbing is implementation-agnostic.  The constants below are the shared contract
# (state 71, action 29, history 10); the MuJoCo/mjlab-specific helpers that used to be imported
# at module scope are pulled in **lazily inside the legacy functions only**, so the Isaac path
# (``diff_vla_wbc.isaac_joint``) never imports the mjlab stack.
ACTION_DIM = 29
HISTORY = 10
STATE_DIM = 71
GOAL_DIM = 680
XI_DIM = 33


def _default_xi(batch: int, device) -> torch.Tensor:
    xi = torch.zeros(batch, XI_DIM, device=device)
    xi[:, 0] = 0.75   # nominal friction, zero encoder bias, zero CoM offset
    return xi


def _read_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None]
    return arr


def _batch_fk(fk, qpos: torch.Tensor, qvel: torch.Tensor, batch: int = 1024):
    out = {"pos": [], "quat": [], "lin_vel": [], "ang_vel": []}
    with torch.no_grad():
        for start in range(0, qpos.shape[0], batch):
            r = fk(qpos[start : start + batch], qvel[start : start + batch])
            for key in out:
                out[key].append(r[key].cpu())
    return {k: torch.cat(v, 0).numpy().astype(np.float32) for k, v in out.items()}


def motion_npz_from_body_csv_dir(
    csv_dir: str | Path,
    output: str | Path,
    *,
    xml_path: str | Path | None = None,
    fps: float = 50.0,
    device: str = "cpu",
) -> Path:
    """Convert a SONIC ``joint_pos.csv`` + ``body_*.csv`` clip into a MotionLibrary npz.

    The per-body CSV order is not trusted.  Root pose + joint angles are fed
    through the G1 MJCF FK, which produces body states in the canonical MuJoCo
    body order.  Joint velocities are finite differences of the joint angles
    (same convention as mjlab's converter).
    """
    from .compat import G1_JOINT_NAMES, ensure_supermimic_importable, load_g1_model, quat_apply_inverse

    ensure_supermimic_importable()
    csv_dir = Path(csv_dir)
    out = Path(output)
    joint_q = _read_csv(csv_dir / "joint_pos.csv")
    body_pos = _read_csv(csv_dir / "body_pos.csv")
    body_quat = _read_csv(csv_dir / "body_quat.csv")
    body_lin = _read_csv(csv_dir / "body_lin_vel.csv")
    body_ang = _read_csv(csv_dir / "body_ang_vel.csv")
    if joint_q.shape[1] != len(G1_JOINT_NAMES):
        raise ValueError(f"{joint_q}: expected {len(G1_JOINT_NAMES)} joints, got {joint_q.shape[1]}")
    if not (joint_q.shape[0] == body_pos.shape[0] == body_quat.shape[0] == body_lin.shape[0] == body_ang.shape[0]):
        raise ValueError("CSV frame counts differ")
    # body_0 is the pelvis in the SONIC exports; use it for the free joint.
    root_pos = body_pos[:, 0:3]
    root_quat = body_quat[:, 0:4]
    root_quat = root_quat / np.linalg.norm(root_quat, axis=-1, keepdims=True)
    # Joint velocities: central differences (same convention as mjlab).
    joint_vel = np.gradient(joint_q, 1.0 / fps, axis=0)
    root_lin_vel = body_lin[:, 0:3]
    root_ang_w = body_ang[:, 0:3]
    q = torch.tensor(root_quat, dtype=torch.float32)
    root_ang_b = quat_apply_inverse(q, torch.tensor(root_ang_w, dtype=torch.float32)).numpy()

    qpos_np = np.concatenate([root_pos, root_quat, joint_q], axis=1).astype(np.float32)
    qvel_np = np.concatenate([root_lin_vel, root_ang_b, joint_vel], axis=1).astype(np.float32)
    if qpos_np.shape[1] != 36 or qvel_np.shape[1] != 35:
        raise ValueError(f"bad qpos/qvel shapes: {qpos_np.shape}, {qvel_np.shape}")

    model = load_g1_model(xml_path)
    from supermimic.kinematics.g1_fk import G1Kinematics

    fk = G1Kinematics(model, device=device)
    bodies = _batch_fk(
        fk,
        torch.tensor(qpos_np, dtype=torch.float32, device=device),
        torch.tensor(qvel_np, dtype=torch.float32, device=device),
    )
    # FK body 0 must be the pelvis and should agree with root_pos (within numerical noise).
    if fk.body_names[0] != "pelvis":
        raise ValueError(f"unexpected FK root body {fk.body_names[0]}")
    root_mae = float(np.abs(bodies["pos"][:, 0] - root_pos).mean())
    if root_mae > 5e-3:
        raise ValueError(
            f"root FK/trajectory mismatch (MAE {root_mae:.4f} m); joint or quaternion convention is wrong"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fps=np.float32(fps),
        joint_pos=joint_q.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=bodies["pos"],
        body_quat_w=bodies["quat"],
        body_lin_vel_w=bodies["lin_vel"],
        body_ang_vel_w=bodies["ang_vel"],
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(fk.body_names),
        source=str(csv_dir),
        input_fps=np.float32(fps),
    )
    return out


def motion_npz_from_lafan1_csv(
    csv_path: str | Path,
    output: str | Path,
    *,
    xml_path: str | Path | None = None,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    device: str = "cpu",
) -> Path:
    """Convert a 36-column LAFAN1/G1 retarget CSV to the motion-library npz.

    CSV layout: root xyz (m), root quaternion **xyzw**, 29 joint angles (rad).
    Resampling/velocity conventions follow the upstream ``motion_convert``
    module (linear interpolation, finite-difference velocities, SO(3) mean for
    angular velocity).
    """
    from .compat import G1_JOINT_NAMES, ensure_supermimic_importable, load_g1_model, quat_apply_inverse

    ensure_supermimic_importable()
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 36:
        raise ValueError(f"{csv_path}: expected (N,36), got {raw.shape}")
    from supermimic.data.motion_convert import resample_motion, quat_xyzw_to_wxyz

    motion = resample_motion(raw, input_fps=input_fps, output_fps=output_fps)
    joint_q = motion.joint_pos.astype(np.float32)
    qpos_np = np.concatenate([motion.root_pos, motion.root_quat, motion.joint_pos], axis=1).astype(np.float32)
    root_ang_w = motion.root_ang_vel_w
    q = torch.tensor(motion.root_quat, dtype=torch.float32)
    root_ang_b = quat_apply_inverse(q, torch.tensor(root_ang_w, dtype=torch.float32)).numpy()
    qvel_np = np.concatenate(
        [motion.root_lin_vel_w, root_ang_b, motion.joint_vel], axis=1
    ).astype(np.float32)
    model = load_g1_model(xml_path)
    from supermimic.kinematics.g1_fk import G1Kinematics

    fk = G1Kinematics(model, device=device)
    bodies = _batch_fk(
        fk,
        torch.tensor(qpos_np, device=device),
        torch.tensor(qvel_np, device=device),
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fps=np.float32(output_fps),
        joint_pos=joint_q,
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies["pos"],
        body_quat_w=bodies["quat"],
        body_lin_vel_w=bodies["lin_vel"],
        body_ang_vel_w=bodies["ang_vel"],
        joint_names=np.array(G1_JOINT_NAMES),
        body_names=np.array(fk.body_names),
        source=str(csv_path),
        input_fps=np.float32(input_fps),
    )
    return out


@dataclass
class MotionBatch:
    fm_tokens: torch.Tensor
    teacher_goals: torch.Tensor
    teacher_states: torch.Tensor
    start_state: torch.Tensor
    obs_hist: torch.Tensor
    history: torch.Tensor
    prev_action: torch.Tensor
    xi: torch.Tensor
    frames: torch.Tensor
    references: list
    teacher_goal_sequence: torch.Tensor | None = None
    model_inputs: dict | None = None
    hand_ref: torch.Tensor | None = None   # (B, T, 14) 录制手部标签（左右各 7），用于辅助监督

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class RecordedTokenSource:
    """Replay the dataset's own motion tokens (``g1_motion_token_64``) as the conditioning.

    The ft318 recordings store, per frame, the 64-d SONIC motion token that the deploy-time VLA is
    trained to emit.  Feeding *those* instead of random vectors makes the joint rollout see the
    real token distribution — this is the "real data" counterpart of the VLM/FM sampler.
    """

    def __init__(
        self,
        npz_dir: str | Path,
        library,
        *,
        token_horizon: int = 40,
        token_dim: int = 64,
        device: str | torch.device = "cuda",
        port_dir: str | Path | None = None,
    ) -> None:
        import bisect

        self.npz_dir = Path(npz_dir)
        self.library = library
        self.horizon = int(token_horizon)
        self.token_dim = int(token_dim)
        self.device = torch.device(device)
        # ---- token 空间口径（2026-10-25）----------------------------------------
        # HA npz 里的 ``g1_motion_token_64`` 来自部署侧 ``encoder_input (1762)`` 前端，而
        # trackerlab WBC 的条件空间是 640 维端口装配（未来 10 帧参考 + anchor）。实测（
        # ``scripts/probe_port_tokens.py``，动作尺度修好之后）端口 token 明显更好：
        #   端口 token 0.070 / 2.87   vs   存储标签 0.052 / 6.48
        # 所以默认优先用 ``data/port_tokens/<dataset>/<clip>.npy`` 侧车（
        # ``scripts/build_port_tokens.py`` 生成），缺失时回退到 npz 标签。
        import os as _os
        env_dir = _os.environ.get("COMIMIC_PORT_TOKENS")
        if port_dir is not None:
            self.port_dir = Path(port_dir)
        elif env_dir:
            self.port_dir = Path(env_dir)
        else:
            self.port_dir = self.npz_dir.parent.parent / "port_tokens" / self.npz_dir.name
        self.port_used = 0
        self.label_used = 0
        self._starts = [int(c.start) for c in library.clips]
        self._names = [c.name for c in library.clips]
        self._cache: dict[str, torch.Tensor] = {}
        self._bisect = bisect.bisect_right

    def _load_token(self, name: str) -> torch.Tensor:
        port = self.port_dir / f"{name}.npy"
        if port.is_file():
            tok = torch.as_tensor(np.load(port), dtype=torch.float32)
            self.port_used += 1
            return tok
        with np.load(self.npz_dir / f"{name}.npz") as d:
            if "g1_motion_token_64" not in d:
                raise KeyError(f"{name}.npz has no g1_motion_token_64 (recorded token)")
            self.label_used += 1
            return torch.as_tensor(np.ascontiguousarray(d["g1_motion_token_64"]), dtype=torch.float32)

    def _tokens(self, ci: int) -> torch.Tensor:
        name = self._names[ci]
        tok = self._cache.get(name)
        if tok is None:
            tok = self._load_token(name)
            if int(tok.shape[-1]) != self.token_dim:
                raise ValueError(f"{name}: token dim {tok.shape[-1]} != {self.token_dim}")
            if len(self._cache) > 128:
                self._cache.clear()
            self._cache[name] = tok
        return tok

    def __call__(self, meta: dict) -> torch.Tensor:
        frames = meta["frames"]
        frames = frames.detach().cpu().tolist() if torch.is_tensor(frames) else list(frames)
        out = torch.zeros(len(frames), self.horizon, self.token_dim)
        for i, f in enumerate(frames):
            ci = max(self._bisect(self._starts, int(f)) - 1, 0)
            local = int(f) - self._starts[ci]
            tok = self._tokens(ci)
            seg = tok[local : local + self.horizon]
            if seg.shape[0] < self.horizon:  # 片段尾部不足：重复最后一帧
                pad = seg[-1:].expand(self.horizon - seg.shape[0], -1)
                seg = torch.cat([seg, pad], 0)
            out[i] = seg
        return out.to(self.device)


class MotionBatchProvider:
    """Sample paired (start state, teacher goal/reference) motion windows."""

    def __init__(
        self,
        bundle: Any,
        library,
        *,
        batch_size: int = 2,
        horizon: int = 8,
        token_source: Callable[[dict], torch.Tensor] | None = None,
        token_dim: int = 64,
        token_horizon: int = 40,
        seed: int | None = None,
        hand_dir: str | Path | None = None,
    ):
        self.bundle = bundle
        self.library = library
        self.hand_dir = None if hand_dir is None else Path(hand_dir)
        self._hand_cache: dict[str, torch.Tensor] = {}
        self._hand_starts = [int(c.start) for c in library.clips]
        self._hand_names = [c.name for c in library.clips]
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self.token_source = token_source
        self.token_dim = token_dim
        self.token_horizon = token_horizon
        self.rng = torch.Generator(device="cpu")
        if seed is not None:
            self.rng.manual_seed(seed)

    def sample(self) -> MotionBatch:
        max_start = self.library.num_frames - self.horizon - 2
        if max_start <= HISTORY:
            raise ValueError("motion library is shorter than the observation/rollout horizon")
        frames = torch.randint(
            HISTORY, max_start, (self.batch_size,), generator=self.rng, dtype=torch.long
        ).to(self.bundle.device)
        start = self.bundle.reference_start(self.library, frames)
        refs = self.bundle.teacher_references(self.library, frames, self.horizon)
        # Token route (the VLA's own 64-d token IS the conditioning): there is no 680-d motion
        # goal, so building teacher goals would be wasted work — and the Isaac bundle has no
        # goal encoder at all (it raises rather than faking one).
        token_route = int(getattr(self.bundle.policy, "token_dim", 0) or 0) > 0
        if token_route:
            teacher_goals = None
            teacher_goal_sequence = None
        else:
            teacher_goals = self.bundle.teacher_goal(self.library, frames)
            teacher_goal_sequence = torch.stack(
                [
                    self.bundle.teacher_goal(self.library, frames + t)
                    for t in range(self.horizon)
                ],
                dim=1,
            )
        teacher_states = self.bundle.teacher_states(self.library, frames, self.horizon)
        meta = {
            "frames": frames,
            "start_state": start["state"],
            "history": start["observations"],
            "obs_hist": start["obs_hist"],
            "prev_action": start["prev_action"],
            "xi": start["xi"],
            "teacher_goals": teacher_goals,
            "references": refs,
        }
        if self.token_source is not None:
            tokens = self.token_source(meta)
        else:
            tokens = torch.randn(
                self.batch_size,
                self.token_horizon,
                self.token_dim,
                device=self.bundle.device,
            )
        if tokens.ndim != 3 or tokens.shape[0] != self.batch_size:
            raise ValueError(f"token source returned {tuple(tokens.shape)}")
        hand_ref = None
        if self.hand_dir is not None:
            import bisect

            hz = torch.zeros(self.batch_size, self.token_horizon, 14)
            for i, f in enumerate(frames.detach().cpu().tolist()):
                ci = max(bisect.bisect_right(self._hand_starts, int(f)) - 1, 0)
                name = self._hand_names[ci]
                tok = self._hand_cache.get(name)
                if tok is None:
                    with np.load(self.hand_dir / f"{name}.npz") as z:
                        tok = torch.as_tensor(np.ascontiguousarray(z["g1_hand_joints_14"]), dtype=torch.float32)
                    if len(self._hand_cache) > 128:
                        self._hand_cache.clear()
                    self._hand_cache[name] = tok
                local = int(f) - self._hand_starts[ci]
                seg = tok[local : local + self.token_horizon]
                if seg.shape[0] < self.token_horizon:
                    seg = torch.cat([seg, seg[-1:].expand(self.token_horizon - seg.shape[0], -1)], 0)
                hz[i] = seg
            hand_ref = hz.to(self.bundle.device)
        return MotionBatch(
            fm_tokens=tokens,
            hand_ref=hand_ref,
            teacher_goals=teacher_goals,
            teacher_states=teacher_states,
            start_state=start["state"],
            obs_hist=start["obs_hist"],
            history=start["observations"],
            prev_action=start["prev_action"],
            xi=start["xi"],
            frames=frames,
            references=refs,
            teacher_goal_sequence=teacher_goal_sequence,
        )


def _reference_state_at(bundle: Any, library, frames: torch.Tensor) -> torch.Tensor:
    r = library.ref(frames)
    return torch.cat(
        (r.root_pos_w, r.root_quat_w, r.joint_pos, r.root_lin_vel_w, r.root_ang_vel_w, r.joint_vel),
        dim=-1,
    )


def sample_reference_windows(
    bundle: Any,
    library,
    *,
    batch_size: int = 8,
    window: int = 8,
    seed: int | None = None,
):
    """Reference windows for world-model pretraining (no simulator required).

    Start frames are sampled **inside a single clip**: a window ``[f, f+window]`` that
    crossed a clip boundary would mix two unrelated motions and feed the world model an
    impossible transition.  SuperMimic guards this with ``ReplayBuffer.window_mask``
    (same episode, no terminal inside the window); here the library clips play that role.

    A bundle may provide ``sample_windows`` itself (the Isaac-native bundle does: its windows
    must come from the Isaac motion library with world-frame root velocities, not from the
    MuJoCo-style reference state builder below).
    """
    own = getattr(bundle, "sample_windows", None)
    if own is not None:
        return own(batch_size=batch_size, window=window, seed=seed)
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    starts, counts = [], []
    for clip in library.clips:
        hi = int(clip.start) + int(clip.length) - 1 - window
        if hi >= int(clip.start):
            starts.append(int(clip.start))
            counts.append(hi - int(clip.start) + 1)
    if not starts:
        raise ValueError(f"no clip is longer than window+1={window + 1} frames")
    counts_t = torch.tensor(counts, dtype=torch.float32, device="cpu")
    starts_t = torch.tensor(starts, dtype=torch.long, device="cpu")
    pick = torch.multinomial(counts_t, batch_size, replacement=True, generator=g)
    offs_in_clip = (torch.rand(batch_size, generator=g) * counts_t[pick]).long()
    frames = (starts_t[pick] + offs_in_clip).to(bundle.device)
    offs = torch.arange(window + 1, device=bundle.device)
    idx = frames[:, None] + offs[None, :]
    states = _reference_state_at(bundle, library, idx.reshape(-1)).reshape(batch_size, window + 1, STATE_DIM)
    actions = (states[:, 1:, 7:36] - bundle.default_joint_pos) / bundle.action_scale
    xi = _default_xi(batch_size, bundle.device)
    done = torch.zeros(batch_size, window, device=bundle.device)
    root = library.ref(idx.reshape(-1))
    from .compat import ensure_supermimic_importable
    from supermimic.world_model.heads import ReferenceFrame

    ensure_supermimic_importable()

    root_pos = root.root_pos_w.reshape(batch_size, window + 1, 3)
    root_quat = root.root_quat_w.reshape(batch_size, window + 1, 4)
    refs = [
        ReferenceFrame(None, None, None, None, root_pos[:, t], root_quat[:, t])
        for t in range(window)
    ]
    return {
        "states": states.permute(1, 0, 2).contiguous(),  # (k+1,B,71)
        "actions": actions.permute(1, 0, 2).contiguous(),  # (k,B,29)
        "xi": xi,
        "done": done.permute(1, 0).contiguous(),
        "refs": refs,
    }


__all__ = [
    "MotionBatch",
    "MotionBatchProvider",
    "motion_npz_from_body_csv_dir",
    "motion_npz_from_lafan1_csv",
    "sample_reference_windows",
]


class CachedTokenSource:
    """Use offline-cached GR00T tokens for a motion library.

    ``tokens`` is indexed by the *global* motion-library frame, e.g. the result
    of running ``sample_model_with_grad`` on the RGB/language clip paired with
    each frame.  This is the recommended way to train the adapter/WBC without
    keeping the 2B VLM in the loop.
    """

    def __init__(self, tokens: torch.Tensor):
        if tokens.ndim != 3:
            raise ValueError("cached tokens must be (num_frames, token_horizon, token_dim)")
        self.tokens = tokens

    def __call__(self, meta: dict) -> torch.Tensor:
        frames = meta["frames"].to(self.tokens.device)
        return self.tokens[frames]
