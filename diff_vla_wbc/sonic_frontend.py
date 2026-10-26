"""SONIC token 前端（S3 用）：hold 窗口 + **环内机器人 anchor**。

背景（`docs/TOKEN_FRONTEND_HANDOVER_FOR_S3_20261026.md`）：64 维 token = 冻结 encoder(1762 obs)，
其中 580 维是「10 帧命令窗口」。HA 录制/部署填 **hold**（当前帧重复 10 次），我们旧实现填
**未来 10 帧** → 两套 token corr 只有 0.05。所以：

* S3 的 BC 标签 = npz 的 `g1_motion_token_64`（hold，录制 anchor）；
* S3 的 burn-in / 奖励回路的 token 必须用**同一个 hold builder**，且 anchor 用**当时仿真机器人的
  root 姿态**（B 模式）—— 离线 identity anchor 复现不了标签（corr 0.97/0.50/0.40，anchor 是实质成分）；
* 参考库必须是**命令流**库（`runs/isaac_motion/ha_cmd/packed.pt`）：`data/npz/ha_tokens` 的
  `joint_pos` 是执行轨迹，与命令流 mean|Δ|=0.41、最佳 lag +8 帧。

本模块只做一件事：把 `CompactMotionLib`（canonical rows）包成 `SonicTokenBuilder` 需要的
`q / qd / root_quat / last_frame_of` 视图，并暴露 `tokens(frame, robot_quat|state)`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

TOKEN_DIM = 64
NUM_JOINTS = 29

#: 哪些库是"命令流"库（q/qd 来自 encoder 的 command，而不是机器人执行轨迹）
_CMD_LIB_HINTS = ("ha_cmd",)


def library_kind(path: str | Path | None) -> str:
    """``"command"`` / ``"executed"`` / ``"unknown"`` —— 只按文件名判断，够用且不会说谎。"""
    text = str(path or "").lower()
    if any(h in text for h in _CMD_LIB_HINTS):
        return "command"
    if "ha_tokens" in text or not text:
        return "executed"
    return "unknown"


class CompactLibView:
    """`SonicTokenBuilder` 需要的最小视图（q/qd 是 MuJoCo 关节序，root_quat 是 wxyz）。"""

    def __init__(self, motion: Any, device: str | torch.device):
        self.device = torch.device(device)
        ref = motion.ref
        self.q = ref[:, 13 : 13 + NUM_JOINTS].float().to(self.device)
        self.qd = ref[:, 13 + NUM_JOINTS : 13 + 2 * NUM_JOINTS].float().to(self.device)
        self._root_quat = ref[:, 3:7].float().to(self.device)
        self._clip_end = motion.clip_end.to(self.device)
        self._clip_id = motion.clip_id.to(self.device)
        self.n = int(ref.shape[0])

    def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
        return self._clip_end[self._clip_id[frame]]

    def root_quat(self, frames: torch.Tensor) -> torch.Tensor:
        return self._root_quat[frames]


class LibraryTokenBuilder:
    """hold 窗口 SONIC token：``tokens(frames, robot_quat=...)`` 或 ``tokens(frames, state=flat_state)``。"""

    def __init__(
        self,
        motion: Any,
        *,
        device: str | torch.device = "cuda",
        window: str = "hold",
        encoder_path: str | Path | None = None,
        library_path: str | Path | None = None,
    ) -> None:
        from trackerLab.algorithms.supermimic.sonic_token import SonicTokenBuilder  # noqa: PLC0415

        self.view = CompactLibView(motion, device)
        self.builder = SonicTokenBuilder(
            self.view, device, window=window, encoder_path=encoder_path
        )
        self.window = str(self.builder.window)
        self.library_kind = library_kind(library_path)
        self.device = torch.device(device)

    # ---- 单帧 / 整段 ---------------------------------------------------------
    def __call__(
        self,
        frames: torch.Tensor,
        *,
        robot_quat: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``frames (N,)`` → ``(N,64)``。``state`` 是扁平状态 ``(N,71)``（quat 在 3:7）。"""
        frames = frames.to(self.device)
        if robot_quat is None and state is not None:
            robot_quat = state[..., 3:7].to(self.device)
        return self.builder(frames, robot_quat=robot_quat)

    def chunk(
        self,
        frames: torch.Tensor,
        *,
        robot_quat: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``frames (B,T)`` → ``(B,T,64)``：整段用**同一个**机器人 root（部署口径：一次查询出 40 帧）。"""
        if frames.ndim != 2:
            raise ValueError(f"frames must be (B,T), got {tuple(frames.shape)}")
        b, t = frames.shape
        if robot_quat is None:
            if state is None:
                raise ValueError("pass robot_quat or state")
            robot_quat = state[..., 3:7]
        rq = robot_quat.reshape(-1, 4)
        if rq.shape[0] == 1:
            rq = rq.expand(b, 4)
        elif rq.shape[0] != b:
            raise ValueError(f"robot_quat batch {rq.shape[0]} != {b}")
        flat = frames.reshape(-1)
        rq_flat = rq.repeat_interleave(t, dim=0)
        tok = self.builder(flat, robot_quat=rq_flat)
        return tok.reshape(b, t, TOKEN_DIM)

    def audit(self) -> str:
        return (f"[AUDIT][token-frontend] window={self.window} library_kind={self.library_kind} "
                f"frames={self.view.n} dim={TOKEN_DIM} "
                f"(handover: HA 标签用 hold + 录制 anchor；burn-in 用 hold + 仿真 anchor)")


__all__ = ["CompactLibView", "LibraryTokenBuilder", "TOKEN_DIM", "library_kind"]
