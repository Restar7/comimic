"""Read / write the full simulator state (qpos | qvel) as one 71-d tensor."""

from __future__ import annotations

import torch

STATE_DIM = 36 + 35
NQ = 36          # 状态里保留的关节数：29 身体 + 7 浮动基
NV = 35

# 43dof（带手）资产下，仿真 qpos 是 50（7 + 29 + 14），而 WBC/WM 的状态契约固定 71 维
# ⇒ 这里按**关节名**挑出 29 个身体关节，读时取子集、写时散回去，手 14 个关节原样保留。
_HAND_TAG = "_hand_"


class StateIO:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)

        qpos_w = int(self.sim.data.qpos.shape[-1])
        qvel_w = int(self.sim.data.qvel.shape[-1])
        self._qpos_idx: list[int] | None = None
        self._qvel_idx: list[int] | None = None
        if qpos_w != NQ or qvel_w != NV:
            import mujoco

            m = self.sim.mj_model
            names = [m.joint(i).name for i in range(m.njnt)]
            # ★ 只取**机器人**的关节：场景里的抽屉(slide)/物体(free) 不能算进来。
            #   mjlab 的 robot body 名带 `robot/` 前缀（2026-10-29 实测：不滤的话 task2 的
            #   drawer_slide 会让索引变成 37/36 而不是 36/35，StateIO 直接断言失败）。
            robot_bodies = {i for i in range(m.nbody) if m.body(i).name.startswith("robot/")}
            body = [
                i for i, n in enumerate(names)
                if _HAND_TAG not in n
                and m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
                and int(m.jnt_bodyid[i]) in robot_bodies
            ]
            self._qpos_idx = list(range(7)) + [int(m.jnt_qposadr[i]) for i in body]
            self._qvel_idx = list(range(6)) + [int(m.jnt_dofadr[i]) for i in body]
            assert len(self._qpos_idx) == NQ and len(self._qvel_idx) == NV, (
                len(self._qpos_idx), len(self._qvel_idx), qpos_w, qvel_w
            )
            print(
                f"[StateIO] 43dof: qpos {qpos_w}→{NQ}, qvel {qvel_w}→{NV}"
                f"（手 {sum(_HAND_TAG in n for n in names)} 关节不参与状态契约）",
                flush=True,
            )

    def read(self) -> torch.Tensor:
        """(N, 71) copy of [qpos(36) | qvel(35)]，只含 29 个身体关节。"""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        qpos, qvel = self.sim.data.qpos[:], self.sim.data.qvel[:]
        if self._qpos_idx is not None:
            qpos = qpos[:, self._qpos_idx]
            qvel = qvel[:, self._qvel_idx]
        s = torch.cat([qpos, qvel], dim=-1).clone()
        s[:, 0:3] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        if self._qpos_idx is None:
            self.sim.data.qpos[env_ids] = state[:, :NQ]
            self.sim.data.qvel[env_ids] = state[:, NQ:]
        else:  # 43dof：散回身体关节，手保持当前值
            qpos = self.sim.data.qpos[env_ids]
            qvel = self.sim.data.qvel[env_ids]
            qpos[:, self._qpos_idx] = state[:, :NQ]
            qvel[:, self._qvel_idx] = state[:, NQ:]
            self.sim.data.qpos[env_ids] = qpos
            self.sim.data.qvel[env_ids] = qvel
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()
