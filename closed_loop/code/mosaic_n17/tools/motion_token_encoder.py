"""Frozen SONIC encoder adapter: trajectories -> 64D motion tokens (G1 mode).

The released ONNX encoder is used **as-is**; this module only assembles its
1762-dim input.  Layout is pinned by
``GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/sonic_g1_runtime_contract.hpp``:
selector@0=0, encoder_index@1=1, future_q[10,29]@4, future_dq[10,29]@294,
future_root_ori6d[10,6]@601, everything else zero.
"""

from __future__ import annotations

from pathlib import Path

from mosaic_n17.contracts.state46 import quaternion_wxyz_to_matrix
import numpy as np


#: MuJoCo DOF order -> IsaacLab/policy slot order, verbatim from
#: ``policy_parameters.hpp::mujoco_to_isaaclab``.  The encoder input q/dq slots
#: are in the IsaacLab order (the training motion lib applies this permutation
#: in ``motion_lib_base.py`` before exposing ``get_dof_pos``).
MUJOCO_TO_ISAACLAB = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
)

ENCODER_INPUT_DIM = 1762
BODY_JOINTS = 29
FUTURE_FRAMES = 10
FUTURE_STEP = 5
OFF_SELECTOR = 0
OFF_ENCODER_INDEX = 1
OFF_Q = 4
OFF_DQ = 294
OFF_ROOT_ORI6D = 601


def assemble_g1_input(
    future_q: np.ndarray,
    future_dq: np.ndarray,
    future_root_ori6d: np.ndarray,
    *,
    encoder_index: float = 1.0,
    selector: float = 0.0,
) -> np.ndarray:
    """Assemble ``[1,1762]`` from ``[10,29]`` q/dq and ``[10,6]`` root 6D."""

    q = np.asarray(future_q, dtype=np.float32)
    dq = np.asarray(future_dq, dtype=np.float32)
    r6 = np.asarray(future_root_ori6d, dtype=np.float32)
    expected_shapes = (
        ("future_q", q, (FUTURE_FRAMES, BODY_JOINTS)),
        ("future_dq", dq, (FUTURE_FRAMES, BODY_JOINTS)),
        ("future_root_ori6d", r6, (FUTURE_FRAMES, 6)),
    )
    for name, value, shape in expected_shapes:
        if value.shape != shape:
            raise ValueError(f"{name} must be {shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    out = np.zeros((1, ENCODER_INPUT_DIM), dtype=np.float32)
    out[0, OFF_SELECTOR] = float(selector)
    out[0, OFF_ENCODER_INDEX] = float(encoder_index)
    out[0, OFF_Q : OFF_Q + FUTURE_FRAMES * BODY_JOINTS] = q.reshape(-1)
    out[0, OFF_DQ : OFF_DQ + FUTURE_FRAMES * BODY_JOINTS] = dq.reshape(-1)
    out[0, OFF_ROOT_ORI6D : OFF_ROOT_ORI6D + FUTURE_FRAMES * 6] = r6.reshape(-1)
    return out


def rotation_6d(rel: np.ndarray) -> np.ndarray:
    """``[...,3,3]`` -> ``[...,6]``: the first two matrix columns, row-major.

    The release C++ (``motion_anchor_ori_b_mf_nonflat``) writes
    ``[R00, R01, R10, R11, R20, R21]``.  Stacking the two columns
    (``[R00, R10, R20, R01, R11, R21]``) is a different contract: replaying a
    token assembled that way lands ~0.75 rad away from the recorded target,
    while this layout lands at the recorded-token floor.
    """

    matrix = np.asarray(rel, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {matrix.shape}")
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,)).copy()


def relative_orientation_6d(future_quat_wxyz: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
    """``inv(base) * future`` for ``[N,4]`` future and one ``[4]`` base quaternion."""

    future = np.asarray(future_quat_wxyz, dtype=np.float64)
    base = np.asarray(base_quat_wxyz, dtype=np.float64)
    if future.ndim != 2 or future.shape[-1] != 4:
        raise ValueError(f"future quaternions must be [N,4], got {future.shape}")
    if base.shape != (4,):
        raise ValueError(f"base quaternion must be [4], got {base.shape}")
    future_matrix = quaternion_wxyz_to_matrix(future)
    base_matrix = quaternion_wxyz_to_matrix(base)
    return rotation_6d(np.einsum("ij,njk->nik", base_matrix.T, future_matrix))


def make_session(onnx_path: str | Path):
    import onnxruntime as ort

    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def encode_windows(session, q: np.ndarray, dq: np.ndarray, root_ori6d: np.ndarray) -> np.ndarray:
    """Encode ``N`` aligned windows -> ``[N,64]`` tokens."""

    tokens = []
    for index in range(q.shape[0]):
        inp = assemble_g1_input(q[index], dq[index], root_ori6d[index])
        tokens.append(session.run(None, {session.get_inputs()[0].name: inp})[0][0])
    return np.stack(tokens)


__all__ = [
    "MUJOCO_TO_ISAACLAB",
    "assemble_g1_input",
    "encode_windows",
    "make_session",
    "relative_orientation_6d",
    "rotation_6d",
]
