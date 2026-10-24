#!/usr/bin/env python3
"""Decode-side acceptance for SONIC motion tokens (frozen release models).

Two sources are supported:

``--sonic-parquet``
    Paired oracle.  A recorded ``action.motion_token`` stream exists next to the
    measured state and ``action.wbc``.  The tool rebuilds the 1762D encoder
    input from the measured state and reports (a) the token cosine against the
    recorded token and (b) the frozen decoder's target error against
    ``action.wbc`` for the recorded token and for the rebuilt token.  The
    recorded token is the floor: a converted stream is expected to land close.

``--tokens-h5``
    Converted tokens for FlowBody demonstrations.  There is no recorded decoder
    output, so the decoder is driven with the demonstration's measured-state
    history and its target is compared with the demonstration command
    (``actions_49d`` body) and the measured ``states46`` body.  A sequence that
    leaves the joint range or produces non-finite output is flagged as diverged.

The decoder input layout is the release contract: token[64]@0, base_ang_vel
10x3@64, body_q 10x29@94 (IsaacLab order, default-relative), body_dq 10x29@384,
last_action 10x29@674, gravity 10x3@964; history is oldest-first at 50 Hz and
``last_action`` is the decoder's own previous output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # direct `python3 .../tools/verify_motion_tokens.py`
    sys.path.insert(0, str(_REPO_ROOT))

from mosaic_n17.contracts.state46 import quaternion_wxyz_to_matrix  # noqa: E402
from mosaic_n17.tools.convert_motion_tokens import rpy_to_quat_wxyz  # noqa: E402
from mosaic_n17.tools.motion_token_encoder import (  # noqa: E402
    FUTURE_FRAMES,
    FUTURE_STEP,
    MUJOCO_TO_ISAACLAB,
    assemble_g1_input,
    make_session,
)


FPS = 50.0
HISTORY = 10
BODY_JOINTS = 29
LOOKAHEAD = (FUTURE_FRAMES - 1) * FUTURE_STEP
ISAACLAB_TO_MUJOCO = tuple(int(index) for index in np.argsort(MUJOCO_TO_ISAACLAB))
SONIC_WBC43_BODY_INDICES = tuple(range(0, 22)) + tuple(range(29, 36))
JOINT_GROUPS = {"leg12": slice(0, 12), "waist3": slice(12, 15), "arm14": slice(15, 29)}
DIVERGENCE_LIMIT_RAD = 6.0


def _double_array(text: str, name: str, size: int) -> np.ndarray:
    block = text.split(f"{name} = {{", 1)[1].split("};", 1)[0]
    values = [float(value) for value in re.findall(r"(?<![\w.])-?\d+\.\d+", block)]
    if len(values) != size:
        raise ValueError(f"{name}: expected {size} values, parsed {len(values)}")
    return np.asarray(values)


def parse_policy_parameters(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Release default angles and per-joint action scale (MuJoCo order)."""

    text = path.read_text()
    defaults = _double_array(text, "default_angles", BODY_JOINTS)
    armature = {
        name: float(re.search(rf"const double ARMATURE_{name} = ([0-9.]+);", text).group(1))
        for name in ("5020", "7520_14", "7520_22", "4010")
    }
    effort = {
        name: float(re.search(rf"const double EFFORT_LIMIT_{name} = ([0-9.]+);", text).group(1))
        for name in ("5020", "7520_14", "7520_22", "4010")
    }
    natural_freq = 10 * 2.0 * np.pi
    environment = {f"EFFORT_LIMIT_{name}": value for name, value in effort.items()}
    environment.update(
        {f"STIFFNESS_{name}": value * natural_freq * natural_freq for name, value in armature.items()}
    )
    block = text.split("g1_action_scale = {", 1)[1].split("};", 1)[0]
    cleaned = "\n".join(line.split("//")[0] for line in block.splitlines())
    expressions = [expression.strip() for expression in cleaned.split(",") if expression.strip()]
    scales = np.asarray(
        [eval(expression, {"__builtins__": {}}, environment) for expression in expressions]
    )
    if scales.shape != (BODY_JOINTS,) or not np.all(scales > 0):
        raise ValueError(f"g1_action_scale: expected {BODY_JOINTS} positive values, got {scales}")
    return defaults, scales


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.asarray((quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]))


def projected_gravity(quaternion: np.ndarray) -> np.ndarray:
    inverse = conjugate(quaternion)
    return quaternion_multiply(
        quaternion_multiply(inverse, np.asarray((0.0, 0.0, 0.0, -1.0))), quaternion
    )[1:]


def body_angular_velocity(quaternions: np.ndarray, dt: float = 1.0 / FPS) -> np.ndarray:
    angular = np.zeros((quaternions.shape[0], 3))
    for index in range(1, quaternions.shape[0]):
        relative = quaternion_multiply(conjugate(quaternions[index - 1]), quaternions[index])
        if relative[0] < 0:
            relative = -relative
        angular[index] = 2.0 * relative[1:] / dt
    return angular


class DecoderReplay:
    """Teacher-forced release decoder replay over one measured-state history."""

    def __init__(self, decoder_session, defaults: np.ndarray, scales: np.ndarray) -> None:
        self.session = decoder_session
        self.defaults = defaults
        self.scales = scales
        self.input_name = decoder_session.get_inputs()[0].name

    def targets(self, tokens: np.ndarray, states_body: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
        frames = tokens.shape[0]
        angular = body_angular_velocity(root_quat)
        gravity = np.stack([projected_gravity(root_quat[index]) for index in range(frames)])
        q_history = np.repeat(
            (states_body[0][list(MUJOCO_TO_ISAACLAB)] - self.defaults[list(MUJOCO_TO_ISAACLAB)])[None],
            HISTORY,
            axis=0,
        )
        dq_history = np.zeros((HISTORY, BODY_JOINTS))
        action_history = np.zeros((HISTORY, BODY_JOINTS))
        angular_history = np.repeat(angular[0][None], HISTORY, axis=0)
        gravity_history = np.repeat(gravity[0][None], HISTORY, axis=0)
        targets = np.full((frames, BODY_JOINTS), np.nan)
        for tick in range(frames):
            observation = np.zeros(994)
            observation[:64] = tokens[tick]
            for offset in range(HISTORY):
                observation[64 + 3 * offset:67 + 3 * offset] = angular_history[offset]
                observation[94 + 29 * offset:123 + 29 * offset] = q_history[offset]
                observation[384 + 29 * offset:413 + 29 * offset] = dq_history[offset]
                observation[674 + 29 * offset:703 + 29 * offset] = action_history[offset]
                observation[964 + 3 * offset:967 + 3 * offset] = gravity_history[offset]
            action = self.session.run(
                None, {self.input_name: observation.reshape(1, 994).astype(np.float32)}
            )[0][0]
            if not np.isfinite(action).all():
                break
            target = self.defaults + action[list(ISAACLAB_TO_MUJOCO)] * self.scales
            targets[tick] = target
            q_now = states_body[tick][list(MUJOCO_TO_ISAACLAB)] - self.defaults[list(MUJOCO_TO_ISAACLAB)]
            dq_now = np.zeros(BODY_JOINTS)
            if tick > 0:
                dq_now = (
                    (states_body[tick] - states_body[tick - 1]) * FPS
                )[list(MUJOCO_TO_ISAACLAB)]
            q_history = np.concatenate((q_history[1:], q_now[None]))
            dq_history = np.concatenate((dq_history[1:], dq_now[None]))
            action_history = np.concatenate((action_history[1:], action[None]))
            angular_history = np.concatenate((angular_history[1:], angular[tick][None]))
            gravity_history = np.concatenate((gravity_history[1:], gravity[tick][None]))
        return targets


def _six_dimensional(relative: np.ndarray, layout: str) -> np.ndarray:
    """``[N,3,3]`` relative rotations -> ``[N,6]`` in the requested release layout."""

    if layout == "row-major":
        return relative[:, :, :2].reshape(relative.shape[0], 6)
    if layout == "column-major":
        return np.concatenate((relative[:, :, 0], relative[:, :, 1]), axis=1)
    raise ValueError(f"unknown orientation layout {layout!r}")


def target_metrics(predicted: np.ndarray, reference: np.ndarray, valid: int) -> dict:
    error = np.abs(predicted[:valid] - reference[:valid])
    finite = np.isfinite(error)
    if not finite.all():
        return {"frames": int(valid), "diverged": True}
    per_joint = error.mean(axis=0)
    return {
        "frames": int(valid),
        "mae": float(per_joint.mean()),
        "median_frame_mae": float(np.median(error.mean(axis=1))),
        "per_joint_mae": [float(value) for value in per_joint],
        "groups": {
            name: float(per_joint[span].mean()) for name, span in JOINT_GROUPS.items()
        },
        "diverged": bool(
            not np.isfinite(predicted[:valid]).all()
            or float(np.abs(predicted[:valid]).max(initial=0.0)) > DIVERGENCE_LIMIT_RAD
        ),
    }


def _replay_token_h5(args, replay: DecoderReplay, sink: dict) -> None:
    with h5py.File(args.mixed_h5, "r") as source, h5py.File(args.tokens_h5, "r") as tokens_file:
        records = source[args.group]
        tokens_group = tokens_file["tokens"]
        all_keys = [key for key in tokens_group if key.startswith(args.prefix)]
        keys = [
            key
            for key in all_keys
            if tokens_group[key].attrs.get("copy_of", key) == key
        ]
        sink["duplicate_records"] = len(all_keys) - len(keys)
        if args.limit:
            keys = keys[: args.limit]
        if not keys:
            raise ValueError(f"no token record starts with prefix {args.prefix!r}")
        sequences: dict = {}
        for key in keys:
            states = records[key]["states46"][...].astype(np.float64)
            actions = records[key]["actions_49d"][...].astype(np.float64)
            states_body = states[:, 3:32]
            action_body = np.concatenate((actions[:, 28:43], actions[:, 0:14]), axis=1)
            tokens = tokens_group[key][...].astype(np.float64)
            valid = int(tokens_group[key].attrs["valid_frames"])
            root_quat = rpy_to_quat_wxyz(states[:, 0:3])
            targets = replay.targets(tokens[:valid], states_body, root_quat)
            if not np.isfinite(targets).all():
                sequences[key] = {"diverged": True}
                continue
            sequences[key] = {
                "against_action": target_metrics(targets, action_body, valid),
                "against_state": target_metrics(targets, states_body, valid),
            }
            print(
                f"{key}: action {sequences[key]['against_action']['mae']:.4f} "
                f"state {sequences[key]['against_state']['mae']:.4f} frames {valid}",
                flush=True,
            )
    sink["mode"] = "tokens_h5"
    sink["sequences"] = sequences
    sink["aggregate"] = _aggregate(sequences)


def _aggregate(sequences: dict) -> dict:
    action = [entry["against_action"]["mae"] for entry in sequences.values() if "against_action" in entry]
    state = [entry["against_state"]["mae"] for entry in sequences.values() if "against_state" in entry]
    diverged = sum(1 for entry in sequences.values() if entry.get("diverged"))
    if not action:
        return {"sequences": len(sequences), "diverged": diverged}
    return {
        "sequences": len(sequences),
        "diverged": diverged,
        "action_mae_mean": float(np.mean(action)),
        "action_mae_median": float(np.median(action)),
        "action_mae_p90": float(np.percentile(action, 90)),
        "state_mae_mean": float(np.mean(state)),
        "state_mae_median": float(np.median(state)),
    }


def _oracle(args, replay: DecoderReplay, encoder_session, sink: dict) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(
        args.sonic_parquet,
        columns=["action.motion_token", "action.wbc", "observation.state", "observation.root_orientation"],
    )
    recorded = np.asarray(table["action.motion_token"].to_pylist(), dtype=np.float64)
    wbc = np.asarray(table["action.wbc"].to_pylist(), dtype=np.float64)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    root_quat = np.asarray(table["observation.root_orientation"].to_pylist(), dtype=np.float64)
    frames = recorded.shape[0] if args.max_frames <= 0 else min(args.max_frames, recorded.shape[0])
    recorded = recorded[:frames]
    wbc_body = wbc[:frames][:, list(SONIC_WBC43_BODY_INDICES)]
    states_body = state[:frames][:, list(SONIC_WBC43_BODY_INDICES)]
    root_quat = root_quat[:frames]

    q = states_body
    dq = np.zeros_like(q)
    dq[1:] = (q[1:] - q[:-1]) * FPS
    q_isaaclab = q[:, list(MUJOCO_TO_ISAACLAB)]
    dq_isaaclab = dq[:, list(MUJOCO_TO_ISAACLAB)]
    offsets = FUTURE_STEP * np.arange(FUTURE_FRAMES)
    valid = frames - LOOKAHEAD
    rebuilt = np.full((frames, 64), np.nan)
    input_name = encoder_session.get_inputs()[0].name
    for start in range(valid):
        window = start + offsets
        relative = np.einsum(
            "ij,njk->nik",
            quaternion_wxyz_to_matrix(root_quat[start]).T,
            quaternion_wxyz_to_matrix(root_quat[window]),
        )
        anchor = _six_dimensional(relative, args.orientation_layout)
        encoder_input = assemble_g1_input(q_isaaclab[window], dq_isaaclab[window], anchor)
        rebuilt[start] = encoder_session.run(None, {input_name: encoder_input})[0][0]
    cosine = np.sum(rebuilt[:valid] * recorded[:valid], axis=1) / (
        np.linalg.norm(rebuilt[:valid], axis=1) * np.linalg.norm(recorded[:valid], axis=1) + 1e-12
    )
    recorded_targets = replay.targets(recorded[:valid], states_body[:valid], root_quat[:valid])
    rebuilt_targets = replay.targets(rebuilt[:valid], states_body[:valid], root_quat[:valid])
    sink["mode"] = "sonic_oracle"
    sink["parquet"] = str(args.sonic_parquet)
    sink["frames"] = int(valid)
    sink["token_cosine"] = {
        "mean": float(np.mean(cosine)),
        "median": float(np.median(cosine)),
        "p05": float(np.percentile(cosine, 5)),
    }
    sink["recorded_token"] = {
        "against_wbc": target_metrics(recorded_targets, wbc_body, valid),
        "against_state": target_metrics(recorded_targets, states_body, valid),
    }
    sink["rebuilt_token"] = {
        "against_wbc": target_metrics(rebuilt_targets, wbc_body, valid),
        "against_state": target_metrics(rebuilt_targets, states_body, valid),
    }
    for label in ("recorded_token", "rebuilt_token"):
        print(
            f"{label}: wbc {sink[label]['against_wbc']['mae']:.4f} "
            f"state {sink[label]['against_state']['mae']:.4f}",
            flush=True,
        )
    print(f"token cosine mean {sink['token_cosine']['mean']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-onnx", type=Path, required=True)
    parser.add_argument("--policy-parameters", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sonic-parquet", type=Path)
    source.add_argument("--tokens-h5", type=Path)
    parser.add_argument("--encoder-onnx", type=Path, help="required with --sonic-parquet")
    parser.add_argument("--mixed-h5", type=Path, help="required with --tokens-h5")
    parser.add_argument("--group", default="sequences")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means every frame")
    parser.add_argument(
        "--orientation-layout",
        choices=("row-major", "column-major"),
        default="row-major",
        help="row-major is the release contract; column-major reproduces the old converter",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    defaults, scales = parse_policy_parameters(args.policy_parameters)
    decoder = make_session(args.decoder_onnx)
    replay = DecoderReplay(decoder, defaults, scales)
    sink: dict = {
        "decoder_onnx": str(args.decoder_onnx),
        "policy_parameters": str(args.policy_parameters),
        "default_angles": defaults.tolist(),
        "action_scale": scales.tolist(),
    }
    if args.sonic_parquet is not None:
        if args.encoder_onnx is None:
            raise SystemExit("--encoder-onnx is required with --sonic-parquet")
        _oracle(args, replay, make_session(args.encoder_onnx), sink)
    else:
        if args.mixed_h5 is None:
            raise SystemExit("--mixed-h5 is required with --tokens-h5")
        _replay_token_h5(args, replay, sink)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sink, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregate": sink.get("aggregate", sink.get("token_cosine"))}, indent=1))


if __name__ == "__main__":
    main()
