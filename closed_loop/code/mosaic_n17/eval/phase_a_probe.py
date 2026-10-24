"""Phase A cross-domain probe (M2 exit gate).

The typed extractor was trained on the 49D simulation domain; before its slots
may drive the consequence model on G1 ego frames we need evidence that they did
not collapse and still carry state/action information.  This module measures, per
slot branch ``C/B/E/S``:

* temporal variation (mean per-dimension std across frames) and effective rank;
* lag-1 cosine between consecutive frames (smoothness, not white noise);
* a time-split ridge readout of ``state46`` from the branch slots, plus a
  time-shuffled control that must not score as well;
* the correlation between slot-change magnitude and state-change magnitude.

Thresholds are heuristics recorded in the report, not physical constants: the
gate is "not collapsed AND readable AND action-related", and every number is
printed so a failure can be argued with data instead of folklore.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np

from ..ac.slots import SLOT_DIM, TOTAL_SLOTS, slot_slices
from ..contracts.state46 import joints43_to_ac_state46


MIN_TEMPORAL_STD = 1e-3
MIN_EFFECTIVE_RANK = 2.0
MIN_LAG1_COSINE = 0.5
MIN_READOUT_R2 = 0.3
MAX_SHUFFLED_R2 = 0.1
MIN_ACTION_CORR = 0.2
LOEO_RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LOEO_SHIFT_FRACTION = 0.5


@dataclass(frozen=True)
class BranchProbe:
    temporal_std: float
    effective_rank: float
    lag1_cosine: float
    readout_r2: float
    shuffled_r2: float
    non_collapsed: bool
    readable: bool
    smooth: bool


@dataclass(frozen=True)
class ProbeReport:
    branches: dict[str, BranchProbe]
    action_corr: float
    action_related: bool
    verdict: str
    reasons: list[str] = field(default_factory=list)


def _validate(slots: np.ndarray) -> np.ndarray:
    array = np.asarray(slots, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (TOTAL_SLOTS, SLOT_DIM):
        raise ValueError(f"slots must be [T,{TOTAL_SLOTS},{SLOT_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("slots must be finite")
    return array


def _lag1_cosine(features: np.ndarray) -> float:
    if features.shape[0] < 2:
        return 0.0
    a = features[:-1]
    b = features[1:]
    denom = np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12, None)
    return float(((a * b).sum(axis=1) / denom).mean())


def _effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    if energy.sum() <= 0:
        return 0.0
    return float(energy.sum() ** 2 / (energy**2).sum())


def _ridge_r2(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, ridge: float
) -> float:
    x_mean = train_x.mean(axis=0, keepdims=True)
    y_mean = train_y.mean(axis=0, keepdims=True)
    xc = train_x - x_mean
    yc = train_y - y_mean
    gram = xc.T @ xc + ridge * np.eye(xc.shape[1])
    weights = np.linalg.solve(gram, xc.T @ yc)
    prediction = (test_x - x_mean) @ weights + y_mean
    residual = ((test_y - prediction) ** 2).sum()
    total = ((test_y - test_y.mean(axis=0, keepdims=True)) ** 2).sum()
    if total <= 0:
        return 0.0
    return float(1.0 - residual / total)


def probe_slots(
    slots: np.ndarray, state46: np.ndarray, *, split: float = 0.8, ridge: float = 1e-2
) -> ProbeReport:
    """Run the M2 probe on one episode's slots and its ``state46`` trajectory."""

    slots = _validate(slots)
    state = np.asarray(state46, dtype=np.float64)
    if state.shape != (slots.shape[0], 46):
        raise ValueError(f"state46 must be [T,46] with T={slots.shape[0]}, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state46 must be finite")
    if not 0.5 <= split < 1.0:
        raise ValueError("split must be in [0.5, 1)")

    cutoff = max(4, int(round(slots.shape[0] * split)))
    branches: dict[str, BranchProbe] = {}
    for name, span in slot_slices().items():
        features = slots[:, span, :].reshape(slots.shape[0], -1)
        temporal_std = float(features.std(axis=0).mean())
        effective_rank = _effective_rank(features)
        lag1 = _lag1_cosine(features)
        readout = _ridge_r2(
            features[:cutoff], state[:cutoff], features[cutoff:], state[cutoff:], ridge
        )
        permuted = np.random.default_rng(0).permutation(features.shape[0])
        shuffled = _ridge_r2(
            features[permuted][:cutoff],
            state[:cutoff],
            features[permuted][cutoff:],
            state[cutoff:],
            ridge,
        )
        branches[name] = BranchProbe(
            temporal_std=temporal_std,
            effective_rank=effective_rank,
            lag1_cosine=lag1,
            readout_r2=readout,
            shuffled_r2=shuffled,
            non_collapsed=temporal_std >= MIN_TEMPORAL_STD and effective_rank >= MIN_EFFECTIVE_RANK,
            readable=readout >= MIN_READOUT_R2 and readout > shuffled + 0.1,
            smooth=lag1 >= MIN_LAG1_COSINE,
        )

    slot_change = np.linalg.norm(np.diff(slots.reshape(slots.shape[0], -1), axis=0), axis=1)
    state_change = np.linalg.norm(np.diff(state, axis=0), axis=1)
    if slot_change.std() > 0 and state_change.std() > 0:
        action_corr = float(np.corrcoef(slot_change, state_change)[0, 1])
    else:
        action_corr = 0.0
    action_related = abs(action_corr) >= MIN_ACTION_CORR

    reasons: list[str] = []
    for name in ("C", "B"):
        if not branches[name].non_collapsed:
            reasons.append(
                f"{name} collapsed (std<{MIN_TEMPORAL_STD} or rank<{MIN_EFFECTIVE_RANK})"
            )
    if not branches["B"].readable:
        reasons.append("B branch is not readable from state46")
    if not all(branches[name].smooth for name in branches):
        reasons.append("slots are not temporally smooth")
    if not action_related:
        reasons.append(f"slot change is not related to state change (|r|<{MIN_ACTION_CORR})")
    verdict = "pass" if not reasons else "fail"
    return ProbeReport(
        branches=branches,
        action_corr=action_corr,
        action_related=action_related,
        verdict=verdict,
        reasons=reasons,
    )


def read_state46(dataset_root: str | Path, episode: int) -> np.ndarray:
    import pyarrow.parquet as pq

    root = Path(dataset_root)
    info = json.loads((root / "meta" / "info.json").read_text())
    chunk = episode // int(info.get("chunks_size", 1000))
    relative = info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    table = pq.read_table(
        root / relative, columns=["observation.state", "observation.root_orientation"]
    )
    joints = np.stack(table["observation.state"].to_pylist()).astype(np.float64)
    quat = np.stack(table["observation.root_orientation"].to_pylist()).astype(np.float64)
    return joints43_to_ac_state46(joints, quat)


def read_slots(path: str | Path) -> np.ndarray:
    from ..data.slots_h5 import read_slots_h5

    return read_slots_h5(path)["slots"]


def run_probe(cases: list[tuple[str, Path, int, Path]], *, split: float = 0.8) -> dict:
    """``cases`` = list of (label, dataset_root, episode, slots_path)."""

    results = {}
    for label, dataset_root, episode, slots_path in cases:
        slots = read_slots(slots_path)
        state = read_state46(dataset_root, episode)
        length = min(slots.shape[0], state.shape[0])
        report = probe_slots(slots[:length], state[:length], split=split)
        results[label] = {
            "episode": episode,
            "frames": int(length),
            "action_corr": report.action_corr,
            "verdict": report.verdict,
            "reasons": report.reasons,
            "branches": {name: asdict(value) for name, value in report.branches.items()},
        }
    verdicts = [item["verdict"] for item in results.values()]
    overall = "pass" if verdicts and all(v == "pass" for v in verdicts) else "fail"
    return {"schema": "mosaic.n17.phase_a_probe.v1", "overall": overall, "cases": results}


def _standardize(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return mean, np.where(std < 1e-8, 1.0, std)


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, ridge: float) -> dict[str, object]:
    """Z-scored ridge fit, so one ridge grid means the same thing on every slot branch.

    Uses the dual form when there are fewer rows than slots: the discriminant lives in an
    ``n x n`` system then, while the primal system is ``D x D`` (D=2304 for the C branch).
    Both forms compute the same regularized least-squares solution on centered data.
    """

    x_mean, x_std = _standardize(train_x)
    y_mean, y_std = _standardize(train_y)
    xc = (train_x - x_mean) / x_std
    yc = (train_y - y_mean) / y_std
    if xc.shape[0] <= xc.shape[1]:
        gram = xc @ xc.T + float(ridge) * np.eye(xc.shape[0], dtype=xc.dtype)
        return {
            "form": "dual",
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "train_design": xc,
            "weights": np.linalg.solve(gram, yc),
        }
    gram = xc.T @ xc + float(ridge) * np.eye(xc.shape[1], dtype=xc.dtype)
    return {
        "form": "primal",
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "weights": np.linalg.solve(gram, xc.T @ yc),
    }


def _ridge_predict(model: dict[str, object], features: np.ndarray) -> np.ndarray:
    normalized = (features - model["x_mean"]) / model["x_std"]
    if model["form"] == "dual":
        prediction = (normalized @ model["train_design"].T) @ model["weights"]
    else:
        prediction = normalized @ model["weights"]
    return prediction * model["y_std"] + model["y_mean"]


def _joint_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = float(((target - prediction) ** 2).sum())
    total = float(((target - target.mean(axis=0, keepdims=True)) ** 2).sum())
    return 0.0 if total <= 0 else 1.0 - residual / total


def _concat(blocks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(blocks, axis=0)


def _cross_episode_branch(
    features: list[np.ndarray], states: list[np.ndarray], *, ridge_grid: tuple[float, ...]
) -> dict[str, object]:
    """Leave-one-episode-out readout with an inner-LOEO ridge grid.

    Fitting on frames of the held-out episode itself (the original time-split probe) confounds
    "the slot carries state" with "state barely changes over the episode", and random row splits
    leak through adjacent-frame similarity.  Holding a whole episode out tests the property the
    deployment path needs: one state->slot map that stays valid on unseen episodes.
    """

    per_case: dict[str, float] = {}
    shifted_case: dict[str, float] = {}
    constant_case: dict[str, float] = {}
    ridge_case: dict[str, float] = {}
    for held in range(len(features)):
        train_indices = [index for index in range(len(features)) if index != held]
        best_ridge, best_inner = None, None
        for ridge in ridge_grid:
            inner_scores = []
            for inner in train_indices:
                fit_indices = [index for index in train_indices if index != inner]
                model = _fit_ridge(
                    _concat([features[index] for index in fit_indices]),
                    _concat([states[index] for index in fit_indices]),
                    ridge,
                )
                inner_scores.append(_joint_r2(_ridge_predict(model, features[inner]), states[inner]))
            mean_inner = float(np.mean(inner_scores))
            if best_inner is None or mean_inner > best_inner:
                best_ridge, best_inner = ridge, mean_inner
        model = _fit_ridge(
            _concat([features[index] for index in train_indices]),
            _concat([states[index] for index in train_indices]),
            float(best_ridge),
        )
        prediction = _ridge_predict(model, features[held])
        per_case[held] = _joint_r2(prediction, states[held])
        constant = _concat([states[index] for index in train_indices]).mean(axis=0, keepdims=True)
        constant_case[held] = _joint_r2(np.broadcast_to(constant, states[held].shape), states[held])
        ridge_case[held] = float(best_ridge)
        shifted = np.roll(features[held], int(features[held].shape[0] * LOEO_SHIFT_FRACTION), axis=0)
        shifted_case[held] = _joint_r2(_ridge_predict(model, shifted), states[held])
    values = list(per_case.values())
    return {
        "readout_r2_by_case": per_case,
        "readout_r2_mean": float(np.mean(values)),
        "readout_r2_min": float(np.min(values)),
        "shifted_r2_by_case": shifted_case,
        "shifted_r2_mean": float(np.mean(list(shifted_case.values()))),
        "constant_r2_by_case": constant_case,
        "constant_r2_mean": float(np.mean(list(constant_case.values()))),
        "ridge_by_case": ridge_case,
    }


def probe_cross_episode(
    cases: list[tuple[str, np.ndarray, np.ndarray]], *, ridge_grid: tuple[float, ...] = LOEO_RIDGE_GRID
) -> dict:
    """Cross-episode probe: one readout map fitted on whole episodes, scored on held-out ones."""

    if len(cases) < 3:
        raise ValueError("cross-episode probe needs at least three episodes")
    prepared: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, slots, state in cases:
        slots = _validate(slots)
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 46:
            raise ValueError(f"{label}: state46 must be [T,46], got {state.shape}")
        length = min(slots.shape[0], state.shape[0])
        if length < 8:
            raise ValueError(f"{label}: need at least 8 aligned frames, got {length}")
        prepared.append((label, slots[:length], state[:length]))
    labels = [item[0] for item in prepared]

    branches: dict[str, dict[str, object]] = {}
    per_case_action_corr: dict[str, float] = {}
    for name, span in slot_slices().items():
        features = [item[1][:, span, :].reshape(item[1].shape[0], -1).astype(np.float64) for item in prepared]
        states = [item[2] for item in prepared]
        report = _cross_episode_branch(features, states, ridge_grid=ridge_grid)
        label_by_index = dict(enumerate(labels))
        for key in ("readout_r2_by_case", "shifted_r2_by_case", "constant_r2_by_case", "ridge_by_case"):
            report[key] = {label_by_index[index]: value for index, value in report[key].items()}
        temporal_std = [float(feature.std(axis=0).mean()) for feature in features]
        effective_rank = [float(_effective_rank(feature)) for feature in features]
        lag1 = [float(_lag1_cosine(feature)) for feature in features]
        # Per-episode centering removes the between-episode offset (each task starts from a
        # different pose/scene).  The raw readout then measures "can one map predict an unseen
        # episode's absolute state", while the centered readout measures "does the slot track
        # that episode's own state changes" -- the property Phase B actually consumes.
        centered = _cross_episode_branch(
            [feature - feature.mean(axis=0, keepdims=True) for feature in features],
            [state - state.mean(axis=0, keepdims=True) for state in states],
            ridge_grid=ridge_grid,
        )
        report["demeaned_readout_r2_mean"] = float(centered["readout_r2_mean"])
        report["demeaned_readout_r2_min"] = float(centered["readout_r2_min"])
        report["demeaned_shifted_r2_mean"] = float(centered["shifted_r2_mean"])
        report["demeaned_readout_r2_by_case"] = {
            label_by_index[index]: value for index, value in centered["readout_r2_by_case"].items()
        }
        report.update(
            {
                "temporal_std": float(np.mean(temporal_std)),
                "effective_rank": float(np.mean(effective_rank)),
                "lag1_cosine": float(np.mean(lag1)),
                "non_collapsed": float(np.mean(temporal_std)) >= MIN_TEMPORAL_STD
                and float(np.mean(effective_rank)) >= MIN_EFFECTIVE_RANK,
                "smooth": float(np.mean(lag1)) >= MIN_LAG1_COSINE,
                "readable": float(report["readout_r2_mean"]) >= MIN_READOUT_R2
                and float(report["readout_r2_mean"]) > float(report["shifted_r2_mean"]) + 0.1,
                "demeaned_readable": float(report["demeaned_readout_r2_mean"]) >= MIN_READOUT_R2
                and float(report["demeaned_readout_r2_mean"])
                > float(report["demeaned_shifted_r2_mean"]) + 0.1,
            }
        )
        branches[name] = report
        correlations = []
        for feature, state in zip(features, states):
            slot_change = np.linalg.norm(np.diff(feature, axis=0), axis=1)
            state_change = np.linalg.norm(np.diff(state, axis=0), axis=1)
            if slot_change.std() > 0 and state_change.std() > 0:
                correlations.append(float(np.corrcoef(slot_change, state_change)[0, 1]))
        per_case_action_corr.update(
            {labels[index]: value for index, value in enumerate(correlations)}
        )

    action_corr = float(np.mean(list(per_case_action_corr.values()))) if per_case_action_corr else 0.0
    action_related = abs(action_corr) >= MIN_ACTION_CORR
    reasons: list[str] = []
    for name in ("C", "B"):
        if not branches[name]["non_collapsed"]:
            reasons.append(
                f"{name} collapsed (std<{MIN_TEMPORAL_STD} or rank<{MIN_EFFECTIVE_RANK})"
            )
    if not branches["B"]["readable"]:
        reasons.append("B branch is not readable from held-out episodes")
    if not all(branches[name]["smooth"] for name in branches):
        reasons.append("slots are not temporally smooth")
    if not action_related:
        reasons.append(f"slot change is not related to state change (|r|<{MIN_ACTION_CORR})")
    return {
        "schema": "mosaic.n17.phase_a_probe.cross_episode.v1",
        "protocol": "leave_one_episode_out",
        "overall": "pass" if not reasons else "fail",
        "cases": {label: {"frames": int(slots.shape[0])} for label, slots, _ in prepared},
        "action_corr": action_corr,
        "action_corr_by_case": per_case_action_corr,
        "action_related": action_related,
        "branches": branches,
        "reasons": reasons,
    }


def run_cross_episode_probe(
    cases: list[tuple[str, Path, int, Path]],
    *,
    ridge_grid: tuple[float, ...] = LOEO_RIDGE_GRID,
    case_stride: int = 1,
) -> dict:
    """``cases`` = list of (label, dataset_root, episode, slots_path)."""

    if case_stride <= 0:
        raise ValueError("case stride must be positive")
    prepared = []
    for label, dataset_root, episode, slots_path in cases:
        slots = read_slots(slots_path)[::case_stride]
        state = read_state46(dataset_root, episode)[::case_stride]
        prepared.append((label, slots, state))
    return probe_cross_episode(prepared, ridge_grid=ridge_grid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs=4,
        action="append",
        metavar=("LABEL", "DATASET_ROOT", "EPISODE", "SLOTS"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument(
        "--protocol",
        choices=("within_episode", "leave_one_episode_out"),
        default="within_episode",
        help=(
            "within_episode keeps the original time-split readout.  leave_one_episode_out fits the "
            "readout on whole episodes and reports the held-out ones, which is the only readout that "
            "is not confounded by a single trajectory's non-stationarity or by adjacent-frame leakage."
        ),
    )
    parser.add_argument(
        "--ridge-grid",
        nargs="+",
        type=float,
        default=list(LOEO_RIDGE_GRID),
        help="Regularization grid selected by inner leave-one-episode-out on the training episodes",
    )
    parser.add_argument(
        "--case-stride",
        type=int,
        default=1,
        help="Keep every Nth row of each episode; the LOEO readout is quadratic in rows",
    )
    args = parser.parse_args()
    cases = [
        (label, Path(root), int(episode), Path(slots)) for label, root, episode, slots in args.case
    ]
    if args.protocol == "leave_one_episode_out":
        if args.case_stride <= 0:
            raise ValueError("case stride must be positive")
        report = run_cross_episode_probe(
            cases, ridge_grid=tuple(args.ridge_grid), case_stride=int(args.case_stride)
        )
    else:
        report = run_probe(cases, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "output": str(args.output)}, indent=1))


if __name__ == "__main__":
    main()
