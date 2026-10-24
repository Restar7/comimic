"""Phase B training entry: typed consequence model on SONIC78 data (M3).

Training pairs are built from two on-disk assets that already exist in this
workspace:

* a LeRobot SONIC root (``observation.state`` 43D, ``observation.root_orientation``
  4D, ``action.motion_token`` 64D and the two 7D hand columns);
* the precomputed typed-slot file per episode (:mod:`mosaic_n17.data.slots_h5`).

Supervision is the *executed prefix* only: with ``H=40`` the robot executes 20
steps before replanning, so the target slot index must be ``1..20``.  The training
step asserts this instead of silently labelling an unexecuted future.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from ..ac.checkpoint import PhaseBContract, load_phase_b_checkpoint, save_phase_b_checkpoint
from ..ac.phase_b_runtime import PhaseBConfig, Sonic78PhaseB, validate_consequence_supervision
from ..ac.slots import SLOT_DIM, TOTAL_SLOTS, masked_huber, slot_slices
from ..contracts.sonic78 import SONIC78_V1
from ..contracts.state46 import joints43_to_ac_state46
from ..data.slots_h5 import read_slots_h5
from ..eval.phase_b_diagnostics import DiagnosticsConfig, evaluate_phase_b


SOURCE_MANIFEST_SCHEMA = "mosaic.n17.phase_b_sources.v1"


@dataclass(frozen=True)
class PhaseBSource:
    """One LeRobot root plus the episode split it contributes to."""

    dataset_root: Path
    slots_dir: Path
    train_episodes: tuple[int, ...] | None = None
    val_episodes: tuple[int, ...] | None = None


def _episode_tuple(value: object, *, key: str) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return parse_episode_spec(value)
    if isinstance(value, (list, tuple)):
        return parse_episode_spec(",".join(str(int(item)) for item in value))
    raise ValueError(f"{key} must be a list of ints or an episode spec string")


def load_source_manifest(path: str | Path) -> list[PhaseBSource]:
    """Read the explicit multi-root source table; no implicit episode guessing."""

    payload = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError(f"source manifest must declare schema {SOURCE_MANIFEST_SCHEMA}")
    entries = payload.get("sources")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source manifest requires a non-empty sources list")
    sources: list[PhaseBSource] = []
    seen_slots: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("source entry must be an object")
        dataset_root = Path(str(entry.get("dataset_root", ""))).resolve(strict=True)
        slots_dir = Path(str(entry.get("slots_dir", ""))).resolve(strict=True)
        key = str(slots_dir)
        if key in seen_slots:
            raise ValueError(f"source manifest lists slots_dir twice: {slots_dir}")
        seen_slots.add(key)
        train = _episode_tuple(entry.get("train_episodes"), key="train_episodes")
        val = _episode_tuple(entry.get("val_episodes"), key="val_episodes")
        if train is None and val is None:
            raise ValueError(f"source {dataset_root} contributes neither train nor val episodes")
        sources.append(
            PhaseBSource(
                dataset_root=dataset_root,
                slots_dir=slots_dir,
                train_episodes=None if train is None else tuple(train),
                val_episodes=None if val is None else tuple(val),
            )
        )
    return sources


def build_split(
    sources: list[PhaseBSource],
    split: str,
    *,
    phase_a_sha256: str,
    max_frames: int | None = None,
) -> list[EpisodeArrays]:
    """Concatenate one split across sources; episode indices stay source-local."""

    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    episodes: list[EpisodeArrays] = []
    for source in sources:
        selected = source.train_episodes if split == "train" else source.val_episodes
        if selected is None and split == "val":
            continue
        episodes.extend(
            build_dataset(
                source.dataset_root,
                source.slots_dir,
                episodes=None if selected is None else list(selected),
                phase_a_sha256=phase_a_sha256,
                max_frames=max_frames,
            )
        )
    return episodes


@dataclass(frozen=True)
class EpisodeArrays:
    actions: np.ndarray  # [T,78]
    state46: np.ndarray  # [T,46]
    slots: np.ndarray  # [T,14,384]
    valid: np.ndarray  # [T]
    source: str
    #: 50 Hz ticks per stored slot row (1 for per-tick stores, 2 for stride-2 Phase-A stores)
    slots_stride: int = 1


def _read_episode(
    root: Path, episode: int, slots_path: Path, *, expected_phase_a_sha256: str | None = None
) -> EpisodeArrays:
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    chunk = episode // int(info.get("chunks_size", 1000))
    relative = info["data_path"].format(episode_chunk=chunk, episode_index=episode)
    table = pq.read_table(root / relative)
    columns = set(table.column_names)
    required = {
        "observation.state",
        "observation.root_orientation",
        "action.motion_token",
        "teleop.left_hand_joints",
        "teleop.right_hand_joints",
    }
    missing = required - columns
    if missing:
        raise ValueError(f"{root} episode {episode} is missing columns {sorted(missing)}")
    joints43 = np.stack(table["observation.state"].to_pylist()).astype(np.float64)
    quat = np.stack(table["observation.root_orientation"].to_pylist()).astype(np.float64)
    motion = np.stack(table["action.motion_token"].to_pylist()).astype(np.float64)
    left = np.stack(table["teleop.left_hand_joints"].to_pylist()).astype(np.float64)
    right = np.stack(table["teleop.right_hand_joints"].to_pylist()).astype(np.float64)
    actions = np.concatenate((motion, left, right), axis=-1)
    SONIC78_V1.validate(actions)
    state46 = joints43_to_ac_state46(joints43, quat)
    slot_payload = read_slots_h5(slots_path)
    slots = slot_payload["slots"]
    valid = slot_payload["valid"]
    declared = str(slot_payload["attrs"].get("phase_a_sha256", ""))
    if expected_phase_a_sha256 is not None and declared != expected_phase_a_sha256:
        raise ValueError(
            f"{slots_path} was produced by Phase A {declared or 'undeclared'}, not "
            f"{expected_phase_a_sha256}; re-export the slots before training"
        )
    length = min(actions.shape[0], state46.shape[0], slots.shape[0])
    if length < SONIC78_V1.horizon + 2:
        raise ValueError(f"episode {episode} is too short for H={SONIC78_V1.horizon}")
    slots_stride = int(slot_payload["attrs"].get("stride", 1))
    if slots_stride <= 0:
        raise ValueError(f"{slots_path} declares an invalid slot stride {slots_stride}")
    if slots_stride > 1:
        # The action/state arrays are 50 Hz; a stride-2 slot store covers every other tick.
        length = min(actions.shape[0], state46.shape[0], slots.shape[0] * slots_stride)
        if length < SONIC78_V1.horizon + 2:
            raise ValueError(f"episode {episode} is too short for H={SONIC78_V1.horizon}")
        usable_slots = length // slots_stride
        return EpisodeArrays(
            actions=actions[:length],
            state46=state46[:length],
            slots=slots[:usable_slots],
            valid=valid[:usable_slots],
            source=str(root),
            slots_stride=slots_stride,
        )
    return EpisodeArrays(
        actions=actions[:length],
        state46=state46[:length],
        slots=slots[:length],
        valid=valid[:length],
        source=str(root),
        slots_stride=slots_stride,
    )


def parse_episode_spec(spec: str | None) -> list[int] | None:
    """Parse ``0-3,7,9`` into an ascending episode list; ``None`` means "all"."""

    if spec is None:
        return None
    episodes: set[int] = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, _, stop = token.partition("-")
            low, high = int(start), int(stop)
            if low < 0 or high < low:
                raise ValueError(f"invalid episode range {token!r}")
            episodes.update(range(low, high + 1))
        else:
            value = int(token)
            if value < 0:
                raise ValueError(f"invalid episode index {token!r}")
            episodes.add(value)
    if not episodes:
        raise ValueError("episode spec selected no episodes")
    return sorted(episodes)


def build_dataset(
    dataset_root: str | Path,
    slots_dir: str | Path,
    *,
    episodes: list[int] | tuple[int, ...] | None = None,
    phase_a_sha256: str | None = None,
    max_episodes: int | None = None,
    max_frames: int | None = None,
) -> list[EpisodeArrays]:
    """Load episodes; ``max_episodes``/``max_frames`` are explicit smoke limits."""

    root = Path(dataset_root)
    slots_root = Path(slots_dir)
    info = json.loads((root / "meta" / "info.json").read_text())
    total = int(info["total_episodes"])
    if episodes is None:
        selected = list(range(total))
    else:
        selected = [int(value) for value in episodes]
        if any(value < 0 or value >= total for value in selected):
            raise ValueError(f"episode selection leaves [0,{total})")
        if not selected:
            raise ValueError("episode selection is empty")
        if len(set(selected)) != len(selected):
            raise ValueError("episode selection contains duplicates")
    if max_episodes is not None:
        selected = selected[: int(max_episodes)]
    episodes_list: list[EpisodeArrays] = []
    for episode in selected:
        slots_path = slots_root / f"episode_{episode:06d}.slots.h5"
        if not slots_path.exists():
            raise FileNotFoundError(slots_path)
        arrays = _read_episode(
            root, episode, slots_path, expected_phase_a_sha256=phase_a_sha256
        )
        if max_frames is not None:
            limit = min(int(max_frames), arrays.actions.shape[0])
            arrays = EpisodeArrays(
                actions=arrays.actions[:limit],
                state46=arrays.state46[:limit],
                slots=arrays.slots[:limit],
                valid=arrays.valid[:limit],
                source=arrays.source,
            )
        episodes_list.append(arrays)
    if not episodes_list:
        raise ValueError(f"no episodes under {root}")
    return episodes_list


def _delta_magnitude_weights(
    episodes: list[EpisodeArrays], supervision_step: int = SONIC78_V1.execute_steps
) -> torch.Tensor:
    """Per-dimension mean ``|future slot - current slot|`` over decision rows.

    The slot L1 is dominated by dimensions that barely move; weighting by the observed delta puts
    the gradient budget on the action-sensitive part instead of the static background.
    """

    total: np.ndarray | None = None
    count = 0
    for episode in episodes:
        stride = int(episode.slots_stride)
        supervision_dense = supervision_step // stride
        if supervision_dense <= 0:
            raise ValueError(f"supervision step {supervision_step} is shorter than one slot stride")
        usable = episode.slots.shape[0] - supervision_dense
        if usable <= 0:
            continue
        for t in range(0, usable, supervision_dense * 8):
            target = t + supervision_dense
            if not bool(episode.valid[t]) or not bool(episode.valid[target]):
                continue
            delta = np.abs(episode.slots[target] - episode.slots[t])
            total = delta if total is None else total + delta
            count += 1
    if count == 0 or total is None:
        raise ValueError("no valid decision row for delta weighting")
    weights = torch.from_numpy(total / count).float()
    return weights / weights.mean()


def phase_b_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "equal",
    delta: float = 1.0,
    dim_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked Huber over slots; ``branch_balanced`` weights C/B/E/S equally.

    The equal-dimension loss gives the 6 C slots (43% of the width) the same budget as the
    3 B slots, so scene-appearance terms can dominate the gradient while the action-induced
    B/E change is small.  ``branch_balanced`` averages within each branch first so every
    branch contributes one quarter of the loss.
    """

    if predicted.shape != target.shape:
        raise ValueError("predicted and target slots must share a shape")
    if mode == "delta_weighted":
        if dim_weight is None:
            raise ValueError("delta_weighted loss needs a per-dimension dim_weight")
        if dim_weight.shape != predicted.shape[1:]:
            raise ValueError(
                f"dim_weight must be {tuple(predicted.shape[1:])}, got {tuple(dim_weight.shape)}"
            )
        abs_error = (predicted - target).abs()
        quadratic = torch.clamp(abs_error, max=delta)
        element = 0.5 * quadratic.square() + delta * (abs_error - quadratic)
        weight = dim_weight.to(element.dtype).unsqueeze(0)
        return (element * weight).sum() / weight.expand_as(element).sum()
    mask = torch.ones_like(predicted, dtype=torch.bool)
    if mode == "equal":
        return masked_huber(predicted, target, mask, delta=delta)
    if mode != "branch_balanced":
        raise ValueError(f"unknown Phase B loss mode {mode!r}")
    slices = slot_slices()
    terms = [
        masked_huber(predicted[:, span], target[:, span], mask[:, span], delta=delta)
        for span in slices.values()
    ]
    return torch.stack(terms).mean()


def ranking_margin_loss(
    recorded_error: torch.Tensor,
    wrong_errors: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Hinge ranking loss: the recorded (executed) action must beat every wrong action.

    ``recorded_error`` is ``[B]``; ``wrong_errors`` is ``[B,K-1]``.  This directly optimizes the
    candidate-selection ordering used by the diagnostics instead of only absolute slot L1.
    """

    if recorded_error.ndim != 1 or wrong_errors.ndim != 2:
        raise ValueError("recorded_error must be [B] and wrong_errors [B,K-1]")
    if recorded_error.shape[0] != wrong_errors.shape[0]:
        raise ValueError("recorded_error and wrong_errors must share the batch size")
    if not torch.isfinite(recorded_error).all() or not torch.isfinite(wrong_errors).all():
        raise ValueError("ranking errors must be finite")
    if margin < 0:
        raise ValueError("ranking margin must be non-negative")
    return torch.relu(margin + recorded_error.unsqueeze(1) - wrong_errors).mean()


def _sample_batch(
    episodes: list[EpisodeArrays],
    batch: int,
    *,
    action_horizon: int,
    supervision_step: int,
    rng: np.random.Generator,
    history_slots: int = 0,
    prev_action_ticks: int = 0,
) -> dict[str, torch.Tensor]:
    current_slots, state46, actions, targets, target_states = [], [], [], [], []
    history, prev_actions = [], []
    for _ in range(batch):
        episode = episodes[int(rng.integers(len(episodes)))]
        stride = int(episode.slots_stride)
        supervision_dense = supervision_step // stride
        if supervision_dense <= 0:
            raise ValueError(f"supervision step {supervision_step} is shorter than one slot stride")
        max_slot = min(
            episode.slots.shape[0] - supervision_dense,
            (episode.actions.shape[0] - action_horizon) // stride,
        )
        if max_slot <= 0:
            continue
        # The policy decides every ``supervision_step`` 50 Hz ticks.  Starting a window at an
        # arbitrary slot row can straddle two replanned chunks, so the action block would mix
        # tokens from different decisions; only decision-grid starts are valid.
        starts = np.arange(0, max_slot, supervision_dense, dtype=np.int64)
        if starts.size == 0:
            continue
        t = int(starts[int(rng.integers(0, starts.size))])
        target = t + supervision_dense
        if not episode.valid[target]:
            continue
        tick = t * stride
        current_slots.append(episode.slots[t])
        state46.append(episode.state46[tick])
        actions.append(episode.actions[tick : tick + action_horizon])
        targets.append(episode.slots[target])
        target_states.append(episode.state46[target * stride])
        if history_slots:
            history.append(
                np.stack([episode.slots[max(t - h, 0)] for h in range(1, history_slots + 1)])
            )
        if prev_action_ticks:
            past = np.zeros((prev_action_ticks, SONIC78_V1.action_dim), dtype=np.float32)
            available = min(tick, prev_action_ticks)
            if available:
                past[prev_action_ticks - available :] = episode.actions[tick - available : tick]
            prev_actions.append(past)
    if not actions:
        raise RuntimeError("no valid sample in this batch; check slot validity masks")
    return {
        "current_slots": torch.from_numpy(np.stack(current_slots)).float(),
        "state46": torch.from_numpy(np.stack(state46)).float(),
        "actions": torch.from_numpy(np.stack(actions)).float(),
        "target_slots": torch.from_numpy(np.stack(targets)).float(),
        "target_state46": torch.from_numpy(np.stack(target_states)).float(),
        **(
            {"history_slots": torch.from_numpy(np.stack(history)).float()}
            if history_slots
            else {}
        ),
        **(
            {"prev_actions": torch.from_numpy(np.stack(prev_actions)).float()}
            if prev_action_ticks
            else {}
        ),
    }


def train(
    *,
    dataset_root: Path | None = None,
    slots_dir: Path | None = None,
    sources: list[PhaseBSource] | None = None,
    output_path: Path,
    phase_a_sha256: str,
    max_steps: int = 200,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    seed: int = 0,
    device: str = "cpu",
    max_episodes: int | None = None,
    max_frames: int | None = None,
    report_path: Path | None = None,
    eval_stride: int = 5,
    eval_max_rows: int | None = None,
    action_horizon: int = SONIC78_V1.horizon,
    loss_mode: str = "equal",
    init_from: Path | None = None,
    save_every_steps: int = 0,
    aux_state46_weight: float = 0.0,
    aux_state46_shuffle: bool = False,
    history_slots: int = 0,
    prev_action_ticks: int = 0,
    depth: int = 2,
    num_heads: int = 8,
    ranking_weight: float = 0.0,
    ranking_k: int = 4,
    ranking_margin: float = 0.0,
) -> dict:
    if loss_mode not in ("equal", "branch_balanced", "delta_weighted"):
        raise ValueError("loss_mode must be equal, branch_balanced or delta_weighted")
    if aux_state46_weight < 0:
        raise ValueError("aux_state46_weight must be non-negative")
    if aux_state46_shuffle and aux_state46_weight == 0:
        raise ValueError("aux_state46_shuffle needs a positive aux_state46_weight")
    if not 0 <= history_slots <= 4:
        raise ValueError("history_slots must be in [0,4]")
    if prev_action_ticks not in (0, SONIC78_V1.execute_steps):
        raise ValueError(f"prev_action_ticks must be 0 or {SONIC78_V1.execute_steps}")
    if ranking_weight < 0 or ranking_margin < 0 or ranking_k < 2:
        raise ValueError("ranking_weight/margin must be non-negative and ranking_k >= 2")
    if action_horizon not in (SONIC78_V1.execute_steps, SONIC78_V1.horizon):
        raise ValueError(
            f"action_horizon must be {SONIC78_V1.execute_steps} (executed prefix) or "
            f"{SONIC78_V1.horizon} (full candidate)"
        )
    if len(phase_a_sha256) != 64:
        raise ValueError("phase_a_sha256 must be a 64-char digest")
    validate_consequence_supervision(supervision_step=SONIC78_V1.execute_steps)
    if sources is None:
        if dataset_root is None or slots_dir is None:
            raise ValueError("train() needs either sources or dataset_root+slots_dir")
        sources = [
            PhaseBSource(
                dataset_root=Path(dataset_root),
                slots_dir=Path(slots_dir),
                train_episodes=None,
                val_episodes=None,
            )
        ]
    train_episodes = build_split(
        sources, "train", phase_a_sha256=phase_a_sha256, max_frames=max_frames
    )
    if max_episodes is not None:
        train_episodes = train_episodes[: int(max_episodes)]
    if not train_episodes:
        raise ValueError("no training episode was selected")
    val_episodes = build_split(
        sources, "val", phase_a_sha256=phase_a_sha256, max_frames=max_frames
    )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    phase_b_config = PhaseBConfig(
        control_horizon=action_horizon,
        prefix_only=action_horizon == SONIC78_V1.execute_steps,
        aux_state46=aux_state46_weight > 0,
        history_slots=history_slots,
        prev_action_ticks=prev_action_ticks,
        depth=depth,
        num_heads=num_heads,
    )
    dim_weight = None
    if loss_mode == "delta_weighted":
        dim_weight = _delta_magnitude_weights(train_episodes).to(device)
    model = Sonic78PhaseB(phase_b_config).to(device)
    if init_from is not None:
        # Continue a run without changing its observation/action contract; a fresh AdamW
        # state is acceptable because only the number of optimization steps is being extended.
        payload = load_phase_b_checkpoint(init_from, expected_phase_a_sha256=phase_a_sha256)
        if PhaseBConfig(**payload["config"]) != phase_b_config:
            raise ValueError(
                f"{init_from} was trained with a different PhaseBConfig; refusing to resume"
            )
        model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    history: list[float] = []
    for step in range(max_steps):
        batch = _sample_batch(
            train_episodes,
            batch_size,
            action_horizon=action_horizon,
            supervision_step=SONIC78_V1.execute_steps,
            rng=rng,
            history_slots=history_slots,
            prev_action_ticks=prev_action_ticks,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(
            batch["current_slots"],
            batch["state46"],
            batch["actions"],
            history_slots=batch.get("history_slots"),
            prev_actions=batch.get("prev_actions"),
        )
        loss = phase_b_loss(
            output["predicted_slots"],
            batch["target_slots"],
            mode=loss_mode,
            dim_weight=dim_weight,
        )
        if aux_state46_weight:
            aux_target = batch["target_state46"]
            if aux_state46_shuffle:
                aux_target = aux_target[
                    torch.randperm(aux_target.shape[0], device=aux_target.device)
                ]
            loss = loss + aux_state46_weight * torch.nn.functional.smooth_l1_loss(
                output["predicted_state46"], aux_target
            )
        if ranking_weight and batch["actions"].shape[0] > 1:
            recorded_error = (output["predicted_slots"] - batch["target_slots"]).abs().mean(
                dim=(1, 2)
            )
            wrong_errors = []
            for shift in range(1, ranking_k):
                permutation = torch.roll(
                    torch.arange(batch["actions"].shape[0], device=device), shifts=shift
                )
                wrong_output = model(
                    batch["current_slots"],
                    batch["state46"],
                    batch["actions"][permutation],
                    history_slots=batch.get("history_slots"),
                    prev_actions=batch.get("prev_actions"),
                )
                wrong_errors.append(
                    (wrong_output["predicted_slots"] - batch["target_slots"])
                    .abs()
                    .mean(dim=(1, 2))
                )
            loss = loss + ranking_weight * ranking_margin_loss(
                recorded_error, torch.stack(wrong_errors, dim=1), margin=ranking_margin
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Phase B loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if save_every_steps and (step + 1) % save_every_steps == 0:
            checkpoint_path = output_path.with_name(
                f"{output_path.stem}.step{step + 1:06d}.pt"
            )
            save_phase_b_checkpoint(
                checkpoint_path,
                model=model,
                config=phase_b_config,
                contract=PhaseBContract(phase_a_sha256=phase_a_sha256),
                extra={
                    "steps": step + 1,
                    "init_from": None if init_from is None else str(init_from),
                    "loss_history_tail": history[-256:],
                },
            )
    def checkpoint_extra(verdicts: dict[str, str]) -> dict[str, object]:
        return {
            "dataset_root": None if dataset_root is None else str(dataset_root),
            "slots_dir": None if slots_dir is None else str(slots_dir),
            "sources": [
                {
                    "dataset_root": str(source.dataset_root),
                    "slots_dir": str(source.slots_dir),
                    "train_episodes": source.train_episodes,
                    "val_episodes": source.val_episodes,
                }
                for source in sources
            ],
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes),
            "steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "loss_mode": loss_mode,
            "aux_state46_weight": aux_state46_weight,
            "aux_state46_shuffle": aux_state46_shuffle,
            "history_slots": history_slots,
            "prev_action_ticks": prev_action_ticks,
            "depth": depth,
            "num_heads": num_heads,
            "ranking_weight": ranking_weight,
            "ranking_k": ranking_k,
            "ranking_margin": ranking_margin,
            "final_loss": history[-1] if history else None,
            "init_from": None if init_from is None else str(init_from),
            "loss_history": history,
            "diagnostics_verdict": verdicts,
        }

    contract = PhaseBContract(phase_a_sha256=phase_a_sha256)
    # Persist the trained weights before diagnostics: a diagnostics failure must not discard a
    # completed run, and diagnostics are the part most likely to change between runs.
    save_phase_b_checkpoint(
        output_path,
        model=model,
        config=phase_b_config,
        contract=contract,
        extra=checkpoint_extra({}),
    )
    diagnostics: dict[str, object] = {}
    if val_episodes:
        config = DiagnosticsConfig(
            stride=eval_stride,
            batch_size=32,
            max_rows=eval_max_rows,
            seed=seed,
            action_horizon=action_horizon,
        )
        diagnostics["val"] = evaluate_phase_b(
            model, val_episodes, config=config, device=device
        )
        diagnostics["train"] = evaluate_phase_b(
            model,
            train_episodes,
            config=DiagnosticsConfig(
                stride=eval_stride,
                batch_size=32,
                max_rows=eval_max_rows,
                seed=seed,
                action_horizon=action_horizon,
            ),
            device=device,
        )
    digest = save_phase_b_checkpoint(
        output_path,
        model=model.cpu(),
        config=phase_b_config,
        contract=contract,
        extra=checkpoint_extra(
            {split: report["verdict"] for split, report in diagnostics.items()}
        ),
    )
    summary: dict[str, object] = {
        "checkpoint_sha256": digest,
        "steps": max_steps,
        "final_loss": history[-1] if history else None,
        "train_episodes": len(train_episodes),
        "val_episodes": len(val_episodes),
        "diagnostics": diagnostics,
    }
    if diagnostics:
        report_target = (
            output_path.with_suffix(".diagnostics.json")
            if report_path is None
            else Path(report_path)
        )
        report_target.parent.mkdir(parents=True, exist_ok=True)
        report_target.write_text(
            json.dumps(
                {
                    "schema": "mosaic.n17.phase_b_diagnostics_report.v1",
                    "checkpoint": str(output_path),
                    "checkpoint_sha256": digest,
                    "phase_a_sha256": phase_a_sha256,
                    "splits": diagnostics,
                },
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        summary["report"] = str(report_target)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, help=f"Multi-root source table ({SOURCE_MANIFEST_SCHEMA})")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--slots-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-a-sha256", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--report", type=Path, help="Where to write the held-out diagnostics JSON")
    parser.add_argument("--eval-stride", type=int, default=5)
    parser.add_argument("--eval-max-rows", type=int, default=None)
    parser.add_argument(
        "--action-horizon",
        type=int,
        choices=(20, 40),
        default=40,
        help="20 scores only the executed prefix; 40 feeds the full candidate chunk",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("equal", "branch_balanced", "delta_weighted"),
        default="equal",
        help=(
            "equal: all-dimension Huber; branch_balanced: C/B/E/S each contribute one quarter; "
            "delta_weighted: per-dimension weight by mean |future slot - current slot|"
        ),
    )
    parser.add_argument(
        "--aux-state46-weight",
        type=float,
        default=0.0,
        help="Weight of the auxiliary future-state46 head (0 disables it)",
    )
    parser.add_argument(
        "--aux-state46-shuffle",
        action="store_true",
        help="Control: shuffle the auxiliary state46 targets across the batch",
    )
    parser.add_argument(
        "--history-slots",
        type=int,
        default=0,
        choices=(0, 1, 2, 3, 4),
        help="Past slot frames (t-1..t-N) fed as extra memory tokens",
    )
    parser.add_argument(
        "--prev-action-ticks",
        type=int,
        default=0,
        choices=(0, 20),
        help="Already-executed action ticks (actions[t-20:t]) fed as past action tokens",
    )
    parser.add_argument("--depth", type=int, default=2, help="Transformer encoder layers")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-k", type=int, default=4)
    parser.add_argument("--ranking-margin", type=float, default=0.0)
    parser.add_argument(
        "--init-from",
        type=Path,
        help="Continue training from a saved Phase B checkpoint with the same contract",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="Write <output>.stepNNNNNN.pt every N steps (0 disables)",
    )
    args = parser.parse_args()
    if args.sources is None and (args.dataset_root is None or args.slots_dir is None):
        parser.error("pass --sources, or both --dataset-root and --slots-dir")
    summary = train(
        dataset_root=args.dataset_root,
        slots_dir=args.slots_dir,
        sources=None if args.sources is None else load_source_manifest(args.sources),
        output_path=args.output,
        phase_a_sha256=args.phase_a_sha256,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_episodes=args.max_episodes,
        max_frames=args.max_frames,
        report_path=args.report,
        eval_stride=args.eval_stride,
        eval_max_rows=args.eval_max_rows,
        action_horizon=args.action_horizon,
        loss_mode=args.loss_mode,
        init_from=args.init_from,
        save_every_steps=args.save_every_steps,
        aux_state46_weight=args.aux_state46_weight,
        aux_state46_shuffle=args.aux_state46_shuffle,
        history_slots=args.history_slots,
        prev_action_ticks=args.prev_action_ticks,
        depth=args.depth,
        num_heads=args.num_heads,
        ranking_weight=args.ranking_weight,
        ranking_k=args.ranking_k,
        ranking_margin=args.ranking_margin,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "diagnostics"}, indent=1))
    for split, report in summary.get("diagnostics", {}).items():
        print(json.dumps({"split": split, "verdict": report["verdict"], "overall": report["overall"]}, indent=1))


if __name__ == "__main__":
    main()
