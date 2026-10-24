"""Phase B held-out diagnostics: copy-current and shuffled-action controls.

Training loss alone cannot say whether the consequence model learned an action-conditioned
transition: a model that predicts ``current_slots`` (or the dataset mean) minimises a short
delta just as well.  This module therefore scores a checkpoint on whole held-out episodes with
the two controls the M3 exit gate names:

* ``copy_current_l1`` -- the trivial predictor ``slot[t]``; a model that cannot beat it has not
  learned that the future differs from the present;
* ``shuffled_action_l1`` -- the same model fed a seeded *global permutation* of the evaluation
  actions.  If the prediction does not change, the action memory is not being used.

Rows are selected only where ``t + supervision_step`` is valid; the supervision step stays the
SONIC78 executed prefix (20), never the unexecuted tail of H=40.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np
import torch

from ..ac.phase_b_runtime import validate_consequence_supervision
from ..ac.slots import SLOT_DIM, TOTAL_SLOTS, slot_slices
from ..contracts.sonic78 import SONIC78_V1


class EpisodeWindow(Protocol):
    """The fields diagnostics need from one episode; ``train_phase_b`` owns the reader."""

    actions: np.ndarray
    state46: np.ndarray
    slots: np.ndarray
    valid: np.ndarray
    source: str


@dataclass(frozen=True)
class DiagnosticsConfig:
    horizon: int = SONIC78_V1.horizon
    supervision_step: int = SONIC78_V1.execute_steps
    action_horizon: int = SONIC78_V1.horizon
    stride: int = 5
    batch_size: int = 32
    max_rows: int | None = None
    seed: int = 0
    ranking_candidates: int = 4
    #: Margin (best wrong candidate error - recorded candidate error) above which a top-1 hit
    #: counts as a clear identification instead of a noise-level flip.
    clear_margin: float = 0.005

    def __post_init__(self) -> None:
        validate_consequence_supervision(supervision_step=self.supervision_step)
        if self.action_horizon not in (SONIC78_V1.execute_steps, SONIC78_V1.horizon):
            raise ValueError("action_horizon must be 20 or 40")
        if self.stride <= 0 or self.batch_size <= 0:
            raise ValueError("stride and batch size must be positive")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive when set")
        if self.ranking_candidates < 2:
            raise ValueError("ranking_candidates must be at least two")
        if self.clear_margin < 0:
            raise ValueError("clear_margin must be non-negative")


def select_rows(
    episodes: Iterable[EpisodeWindow], config: DiagnosticsConfig
) -> list[tuple[int, int]]:
    """``(episode_index, t)`` rows whose executed-prefix target exists."""

    rows: list[tuple[int, int]] = []
    for index, episode in enumerate(episodes):
        stride = int(getattr(episode, "slots_stride", 1))
        length = int(episode.slots.shape[0])
        supervision_dense = config.supervision_step // stride
        if supervision_dense <= 0:
            raise ValueError("supervision step is shorter than one slot stride")
        stop = min(length - supervision_dense, (episode.actions.shape[0] - config.action_horizon) // stride)
        for t in range(0, max(stop, 0), config.stride):
            # Evaluate at the same decision-grid starts the trainer samples: a window that
            # straddles two replanned chunks is not a deployment decision.
            if (t * stride) % config.supervision_step:
                continue
            target = t + supervision_dense
            if target >= length or not bool(episode.valid[t]) or not bool(episode.valid[target]):
                continue
            rows.append((index, t))
            if config.max_rows is not None and len(rows) >= config.max_rows:
                return rows
    if not rows:
        raise ValueError("no evaluation row has the executed-prefix target")
    return rows


def _deranged_actions(actions: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Return the action blocks of other rows, never a row's own block.

    The candidate-ranking test must not count "a row was randomly assigned its own action"
    as a failure; a plain permutation has fixed points, so repair them with a cyclic swap.
    """

    size = len(actions)
    if size < 2:
        raise ValueError("candidate ranking needs at least two rows per batch")
    perm = torch.randperm(size, generator=generator)
    fixed = (perm == torch.arange(size)).nonzero().flatten().tolist()
    for index in fixed:
        other = (index + 1) % size
        perm[index], perm[other] = perm[other], perm[index]
    return actions[perm]


def _wrong_action_outside_batch(
    episodes: list[EpisodeWindow],
    rows: list[tuple[int, int]],
    offset: int,
    current_row: tuple[int, int],
    config: DiagnosticsConfig,
) -> torch.Tensor:
    """Wrong action block for a batch with a single row (the split tail)."""

    other = (offset + 1) % len(rows)
    if rows[other] == current_row:
        other = (other + 1) % len(rows)
    if rows[other] == current_row:
        raise ValueError("candidate ranking needs at least two distinct rows")
    index, t = rows[other]
    stride = int(getattr(episodes[index], "slots_stride", 1))
    return torch.from_numpy(
        episodes[index].actions[t * stride : t * stride + config.action_horizon]
    ).float().unsqueeze(0)


def _branch_errors(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    errors = (prediction - target).abs().mean(dim=0)  # [14,384]
    return {name: float(errors[span].mean().item()) for name, span in slot_slices().items()}


@torch.no_grad()
def evaluate_phase_b(
    model: torch.nn.Module,
    episodes: list[EpisodeWindow],
    *,
    config: DiagnosticsConfig = DiagnosticsConfig(),
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Score one Phase B model against copy-current and a shuffled-action control."""

    rows = select_rows(episodes, config)
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    totals: dict[str, dict[str, float]] = {
        name: {"model": 0.0, "copy": 0.0, "shuffled": 0.0} for name in slot_slices()
    }
    sensitivity_sum = 0.0
    sensitivity_max = 0.0
    tail_sensitivity_sum = 0.0
    tail_sensitivity_max = 0.0
    ranking_top1 = 0
    ranking_rank_sum = 0.0
    ranking_mrr = 0.0
    ranking_rows = 0
    margins: list[float] = []
    model.eval()
    slices = slot_slices()
    for offset in range(0, len(rows), config.batch_size):
        batch_rows = rows[offset : offset + config.batch_size]
        stride = int(getattr(episodes[batch_rows[0][0]], "slots_stride", 1))
        supervision_dense = config.supervision_step // stride
        current = torch.from_numpy(
            np.stack([episodes[index].slots[t] for index, t in batch_rows])
        ).float().to(device)
        state = torch.from_numpy(
            np.stack([episodes[index].state46[t * stride] for index, t in batch_rows])
        ).float().to(device)
        actions = torch.from_numpy(
            np.stack(
                [
                    episodes[index].actions[
                        t * stride : t * stride + config.action_horizon
                    ]
                    for index, t in batch_rows
                ]
            )
        ).float().to(device)
        target = torch.from_numpy(
            np.stack(
                [
                    episodes[index].slots[t + supervision_dense]
                    for index, t in batch_rows
                ]
            )
        ).float().to(device)
        model_config = getattr(model, "config", None)
        history_count = int(getattr(model_config, "history_slots", 0))
        past_ticks = int(getattr(model_config, "prev_action_ticks", 0))
        history = None
        prev_actions = None
        if history_count:
            history = torch.from_numpy(
                np.stack(
                    [
                        np.stack(
                            [
                                episodes[index].slots[max(t - h, 0)]
                                for h in range(1, history_count + 1)
                            ]
                        )
                        for index, t in batch_rows
                    ]
                )
            ).float().to(device)
        if past_ticks:
            past = np.zeros((len(batch_rows), past_ticks, 78), dtype=np.float32)
            for row_index, (index, t) in enumerate(batch_rows):
                tick = t * stride
                available = min(tick, past_ticks)
                if available:
                    past[row_index, past_ticks - available :] = episodes[index].actions[
                        tick - available : tick
                    ]
            prev_actions = torch.from_numpy(past).float().to(device)

        def _predict(candidate_actions: torch.Tensor) -> torch.Tensor:
            if history_count or past_ticks:
                return model(
                    current,
                    state,
                    candidate_actions,
                    history_slots=history,
                    prev_actions=prev_actions,
                )["predicted_slots"]
            return model(current, state, candidate_actions)["predicted_slots"]

        prediction = _predict(actions)
        permutation = torch.randperm(len(batch_rows), generator=generator).to(device)
        shuffled = _predict(actions[permutation])
        tail_sensitivity_sum = 0.0
        tail_sensitivity_max = 0.0
        if config.action_horizon == SONIC78_V1.horizon:
            # Causal-invariance probe: the last 20 tokens execute only after the next replan,
            # so they cannot change the t+20 consequence.  A model that reads them anyway has
            # learned the closed-loop recording pattern, not the candidate's executed prefix.
            tail_swapped = actions.clone()
            tail_swapped[:, SONIC78_V1.execute_steps :] = actions[
                permutation, SONIC78_V1.execute_steps :
            ]
            tail_pred = _predict(tail_swapped)
            tail_delta = (prediction - tail_pred).abs()
            tail_sensitivity_sum += float(tail_delta.sum().item())
            tail_sensitivity_max = max(tail_sensitivity_max, float(tail_delta.max().item()))
        copy = current
        for name, span in slices.items():
            totals[name]["model"] += float((prediction[:, span] - target[:, span]).abs().sum().item())
            totals[name]["copy"] += float((copy[:, span] - target[:, span]).abs().sum().item())
            totals[name]["shuffled"] += float((shuffled[:, span] - target[:, span]).abs().sum().item())
        delta = (prediction - shuffled).abs()
        sensitivity_sum += float(delta.sum().item())
        sensitivity_max = max(sensitivity_max, float(delta.max().item()))
        # Candidate ranking: candidate 0 is the recorded action (the one that produced the
        # target); the others are wrong actions sourced from other rows.  The deployment
        # question is "can the AC pick the action whose consequence matches?", not "is the
        # absolute L1 lower than copy-current".  Ties count against the recorded action, so a
        # copy-current model cannot look perfect here.
        if len(rows) >= 2:
            candidate_errors = []
            for candidate in range(config.ranking_candidates):
                if candidate == 0:
                    candidate_actions = actions
                elif len(batch_rows) < 2:
                    candidate_actions = _wrong_action_outside_batch(
                        episodes, rows, offset, batch_rows[0], config
                    ).to(device)
                else:
                    candidate_actions = _deranged_actions(actions, generator).to(device)
                candidate_prediction = _predict(candidate_actions)
                candidate_errors.append(
                    (candidate_prediction - target).abs().mean(dim=(1, 2))
                )
            errors = torch.stack(candidate_errors, dim=1)
            better = (errors[:, 1:] < errors[:, 0:1]).sum(dim=1)
            ties = (errors[:, 1:] == errors[:, 0:1]).sum(dim=1)
            ranks = 1.0 + better.float() + 0.5 * ties.float()
            best_wrong = errors[:, 1:].min(dim=1).values
            margins.extend((best_wrong - errors[:, 0]).tolist())
            ranking_top1 += int(((better == 0) & (ties == 0)).sum().item())
            ranking_rank_sum += float(ranks.sum().item())
            ranking_mrr += float((1.0 / ranks).sum().item())
            ranking_rows += int(len(batch_rows))
    branch_denominator = {
        name: float(len(rows) * (span.stop - span.start) * SLOT_DIM)
        for name, span in slices.items()
    }
    branches = {
        name: {
            "l1": values["model"] / branch_denominator[name],
            "copy_current_l1": values["copy"] / branch_denominator[name],
            "shuffled_action_l1": values["shuffled"] / branch_denominator[name],
            "beats_copy_current": values["model"] < values["copy"],
            "shuffled_action_hurts": values["shuffled"] > values["model"],
        }
        for name, values in totals.items()
    }
    overall_denominator = float(len(rows) * TOTAL_SLOTS * SLOT_DIM)
    model_total = sum(values["model"] for values in totals.values()) / overall_denominator
    copy_total = sum(values["copy"] for values in totals.values()) / overall_denominator
    shuffled_total = sum(values["shuffled"] for values in totals.values()) / overall_denominator
    beats_copy = all(entry["beats_copy_current"] for entry in branches.values())
    shuffled_hurts = all(entry["shuffled_action_hurts"] for entry in branches.values())
    reasons = []
    if not beats_copy:
        reasons.append("at least one branch is not better than copy-current")
    if not shuffled_hurts:
        reasons.append("shuffling the actions does not raise the error in at least one branch")
    return {
        "schema": "mosaic.n17.phase_b_diagnostics.v1",
        "rows": len(rows),
        "episodes": len(episodes),
        "config": {
            "horizon": int(config.horizon),
            "supervision_step": int(config.supervision_step),
            "action_horizon": int(config.action_horizon),
            "stride": int(config.stride),
            "seed": int(config.seed),
        },
        "branches": branches,
        "overall": {
            "model_l1": model_total,
            "copy_current_l1": copy_total,
            "shuffled_action_l1": shuffled_total,
            "beats_copy_current": beats_copy,
            "shuffled_action_hurts": shuffled_hurts,
            "action_sensitivity_mean": sensitivity_sum / (len(rows) * 14 * 384),
            "action_sensitivity_max": sensitivity_max,
            "tail_sensitivity_mean": tail_sensitivity_sum / (len(rows) * 14 * 384),
            "tail_sensitivity_max": tail_sensitivity_max,
            "ranking_rows": int(ranking_rows),
            "ranking_candidates": int(config.ranking_candidates),
            "ranking_top1_accuracy": ranking_top1 / max(ranking_rows, 1),
            "ranking_mean_rank": ranking_rank_sum / max(ranking_rows, 1),
            "ranking_mrr": ranking_mrr / max(ranking_rows, 1),
            "ranking_margin_median": float(np.median(margins)) if margins else 0.0,
            "ranking_clear_fraction": float(np.mean(np.asarray(margins) > config.clear_margin))
            if margins
            else 0.0,
            "ranking_random_top1": 1.0 / int(config.ranking_candidates),
        },
        "verdict": "pass" if not reasons else "fail",
        "reasons": reasons,
    }


__all__ = ["DiagnosticsConfig", "EpisodeWindow", "evaluate_phase_b", "select_rows"]
