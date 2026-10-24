"""Turn verified bank slots into teacher LeRobot samples.

The bank stores candidate blocks keyed by a state id; the *observation* that
produced them (ego frame, state43, root orientation, instruction) lives in the
source LeRobot episode.  This module joins the two and refuses any slot whose
state id has no matching source row, whose receipt is missing, or whose
candidate is not the arg-max verified one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..contracts.provenance import ProvenanceRecord
from ..contracts.sonic78 import SONIC78_V1, Sonic78Spec
from ..data.teacher_lerobot import TeacherSample
from .teacher_bank import TeacherBank


STATE_GROUPS = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
)


@dataclass(frozen=True)
class SourceRow:
    dataset_root: str
    episode_index: int
    tick: int
    instruction: str
    state43: np.ndarray
    projected_gravity: np.ndarray
    root_orientation: np.ndarray
    frame: np.ndarray
    video_key: str = "ego_view"


def _read_root_orientation(root: Path, episode_index: int, tick: int) -> np.ndarray:
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    chunk = episode_index // int(info.get("chunks_size", 1000))
    relative = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    table = pq.read_table(root / relative, columns=["observation.root_orientation"])
    values = np.asarray(table["observation.root_orientation"][tick].as_py(), dtype=np.float64)
    if values.shape != (4,):
        raise ValueError(f"root orientation must be [4], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("root orientation must be finite")
    return values


def read_source_row(
    dataset_root: str | Path,
    episode_index: int,
    tick: int,
    *,
    video_key: str = "ego_view",
) -> SourceRow:
    """Read one observation row (state + ego frame + instruction) from LeRobot."""

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag

    root = Path(dataset_root)
    modality = MODALITY_CONFIGS[EmbodimentTag.UNITREE_G1_SONIC.value]
    loader = LeRobotEpisodeLoader(root, modality)
    length = loader.get_episode_length(episode_index)
    if not 0 <= tick < length:
        raise ValueError(f"tick {tick} outside episode {episode_index} of length {length}")
    frame_data = loader[episode_index]
    row = frame_data.iloc[tick]
    state43 = np.concatenate(
        [np.asarray(row[f"state.{group}"], dtype=np.float64).reshape(-1) for group in STATE_GROUPS]
    )
    if state43.shape != (43,):
        raise ValueError(f"reconstructed state must be [43], got {state43.shape}")
    gravity = np.asarray(row["state.projected_gravity"], dtype=np.float64).reshape(-1)
    frames = loader._load_video_data(episode_index, np.array([tick], dtype=np.int64))  # noqa: SLF001
    if video_key not in frames:
        raise KeyError(f"episode has no video key {video_key!r}; available {sorted(frames)}")
    image = np.asarray(frames[video_key])[0]
    instruction = str(row["language.annotation.human.task_description"])
    if not instruction:
        raise ValueError(f"empty instruction at episode {episode_index} tick {tick}")
    # ``root_orientation`` is not part of the N1.7 state modality config, so the
    # loader does not project it; read the raw parquet column instead.  Falling
    # back to identity is forbidden because the AC state46 depends on it.
    root_quat = _read_root_orientation(root, episode_index, tick)
    return SourceRow(
        dataset_root=str(root),
        episode_index=int(episode_index),
        tick=int(tick),
        instruction=instruction,
        state43=state43,
        projected_gravity=gravity,
        root_orientation=root_quat,
        frame=np.asarray(image, dtype=np.uint8),
        video_key=video_key,
    )


def materialize_samples(
    bank: TeacherBank,
    sources: Mapping[str, SourceRow],
    *,
    spec: Sonic78Spec = SONIC78_V1,
    generator_sha256: str,
    processor_sha256: str,
    stats_sha256: str,
    ac_sha256: str = "",
) -> list[TeacherSample]:
    """Join verified bank slots with their source observations."""

    if not sources:
        raise ValueError("no source rows")
    samples: list[TeacherSample] = []
    for state_id in bank.eligible_ids():
        if state_id not in sources:
            raise KeyError(f"verified teacher state {state_id!r} has no source observation")
        slot = bank.get(state_id, require_verified=True)
        if not slot.receipt_sha256:
            raise ValueError(f"teacher state {state_id!r} has no verification receipt")
        best = int(np.argmax(slot.scores))
        action = slot.candidates[best]
        if not slot.mask[best].all():
            raise ValueError(
                f"teacher state {state_id!r} best candidate carries masked channels; "
                "a partial action must not become full supervision"
            )
        source = sources[state_id]
        samples.append(
            TeacherSample(
                episode_index=source.episode_index,
                instruction=source.instruction,
                state43=source.state43,
                projected_gravity=source.projected_gravity,
                root_orientation=source.root_orientation,
                action=action,
                frame=source.frame,
                provenance=ProvenanceRecord(
                    candidate_id=f"{state_id}:{slot.checksums[best][:12]}",
                    provenance="selected_teacher",
                    generator_sha256=generator_sha256,
                    processor_sha256=processor_sha256,
                    stats_sha256=stats_sha256,
                    ac_sha256=ac_sha256,
                    sim_receipt_sha256=slot.receipt_sha256,
                    bank_generation=slot.generation,
                ),
            )
        )
    if not samples:
        raise ValueError("bank has no eligible (verified, unprotected) teachers")
    return samples


def write_source_index(path: str | Path, rows: Mapping[str, SourceRow]) -> None:
    """Persist the state-id -> source-row index used to build a teacher set."""

    payload = {
        "schema": "mosaic.n17.teacher_source_index.v1",
        "rows": {
            state_id: {
                "dataset_root": row.dataset_root,
                "episode_index": row.episode_index,
                "tick": row.tick,
                "instruction": row.instruction,
                "video_key": row.video_key,
            }
            for state_id, row in rows.items()
        },
    }
    Path(path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


__all__ = [
    "STATE_GROUPS",
    "SourceRow",
    "materialize_samples",
    "read_source_row",
    "write_source_index",
]
