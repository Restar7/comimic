"""Offline typed-slot target builder (Phase A over a LeRobot ego video).

The frozen V-JEPA dense encoder is *not* re-implemented here: the caller passes a
``dense_fn(frames) -> [N,P,D]`` callable, typically wrapping the FlowBody encoder
that Phase A was trained against.  Mismatched preprocessing would silently
produce garbage slots, so the builder records the encoder identity and refuses a
slot file whose Phase A digest does not match the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..ac.phase_a_runtime import PhaseAAdapter
from .slots_h5 import write_slots_h5


@dataclass(frozen=True)
class AuxTargetConfig:
    video_key: str = "ego_view"
    stride: int = 1
    start: int = 0

    def __post_init__(self) -> None:
        if not self.video_key:
            raise ValueError("video_key must be non-empty")
        if isinstance(self.stride, bool) or not isinstance(self.stride, int) or self.stride <= 0:
            raise ValueError("stride must be a positive int")
        if isinstance(self.start, bool) or not isinstance(self.start, int) or self.start < 0:
            raise ValueError("start must be a non-negative int")


def build_aux_targets(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    dense_fn,
    phase_a: PhaseAAdapter,
    config: AuxTargetConfig = AuxTargetConfig(),
    dense_dim: int,
) -> list[dict]:
    """Encode every ``config.stride``-th ego frame of one dataset into slot files."""

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag

    if not callable(dense_fn):
        raise TypeError("dense_fn must be callable")
    if int(dense_dim) != phase_a.contract.dense_dim:
        raise ValueError(f"dense_dim {dense_dim} != Phase A contract {phase_a.contract.dense_dim}")
    root = Path(dataset_root)
    out = Path(output_dir)
    modality = MODALITY_CONFIGS[EmbodimentTag.UNITREE_G1_SONIC.value]
    loader = LeRobotEpisodeLoader(root, modality)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = int(info["fps"])

    records: list[dict] = []
    for episode_index in range(len(loader)):
        length = loader.get_episode_length(episode_index)
        indices = np.arange(config.start, length, config.stride, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"episode {episode_index} has no frames under the stride config")
        frames = loader._load_video_data(episode_index, indices)  # noqa: SLF001
        if config.video_key not in frames:
            raise KeyError(
                f"episode {episode_index} has no video key {config.video_key!r}; "
                f"available {sorted(frames)}"
            )
        images = np.asarray(frames[config.video_key])
        if images.ndim != 4:
            raise ValueError(f"frames must be [N,H,W,C], got {images.shape}")
        dense = dense_fn(images)
        slots = phase_a.encode(dense)
        if slots.shape[0] != images.shape[0]:
            raise ValueError("Phase A returned a different number of steps than frames")
        valid = np.ones(slots.shape[0], dtype=bool)
        target = out / f"episode_{episode_index:06d}.slots.h5"
        digest = write_slots_h5(
            target,
            episode_index=episode_index,
            slots=slots,
            valid=valid,
            meta={
                "phase_a_sha256": phase_a.contract.checkpoint_sha256,
                "dense_dim": int(dense_dim),
                "video_key": config.video_key,
                "fps": fps,
                "stride": config.stride,
                "dataset_root": str(root),
                "frame_indices_head": int(indices[0]),
                "frame_indices_tail": int(indices[-1]),
            },
        )
        records.append(
            {
                "episode_index": episode_index,
                "path": str(target),
                "sha256": digest,
                "frames": int(indices.size),
            }
        )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "mosaic.n17.aux_targets_manifest.v1",
                "dataset_root": str(root),
                "phase_a_sha256": phase_a.contract.checkpoint_sha256,
                "dense_dim": int(dense_dim),
                "video_key": config.video_key,
                "stride": config.stride,
                "records": records,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


__all__ = ["AuxTargetConfig", "build_aux_targets"]
