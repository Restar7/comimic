"""Bridge the real G1 task recordings to the checkpoint's own VLM preprocessing.

``Gr00tVLM.build_batch`` runs a bare ``AutoProcessor`` on a PIL image.  That is
right for the no-checkpoint path but **wrong once a GR00T checkpoint is loaded**:
the checkpoint was trained through ``Gr00tN1d7Processor``, which resizes
(shortest edge 256 -> crop 230 -> 256), percentil-normalises the state, formalises
the language and pads state to ``max_state_dim``.  Feeding it a raw image and raw
joint angles puts the model off-distribution.

This module restores that path:

    MotionLibrary frame index  ->  episode npz  ->  ego_view mp4 frame
                                              \\->  g1_state_46 / instruction
                                                      |
                                                      v
                              Gr00tN1d7Processor.process_observation(...)
                                                      |
                                                      v
                              BatchFeature(input_ids, pixel_values, state, ...)

The npz files come from ``scripts/prepare_g1_tasks_npz.py``; each one is a single
episode, so an in-episode frame index is also the video frame index (50 fps).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_SONIC_TAG = "unitree_g1_sonic"
_STATE_KEYS = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "projected_gravity",
)
#: Widths of the groups, used to re-split the stored ``g1_state_46``.
_STATE_WIDTHS = (6, 6, 3, 7, 7, 7, 7, 3)


@dataclass(frozen=True)
class FrameMeta:
    """Everything the VLM needs about one frame, resolved from the npz index."""

    file_index: int
    local_frame: int
    episode_index: int
    ego_view: str
    instruction: str


class TaskFrameIndex:
    """Map a global ``MotionLibrary`` frame back to its episode and in-episode frame.

    ``files`` must be in the same order used to build the library (sorted paths),
    which is what :func:`diff_vla_wbc.data.load_motion_library` receives.
    """

    def __init__(self, files: Sequence[str | Path]):
        self.files = [Path(f) for f in files]
        self._lengths: list[int] = []
        self._ego: list[str] = []
        self._instruction: list[str] = []
        self._episode: list[int] = []
        for path in self.files:
            with np.load(path, allow_pickle=False) as d:
                self._lengths.append(int(d["joint_pos"].shape[0]))
                self._ego.append(str(d["g1_ego_view"]))
                self._instruction.append(str(d["g1_instruction"]))
                self._episode.append(int(d["g1_episode_index"]))
        self._starts = np.cumsum([0] + self._lengths[:-1]).tolist()
        self.num_frames = int(sum(self._lengths))

    @property
    def num_episodes(self) -> int:
        return len(self.files)

    def lookup(self, frame: int) -> FrameMeta:
        frame = int(frame)
        if not 0 <= frame < self.num_frames:
            raise IndexError(f"frame {frame} outside [0, {self.num_frames})")
        # ``bisect`` style search over the (sorted) clip starts.
        lo, hi = 0, len(self._lengths) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._starts[mid] <= frame:
                lo = mid
            else:
                hi = mid - 1
        local = frame - self._starts[lo]
        return FrameMeta(
            file_index=lo,
            local_frame=local,
            episode_index=self._episode[lo],
            ego_view=self._ego[lo],
            instruction=self._instruction[lo],
        )

    def state_46(self, file_index: int) -> np.ndarray:
        """``(T, 46)`` state in ``Gr00tN1d7Processor`` key order (LRU cached)."""
        return _load_state(self.files[file_index])


_STATE_CACHE: OrderedDict[Path, np.ndarray] = OrderedDict()
_STATE_CACHE_SIZE = 8


def _load_state(path: Path) -> np.ndarray:
    cached = _STATE_CACHE.get(path)
    if cached is not None:
        _STATE_CACHE.move_to_end(path)
        return cached
    with np.load(path, allow_pickle=False) as d:
        state = np.ascontiguousarray(d["g1_state_46"])
    _STATE_CACHE[path] = state
    if len(_STATE_CACHE) > _STATE_CACHE_SIZE:
        _STATE_CACHE.popitem(last=False)
    return state


class EgoViewImages:
    """Random-access RGB frames from the recorded ``ego_view`` mp4 files.

    Seeking to a keyframe and decoding forward matches the requested frame exactly
    (verified against ``pts``) and costs ~40 ms per access, which is far cheaper
    than decoding a 3000-frame episode for every sample.
    """

    def __init__(self, cache_size: int = 4):
        self.cache_size = int(cache_size)
        self._open: OrderedDict[Path, Any] = OrderedDict()

    def _container(self, path: Path):
        import av

        hit = self._open.get(path)
        if hit is not None:
            self._open.move_to_end(path)
            return hit
        container = av.open(str(path))
        self._open[path] = container
        if len(self._open) > self.cache_size:
            _, evicted = self._open.popitem(last=False)
            evicted.close()
        return container

    def read(self, path: str | Path, index: int) -> np.ndarray:
        path = Path(path)
        container = self._container(path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        time_base = float(stream.time_base)
        target = index / fps
        container.seek(int(target / time_base), stream=stream, backward=True, any_frame=False)
        frame = None
        for candidate in container.decode(stream):
            frame = candidate
            pts = candidate.pts
            if pts is not None and float(pts * time_base) + 1e-6 >= target - 0.5 / fps:
                break
        if frame is None:
            raise ValueError(f"{path}: no frame decoded at index {index}")
        return frame.to_ndarray(format="rgb24")

    def close(self) -> None:
        for container in self._open.values():
            container.close()
        self._open.clear()


class TaskObservationBuilder:
    """Assemble the ``observation`` dict ``process_observation`` expects."""

    def __init__(self, index: TaskFrameIndex, images: EgoViewImages | None = None):
        self.index = index
        self.images = images or EgoViewImages()

    def build(self, frames: Sequence[int] | torch.Tensor) -> dict[str, Any]:
        if torch.is_tensor(frames):
            frames = frames.detach().cpu().tolist()
        frames = [int(f) for f in frames]
        metas = [self.index.lookup(f) for f in frames]
        pixel_frames = []
        for meta in metas:
            rgb = self.images.read(meta.ego_view, meta.local_frame)
            pixel_frames.append(rgb[None])  # (T=1, H, W, C)
        observation: dict[str, Any] = {
            "video.ego_view": np.stack(pixel_frames, axis=0).astype(np.uint8),
            "annotation.human.task_description": [meta.instruction for meta in metas],
        }
        states = [self.index.state_46(meta.file_index)[meta.local_frame] for meta in metas]
        stacked = np.stack(states, axis=0).astype(np.float32)
        offset = 0
        for key, width in zip(_STATE_KEYS, _STATE_WIDTHS):
            observation[f"state.{key}"] = np.ascontiguousarray(stacked[:, offset : offset + width])
            offset += width
        if offset != stacked.shape[1]:
            raise ValueError(f"state_46 width {stacked.shape[1]} != {offset}")
        return observation


def load_gr00t_processor(
    checkpoint: str | Path,
    *,
    vlm_path: str | Path | None = None,
    local_files_only: bool = True,
):
    """Load the checkpoint's ``Gr00tN1d7Processor`` (offline, local Cosmos dir)."""
    from .gr00t_tiny import ensure_gr00t_importable
    from .gr00t_vlm import resolve_vlm_source

    ensure_gr00t_importable()
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    checkpoint = Path(checkpoint)
    if not (checkpoint / "processor_config.json").is_file():
        raise FileNotFoundError(
            f"{checkpoint} has no processor_config.json; point at the checkpoint directory "
            "itself (it stores processor_config/statistics/embodiment_id next to config.json)"
        )
    # The stored config pins the gated repo id; point it at the local weights view.
    source, _origin = resolve_vlm_source(vlm_path)
    return Gr00tN1d7Processor.from_pretrained(
        checkpoint,
        model_name=str(source),
        local_files_only=local_files_only,
        transformers_loading_kwargs={
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        },
    )


def processed_batch(
    processor: Any,
    builder: TaskObservationBuilder,
    frames: Sequence[int] | torch.Tensor,
    *,
    embodiment: str = _SONIC_TAG,
):
    """``frames`` -> tokenised VLM batch, through the checkpoint's own preprocessing."""
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = EmbodimentTag(embodiment)
    observation = builder.build(frames)
    batch = processor.process_observation(observation, tag)
    # ``process_observation`` returns ``state`` as ``(B, max_state_dim)``; the action
    # head wants a state-history axis, ``(B, state_history_length, max_state_dim)``.
    state = batch["state"]
    if state.ndim == 2:
        batch["state"] = state.unsqueeze(1)
    return batch


__all__ = [
    "EgoViewImages",
    "FrameMeta",
    "TaskFrameIndex",
    "TaskObservationBuilder",
    "load_gr00t_processor",
    "processed_batch",
]
