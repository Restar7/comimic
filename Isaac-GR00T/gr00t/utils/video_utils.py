# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional, Tuple

import numpy as np


_TORCHCODEC_INSTALL_HINT = (
    "torchcodec is required for video decoding. Install it via the platform "
    "deployment script (scripts/deployment/{dgpu,orin,thor,spark}/install_deps.sh) "
    "or `uv pip install torchcodec`."
)

_TORCHCODEC_FFMPEG_HINT = (
    "torchcodec is installed but its native library could not be loaded. This is "
    "almost always an FFmpeg version mismatch: torchcodec 0.8.0 only ships loaders "
    "for FFmpeg 4-7 and cannot run against FFmpeg 8 (the default on Ubuntu 25.10+ "
    "and 26.04). Install an FFmpeg<8 runtime (e.g. `conda install -c conda-forge "
    "'ffmpeg<8'` or a distro package) and ensure its shared libraries are on "
    "LD_LIBRARY_PATH. Original error: {error}"
)

_DEFAULT_DECODER_KWARGS = {
    "device": "cpu",
    "dimension_order": "NHWC",
    "num_ffmpeg_threads": 0,
}


def _decoder_import_error(exc: BaseException) -> ImportError:
    """Map a torchcodec import/load failure to an actionable ImportError.

    A plain ``ImportError`` means torchcodec is not installed. An ``OSError`` or
    ``RuntimeError`` means torchcodec is present but its native library failed to
    load, which is most commonly an FFmpeg major-version mismatch.
    """
    if isinstance(exc, ImportError):
        return ImportError(_TORCHCODEC_INSTALL_HINT)
    return ImportError(_TORCHCODEC_FFMPEG_HINT.format(error=exc))


# ---------------------------------------------------------------------------
# PyAV fallback（本机没有 CUDA-13 的 torchcodec；ft318 的 ego_view 是普通 H.264 mp4）
# ---------------------------------------------------------------------------
class _AVFrames:
    """torchcodec ``FrameBatch`` 的最小替身：只有 ``.data`` / ``.pts_seconds``。"""

    def __init__(self, frames: np.ndarray, pts: np.ndarray):
        self._f, self._p = frames, pts

    @property
    def data(self):
        import torch

        return torch.from_numpy(self._f)

    @property
    def pts_seconds(self):
        import torch

        return torch.from_numpy(self._p)


class _AVVideoDecoder:
    """用 PyAV 复刻 torchcodec.VideoDecoder 用到的那几个接口。"""

    def __init__(self, video_path: str, **_: object):
        import av

        self.path = str(video_path)
        self._container = av.open(self.path)
        self._stream = self._container.streams.video[0]
        self._fps = float(self._stream.average_rate) if self._stream.average_rate else 30.0
        self._cache: dict[int, np.ndarray] = {}
        self._pts: dict[int, float] = {}
        self._n: int | None = None

    # -- 基础信息 --
    def __len__(self) -> int:
        if self._n is None:
            n = 0
            for _ in self._container.decode(self._stream):
                n += 1
            self._container.seek(0)
            self._n = max(n, 1)
        return self._n

    class _Meta:
        def __init__(self, fps: float):
            self.average_fps = fps

    @property
    def metadata(self):
        return self._Meta(self._fps)

    # -- 取帧 --
    def _load(self, indices) -> None:
        want = {int(i) for i in indices}
        if all(i in self._cache for i in want):
            return
        import av

        self._container.close()
        self._container = av.open(self.path)
        self._stream = self._container.streams.video[0]
        hi = max(want)
        for i, frame in enumerate(self._container.decode(self._stream)):
            if i > hi:
                break
            if i in want and i not in self._cache:
                self._cache[i] = frame.to_ndarray(format="rgb24")  # HWC uint8
                self._pts[i] = float(frame.pts * self._stream.time_base) if frame.pts is not None else i / self._fps
            if i >= hi:
                break

    def get_frames_at(self, indices):
        idx = [int(i) for i in indices]
        self._load(idx)
        missing = [i for i in idx if i not in self._cache]
        if missing:
            raise IndexError(f"{self.path}: frame indices {missing[:5]} out of range")
        return _AVFrames(np.stack([self._cache[i] for i in idx], 0), np.asarray([self._pts[i] for i in idx], np.float64))

    def get_frames_played_at(self, seconds):
        idx = [int(round(float(t) * self._fps)) for t in seconds]
        return self.get_frames_at(idx)


def _av_available() -> bool:
    try:
        import av  # noqa: F401

        return True
    except Exception:
        return False


def _get_video_decoder_cls():
    try:
        from torchcodec.decoders import VideoDecoder
    except (ImportError, OSError, RuntimeError) as exc:
        if _av_available():
            return _AVVideoDecoder
        raise _decoder_import_error(exc) from exc
    return VideoDecoder


def _build_decoder(video_path: str, decoder_kwargs: Optional[dict]):
    video_decoder_cls = _get_video_decoder_cls()
    kwargs = {**_DEFAULT_DECODER_KWARGS, **(decoder_kwargs or {})}
    return video_decoder_cls(video_path, **kwargs)


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    decoder_kwargs: Optional[dict] = None,
) -> np.ndarray:
    decoder = _build_decoder(video_path, decoder_kwargs)
    return decoder.get_frames_at(indices=indices).data.numpy()


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    decoder_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Get frames from a video at specified timestamps.

    Args:
        video_path (str): Path to the video file.
        timestamps (list[float] | np.ndarray): Timestamps to retrieve frames for, in seconds.

    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    decoder = _build_decoder(video_path, decoder_kwargs)

    # https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoStreamMetadata.html#torchcodec.decoders.VideoStreamMetadata
    fps = decoder.metadata.average_fps
    interval = 1 / fps
    timestamps = np.array(timestamps).astype(np.float64)

    # Correct float precision issues in timestamps
    # E.g. for 5fps video: [1.0, 1.20000005, 1.39999998] -> [1.0, 1.2, 1.4]
    # Without this, torchcodec will read the delayed frame (e.g. 1.39999998 -> 1.2)
    # Round to nearest frame interval to prevent torchcodec from reading wrong frames.
    # Allow max 1% error from expected interval.
    closest_timestamps = np.round(timestamps / interval) * interval
    timestamp_errors = np.abs(closest_timestamps - timestamps) / interval
    invalid_mask = timestamp_errors >= 0.01
    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        invalid_timestamps = timestamps[invalid_indices]
        raise ValueError(
            f"Try to read invalid timestamps {invalid_timestamps} from video {video_path} (FPS: {fps})"
        )

    timestamps = closest_timestamps
    return decoder.get_frames_played_at(seconds=timestamps).data.numpy()


def get_all_frames(
    video_path: str,
    decoder_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Get all frames from a video.

    Returns:
        tuple[np.ndarray, np.ndarray]: Frames and timestamps.
    """
    decoder = _build_decoder(video_path, decoder_kwargs)
    frames = decoder.get_frames_at(indices=range(len(decoder)))
    return frames.data.numpy(), frames.pts_seconds.numpy()


def get_accumulate_timestamp_idxs(
    timestamps: List[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: Optional[int] = 0,
    allow_negative=False,
) -> Tuple[List[int], List[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx
