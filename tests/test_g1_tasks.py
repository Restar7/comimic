"""Tests for the real-recordings bridge (``diff_vla_wbc.g1_tasks``).

No checkpoint, GPU or video file is needed: the index and the state split are
pure numpy, and the image reader is stubbed out.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from diff_vla_wbc.g1_tasks import (
    EgoViewImages,
    TaskFrameIndex,
    TaskObservationBuilder,
    _STATE_KEYS,
    _STATE_WIDTHS,
)


def _write_npz(path: Path, frames: int, *, episode: int, instruction: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        joint_pos=np.zeros((frames, 29), np.float32),
        g1_state_46=np.arange(frames * 46, dtype=np.float32).reshape(frames, 46),
        g1_instruction=np.array(instruction),
        g1_ego_view=np.array(f"/tmp/episode_{episode:06d}.mp4"),
        g1_episode_index=np.int32(episode),
    )


class _FakeImages(EgoViewImages):
    """Returns a constant frame so no mp4 is touched."""

    def __init__(self):
        self.seen: list[tuple[str, int]] = []

    def read(self, path, index):  # noqa: ANN001
        self.seen.append((str(path), int(index)))
        return np.full((8, 8, 3), int(index) % 256, dtype=np.uint8)


class StateLayoutTest(unittest.TestCase):
    def test_state_groups_cover_46_in_processor_order(self):
        self.assertEqual(sum(_STATE_WIDTHS), 46)
        self.assertEqual(
            _STATE_KEYS,
            (
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ),
        )


class TaskFrameIndexTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.a = self.tmp / "a.npz"
        self.b = self.tmp / "b.npz"
        _write_npz(self.a, 5, episode=3, instruction="pick A")
        _write_npz(self.b, 4, episode=7, instruction="pick B")
        self.index = TaskFrameIndex([self.a, self.b])

    def test_totals_and_boundaries(self):
        self.assertEqual(self.index.num_episodes, 2)
        self.assertEqual(self.index.num_frames, 9)
        first = self.index.lookup(0)
        self.assertEqual((first.file_index, first.local_frame, first.episode_index), (0, 0, 3))
        last_a = self.index.lookup(4)
        self.assertEqual((last_a.file_index, last_a.local_frame), (0, 4))
        first_b = self.index.lookup(5)
        self.assertEqual((first_b.file_index, first_b.local_frame, first_b.episode_index), (1, 0, 7))
        self.assertEqual(self.index.lookup(8).local_frame, 3)

    def test_out_of_range(self):
        with self.assertRaises(IndexError):
            self.index.lookup(9)
        with self.assertRaises(IndexError):
            self.index.lookup(-1)

    def test_state_and_instruction_resolve(self):
        meta = self.index.lookup(6)
        state = self.index.state_46(meta.file_index)
        self.assertEqual(state.shape, (4, 46))
        self.assertAlmostEqual(float(state[1, 0]), float(1 * 46 + 0))
        self.assertEqual(meta.instruction, "pick B")

    def test_builder_splits_state_and_reads_frames(self):
        images = _FakeImages()
        builder = TaskObservationBuilder(self.index, images)
        obs = builder.build(torch.tensor([0, 6]))
        self.assertEqual(obs["video.ego_view"].shape, (2, 1, 8, 8, 3))
        self.assertEqual(obs["annotation.human.task_description"], ["pick A", "pick B"])
        offset = 0
        for key, width in zip(_STATE_KEYS, _STATE_WIDTHS):
            block = obs[f"state.{key}"]
            self.assertEqual(block.shape, (2, width))
            expected = np.stack(
                [self.index.state_46(0)[0, offset : offset + width],
                 self.index.state_46(1)[1, offset : offset + width]]
            )
            np.testing.assert_allclose(block, expected)
            offset += width
        # (path, in-episode frame) — frame 6 is episode B's second frame (B starts at 5).
        self.assertEqual(images.seen, [("/tmp/episode_000003.mp4", 0), ("/tmp/episode_000007.mp4", 1)])


if __name__ == "__main__":
    unittest.main()
