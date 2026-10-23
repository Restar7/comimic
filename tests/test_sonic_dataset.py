"""Tests for the pinned SONIC four-task dataset resolver (``diff_vla_wbc.sonic_dataset``).

The dataset choice is frozen in ``data/sonic_datasets.json``: the
``groot_sonic_vla_ft318_frozen_20261023`` snapshot (failed episodes removed),
318 episodes across four task groups.  These tests stop the pipeline from
silently drifting back to the old ``SONIC_DATA`` roots or a stale count.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from diff_vla_wbc import sonic_dataset as sd

MANIFEST = sd.load_manifest() if sd.MANIFEST_PATH.is_file() else None
ROOT = Path(MANIFEST["sets"][MANIFEST["active"]]["root"]) if MANIFEST else None

EXPECTED_TASKS = {
    "task_1_meeting_room_milk_basket": 89,
    "task_2_office_red_cup_drawer": 70,
    "task_3_kitchen_plate_milk_transfer": 65,
    "task_4_diningroom_can_low_chair": 94,
}


@unittest.skipUnless(ROOT is not None and ROOT.is_dir(), "pinned SONIC dataset not mounted")
class PinnedManifestTest(unittest.TestCase):
    def test_manifest_shape(self):
        m = sd.load_manifest()
        self.assertEqual(m["active"], "ft318_frozen_20261023")
        active = m["sets"][m["active"]]
        self.assertEqual(active["totals"]["episodes"], 318)
        self.assertEqual(active["totals"]["frames"], 473876)
        self.assertEqual(active["totals"]["datasets"], 9)

    def test_task_counts(self):
        self.assertEqual(sd.task_counts(), EXPECTED_TASKS)
        self.assertEqual(sum(EXPECTED_TASKS.values()), 318)

    def test_active_datasets_resolve(self):
        paths = sd.active_datasets()
        self.assertEqual(len(paths), 9)
        for p in paths:
            self.assertTrue(p.is_dir(), p)
            self.assertIn("task_", str(p))

    def test_episode_counts_match_manifest(self):
        for entry in sd.dataset_entries():
            files = sd.episode_files(ROOT / entry["path"])
            self.assertEqual(len(files), entry["episodes"], entry["path"])

    def test_token_and_hand_columns(self):
        import pyarrow.parquet as pq

        first = sd.episode_files(sd.active_datasets()[0])[0]
        schema = {f.name: str(f.type) for f in pq.ParquetFile(first).schema_arrow}
        self.assertIn("64", schema["action.motion_token"])
        self.assertIn("43", schema["action.wbc"])
        self.assertIn("7", schema["teleop.left_hand_joints"])
        self.assertIn("7", schema["teleop.right_hand_joints"])

    def test_verify_reports_ok(self):
        report = sd.verify()
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["episodes"], 318)
        self.assertEqual(report["frames"], 473876)
        self.assertAlmostEqual(report["hours_at_50fps"], 2.633, places=3)
        self.assertEqual(report["tasks"], EXPECTED_TASKS)

    def test_fingerprint_stable(self):
        combined = sd.fingerprint_set()
        self.assertEqual(len(combined), 64)
        stored = {e["path"]: e["fingerprint_sha256"] for e in sd.dataset_entries()}
        self.assertEqual(len(stored), 9)
        self.assertTrue(all(len(v) == 64 for v in stored.values()))

    def test_instructions_recorded(self):
        ins = sd.instructions()
        self.assertEqual(len(ins), 4)
        self.assertIn("milk carton", ins["task_1_meeting_room_milk_basket"].lower())

    def test_unknown_set_raises(self):
        with self.assertRaises(KeyError):
            sd.dataset_paths("does_not_exist")

    def test_alternate_set_recorded(self):
        m = sd.load_manifest()
        alt = m["sets"]["ft271_frozen_20261023"]
        self.assertEqual(alt["totals"]["episodes"], 271)
        self.assertEqual(alt["totals"]["datasets"], 7)

    def test_manifest_is_json_serialisable(self):
        json.dumps(sd.load_manifest())


if __name__ == "__main__":
    unittest.main()
