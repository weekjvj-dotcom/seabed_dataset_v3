"""Pure-Python checks for the additive v3 500-scene batch contract."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v3_500_scenes.json"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def seed(root_seed, *parts):
    return int(hashlib.sha256(canonical([int(root_seed), *parts]).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def load():
    return json.loads(CONFIG.read_text())


def plan(cfg):
    modes = cfg["lighting_modes"]
    offset = cfg["single_lighting_schedule"]["offset"]
    split = cfg["batch_split_schedule"]
    rows = []
    for number in range(1, 501):
        split_id = "train" if number <= split["train"] else ("validation" if number <= split["train"] + split["validation"] else "test")
        mode = modes[(number - 1 + offset) % 3]
        scene_id = f"scene_{number:04d}"
        layout_seed = seed(cfg["root_seed"], "scene_layout", number)
        rows.append({"scene_id": scene_id, "scene_number": number, "layout_seed": layout_seed, "split": split_id, "lighting_mode": mode, "samples": [f"{scene_id}_{mode}_d5_{preset['id']}" for preset in cfg["presets"]]})
    return rows


class Batch500Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load()
        cls.rows = plan(cls.cfg)

    def test_target_is_v3_batch_not_new_algorithm(self):
        self.assertEqual(self.cfg["schema_version"], 2)
        self.assertEqual(self.cfg["batch_dataset_id"], "seabed-v3-500-scenes")
        self.assertIn("batch-scale v3", self.cfg["v3_asset_policy"]["version_note"])
        self.assertEqual(self.cfg["output_root"], "outputs/v3_500_scenes_final")

    def test_scene_ids_and_layout_seeds_are_unique(self):
        ids = [row["scene_id"] for row in self.rows]
        seeds = [row["layout_seed"] for row in self.rows]
        self.assertEqual(ids, [f"scene_{number:04d}" for number in range(1, 501)])
        self.assertEqual(len(set(ids)), 500)
        self.assertEqual(len(set(seeds)), 500)
        self.assertEqual(seeds[0], seed(self.cfg["root_seed"], "scene_layout", 1))
        self.assertEqual(seeds[-1], seed(self.cfg["root_seed"], "scene_layout", 500))

    def test_single_mode_is_balanced(self):
        counts = Counter(row["lighting_mode"] for row in self.rows)
        self.assertEqual(counts, Counter({"natural": 167, "artificial": 167, "mixed": 166}))
        for row in self.rows:
            self.assertIn(row["lighting_mode"], self.cfg["lighting_modes"])

    def test_each_scene_has_one_clear_and_three_degraded(self):
        self.assertEqual(tuple(preset["id"] for preset in self.cfg["presets"]), ("mild", "medium", "strong"))
        self.assertTrue(all(len(row["samples"]) == 3 for row in self.rows))
        self.assertEqual(len({sample for row in self.rows for sample in row["samples"]}), 1500)
        self.assertEqual(self.cfg["single_lighting_schedule"]["counts"], {"natural": 167, "artificial": 167, "mixed": 166})

    def test_split_schedule_and_paths_are_explicit(self):
        self.assertEqual(self.cfg["batch_split_schedule"], {"train": 400, "validation": 50, "test": 50})
        self.assertEqual(Counter(row["split"] for row in self.rows), Counter({"train": 400, "validation": 50, "test": 50}))
        self.assertEqual(self.cfg["calibration_gate"], "reports/calibration_v3_500/gate.json")
        self.assertNotIn("seabed_dataset_v2", self.cfg["output_root"])
        self.assertNotIn("seabed_dataset_v3/outputs/v3_first_round", self.cfg["output_root"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
