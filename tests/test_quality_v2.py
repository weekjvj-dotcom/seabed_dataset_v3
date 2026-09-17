"""Real NumPy fixtures for the array-only v2 quality contract."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quality import (  # noqa: E402
    geometry_decision,
    geometry_metrics,
    noise_metrics,
    rgb_decision,
    rgb_metrics,
)


QUALITY = {
    "valid_fraction_min": 0.90,
    "near_fraction_min": 0.05,
    "middle_fraction_min": 0.15,
    "far_fraction_min": 0.05,
    "band_ecology_pixels_min": 16,
    "mean_linear_min": 0.003,
    "clipped_fraction_max": 0.10,
    "display_dynamic_range_min": 0.03,
    "effect_noise_min": 5.0,
    "structure_noise_min": 5.0,
}
CFG = {"quality": QUALITY}


def _registry() -> dict[str, dict[str, object]]:
    return {
        "1": {"semantic_id": 1, "morphology": "sandbed"},
        "2": {"semantic_id": 2, "morphology": "rock_cluster"},
        "3": {"semantic_id": 3, "asset_morphology": "branching"},
        "4": {"semantic_id": 3, "morphology": "brain"},
        "5": {"semantic_id": 4, "morphology": "seagrass_clump"},
        "6": {"semantic_id": 5, "morphology": "disk_fish"},
        "7": {"semantic_id": 5, "morphology": "fusiform_fish"},
    }


def _json_safe(value: object) -> object:
    """Round-trip helper that catches NumPy scalar leakage and NaN/Inf."""

    return json.loads(json.dumps(value, allow_nan=False))


class GeometryQualityTests(unittest.TestCase):
    def test_positive_depth_below_point_eight_is_not_near(self) -> None:
        depth = np.full((8, 8), 0.5, dtype=np.float64)
        semantic = np.full((8, 8), 1, dtype=np.uint16)
        instance = np.full((8, 8), 1, dtype=np.uint16)
        metrics = geometry_metrics(depth, semantic, instance, _registry(), CFG)
        self.assertEqual(metrics["bands"]["near"]["pixel_count"], 0)
        self.assertEqual(metrics["below_near"]["pixel_count"], 64)
        decision = geometry_decision(metrics, CFG)
        self.assertFalse(decision["passed"])
        self.assertIn("near_fraction_below_min", decision["reasons"])
        self.assertIn("near_ecology_pixels_below_min", decision["reasons"])

    def test_sand_only_each_band_fails_ecology_gate(self) -> None:
        depth = np.tile(
            np.repeat(np.array([1.0, 3.0, 7.0, 15.0], dtype=np.float64), 5),
            (20, 1),
        )
        semantic = np.ones((20, 20), dtype=np.uint16)
        instance = np.ones((20, 20), dtype=np.uint16)
        metrics = geometry_metrics(depth, semantic, instance, _registry(), CFG)
        decision = geometry_decision(metrics, CFG)
        self.assertFalse(decision["passed"])
        for band in ("near", "middle", "far"):
            self.assertIn(f"{band}_ecology_pixels_below_min", decision["reasons"])
            self.assertEqual(metrics["bands"][band]["ecology_pixel_count"], 0)
        self.assertEqual(metrics["background_sand_fraction"], 1.0)

    def test_normal_multiband_multiclass_fixture_passes(self) -> None:
        # Every 4x4 tile has 100 pixels.  Each range therefore clears the
        # configured full-frame fraction and ecology count gates.
        depth = np.empty((20, 20), dtype=np.float64)
        depth[:, :5] = 1.0
        depth[:, 5:10] = 3.0
        depth[:, 10:15] = 7.0
        depth[:, 15:] = 15.0
        semantic = np.ones((20, 20), dtype=np.uint16)
        instance = np.ones((20, 20), dtype=np.uint16)
        semantic[:4, :5] = 3
        instance[:4, :5] = 3
        semantic[4:8, :5] = 5
        instance[4:8, :5] = 6
        semantic[:4, 5:10] = 2
        instance[:4, 5:10] = 2
        semantic[4:8, 5:10] = 3
        instance[4:8, 5:10] = 4
        semantic[:4, 10:15] = 4
        instance[:4, 10:15] = 5
        semantic[4:8, 10:15] = 5
        instance[4:8, 10:15] = 7
        # Keep the background range as sand to ensure it cannot satisfy the
        # ecology gates for the three foreground bands.
        metrics = geometry_metrics(depth, semantic, instance, _registry(), CFG)
        decision = geometry_decision(metrics, CFG)
        self.assertTrue(decision["passed"], decision)
        self.assertEqual(metrics["valid_fraction"], 1.0)
        self.assertEqual(metrics["visible_fish_count"], 2)
        self.assertEqual(set(metrics["coral_morphologies"]), {"branching", "brain"})
        self.assertEqual(set(metrics["fish_morphologies"]), {"disk_fish", "fusiform_fish"})
        self.assertEqual(metrics["background_sand_fraction"], 1.0)
        self.assertEqual(metrics["bands"]["near"]["visible_fish_count"], 1)
        self.assertEqual(metrics["bands"]["middle"]["ecology_pixel_count"], 40)

    def test_band_fraction_uses_valid_pixels_at_ninety_percent_coverage(self) -> None:
        # 9,000 of 10,000 pixels are valid.  Near occupies exactly 450 valid
        # pixels (5% of valid, but only 4.5% of the full frame), so this is a
        # boundary fixture that distinguishes the approved denominator.
        depth = np.zeros((100, 100), dtype=np.float64)
        depth[:90, :5] = 1.0
        depth[:90, 5:20] = 3.0
        depth[:90, 20:25] = 7.0
        depth[:90, 25:] = 15.0
        semantic = np.zeros((100, 100), dtype=np.uint16)
        instance = np.zeros((100, 100), dtype=np.uint16)
        semantic[:90] = 1
        instance[:90] = 1
        # Give each foreground band enough ecology pixels while preserving the
        # exact band areas above.
        for columns, instance_id, semantic_id in (
            (slice(0, 1), 2, 2),
            (slice(5, 6), 3, 3),
            (slice(20, 21), 4, 4),
        ):
            semantic[:16, columns] = semantic_id
            instance[:16, columns] = instance_id
        registry = {
            "1": {"semantic_id": 1, "morphology": "sandbed"},
            "2": {"semantic_id": 2, "morphology": "rock_cluster"},
            "3": {"semantic_id": 3, "morphology": "branching"},
            "4": {"semantic_id": 4, "morphology": "seagrass_clump"},
        }
        metrics = geometry_metrics(depth, semantic, instance, registry, CFG)
        self.assertAlmostEqual(metrics["valid_fraction"], 0.90, places=12)
        self.assertAlmostEqual(metrics["bands"]["near"]["fraction"], 0.045, places=12)
        self.assertAlmostEqual(metrics["bands"]["near"]["fraction_of_valid"], 0.05, places=12)
        self.assertTrue(geometry_decision(metrics, CFG)["passed"])

    def test_unknown_id_nonfinite_and_shape_mismatch_are_explained(self) -> None:
        depth = np.ones((2, 2), dtype=np.float64)
        semantic = np.ones((2, 2), dtype=np.uint16)
        instance = np.ones((2, 2), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "absent from registry"):
            geometry_metrics(depth, semantic, instance * 99, _registry(), CFG)
        bad_depth = depth.copy()
        bad_depth[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            geometry_metrics(bad_depth, semantic, instance, _registry(), CFG)
        with self.assertRaisesRegex(ValueError, "shape"):
            geometry_metrics(depth, semantic[:, :1], instance, _registry(), CFG)
        bad_semantic = semantic.astype(np.float64)
        bad_semantic[0, 0] = 3.5
        with self.assertRaisesRegex(ValueError, "non-integer"):
            geometry_metrics(depth, bad_semantic, instance, _registry(), CFG)


class RGBQualityTests(unittest.TestCase):
    def test_standard_linear_rec709_and_srgb_statistics(self) -> None:
        rgb = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float64,
        )
        semantic = np.array([[0, 3], [5, 4]], dtype=np.uint16)
        metrics = rgb_metrics(rgb, np.array([[0, 255], [True, True]]), semantic)
        expected_luma = np.array([0.2126, 0.7152, 0.0722])
        selected_rgb = rgb[np.array([[False, True], [True, True]])]
        display_rgb = np.where(
            selected_rgb <= 0.0031308,
            12.92 * selected_rgb,
            1.055 * selected_rgb ** (1 / 2.4) - 0.055,
        )
        display_luma = (
            0.2126 * display_rgb[:, 0]
            + 0.7152 * display_rgb[:, 1]
            + 0.0722 * display_rgb[:, 2]
        )
        self.assertAlmostEqual(metrics["mean_linear"], float(np.mean(expected_luma)), places=12)
        self.assertAlmostEqual(metrics["displayP5"], float(np.percentile(display_luma, 5)), places=12)
        self.assertAlmostEqual(metrics["displayP95"], float(np.percentile(display_luma, 95)), places=12)
        self.assertAlmostEqual(
            metrics["dynamic_range"],
            float(np.percentile(display_luma, 95) - np.percentile(display_luma, 5)),
            places=12,
        )
        self.assertAlmostEqual(metrics["clipped_fraction"], 1.0, places=12)
        self.assertEqual(metrics["semantic_stats"]["3"]["pixel_count"], 1)
        self.assertEqual(metrics["valid_pixel_count"], 3)
        self.assertEqual(_json_safe(metrics), metrics)

    def test_display_luminance_oetfs_channels_before_rec709_weighting(self) -> None:
        rgb = np.array(
            [
                [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0]],
                [[0.0, 0.0, 0.25], [0.3, 0.2, 0.1]],
            ],
            dtype=np.float64,
        )
        metrics = rgb_metrics(rgb, np.ones((2, 2), dtype=bool))
        linear_luma = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )
        encoded = np.where(
            rgb <= 0.0031308,
            12.92 * rgb,
            1.055 * rgb ** (1 / 2.4) - 0.055,
        )
        expected_display_luma = (
            0.2126 * encoded[..., 0]
            + 0.7152 * encoded[..., 1]
            + 0.0722 * encoded[..., 2]
        )
        wrong_display_luma = np.where(
            linear_luma <= 0.0031308,
            12.92 * linear_luma,
            1.055 * linear_luma ** (1 / 2.4) - 0.055,
        )
        self.assertAlmostEqual(metrics["mean_linear"], float(np.mean(linear_luma)), places=12)
        self.assertAlmostEqual(
            metrics["displayP5"], float(np.percentile(expected_display_luma, 5)), places=12
        )
        self.assertAlmostEqual(
            metrics["displayP95"], float(np.percentile(expected_display_luma, 95)), places=12
        )
        self.assertGreater(float(np.max(np.abs(expected_display_luma - wrong_display_luma))), 1e-3)

    def test_black_saturated_and_uniform_images_fail_global_gates(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        black = rgb_metrics(np.zeros((8, 8, 3), dtype=np.float64), mask)
        saturated = rgb_metrics(np.ones((8, 8, 3), dtype=np.float64), mask)
        uniform = rgb_metrics(np.full((8, 8, 3), 0.2, dtype=np.float64), mask)
        self.assertFalse(rgb_decision(black, CFG, "natural")["passed"])
        self.assertFalse(rgb_decision(saturated, CFG, "natural")["passed"])
        self.assertFalse(rgb_decision(uniform, CFG, "natural")["passed"])
        # Artificial dark-edge diagnostics never alter the frozen gates.
        self.assertEqual(
            rgb_decision(uniform, CFG, "artificial")["reasons"],
            rgb_decision(uniform, CFG, "natural")["reasons"],
        )

    def test_hdr_raw_mean_is_not_preclipped(self) -> None:
        rgb = np.full((2, 2, 3), 2.0, dtype=np.float64)
        metrics = rgb_metrics(rgb, np.ones((2, 2), dtype=np.uint8) * 255)
        self.assertAlmostEqual(metrics["mean_linear"], 2.0, places=12)
        self.assertGreater(metrics["displayP95"], 1.0)
        self.assertEqual(metrics["clipped_fraction"], 1.0)

    def test_rgb_invalid_inputs_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "H.*W.*3"):
            rgb_metrics(np.zeros((2, 2), dtype=np.float64), np.ones((2, 2), dtype=bool))
        bad = np.zeros((2, 2, 3), dtype=np.float64)
        bad[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            rgb_metrics(bad, np.ones((2, 2), dtype=bool))
        with self.assertRaisesRegex(ValueError, "empty"):
            rgb_metrics(np.zeros((2, 2, 3), dtype=np.float64), np.zeros((2, 2), dtype=np.uint8))


class NoiseQualityTests(unittest.TestCase):
    def test_independent_seed_noise_and_effect_are_finite(self) -> None:
        rng = np.random.default_rng(20260908)
        clear = np.zeros((32, 32, 3), dtype=np.float64)
        clear[..., 0] = np.linspace(0.05, 0.8, 32)[None, :]
        clear[..., 1] = np.linspace(0.1, 0.5, 32)[:, None]
        a = clear + 0.2 + rng.normal(0, 0.002, size=clear.shape)
        b = clear + 0.2 + rng.normal(0, 0.002, size=clear.shape)
        metrics = noise_metrics(clear, a, b, np.ones((32, 32), dtype=bool))
        self.assertTrue(metrics["all_finite"])
        self.assertGreater(metrics["noise_estimate"], 0.0)
        self.assertGreater(metrics["effect_rms"], 0.0)
        self.assertTrue(math.isfinite(metrics["effect_noise_ratio"]))
        self.assertTrue(math.isfinite(metrics["structure_noise_ratio"]))
        self.assertEqual(_json_safe(metrics), metrics)

    def test_uniform_colour_plus_independent_noise_has_low_structure_ratio(self) -> None:
        rng = np.random.default_rng(7)
        base = np.full((128, 128, 3), 0.2, dtype=np.float64)
        a = base + rng.normal(0.0, 0.02, size=base.shape)
        b = base + rng.normal(0.0, 0.02, size=base.shape)
        metrics = noise_metrics(base, a, b, np.ones((128, 128), dtype=bool))
        self.assertLess(metrics["structure_noise_ratio"], QUALITY["structure_noise_min"])
        self.assertGreater(metrics["effect_noise_ratio"], 0.0)

    def test_zero_noise_ratio_is_null_and_explicitly_flagged(self) -> None:
        clear = np.zeros((3, 3, 3), dtype=np.float64)
        wet = np.full((3, 3, 3), 0.1, dtype=np.float64)
        metrics = noise_metrics(clear, wet, wet.copy(), np.ones((3, 3), dtype=np.uint8) * 255)
        self.assertEqual(metrics["noise_estimate"], 0.0)
        self.assertTrue(metrics["noise_zero"])
        self.assertIsNone(metrics["effect_noise_ratio"])
        self.assertIsNone(metrics["structure_noise_ratio"])
        self.assertTrue(_json_safe(metrics))

    def test_noise_shape_nonfinite_and_empty_roi_raise(self) -> None:
        rgb = np.zeros((2, 2, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            noise_metrics(rgb, np.zeros((2, 3, 3)), rgb, np.ones((2, 2), dtype=bool))
        bad = rgb.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            noise_metrics(rgb, bad, rgb, np.ones((2, 2), dtype=bool))
        with self.assertRaisesRegex(ValueError, "empty"):
            noise_metrics(rgb, rgb, rgb, np.zeros((2, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
