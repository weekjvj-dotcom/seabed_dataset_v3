"""Pure contracts for v2 lighting sampling.

Blender resource tests belong to the main agent because they require the one
shared Blender render/API-probe token.  These tests intentionally exercise the
JSON plan and deterministic streams with the system Python.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lighting import MODES, build_lighting, lighting_metadata, sample_lighting

try:  # The same file is also run by the system Python for pure-plan checks.
    import bpy  # type: ignore
    from mathutils import Matrix  # type: ignore
except ImportError:  # pragma: no cover - exercised outside Blender
    bpy = None
    Matrix = None


def _cfg() -> dict:
    return {
        "schema_version": 2,
        "root_seed": 72026,
        "layouts": [{"seed": 101, "split": "train"}],
        "lighting_sampling": {
            "sky_strength": [0.08, 0.35],
            "sun_elevation_deg": [20, 75],
            "sun_rotation_deg": [0, 360],
            "spot_energy": [100, 600],
            "spot_full_angle_deg": [35, 70],
            "spot_blend": [0.3, 0.7],
            "temperature_k": [4500, 6500],
        },
    }


class LightingPlan(unittest.TestCase):
    def test_schema_and_native_units(self):
        plan = sample_lighting(_cfg(), 0)
        json.dumps(plan, allow_nan=False, sort_keys=True)
        self.assertEqual(plan["schema_version"], 2)
        self.assertEqual(plan["sky"]["sky_type"], "MULTIPLE_SCATTERING")
        self.assertFalse(plan["sky"]["sun_disc"])
        self.assertEqual([lamp["name"] for lamp in plan["lamps"]], ["LeftSpot", "RightSpot"])
        for lamp in plan["lamps"]:
            self.assertEqual(len(lamp["position_local"]), 3)
            self.assertEqual(len(lamp["target_local"]), 3)
            self.assertAlmostEqual(lamp["spot_size_rad"], math.radians(lamp["spot_full_angle_deg"]))
            self.assertGreaterEqual(lamp["spot_blend"], 0.0)
            self.assertLessEqual(lamp["spot_blend"], 1.0)
            self.assertGreaterEqual(lamp["temperature_k"], 4500.0)
            self.assertLessEqual(lamp["temperature_k"], 6500.0)

    def test_same_seed_is_order_independent(self):
        cfg = _cfg()
        first = sample_lighting(cfg, 1, 0)
        _ = sample_lighting(cfg, 0, 0)
        _ = sample_lighting(cfg, 2, 1)
        second = sample_lighting(cfg, 1, 0)
        self.assertEqual(first, second)

    def test_state_and_attempt_streams_are_distinct(self):
        cfg = _cfg()
        state_a = sample_lighting(cfg, 0, 0)
        state_b = sample_lighting(cfg, 1, 0)
        attempt_b = sample_lighting(cfg, 0, 1)
        self.assertNotEqual(state_a, state_b)
        self.assertNotEqual(state_a, attempt_b)
        self.assertNotEqual(state_a["seed"], state_b["seed"])

    def test_layout_is_part_of_every_lighting_stream(self):
        cfg = _cfg()
        first = sample_lighting(cfg, state_index=0, attempt_index=0)
        cfg["_layout_seed"] = 202
        second = sample_lighting(cfg, state_index=0, attempt_index=0)
        self.assertEqual(first["layout_seed"], 101)
        self.assertEqual(second["layout_seed"], 202)
        self.assertNotEqual(first["seed_streams"], second["seed_streams"])
        self.assertNotEqual(first["sky"], second["sky"])
        self.assertNotEqual(first["lamps"], second["lamps"])

    def test_cfg_overrides_are_used(self):
        cfg = _cfg()
        cfg["lighting_sampling"] = {
            "sky_strength": [0.21, 0.21],
            "sun_elevation_deg": [42, 42],
            "sun_rotation_deg": [123, 123],
            "spot_energy": [275, 275],
            "spot_full_angle_deg": [51, 51],
            "spot_blend": [0.44, 0.44],
            "temperature_k": [5600, 5600],
        }
        plan = sample_lighting(cfg, 0)
        self.assertEqual(plan["sky"]["strength"], 0.21)
        self.assertEqual(plan["sky"]["sun_elevation_deg"], 42.0)
        self.assertEqual(plan["sky"]["sun_rotation_deg"], 123.0)
        for lamp in plan["lamps"]:
            self.assertEqual(lamp["energy_w"], 275.0)
            self.assertEqual(lamp["spot_full_angle_deg"], 51.0)
            self.assertEqual(lamp["spot_blend"], 0.44)
            self.assertEqual(lamp["temperature_k"], 5600.0)

    def test_invalid_ranges_fail_before_sampling(self):
        for key, value in (
            ("spot_blend", [-0.1, 0.4]),
            ("spot_full_angle_deg", [0, 20]),
            ("sun_elevation_deg", [20, 91]),
            ("temperature_k", [0, 5000]),
        ):
            cfg = copy.deepcopy(_cfg())
            cfg["lighting_sampling"][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    sample_lighting(cfg, 0)

    def test_supported_mode_contract_is_explicit(self):
        self.assertEqual(MODES, ("natural", "artificial", "mixed"))


@unittest.skipIf(bpy is None, "requires actual Blender bpy")
class NativeLighting(unittest.TestCase):
    """Integration checks executed by the main agent in one Blender process."""

    def setUp(self):
        self.previous_scene = None
        window = getattr(getattr(bpy, "context", None), "window", None)
        if window is not None:
            self.previous_scene = window.scene
        self.scene = bpy.data.scenes.new("V2_Lighting_Test_Scene")
        camera_data = bpy.data.cameras.new("V2_Lighting_Test_Camera_Data")
        self.camera = bpy.data.objects.new("V2_Lighting_Test_Camera", camera_data)
        self.scene.collection.objects.link(self.camera)
        self.scene.camera = self.camera
        self.camera.matrix_world = Matrix.Identity(4)
        if window is not None:
            window.scene = self.scene
        self.handles = []

    def tearDown(self):
        # Keep the test self-contained when the runner loads more Blender
        # modules after this one.  Every target is explicit and was created by
        # this test, so unrelated user data is not touched.
        for handle in self.handles:
            world = handle.get("world")
            if world is not None:
                world.use_fake_user = False
            for obj in handle.get("objects", ()):
                if obj.name in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            collection = handle.get("collection")
            if collection is not None and collection.name in bpy.data.collections:
                bpy.data.collections.remove(collection, do_unlink=True)
            if world is not None and world.name in bpy.data.worlds:
                # The test scene is still a user until it is detached below.
                self.scene.world = None
        if self.camera is not None and self.camera.name in bpy.data.objects:
            camera_data = self.camera.data
            bpy.data.objects.remove(self.camera, do_unlink=True)
            if camera_data is not None and camera_data.name in bpy.data.cameras and camera_data.users == 0:
                bpy.data.cameras.remove(camera_data)
        window = getattr(getattr(bpy, "context", None), "window", None)
        if window is not None and self.previous_scene is not None:
            window.scene = self.previous_scene
        if self.scene is not None and self.scene.name in bpy.data.scenes:
            bpy.data.scenes.remove(self.scene)
        for handle in self.handles:
            world = handle.get("world")
            if world is not None and world.name in bpy.data.worlds and world.users == 0:
                bpy.data.worlds.remove(world)

    @staticmethod
    def _nodes(world, bl_idname):
        return [node for node in world.node_tree.nodes if node.bl_idname == bl_idname]

    @staticmethod
    def _matrix_close(actual, expected, places=5):
        for row_a, row_e in zip(actual, expected):
            for value_a, value_e in zip(row_a, row_e):
                if abs(float(value_a) - float(value_e)) > 10 ** (-places):
                    return False
        return True

    @staticmethod
    def _drivers_none(owner):
        animation = getattr(owner, "animation_data", None)
        return animation is None or len(getattr(animation, "drivers", ())) == 0

    def _update_scene(self):
        context = getattr(bpy, "context", None)
        if context is not None and hasattr(context, "temp_override"):
            with context.temp_override(scene=self.scene, view_layer=self.scene.view_layers[0]):
                context.view_layer.update()
        else:  # pragma: no cover - retained for older Blender runners
            bpy.context.view_layer.update()

    @staticmethod
    def _normalise_metadata(value, key=None):
        """Ignore datablock names while retaining all semantic readbacks."""
        if key in {"name", "data_name", "world_name", "identifier", "resource_identity"}:
            return "<volatile-resource-name>"
        if key in {"object_names", "data_names", "collections"}:
            return ["<volatile-resource-name>"] * len(value) if isinstance(value, list) else "<volatile-resource-name>"
        if key == "objects" and isinstance(value, list) and all(isinstance(item, str) for item in value):
            return ["<volatile-resource-name>"] * len(value)
        if isinstance(value, dict):
            return {name: NativeLighting._normalise_metadata(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [NativeLighting._normalise_metadata(item, key) for item in value]
        return value

    def _assert_no_rna_pointer_repr(self, metadata):
        encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False, sort_keys=True)
        self.assertNotIn("bpy_struct", encoded)
        self.assertIsNone(re.search(r"0x[0-9a-fA-F]+", encoded))

    def test_native_modes_parenting_temperature_and_metadata(self):
        cfg = _cfg()
        plan = sample_lighting(cfg, state_index=1, attempt_index=0)
        for mode in MODES:
            self.handles.append(build_lighting(self.scene, self.camera, plan, mode))

        natural, artificial, mixed = self.handles
        # Each mode owns its World, collection, object pair and Light data.
        self.assertEqual(len({id(h["world"]) for h in self.handles}), 3)
        self.assertEqual(len({id(h["collection"]) for h in self.handles}), 3)
        self.assertEqual(
            len({id(obj) for handle in self.handles for obj in handle["objects"]}),
            6,
        )
        self.assertEqual(
            len({id(obj.data) for handle in self.handles for obj in handle["objects"]}),
            6,
        )

        sky_strengths = []
        sky_angles = []
        for handle in self.handles:
            self.assertEqual(len(handle["objects"]), 2)
            self.assertEqual(len(handle["collection"].objects), 2)
            self.assertTrue(all(obj.type == "LIGHT" for obj in handle["collection"].objects))
            self.assertFalse(handle["collection"].hide_render)
            self.assertFalse(handle["collection"].hide_viewport)
            backgrounds = self._nodes(handle["world"], "ShaderNodeBackground")
            skies = self._nodes(handle["world"], "ShaderNodeTexSky")
            self.assertEqual(len(backgrounds), 1)
            self.assertEqual(len(skies), 1)
            self.assertEqual(skies[0].sky_type, "MULTIPLE_SCATTERING")
            self.assertFalse(skies[0].sun_disc)
            sky_strengths.append(float(backgrounds[0].inputs["Strength"].default_value))
            sky_angles.append((float(skies[0].sun_elevation), float(skies[0].sun_rotation)))
            for obj in handle["objects"]:
                data = obj.data
                self.assertEqual(data.type, "SPOT")
                self.assertAlmostEqual(
                    data.spot_size,
                    math.radians(plan["lamps"][0]["spot_full_angle_deg"]),
                    delta=1e-5,
                )
                self.assertAlmostEqual(data.spot_blend, plan["lamps"][0]["spot_blend"], delta=1e-5)
                self.assertTrue(data.use_shadow)
                self.assertIs(obj.parent, self.camera)
                self.assertTrue(self._drivers_none(obj))
                self.assertTrue(self._drivers_none(data))
            self.assertTrue(self._drivers_none(handle["world"]))
            self.assertTrue(self._drivers_none(handle["collection"]))

        self.assertAlmostEqual(sky_strengths[0], plan["sky"]["strength"], delta=1e-5)
        self.assertAlmostEqual(sky_strengths[0], sky_strengths[2], delta=1e-5)
        for actual, expected in zip(sky_angles[0], sky_angles[2]):
            self.assertAlmostEqual(actual, expected, delta=1e-5)
        self.assertEqual(sky_strengths[1], 0.0)
        self.assertTrue(all(obj.data.energy == 0.0 for obj in natural["objects"]))
        self.assertTrue(all(obj.data.energy > 0.0 for obj in artificial["objects"]))
        self.assertTrue(all(obj.data.energy > 0.0 for obj in mixed["objects"]))
        for artificial_obj, mixed_obj in zip(artificial["objects"], mixed["objects"]):
            self.assertAlmostEqual(float(artificial_obj.data.energy), float(mixed_obj.data.energy), delta=0.001)
        for artificial_obj, mixed_obj in zip(artificial["objects"], mixed["objects"]):
            self.assertAlmostEqual(artificial_obj.data.spot_size, mixed_obj.data.spot_size, delta=1e-5)
            self.assertAlmostEqual(artificial_obj.data.spot_blend, mixed_obj.data.spot_blend, delta=1e-5)
            self.assertAlmostEqual(artificial_obj.data.temperature, mixed_obj.data.temperature, delta=0.01)
            self.assertTrue(self._matrix_close(artificial_obj.matrix_local, mixed_obj.matrix_local))

        # Blender 5.2.1's native temperature output is separate from base
        # ``Light.color``; both are read back so a white base color is not
        # mislabeled as the effective temperature color.
        for handle in (artificial, mixed):
            for obj in handle["objects"]:
                data = obj.data
                self.assertTrue(data.use_temperature)
                temperature_color = [float(v) for v in data.temperature_color]
                base_color = [float(v) for v in data.color]
                self.assertGreaterEqual(len(temperature_color), 3)
                self.assertGreaterEqual(len(base_color), 3)
                self.assertTrue(all(math.isfinite(v) for v in temperature_color[:3]))
                self.assertTrue(all(math.isfinite(v) for v in base_color[:3]))
                self.assertAlmostEqual(float(data.temperature), plan["lamps"][0]["temperature_k"], delta=0.01)
                self.assertNotEqual(temperature_color[:3], base_color[:3])

        metadata = lighting_metadata(mixed)
        json.dumps(metadata, ensure_ascii=False, allow_nan=False, sort_keys=True)
        self._assert_no_rna_pointer_repr(metadata)
        self.assertEqual(metadata["mode"], "mixed")
        self.assertEqual(len(metadata["objects"]), 2)
        self.assertCountEqual(metadata["collection"]["objects"], [obj.name for obj in mixed["objects"]])
        for item in metadata["objects"]:
            self.assertEqual(item["type"], "SPOT")
            self.assertTrue(item["use_temperature"])
            self.assertEqual(item["temperature"]["source"], "native_light_temperature")
            self.assertEqual(len(item["effective_rgb"]), 3)
            self.assertEqual(len(item["temperature_color"]), 3)
            self.assertEqual(item["temperature"]["effective_rgb_source"], "Light.temperature_color")
            linking = item["light_linking"]
            self.assertIsInstance(linking, dict)
            self.assertIsNone(linking["receiver_collection"])
            self.assertIsNone(linking["blocker_collection"])
            self.assertIsNone(linking["lightgroup"])
            self.assertEqual(item["drivers"]["object"]["has_animation_data"], False)
            self.assertEqual(item["drivers"]["data"]["has_animation_data"], False)
            self.assertEqual(len(item["transform"]["matrix_local"]), 4)
            self.assertEqual(len(item["transform"]["matrix_world"]), 4)
        self.assertEqual(metadata["native_constraints"]["temperature_is_read_back"], True)
        self.assertEqual(metadata["native_constraints"]["temperature_readback_status"], "native_light_temperature")

        # Building the same plan again must preserve semantic metadata even
        # though Blender assigns different datablock names to the new rig.
        repeat_handle = build_lighting(self.scene, self.camera, plan, "mixed")
        self.handles.append(repeat_handle)
        repeat_metadata = lighting_metadata(repeat_handle)
        self._assert_no_rna_pointer_repr(repeat_metadata)
        self.assertEqual(
            self._normalise_metadata(metadata),
            self._normalise_metadata(repeat_metadata),
        )

        # Local matrices remain fixed while the parent camera moves, and the
        # native world matrix follows that parent transform.
        before_local = [obj.matrix_local.copy() for obj in mixed["objects"]]
        before_world = [obj.matrix_world.copy() for obj in mixed["objects"]]
        self.camera.matrix_world = Matrix.Translation((1.25, -0.4, 0.7)) @ Matrix.Rotation(
            math.radians(11.0), 4, "Y"
        )
        self._update_scene()
        for obj, local, previous_world in zip(mixed["objects"], before_local, before_world):
            expected_world = self.camera.matrix_world @ obj.matrix_parent_inverse @ local
            self.assertTrue(self._matrix_close(obj.matrix_local, local))
            self.assertTrue(self._matrix_close(obj.matrix_world, expected_world))
            self.assertFalse(self._matrix_close(obj.matrix_world, previous_world))



if __name__ == "__main__":
    unittest.main()
