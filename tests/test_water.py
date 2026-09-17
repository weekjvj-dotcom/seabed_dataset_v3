"""System-Python checks for the native water contract.

These tests intentionally avoid launching Blender.  The main Agent owns the
single serialized Blender integration run and can use the runtime probe to
exercise the bpy branches.  The checks here cover the part that must remain
deterministic before a scene is touched: schema validation, bounds, public
signatures, and the absence of image/filter shortcuts.
"""

from __future__ import annotations

import ast
import json
import inspect
from pathlib import Path
import unittest

from seabed_dataset_v2.src.water import native


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "water" / "native.py"


def valid_preset(**overrides):
    preset = {
        "id": "medium",
        "absorption": [0.30, 0.10, 0.04],
        "scatter": [0.06, 0.06, 0.06],
        "g": 0.65,
        "ior": 1.333,
    }
    preset.update(overrides)
    return preset


class WaterValidationTests(unittest.TestCase):
    def test_public_signatures_are_locked(self):
        build = inspect.signature(native.build_water)
        self.assertEqual(
            list(build.parameters), ["scene", "surface_z", "preset", "extent", "bottom_z"]
        )
        self.assertEqual(build.parameters["extent"].default, 70.0)
        self.assertEqual(build.parameters["bottom_z"].default, -3.0)
        self.assertEqual(list(inspect.signature(native.set_water_enabled).parameters), ["handle", "enabled"])
        self.assertEqual(list(inspect.signature(native.set_water_preset).parameters), ["handle", "preset"])
        self.assertEqual(list(inspect.signature(native.water_metadata).parameters), ["handle"])

    def test_valid_preset_is_copied_to_json_values(self):
        original = valid_preset()
        result = native._validate_preset(original)
        self.assertEqual(result["id"], "medium")
        self.assertEqual(result["absorption"], [0.30, 0.10, 0.04])
        self.assertEqual(result["scatter"], [0.06, 0.06, 0.06])
        self.assertEqual(result["g"], 0.65)
        self.assertEqual(result["ior"], 1.333)
        result["absorption"][0] = 9.0
        self.assertEqual(original["absorption"][0], 0.30)

    def test_invalid_preset_values_raise_value_error(self):
        cases = [
            {},
            valid_preset(absorption=[0.1, 0.2]),
            valid_preset(scatter=[0.1, -0.2, 0.3]),
            valid_preset(absorption=[0.1, float("nan"), 0.3]),
            valid_preset(g=-1.0),
            valid_preset(g=1.0),
            valid_preset(g=2.0),
            valid_preset(ior=0.0),
            valid_preset(ior=float("inf")),
            valid_preset(id="bad/name"),
        ]
        for preset in cases:
            with self.subTest(preset=preset):
                with self.assertRaises(ValueError):
                    native._validate_preset(preset)

    def test_unit_ior_is_valid_for_no_refraction_control(self):
        result = native._validate_preset(valid_preset(ior=1.0))
        self.assertEqual(result["ior"], 1.0)

    def test_invalid_geometry_values_raise_value_error(self):
        self.assertEqual(native._validate_geometry(5.0, 70.0, -3.0), (5.0, 70.0, -3.0))
        for args in ((-3.0, 70.0, -3.0), (2.0, 0.0, -3.0), (2.0, -1.0, -3.0), (2.0, 70.0, 2.0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    native._validate_geometry(*args)

    def test_build_validates_before_bpy_or_scene_mutation(self):
        # Invalid configuration must report ValueError even when no Blender
        # module is present; no collection/data-block can have been created.
        with self.assertRaises(ValueError):
            native.build_water(None, 5.0, valid_preset(g=1.0))
        with self.assertRaises(ValueError):
            native.build_water(None, 0.0, valid_preset())

        # A valid preset still requires an actual Blender scene.  This is a
        # contract check only; the integration run is owned by the main Agent.
        with self.assertRaises(ValueError):
            native.build_water(None, 5.0, valid_preset())

    def test_metadata_rejects_unbuilt_or_non_water_handles(self):
        with self.assertRaises(ValueError):
            native.water_metadata({"objects": []})
        with self.assertRaises(ValueError):
            native.set_water_enabled({"objects": [{"is_water": False}]}, True)

    def test_nested_idproperty_like_values_keep_structure_for_json(self):
        class IDPropertyLike:
            def __init__(self, values):
                self.values = values

            def keys(self):
                return self.values.keys()

            def __getitem__(self, key):
                return self.values[key]

        value = IDPropertyLike({"top": [1], "sides": IDPropertyLike({"count": 4})})
        converted = native._json_compatible(value)
        self.assertEqual(converted, {"top": [1], "sides": {"count": 4}})
        self.assertEqual(json.loads(json.dumps(converted, allow_nan=False)), converted)

    def test_source_uses_native_nodes_without_image_shortcuts(self):
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        node_ids = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("ShaderNodeVolumeCoefficients", node_ids)
        self.assertIn("ShaderNodeBsdfGlass", node_ids)
        self.assertNotIn("ShaderNodeLightPath", node_ids)
        self.assertNotIn("ShaderNodeBsdfTransparent", node_ids)
        self.assertNotIn("CompositorNode", source)
        self.assertNotIn("driver_add", source)
        self.assertNotIn("bpy.types", source)
        self.assertNotIn("foreach_get", source)
        self.assertNotIn('_socket(volume, "IOR")', source)
        self.assertNotIn('_set_socket_value(coefficients, "IOR"', source)
        self.assertIn("HENYEY_GREENSTEIN", source)

    def test_mesh_contract_is_explicit_in_source(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("one_closed_finite_volume", source)
        self.assertIn("top_polygon_index", source)
        self.assertIn("side_bottom_surface_connected", source)
        self.assertIn("material_slot_volume_inputs_match", source)
        self.assertIn("physical_limitations", source)


@unittest.skipUnless(native.bpy is not None, "requires Blender's bpy runtime")
class BlenderWaterIntegrationTests(unittest.TestCase):
    """Runtime checks for the real 5.2 Blender data API.

    The test does not render.  It creates one temporary empty scene, exercises
    the public water handle, and removes only the data-blocks it created.
    """

    def setUp(self):
        self.blender = native.bpy
        self.scene = self.blender.data.scenes.new("TEST Native Water API")
        self.handle = None
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        if self.handle is not None:
            for obj in list(self.handle["objects"]):
                if obj.name in self.blender.data.objects:
                    self.blender.data.objects.remove(obj, do_unlink=True)
            for material in list(self.handle["materials"]):
                if material.name in self.blender.data.materials:
                    self.blender.data.materials.remove(material)
            group = self.handle["node_group"]
            if group.name in self.blender.data.node_groups:
                self.blender.data.node_groups.remove(group)
            mesh = self.handle["mesh"]
            if mesh.name in self.blender.data.meshes:
                self.blender.data.meshes.remove(mesh)
            collection = self.handle["collection"]
            if collection.name in self.blender.data.collections:
                self.blender.data.collections.remove(collection)
        if self.scene.name in self.blender.data.scenes:
            self.blender.data.scenes.remove(self.scene)

    def test_build_failure_removes_only_new_datablocks(self):
        counts_before = {
            "collections": len(self.blender.data.collections),
            "node_groups": len(self.blender.data.node_groups),
            "meshes": len(self.blender.data.meshes),
            "objects": len(self.blender.data.objects),
            "materials": len(self.blender.data.materials),
        }
        original_make_material = native._make_material

        def injected_failure(*args, **kwargs):
            raise RuntimeError("injected material construction failure")

        native._make_material = injected_failure
        try:
            with self.assertRaisesRegex(RuntimeError, "injected material construction failure"):
                native.build_water(
                    self.scene,
                    surface_z=5.0,
                    preset=valid_preset(),
                    extent=7.0,
                    bottom_z=-3.0,
                )
        finally:
            native._make_material = original_make_material

        counts_after = {
            "collections": len(self.blender.data.collections),
            "node_groups": len(self.blender.data.node_groups),
            "meshes": len(self.blender.data.meshes),
            "objects": len(self.blender.data.objects),
            "materials": len(self.blender.data.materials),
        }
        self.assertEqual(counts_after, counts_before)
        self.assertEqual(
            [obj for obj in self.scene.objects if obj.get("is_water", False)],
            [],
        )

    def test_build_metadata_clear_and_native_preset_update(self):
        self.handle = native.build_water(
            self.scene,
            surface_z=5.0,
            preset=valid_preset(),
            extent=7.0,
            bottom_z=-3.0,
        )
        self.assertEqual(len(self.handle["objects"]), 1)
        metadata = native.water_metadata(self.handle)
        json.dumps(metadata, allow_nan=False)
        self.assertTrue(metadata["boundary"]["closed"])
        self.assertTrue(metadata["boundary"]["top_and_sides_single_mesh"])
        self.assertEqual(metadata["boundary"]["surface_z"], 5.0)
        self.assertEqual(metadata["boundary"]["bottom_z"], -3.0)
        self.assertEqual(metadata["boundary"]["extent_x"], [-7.0, 7.0])
        self.assertEqual(metadata["boundary"]["extent_y"], [-7.0, 7.0])
        top_normal = metadata["boundary"]["top_outward_normal"]
        self.assertAlmostEqual(top_normal[0], 0.0, places=6)
        self.assertAlmostEqual(top_normal[1], 0.0, places=6)
        self.assertAlmostEqual(top_normal[2], 1.0, places=6)
        self.assertFalse(metadata["integrity"]["ecology_labels_present"])
        self.assertEqual(metadata["nodes"]["volume"]["phase"], "HENYEY_GREENSTEIN")
        self.assertTrue(metadata["nodes"]["top_surface_connected"])
        self.assertTrue(metadata["nodes"]["top_volume_connected"])
        self.assertFalse(metadata["nodes"]["side_bottom_surface_connected"])
        self.assertTrue(metadata["nodes"]["side_bottom_volume_connected"])
        self.assertTrue(metadata["nodes"]["volume"]["direct_input_links"]["absorption"]["linked"])
        self.assertTrue(metadata["nodes"]["volume"]["direct_input_links"]["scatter"]["linked"])

        native.set_water_enabled(self.handle, False)
        self.assertTrue(all(obj.hide_render for obj in self.handle["objects"]))
        self.assertTrue(native.water_metadata(self.handle)["enabled"] is False)
        native.set_water_enabled(self.handle, True)
        self.assertTrue(all(not obj.hide_render for obj in self.handle["objects"]))
        self.assertTrue(native.water_metadata(self.handle)["enabled"] is True)

        alternate = valid_preset(
            id="ior145",
            absorption=[0.12, 0.23, 0.34],
            scatter=[0.01, 0.02, 0.03],
            g=-0.40,
            ior=1.45,
        )
        native.set_water_preset(self.handle, alternate)
        updated = native.water_metadata(self.handle)
        self.assertEqual(updated["preset"]["id"], "ior145")
        for actual, expected in zip(updated["preset"]["absorption"], alternate["absorption"]):
            self.assertAlmostEqual(actual, expected, delta=1e-6)
        for actual, expected in zip(updated["preset"]["scatter"], alternate["scatter"]):
            self.assertAlmostEqual(actual, expected, delta=1e-6)
        self.assertAlmostEqual(updated["preset"]["g"], -0.40, delta=1e-6)
        self.assertAlmostEqual(updated["preset"]["ior"], 1.45, delta=1e-6)
        self.assertAlmostEqual(updated["nodes"]["surface"]["ior"], 1.45, delta=1e-6)
        self.assertAlmostEqual(updated["nodes"]["surface"]["roughness"], 0.0, delta=1e-6)
        self.assertEqual(updated["nodes"]["surface"]["color"], [1.0, 1.0, 1.0, 1.0])
        self.assertFalse(updated["nodes"]["volume"]["phase_ior_applicable"])
        self.assertIsNone(updated["nodes"]["volume"]["inputs"]["ior"])
        self.assertIsNone(updated["nodes"]["volume"]["direct_input_links"]["ior"])
        self.assertTrue(updated["nodes"]["material_slot_volume_inputs_match"])
        self.assertIn("Initial node-group socket defaults", updated["nodes"]["node_group"]["interface_defaults_scope"])
        self.assertEqual(updated["integrity"]["custom_driver_count"], 0)


if __name__ == "__main__":
    unittest.main()
