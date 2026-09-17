"""Geometry-state contract tests for the v2 local-pivot sampler.

These tests run inside Blender.  Ordinary CPython can still compile the module
without importing ``bpy``; the main test runner invokes this module explicitly
after the parent agent has integrated the scene pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from random import Random
import unittest

try:  # pragma: no cover - Blender-only integration tests.
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    bpy = None  # type: ignore
    Matrix = None  # type: ignore
    Vector = None  # type: ignore


if bpy is not None:  # pragma: no branch
    from seabed_dataset_v2.src.assets.ecology import build_ecosystem
    from seabed_dataset_v2.src.randomization import _camera_target_from_xy, _relative_pose, apply_geometry_state, prepare_layout


def _cfg() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "first_round.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {
        "root_seed": 72026,
        "layouts": [{"seed": 101, "split": "train"}],
        "depths_m": [5],
        "camera_sampling": {
            "lens_mm": 28,
            "height_m": [1.8, 2.3],
            "x_range": [-0.45, 0.45],
            "y_range": [-3.8, -3.1],
            "look_y_range": [0, 0.7],
            "look_z_range": [-0.6, -0.3],
            "roll_deg": [-3, 3],
        },
        "object_sampling": {
            "fish_scale": [0.75, 1.25],
            "fish_pitch_deg": [-15, 15],
            "fish_roll_deg": [-8, 8],
            "foreground_fish": [0, 2],
            "midground_fish": [4, 8],
            "background_fish": [3, 6],
            "small_scale": [0.9, 1.1],
            "small_yaw_deg": [-20, 20],
        },
    }


def _world_points(obj):
    return [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]


def _matrix_rows(obj):
    return tuple(tuple(float(obj.matrix_world[row][column]) for column in range(4)) for row in range(4))


def _remove_test_scene(scene, ecosystem):
    collection = ecosystem.get("collection") if ecosystem else None
    if collection is not None:
        for obj in list(collection.objects):
            mesh = obj.data if obj.type == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    camera = scene.camera
    if camera is not None:
        data = camera.data
        bpy.data.objects.remove(camera, do_unlink=True)
        if data is not None and data.users == 0:
            bpy.data.cameras.remove(data)
    if scene.users == 0:
        bpy.data.scenes.remove(scene)


@unittest.skipIf(bpy is None, "v2 geometry tests require Blender's bpy module")
class TestGeometryStateV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _cfg()
        cls.scene = bpy.data.scenes.new("geometry_v2_contract_scene")
        cls.ecosystem = build_ecosystem(cls.scene, seed=101, floor_z=0.0)
        cls.camera_data = bpy.data.cameras.new("geometry_v2_contract_camera_data")
        cls.camera = bpy.data.objects.new("geometry_v2_contract_camera", cls.camera_data)
        cls.scene.collection.objects.link(cls.camera)
        cls.scene.camera = cls.camera
        cls.before = {
            int(obj["instance_id"]): _world_points(obj)
            for obj in cls.ecosystem["objects"]
        }
        cls.base_layout = prepare_layout(cls.scene, cls.ecosystem)
        cls.accepted_attempts = {}
        cls.candidate_failures = {}
        for state_index in range(3):
            accepted = None
            failures = []
            for attempt_index in range(3):
                try:
                    accepted = apply_geometry_state(cls.scene, cls.ecosystem, cls.camera, cls.cfg, state_index, attempt_index)
                    cls.accepted_attempts[state_index] = attempt_index
                    break
                except ValueError as exc:
                    failures.append(str(exc))
                    if not str(exc).startswith("CANDIDATE_"):
                        raise AssertionError(str(exc))
            cls.candidate_failures[state_index] = failures
            if accepted is None:
                raise AssertionError(f"no feasible attempt in 0..2 for state {state_index}: {failures}")

    @classmethod
    def tearDownClass(cls):
        _remove_test_scene(cls.scene, cls.ecosystem)

    def _state_plan(self, state_index):
        return apply_geometry_state(
            self.scene,
            self.ecosystem,
            self.camera,
            self.cfg,
            state_index,
            self.accepted_attempts[state_index],
        )

    def test_prepare_keeps_evaluated_world_vertices(self):
        # Other methods intentionally exercise state resets before this
        # alphabetically ordered test. Restore the prepared base matrices so
        # this assertion measures only the world-baked -> local conversion.
        private = self.ecosystem.get("_layout_private", {})
        for record in private.get("records", []):
            record["object"].matrix_world = record["base_matrix"]
        if self.scene.view_layers and callable(getattr(self.scene.view_layers[0], "update", None)):
            self.scene.view_layers[0].update()
        max_error = 0.0
        for obj in self.ecosystem["objects"]:
            expected = self.before[int(obj["instance_id"])]
            actual = _world_points(obj)
            self.assertEqual(len(expected), len(actual), msg=obj.name)
            for old, new in zip(expected, actual):
                max_error = max(max_error, float((old - new).length))
        self.assertLessEqual(max_error, 1.0e-5)
        self.assertLessEqual(float(self.base_layout["world_vertex_max_error_m"]), 1.0e-5)
        for row in self.base_layout["instances"]:
            self.assertEqual(len(row["base_matrix"]), 4)
            self.assertEqual(len(row["actual_matrix"]), 4)

    def test_pool_ids_and_background_ids_are_stable(self):
        ids = [int(row["instance_id"]) for row in self.ecosystem["instances"]]
        self.assertEqual(ids[:77], list(range(1, 78)))
        self.assertEqual(len(ids), len(set(ids)))
        background = [instance for instance in self.ecosystem["instances"] if int(instance["instance_id"]) > 77]
        self.assertGreaterEqual(len(background), 3)
        self.assertEqual(len(background), len({int(row["instance_id"]) for row in background}))
        self.assertTrue({"background_rock_reef", "background_branching", "background_fan", "background_seagrass"}.intersection({str(row["morphology"]) for row in background}))

    def test_same_state_is_byte_stable_and_other_state_changes(self):
        first = self._state_plan(0)
        first_again = self._state_plan(0)
        self.assertEqual(first, first_again)
        second = self._state_plan(1)
        self.assertNotEqual(first["seed_streams"], second["seed_streams"])
        self.assertNotEqual(first["camera"], second["camera"])
        counts = {first["visible_fish_count"], second["visible_fish_count"]}
        third = self._state_plan(2)
        counts.add(third["visible_fish_count"])
        self.assertGreaterEqual(len(counts), 2)
        self.assertEqual(sorted(self.accepted_attempts), [0, 1, 2])
        self.assertTrue(all(str(error).startswith("CANDIDATE_") for errors in self.candidate_failures.values() for error in errors))

        # Exercise the bounded rejection path with a deliberately impossible
        # camera height, without requiring a normal accepted candidate to fail.
        impossible_cfg = json.loads(json.dumps(self.cfg))
        impossible_cfg["camera_sampling"]["height_m"] = [5.0, 5.0]
        with self.assertRaisesRegex(ValueError, r"^CANDIDATE_CAMERA_CLEARANCE:"):
            apply_geometry_state(self.scene, self.ecosystem, self.camera, impossible_cfg, 0, 0)

        # The same state index under another layout seed must receive
        # different camera/fish/object streams; layout_seed is part of each
        # independent derivation rather than only the shared layout stream.
        other_scene = bpy.data.scenes.new("geometry_v2_other_layout_scene")
        other_ecosystem = build_ecosystem(other_scene, seed=202, floor_z=0.0)
        other_camera_data = bpy.data.cameras.new("geometry_v2_other_layout_camera_data")
        other_camera = bpy.data.objects.new("geometry_v2_other_layout_camera", other_camera_data)
        other_scene.collection.objects.link(other_camera)
        other_scene.camera = other_camera
        try:
            other_plan = None
            for other_attempt in range(3):
                try:
                    other_plan = apply_geometry_state(other_scene, other_ecosystem, other_camera, self.cfg, 0, other_attempt)
                    break
                except ValueError as exc:
                    if not str(exc).startswith("CANDIDATE_"):
                        raise
            if other_plan is None:
                raise AssertionError("no feasible attempt in 0..2 for the second layout")
            for stream in ("camera", "fish", "objects"):
                self.assertNotEqual(first["seed_streams"][stream], other_plan["seed_streams"][stream])
        finally:
            _remove_test_scene(other_scene, other_ecosystem)

    def test_hidden_pool_identity_and_label_target_set(self):
        plan = self._state_plan(1)
        visible = set(plan["visible_object_names"])
        all_rows = {str(row["name"]): row for row in plan["instances"]}
        self.assertTrue(visible)
        self.assertEqual(len(visible), len(plan["visible_object_names"]))
        for obj in self.ecosystem["objects"]:
            row = all_rows[obj.name]
            self.assertEqual(bool(row["active"]), obj.name in visible)
            self.assertEqual(bool(obj.hide_render), obj.name not in visible)
            self.assertEqual(str(row["asset_uid"]), str(obj.get("asset_uid")))
        hidden_ids = {int(row["instance_id"]): str(row["asset_uid"]) for row in plan["instances"] if not row["active"]}
        repeat = self._state_plan(1)
        repeat_hidden = {int(row["instance_id"]): str(row["asset_uid"]) for row in repeat["instances"] if not row["active"]}
        self.assertEqual(hidden_ids, repeat_hidden)

    def test_fish_morphologies_normals_and_camera_clearance(self):
        plan = self._state_plan(0)
        fish = [row for row in plan["instances"] if row["active"] and int(row["semantic_id"]) == 5]
        self.assertGreaterEqual(len({str(row["morphology"]) for row in fish}), 2)
        self.assertTrue(plan["collision_checks"]["camera_final"]["passed"])
        self.assertTrue(plan["collision_checks"]["fish_body_fin_proxy"]["passed"])
        self.assertTrue(plan["collision_checks"]["terrain"]["passed"])
        self.assertGreaterEqual(float(plan["camera"]["height_above_seabed_m"]), 1.8 - 1.0e-6)
        self.assertLessEqual(float(plan["camera"]["height_above_seabed_m"]), 2.3 + 1.0e-6)
        for obj in self.ecosystem["objects"]:
            if obj.hide_render:
                continue
            for polygon in obj.data.polygons:
                normal = polygon.normal
                self.assertTrue(all(math.isfinite(float(value)) for value in normal), msg=obj.name)
                self.assertGreater(float(normal.length), 0.0, msg=obj.name)

    def test_local_pose_origin_does_not_orbit_world_origin(self):
        plan = self._state_plan(2)
        for row in plan["instances"]:
            obj = bpy.data.objects.get(str(row["name"]))
            self.assertIsNotNone(obj)
            origin = obj.matrix_world @ Vector((0.0, 0.0, 0.0))
            translation = obj.matrix_world.translation
            self.assertLessEqual(float((origin - translation).length), 1.0e-6)
        self.assertTrue(all(isinstance(value, (int, float)) for value in plan["camera"]["eye"]))

    def test_pose_axes_match_asset_conventions(self):
        # A canonical fish has X=right, Y=up, Z=forward. A 90-degree heading
        # must turn Z toward X while preserving Y; it must not roll around Z.
        fish = _relative_pose(Matrix.Identity(4), Vector((0.0, 0.0, 0.0)), yaw_deg=90.0, pose_space="fish")
        fish_up = fish.to_3x3() @ Vector((0.0, 1.0, 0.0))
        fish_forward = fish.to_3x3() @ Vector((0.0, 0.0, 1.0))
        self.assertLess(float((fish_up - Vector((0.0, 1.0, 0.0))).length), 1.0e-6)
        self.assertLess(float((fish_forward - Vector((1.0, 0.0, 0.0))).length), 1.0e-6)

        # A benthic 90-degree yaw remains a ground-plane rotation: local Z
        # stays vertical, while local X turns toward local Y.
        benthic = _relative_pose(Matrix.Identity(4), Vector((0.0, 0.0, 0.0)), yaw_deg=90.0, pose_space="benthic")
        benthic_up = benthic.to_3x3() @ Vector((0.0, 0.0, 1.0))
        benthic_right = benthic.to_3x3() @ Vector((1.0, 0.0, 0.0))
        self.assertLess(float((benthic_up - Vector((0.0, 0.0, 1.0))).length), 1.0e-6)
        self.assertLess(float((benthic_right - Vector((0.0, 1.0, 0.0))).length), 1.0e-6)

    def test_optional_pitch_down_sampling_keeps_target_xy(self):
        eye = Vector((0.10, -3.40, 2.00))
        sampling = {"pitch_down_deg": [30.5, 31.5]}
        rng = Random(20260908)
        for index in range(24):
            target_x = -0.2 + index * 0.01
            target_y = 0.10 + index * 0.02
            target, actual = _camera_target_from_xy(eye, target_x, target_y, sampling, rng, 0.0, 1.717)
            horizontal = math.hypot(target_x - eye.x, target_y - eye.y)
            self.assertLessEqual(abs(float(target.x) - target_x), 1.0e-6)
            self.assertLessEqual(abs(float(target.y) - target_y), 1.0e-6)
            self.assertGreaterEqual(float(actual), 30.5 - 1.0e-5)
            self.assertLessEqual(float(actual), 31.5 + 1.0e-5)
            self.assertLessEqual(
                abs(float(math.degrees(math.atan2(eye.z - target.z, horizontal))) - float(actual)),
                1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
