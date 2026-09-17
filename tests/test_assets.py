"""契约测试：复杂海底生态资产必须是可求值的静态 Blender 网格。

该测试在 Blender 的 Python 环境中运行。普通 CPython 只会编译该文件，缺少
``bpy`` 时测试类会被跳过；这样可以把接口检查与 Blender 集成检查分开执行。
"""

from __future__ import annotations

import hashlib
import math
from random import Random
import sys
import unittest

try:  # pragma: no cover - bpy only exists inside Blender
    import bpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    bpy = None  # type: ignore


if bpy is not None:  # pragma: no branch - selection is environment dependent
    from seabed_dataset_v2.src.assets.ecology import (
        _build_fish,
        _layout_profile,
        build_ecosystem,
    )
    from seabed_dataset_v2.src.assets.geometry import GeometryData, append_leaf_blade, append_tube


def _face_normal(vertices, face):
    """Return the first-triangle normal for a generated polygon."""

    a, b, c = (vertices[index] for index in face[:3])
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    return (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )


def _geometry_digest(result) -> str:
    """Hash static geometry while ignoring Blender's auto-renamed object names."""

    digest = hashlib.sha256()
    for obj in sorted(result["objects"], key=lambda item: int(item["instance_id"])):
        digest.update(f"{int(obj['instance_id'])}:{obj['asset_morphology']}".encode("utf-8"))
        for vertex in obj.data.vertices:
            for coordinate in vertex.co:
                digest.update(f"{float(coordinate):.9f},".encode("ascii"))
        for polygon in obj.data.polygons:
            digest.update(f"{int(polygon.material_index)}:{int(polygon.use_smooth)},".encode("ascii"))
    return digest.hexdigest()


def _discard_test_build(scene, result) -> None:
    """Dispose of temporary scene data created by the isolation regression."""

    collection = result["collection"]
    for obj in list(collection.objects):
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    bpy.data.scenes.remove(scene)
    if collection.users == 0:
        bpy.data.collections.remove(collection)


@unittest.skipIf(bpy is None, "asset contract tests require Blender's bpy module")
class TestEcosystemAssetContract(unittest.TestCase):
    """Validate the stable public result shape and complexity guardrails."""

    @classmethod
    def setUpClass(cls):
        cls.scene = bpy.context.scene
        cls.result = build_ecosystem(cls.scene, seed=1729, floor_z=0.0)

    def test_public_result_contract(self):
        result = self.result
        self.assertIsNotNone(result.get("collection"))
        self.assertTrue(result.get("objects"))
        self.assertTrue(result.get("instances"))
        self.assertIsInstance(result.get("summary"), dict)
        self.assertIn("morphology_counts", result["summary"])
        self.assertIn("triangle_count", result["summary"])
        self.assertIn("mesh_budget", result["summary"])
        self.assertIn("asset_source", result["summary"])
        self.assertIn("layout_family", result["summary"])
        self.assertIn("layout_style", result["summary"])
        self.assertIn("layout_notes", result["summary"])
        self.assertEqual(result["summary"]["sand_extent"]["x"], [-40.0, 40.0])
        self.assertEqual(result["summary"]["sand_extent"]["y"], [-50.0, 50.0])
        self.assertEqual(result["summary"]["asset_source"], "original_programmatic")

    def test_semantic_and_instance_ids_are_complete(self):
        result = self.result
        target_objects = result["objects"]
        valid_semantics = {1, 2, 3, 4, 5}
        self.assertTrue(valid_semantics.issubset({int(o.get("semantic_id", -1)) for o in target_objects}))

        listed_names = {o.name for o in target_objects}
        instance_rows = result["instances"]
        self.assertTrue(instance_rows)
        seen_ids = set()
        id_semantics = {}
        for row in instance_rows:
            instance_id = int(row["instance_id"])
            semantic_id = int(row["semantic_id"])
            self.assertGreaterEqual(instance_id, 1)
            self.assertLessEqual(instance_id, 65535)
            self.assertIn(semantic_id, valid_semantics)
            self.assertNotIn(instance_id, seen_ids, msg="instances should contain one row per stable ID")
            seen_ids.add(instance_id)
            id_semantics[instance_id] = semantic_id
            self.assertTrue(row.get("objects"))
            self.assertTrue(set(row["objects"]).issubset(listed_names))

        for obj in target_objects:
            self.assertIn(int(obj.get("semantic_id", -1)), valid_semantics)
            self.assertIn(int(obj.get("instance_id", 0)), id_semantics)
            self.assertEqual(id_semantics[int(obj["instance_id"])], int(obj["semantic_id"]))
            self.assertTrue(obj.get("asset_morphology"))

    def test_required_ecology_scale_and_morphologies(self):
        result = self.result
        rows = result["instances"]
        coral = [r for r in rows if int(r["semantic_id"]) == 3]
        fish = [r for r in rows if int(r["semantic_id"]) == 5]
        self.assertGreaterEqual(len(coral), 30)
        self.assertGreaterEqual(len(fish), 18)
        self.assertLessEqual(len(fish), 24)

        coral_shapes = {str(r["morphology"]) for r in coral}
        self.assertTrue({"branching", "brain", "table", "fan"}.issubset(coral_shapes))
        fish_shapes = {str(r["morphology"]) for r in fish}
        self.assertTrue({"fusiform_fish", "disk_fish"}.issubset(fish_shapes))

        counts = result["summary"]["morphology_counts"]
        for required in ("branching", "brain", "table", "fan", "fusiform_fish", "disk_fish"):
            self.assertGreaterEqual(int(counts.get(required, 0)), 1)

    def test_exact_default_asset_counts(self):
        self.assertEqual(int(self.result["summary"]["object_count"]), 77)
        self.assertEqual(
            self.result["summary"]["morphology_counts"],
            {
                "brain": 10,
                "branching": 10,
                "disk_fish": 9,
                "fan": 4,
                "fusiform_fish": 11,
                "rock_cluster": 14,
                "sandbed": 1,
                "seagrass_clump": 12,
                "table": 6,
            },
        )

    def test_triangle_budget_and_static_meshes(self):
        result = self.result
        total_triangles = 0
        for obj in result["objects"]:
            self.assertEqual(obj.type, "MESH")
            self.assertEqual(len(obj.modifiers), 0, msg=obj.name)
            self.assertIsNone(obj.animation_data, msg=obj.name)
            mesh = obj.data
            mesh.calc_loop_triangles()
            total_triangles += len(mesh.loop_triangles)
        self.assertLess(total_triangles, 1_500_000)
        self.assertEqual(int(result["summary"]["triangle_count"]), total_triangles)
        self.assertLessEqual(int(result["summary"]["mesh_budget"]), 1_500_000)

    def test_materials_are_native_procedural_pbr(self):
        materials = {material.name: material for obj in self.result["objects"] for material in obj.data.materials}
        self.assertTrue(materials)
        for material in materials.values():
            self.assertTrue(material.use_nodes)
            node_types = {node.bl_idname for node in material.node_tree.nodes}
            self.assertIn("ShaderNodeBsdfPrincipled", node_types)
            self.assertIn("ShaderNodeTexNoise", node_types)
            self.assertNotIn("ShaderNodeTexImage", node_types)
            self.assertNotIn("ShaderNodeEmission", node_types)
            self.assertFalse(bool(material.get("uses_emission", True)))

    def test_pilot_layout_families_are_distinct_and_smooth(self):
        families = {_layout_profile(seed)[0] for seed in (101, 202, 303)}
        self.assertEqual(families, {"curved_dual_ridge", "left_concentrated_island", "diagonal_scatter"})
        for obj in self.result["objects"]:
            if str(obj.get("asset_morphology")) == "rock_cluster":
                continue
            self.assertTrue(all(polygon.use_smooth for polygon in obj.data.polygons), msg=obj.name)

    def test_static_geometry_is_seed_repeatable_and_scene_isolated(self):
        original_collection = self.result["collection"]
        original_digest = _geometry_digest(self.result)
        original_object_count = len(original_collection.objects)
        same_scene = bpy.data.scenes.new("asset_contract_same_seed")
        different_scene = bpy.data.scenes.new("asset_contract_different_seed")
        same_result = None
        different_result = None
        try:
            same_result = build_ecosystem(same_scene, seed=1729, floor_z=0.0)
            different_result = build_ecosystem(different_scene, seed=101, floor_z=0.0)
            self.assertIsNot(same_result["collection"], original_collection)
            self.assertIsNot(different_result["collection"], original_collection)
            self.assertIsNot(same_result["collection"], different_result["collection"])
            self.assertEqual(self.scene.collection.children.get(original_collection.name), original_collection)
            self.assertEqual(len(original_collection.objects), original_object_count)
            self.assertEqual(_geometry_digest(same_result), original_digest)
            self.assertNotEqual(_geometry_digest(different_result), original_digest)
            self.assertNotEqual(
                same_result["summary"]["layout_family"],
                different_result["summary"]["layout_family"],
            )
        finally:
            if same_result is not None:
                _discard_test_build(same_scene, same_result)
            else:
                bpy.data.scenes.remove(same_scene)
            if different_result is not None:
                _discard_test_build(different_scene, different_result)
            else:
                bpy.data.scenes.remove(different_scene)

    def test_shared_prewired_scene_is_rejected_without_mutation(self):
        original_collection = self.result["collection"]
        original_digest = _geometry_digest(self.result)
        original_object_count = len(original_collection.objects)
        original_children = tuple(child.name for child in self.scene.collection.children)
        shared_scene = bpy.data.scenes.new("asset_contract_shared_scene")
        shared_scene.collection.children.link(original_collection)
        try:
            with self.assertRaisesRegex(
                ValueError,
                "shared ecosystem scene cannot be rebuilt; use a fresh scene",
            ):
                build_ecosystem(shared_scene, seed=1729, floor_z=0.0)
            self.assertEqual(_geometry_digest(self.result), original_digest)
            self.assertEqual(len(original_collection.objects), original_object_count)
            self.assertEqual(tuple(child.name for child in self.scene.collection.children), original_children)
            self.assertEqual(shared_scene.collection.children.get(original_collection.name), original_collection)
        finally:
            shared_scene.collection.children.unlink(original_collection)
            bpy.data.scenes.remove(shared_scene)

    def test_straight_tube_winds_all_faces_outward(self):
        geometry = GeometryData()
        sides = 12
        append_tube(geometry, [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], 0.2, sides=sides, cap=True)
        self.assertEqual(len(geometry.faces), sides + 2 * sides)
        for face in geometry.faces[:sides]:
            normal = _face_normal(geometry.vertices, face)
            center = tuple(sum(geometry.vertices[index][axis] for index in face) / len(face) for axis in range(3))
            radial = (center[0], center[1], 0.0)
            self.assertGreater(sum(normal[axis] * radial[axis] for axis in range(3)), 0.0)
        for side in range(sides):
            start_normal = _face_normal(geometry.vertices, geometry.faces[sides + 2 * side])
            end_normal = _face_normal(geometry.vertices, geometry.faces[sides + 2 * side + 1])
            self.assertLess(start_normal[2], 0.0)
            self.assertGreater(end_normal[2], 0.0)

    def test_leaf_caps_and_fish_stripes_wind_outward(self):
        leaf = GeometryData()
        append_leaf_blade(leaf, [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], [0.12, 0.02], thickness=0.01)
        start_normal = _face_normal(leaf.vertices, leaf.faces[-2])
        end_normal = _face_normal(leaf.vertices, leaf.faces[-1])
        self.assertLess(start_normal[2], 0.0)
        self.assertGreater(end_normal[2], 0.0)

        fish = _build_fish(Random(17), "fusiform_fish", (0.0, 0.0, 0.0), 1.0, -math.pi / 2.0, 0.0, "fish_blue")
        stripe_faces = [
            (face, material_index)
            for face, material_index in zip(fish.faces, fish.material_indices)
            if material_index == 2
        ]
        self.assertTrue(stripe_faces)
        for face, _material_index in stripe_faces:
            normal = _face_normal(fish.vertices, face)
            center = tuple(sum(fish.vertices[index][axis] for index in face) / len(face) for axis in range(3))
            self.assertGreater(normal[0] * center[0], 0.0)

    def test_sandbed_extent_and_flat_id_namespace(self):
        sand = [o for o in self.result["objects"] if str(o.get("asset_morphology")) == "sandbed"]
        self.assertEqual(len(sand), 1)
        bounds = sand[0].bound_box
        xs = [float(v[0]) for v in bounds]
        ys = [float(v[1]) for v in bounds]
        self.assertGreaterEqual(min(xs), -40.01)
        self.assertLessEqual(max(xs), 40.01)
        self.assertGreaterEqual(min(ys), -50.01)
        self.assertLessEqual(max(ys), 50.01)
        self.assertEqual(int(sand[0]["semantic_id"]), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(argv=[sys.argv[0]])
