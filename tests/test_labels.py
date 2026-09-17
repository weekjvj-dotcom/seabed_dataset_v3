"""Independent checks for the geometry-label exporter.

Running this file with the system Python exercises the dependency-free image
formats.  Running it with Blender's Python additionally creates a small,
known scene.  The expected plane depths and probe directions are calculated in
the test from hard-coded camera parameters; they are not copied from the
exporter's hit values.  This file is intentionally supplied for the parent
agent to run in Blender and is not launched by this worker.
"""

from __future__ import annotations

import math
import struct
import tempfile
import unittest
from pathlib import Path

from seabed_dataset_v2.src.labels.formats import FormatError, read_exr, read_png, write_exr, write_png

try:  # Optional independent decoder; available in the dedicated validation env.
    import Imath  # type: ignore
    import OpenEXR  # type: ignore
except ImportError:  # pragma: no cover - system Python does not ship OpenEXR
    Imath = None  # type: ignore
    OpenEXR = None  # type: ignore

try:  # Blender-only tests remain importable under ordinary CPython.
    import bpy  # type: ignore
    from mathutils import Matrix  # type: ignore
except ImportError:  # pragma: no cover - normal CPython path
    bpy = None  # type: ignore
    Matrix = None  # type: ignore


class TestLabelFormats(unittest.TestCase):
    def test_png_and_exr_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="seabed_labels_") as temporary:
            directory = Path(temporary)
            pixels8 = [[0, 1, 127, 255], [8, 64, 192, 254]]
            pixels16 = [[0, 1, 32768, 65535], [11, 2222, 50000, 65534]]
            depth = [[0.0, 1.0, 4.25, 10.0], [0.5, 2.0, 8.0, 16.0]]
            write_png(directory / "mask.png", pixels8, 4, 2, 8)
            write_png(directory / "semantic.png", pixels16, 4, 2, 16)
            write_exr(directory / "depth.exr", depth, 4, 2)
            self.assertEqual(read_png(directory / "mask.png")["pixels"], pixels8)
            self.assertEqual(read_png(directory / "semantic.png")["pixels"], pixels16)
            decoded_depth = read_exr(directory / "depth.exr")["pixels"]
            for decoded_row, expected_row in zip(decoded_depth, depth):
                for decoded, expected in zip(decoded_row, expected_row):
                    self.assertEqual(decoded, expected)

    def test_exr_uses_standard_magic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="seabed_labels_") as temporary:
            path = Path(temporary) / "depth.exr"
            write_exr(path, [[0.0]], 1, 1)
            raw = path.read_bytes()
            # This assertion is independent of the module's reader: it catches
            # a writer and reader accidentally sharing the same wrong constant.
            self.assertEqual(raw[:4], b"\x76\x2f\x31\x01")
            self.assertEqual(struct.unpack("<I", raw[:4])[0], 0x01312F76)

    @unittest.skipIf(OpenEXR is None or Imath is None, "OpenEXR 3 Python bindings are unavailable")
    def test_exr_is_readable_by_openexr3(self) -> None:
        expected = [[0.0, 1.25], [4.0, 16.0]]
        with tempfile.TemporaryDirectory(prefix="seabed_labels_") as temporary:
            path = Path(temporary) / "depth.exr"
            write_exr(path, expected, 2, 2)
            image = OpenEXR.InputFile(str(path))
            try:
                header = image.header()
                self.assertIn("Y", header["channels"])
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                payload = image.channel("Y", pixel_type)
            finally:
                close = getattr(image, "close", None)
                if callable(close):
                    close()
            values = struct.unpack("<4f", payload)
            self.assertEqual(values, (0.0, 1.25, 4.0, 16.0))

    def test_invalid_label_samples_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="seabed_labels_") as temporary:
            directory = Path(temporary)
            with self.assertRaises(ValueError):
                write_png(directory / "bad.png", [[-1]], 1, 1, 8)
            with self.assertRaises(ValueError):
                write_png(directory / "bad.png", [[1.5]], 1, 1, 16)
            with self.assertRaises(ValueError):
                write_png(directory / "bad.png", [[1]], 1, 1, 16.9)
            with self.assertRaises(ValueError):
                write_png(directory / "bad.png", [[1]], 1, 1, True)
            with self.assertRaises(ValueError):
                write_exr(directory / "bad.exr", [[math.inf]], 1, 1)
            directory.joinpath("not-an-image").write_bytes(b"broken")
            with self.assertRaises(FormatError):
                read_png(directory / "not-an-image")

    def test_exr_rejects_invalid_version_flags(self) -> None:
        with tempfile.TemporaryDirectory(prefix="seabed_labels_") as temporary:
            path = Path(temporary) / "depth.exr"
            write_exr(path, [[1.0]], 1, 1)
            original = bytearray(path.read_bytes())
            # Version 2 plus the long-name flag is still a supported scanline
            # file; the other values must be rejected before header parsing.
            long_name = bytearray(original)
            struct.pack_into("<I", long_name, 4, 2 | 0x0400)
            path.write_bytes(long_name)
            self.assertEqual(read_exr(path)["pixels"], [[1.0]])
            for flags_or_version in (2 | 0x0200, 2 | 0x0800, 2 | 0x1000, 2 | 0x2000, 3):
                invalid = bytearray(original)
                struct.pack_into("<I", invalid, 4, flags_or_version)
                path.write_bytes(invalid)
                with self.subTest(version=hex(flags_or_version)), self.assertRaises(FormatError):
                    read_exr(path)


if bpy is not None:  # pragma: no cover - executed only by Blender's Python
    from seabed_dataset_v2.src.labels.geometry import (
        _render_dimensions,
        camera_metadata,
        export_labels,
        ray_for_pixel,
    )


    def _clear_scene(scene) -> None:
        for obj in list(scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)


    def _box(scene, name: str, center: tuple[float, float, float], size: float, semantic_id: int, instance_id: int, morphology: str):
        half = size * 0.5
        cx, cy, cz = center
        vertices = [
            (cx - half, cy - half, cz - half),
            (cx + half, cy - half, cz - half),
            (cx + half, cy + half, cz - half),
            (cx - half, cy + half, cz - half),
            (cx - half, cy - half, cz + half),
            (cx + half, cy - half, cz + half),
            (cx + half, cy + half, cz + half),
            (cx - half, cy + half, cz + half),
        ]
        faces = [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        mesh = bpy.data.meshes.new(f"{name}_mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        obj["semantic_id"] = semantic_id
        obj["instance_id"] = instance_id
        obj["asset_morphology"] = morphology
        return obj


    class TestBlenderGeometryLabels(unittest.TestCase):
        width = 11
        height = 7
        lens_mm = 50.0
        sensor_width_mm = 36.0

        @classmethod
        def setUpClass(cls):
            cls.scene = bpy.context.scene

        def setUp(self):
            _clear_scene(self.scene)
            render = self.scene.render
            render.resolution_x = self.width
            render.resolution_y = self.height
            render.resolution_percentage = 100
            render.pixel_aspect_x = 1.0
            render.pixel_aspect_y = 1.0
            render.use_motion_blur = False

            camera_data = bpy.data.cameras.new("labels_test_camera_data")
            camera_data.type = "PERSP"
            camera_data.lens = self.lens_mm
            camera_data.sensor_width = self.sensor_width_mm
            camera_data.sensor_height = 24.0
            camera_data.sensor_fit = "HORIZONTAL"
            camera_data.shift_x = 0.0
            camera_data.shift_y = 0.0
            camera = bpy.data.objects.new("labels_test_camera", camera_data)
            self.scene.collection.objects.link(camera)
            camera.matrix_world = Matrix.Identity(4)
            self.scene.camera = camera

            # A large plane at z=-4 gives every non-probe pixel a known target.
            floor_mesh = bpy.data.meshes.new("floor_mesh")
            floor_mesh.from_pydata(
                [(-20.0, -20.0, -4.0), (20.0, -20.0, -4.0), (20.0, 20.0, -4.0), (-20.0, 20.0, -4.0)],
                [],
                [(0, 1, 2, 3)],
            )
            floor_mesh.update()
            floor = bpy.data.objects.new("known_sand_floor", floor_mesh)
            self.scene.collection.objects.link(floor)
            floor["semantic_id"] = 1
            floor["instance_id"] = 10
            floor["asset_morphology"] = "plane"
            self.floor = floor

            # Independent pinhole values used only to place probe meshes and
            # calculate expected results below.
            self.fx = self.lens_mm / self.sensor_width_mm * self.width
            self.fy = self.fx
            self.cx = self.width * 0.5
            self.cy = self.height * 0.5

            def point_for_pixel(pixel_u: int, pixel_v: int, depth: float = 3.0):
                slope_x = (pixel_u + 0.5 - self.cx) / self.fx
                slope_y = (self.cy - (pixel_v + 0.5)) / self.fy
                return (slope_x * depth, slope_y * depth, -depth)

            self.probes = {
                "top": _box(self.scene, "probe_top_coral", point_for_pixel(5, 1), 0.25, 3, 101, "branching"),
                "bottom": _box(self.scene, "probe_bottom_grass", point_for_pixel(5, 5), 0.25, 4, 102, "blade"),
                "left": _box(self.scene, "probe_left_rock", point_for_pixel(1, 3), 0.25, 2, 103, "rock"),
                "right_a": _box(self.scene, "probe_right_fish_body", point_for_pixel(9, 3), 0.25, 5, 104, "fusiform"),
                # A separate mesh part with the same fish instance ID.
                "right_b": _box(self.scene, "probe_second_fish_part", point_for_pixel(8, 1), 0.25, 5, 104, "fusiform"),
            }
            # A large water object would cover the probes if it were included;
            # the is_water marker must exclude it from the label BVH.
            water = _box(self.scene, "water_volume_probe", (0.0, 0.0, -2.0), 30.0, 1, 999, "water")
            water["is_water"] = True
            self.water = water
            bpy.context.view_layer.update()

        def tearDown(self):
            _clear_scene(self.scene)
            bpy.context.view_layer.update()

        def test_render_percentage_uses_blender_truncation(self):
            self.scene.render.resolution_x = 103
            self.scene.render.resolution_y = 100
            self.scene.render.resolution_percentage = 33
            self.assertEqual(_render_dimensions(self.scene), (33, 33))

        def test_clip_planes_bound_cast_without_near_plane_occlusion(self):
            # Probe boxes are around z=-3 and the floor is z=-4.  A near clip
            # at 3.5 removes probes while retaining the farther floor.
            self.scene.camera.data.clip_start = 3.5
            with tempfile.TemporaryDirectory(prefix="seabed_blender_labels_") as temporary:
                metadata = export_labels(self.scene, temporary)
                semantic = read_png(Path(temporary) / metadata["files"]["semantic_id"])["pixels"]
                valid = read_png(Path(temporary) / metadata["files"]["valid_mask"])["pixels"]
                self.assertEqual(semantic[1][5], 1)
                self.assertEqual(valid[1][5], 255)

            # A near clip beyond the floor removes every target along the
            # center ray; the miss remains zero rather than being replaced by
            # an artificial clipping-plane hit.
            self.scene.camera.data.clip_start = 4.5
            with tempfile.TemporaryDirectory(prefix="seabed_blender_labels_") as temporary:
                metadata = export_labels(self.scene, temporary)
                semantic = read_png(Path(temporary) / metadata["files"]["semantic_id"])["pixels"]
                valid = read_png(Path(temporary) / metadata["files"]["valid_mask"])["pixels"]
                self.assertEqual(semantic[3][5], 0)
                self.assertEqual(valid[3][5], 0)

        def test_far_clip_is_applied_to_target_intersections(self):
            self.scene.camera.data.clip_end = 3.5
            with tempfile.TemporaryDirectory(prefix="seabed_blender_labels_") as temporary:
                metadata = export_labels(self.scene, temporary)
                semantic = read_png(Path(temporary) / metadata["files"]["semantic_id"])["pixels"]
                valid = read_png(Path(temporary) / metadata["files"]["valid_mask"])["pixels"]
                self.assertEqual(semantic[3][5], 0)
                self.assertEqual(valid[3][5], 0)
                self.assertEqual(semantic[1][5], 3)

        def test_visibility_contract_rejects_explicit_hidden_targets(self):
            self.floor.hide_render = True
            with tempfile.TemporaryDirectory(prefix="seabed_blender_labels_") as temporary:
                with self.assertRaises(ValueError):
                    export_labels(self.scene, temporary, objects=[self.floor])
                metadata = export_labels(self.scene, temporary)
                semantic = read_png(Path(temporary) / metadata["files"]["semantic_id"])["pixels"]
                self.assertEqual(semantic[3][5], 0)

            self.floor.hide_render = False
            self.floor["visible_camera"] = False
            with tempfile.TemporaryDirectory(prefix="seabed_blender_labels_") as temporary:
                with self.assertRaises(ValueError):
                    export_labels(self.scene, temporary, objects=[self.floor])
                metadata = export_labels(self.scene, temporary)
                semantic = read_png(Path(temporary) / metadata["files"]["semantic_id"])["pixels"]
                self.assertEqual(semantic[3][5], 0)

            hidden_collection = bpy.data.collections.new("labels_hidden_collection")
            self.scene.collection.children.link(hidden_collection)
            self.scene.collection.objects.unlink(self.floor)
            hidden_collection.objects.link(self.floor)
            hidden_collection.hide_render = True
            del self.floor["visible_camera"]
            with tempfile.TemporaryDirectory(prefix="seabed_blender_labels_") as temporary:
                with self.assertRaises(ValueError):
                    export_labels(self.scene, temporary, objects=[self.floor])

        def test_camera_and_plane_depths_use_independent_formula(self):
            with tempfile.TemporaryDirectory(prefix="seabed_blender_labels_") as temporary:
                metadata = export_labels(self.scene, temporary)
                camera_meta = camera_metadata(self.scene)
                # Derive expected K from the explicitly configured sensor and
                # resolution, rather than reading K back as the oracle.
                # Blender's view_frame is evaluated in single-precision-like
                # camera coordinates; use an explicit calibration readback
                # tolerance instead of decimal-place rounding at 1e-6.
                self.assertAlmostEqual(camera_meta["K"][0][0], self.fx, delta=1.0e-5)
                self.assertAlmostEqual(camera_meta["K"][1][1], self.fy, delta=1.0e-5)
                self.assertAlmostEqual(camera_meta["K"][0][2], self.cx, delta=1.0e-5)
                self.assertAlmostEqual(camera_meta["K"][1][2], self.cy, delta=1.0e-5)
                # K uses pixel-edge coordinates while ray_for_pixel adds the
                # 0.5 pixel-center offset exactly once.  The odd-sized image's
                # center pixel must therefore point along camera -Z.
                _origin, center_direction = ray_for_pixel(self.scene, 5, 3)
                self.assertAlmostEqual(center_direction.x, 0.0, delta=1.0e-6)
                self.assertAlmostEqual(center_direction.y, 0.0, delta=1.0e-6)
                self.assertAlmostEqual(center_direction.z, -1.0, delta=1.0e-6)
                depth_range = read_exr(Path(temporary) / metadata["files"]["depth_range_m"])["pixels"]
                depth_z = read_exr(Path(temporary) / metadata["files"]["depth_camera_z_m"])["pixels"]
                semantic = read_png(Path(temporary) / metadata["files"]["semantic_id"])["pixels"]
                instance = read_png(Path(temporary) / metadata["files"]["instance_id"])["pixels"]
                valid = read_png(Path(temporary) / metadata["files"]["valid_mask"])["pixels"]

                # Center ray intersects the independent floor at (0,0,-4).
                self.assertEqual(semantic[3][5], 1)
                self.assertEqual(instance[3][5], 10)
                self.assertEqual(valid[3][5], 255)
                self.assertAlmostEqual(depth_range[3][5], 4.0, delta=0.001)
                self.assertAlmostEqual(depth_z[3][5], 4.0, delta=0.001)

                # Pixel (4,3) has a known off-axis range on the same plane.
                pixel_u, pixel_v = 4, 3
                slope_x = (pixel_u + 0.5 - self.cx) / self.fx
                slope_y = (self.cy - (pixel_v + 0.5)) / self.fy
                expected_range = 4.0 * math.sqrt(1.0 + slope_x * slope_x + slope_y * slope_y)
                tolerance = max(0.001, 0.001 * expected_range)
                self.assertEqual(semantic[pixel_v][pixel_u], 1)
                self.assertAlmostEqual(depth_range[pixel_v][pixel_u], expected_range, delta=tolerance)
                self.assertAlmostEqual(depth_z[pixel_v][pixel_u], 4.0, delta=tolerance)

                # Four directional probes verify image-top/image-bottom and
                # image-left/image-right orientation and their labels.
                for (pixel_u, pixel_v), expected_semantic, expected_instance in (
                    ((5, 1), 3, 101),
                    ((5, 5), 4, 102),
                    ((1, 3), 2, 103),
                    ((9, 3), 5, 104),
                    ((8, 1), 5, 104),
                ):
                    self.assertEqual(semantic[pixel_v][pixel_u], expected_semantic)
                    self.assertEqual(instance[pixel_v][pixel_u], expected_instance)
                    self.assertEqual(valid[pixel_v][pixel_u], 255)

                # The two fish meshes intentionally share one instance ID.
                self.assertEqual(metadata["instances"]["104"]["semantic_id"], 5)
                self.assertEqual(set(metadata["instances"]["104"]["objects"]), {"probe_right_fish_body", "probe_second_fish_part"})
                self.assertGreaterEqual(metadata["instances"]["104"]["visible_pixel_count"], 2)
                self.assertEqual(metadata["target_object_count"], 6)

        def test_instance_id_cannot_cross_categories(self):
            conflict = _box(self.scene, "conflicting_instance", (0.0, 0.0, -2.0), 0.1, 1, 104, "bad")
            with tempfile.TemporaryDirectory(prefix="seabed_blender_labels_") as temporary:
                with self.assertRaises(ValueError):
                    export_labels(self.scene, temporary, objects=[self.floor, self.probes["right_a"], conflict])


if __name__ == "__main__":
    unittest.main()
