"""Low-level, deterministic mesh builders for the procedural reef scene.

The functions in this module deliberately build meshes with ``from_pydata``
style data.  They do not add modifiers, operators, image textures, or animated
state, so a generated scene can be evaluated and rendered independently of the
Python process that created it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, floor, pi, sin, sqrt
from typing import Callable, Iterable, Sequence

import bpy  # type: ignore


Vec3 = tuple[float, float, float]
MaterialSelector = Callable[[int, int], int]


@dataclass
class GeometryData:
    """A small append-only mesh accumulator with optional face material IDs."""

    vertices: list[Vec3] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    material_indices: list[int] = field(default_factory=list)

    def add_vertex(self, value: Sequence[float]) -> int:
        self.vertices.append((float(value[0]), float(value[1]), float(value[2])))
        return len(self.vertices) - 1

    def add_face(self, indices: Iterable[int], material_index: int = 0) -> None:
        face = tuple(int(i) for i in indices)
        if len(face) >= 3:
            self.faces.append(face)
            self.material_indices.append(int(material_index))

    def extend(self, other: "GeometryData") -> None:
        """Append another geometry while preserving its face material IDs."""

        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.faces.extend(tuple(index + offset for index in face) for face in other.faces)
        self.material_indices.extend(other.material_indices)


def _add(center: Sequence[float], value: Sequence[float]) -> Vec3:
    return (
        float(center[0] + value[0]),
        float(center[1] + value[1]),
        float(center[2] + value[2]),
    )


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0] - b[0]), float(a[1] - b[1]), float(a[2] - b[2]))


def _mul(value: Sequence[float], scalar: float) -> Vec3:
    return (float(value[0] * scalar), float(value[1] * scalar), float(value[2] * scalar))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        float(a[1] * b[2] - a[2] * b[1]),
        float(a[2] * b[0] - a[0] * b[2]),
        float(a[0] * b[1] - a[1] * b[0]),
    )


def _length(value: Sequence[float]) -> float:
    return sqrt(max(0.0, _dot(value, value)))


def _normalize(value: Sequence[float], fallback: Vec3 = (0.0, 0.0, 1.0)) -> Vec3:
    length = _length(value)
    if length < 1.0e-9:
        return fallback
    return _mul(value, 1.0 / length)


def _local_to_world(
    value: Sequence[float],
    center: Sequence[float],
    heading_yaw: float,
    heading_pitch: float = 0.0,
) -> Vec3:
    """Transform fish local coordinates where local +Z is its heading."""

    cy = cos(heading_yaw)
    sy = sin(heading_yaw)
    cp = cos(heading_pitch)
    sp = sin(heading_pitch)
    forward = (cp * cy, cp * sy, sp)
    right = (-sy, cy, 0.0)
    up = (-sp * cy, -sp * sy, cp)
    world = (
        right[0] * value[0] + up[0] * value[1] + forward[0] * value[2],
        right[1] * value[0] + up[1] * value[1] + forward[1] * value[2],
        right[2] * value[0] + up[2] * value[1] + forward[2] * value[2],
    )
    return _add(center, world)


def transform_geometry(
    geometry: GeometryData,
    center: Sequence[float],
    heading_yaw: float,
    heading_pitch: float = 0.0,
) -> GeometryData:
    """Return a transformed copy, useful for complete fish assemblies."""

    return GeometryData(
        vertices=[_local_to_world(v, center, heading_yaw, heading_pitch) for v in geometry.vertices],
        faces=list(geometry.faces),
        material_indices=list(geometry.material_indices),
    )


def append_uv_sphere(
    geometry: GeometryData,
    center: Sequence[float],
    scale: Sequence[float],
    *,
    segments: int = 12,
    rings: int = 7,
    material_index: int = 0,
    material_selector: MaterialSelector | None = None,
) -> None:
    """Append a UV ellipsoid with explicit caps and no modifier dependency."""

    segments = max(6, int(segments))
    rings = max(3, int(rings))
    bottom = geometry.add_vertex(_add(center, (0.0, 0.0, -float(scale[2]))))
    ring_indices: list[list[int]] = []
    for ring in range(1, rings):
        latitude = -pi / 2.0 + pi * ring / rings
        c_lat = cos(latitude)
        s_lat = sin(latitude)
        current: list[int] = []
        for segment in range(segments):
            longitude = 2.0 * pi * segment / segments
            current.append(
                geometry.add_vertex(
                    _add(
                        center,
                        (
                            float(scale[0]) * c_lat * cos(longitude),
                            float(scale[1]) * c_lat * sin(longitude),
                            float(scale[2]) * s_lat,
                        ),
                    )
                )
            )
        ring_indices.append(current)
    top = geometry.add_vertex(_add(center, (0.0, 0.0, float(scale[2]))))

    for segment in range(segments):
        nxt = (segment + 1) % segments
        material = material_selector(0, segment) if material_selector else material_index
        geometry.add_face((bottom, ring_indices[0][nxt], ring_indices[0][segment]), material)
        for ring in range(len(ring_indices) - 1):
            current = ring_indices[ring]
            following = ring_indices[ring + 1]
            material = material_selector(ring + 1, segment) if material_selector else material_index
            geometry.add_face((current[segment], current[nxt], following[nxt], following[segment]), material)
        material = material_selector(rings, segment) if material_selector else material_index
        geometry.add_face((ring_indices[-1][segment], ring_indices[-1][nxt], top), material)


def append_tube(
    geometry: GeometryData,
    points: Sequence[Sequence[float]],
    radii: float | Sequence[float],
    *,
    sides: int = 8,
    material_index: int = 0,
    material_selector: MaterialSelector | None = None,
    cap: bool = True,
) -> None:
    """Append a tapered tube along a 3-D polyline."""

    if len(points) < 2:
        return
    sides = max(5, int(sides))
    if isinstance(radii, (int, float)):
        radius_values = [float(radii)] * len(points)
    else:
        radius_values = [float(v) for v in radii]
        if len(radius_values) != len(points):
            raise ValueError("radii must have one value per tube point")

    rings: list[list[int]] = []
    for point_index, point in enumerate(points):
        previous = points[max(0, point_index - 1)]
        following = points[min(len(points) - 1, point_index + 1)]
        tangent = _normalize(_sub(following, previous))
        reference = (0.0, 0.0, 1.0) if abs(tangent[2]) < 0.86 else (0.0, 1.0, 0.0)
        side_a = _normalize(_cross(tangent, reference), (1.0, 0.0, 0.0))
        side_b = _normalize(_cross(tangent, side_a), (0.0, 1.0, 0.0))
        ring: list[int] = []
        for side in range(sides):
            angle = 2.0 * pi * side / sides
            offset = _add(_mul(side_a, cos(angle) * radius_values[point_index]), _mul(side_b, sin(angle) * radius_values[point_index]))
            ring.append(geometry.add_vertex(_add(point, offset)))
        rings.append(ring)

    for point_index in range(len(rings) - 1):
        for side in range(sides):
            nxt = (side + 1) % sides
            material = material_selector(point_index, side) if material_selector else material_index
            geometry.add_face(
                (rings[point_index][side], rings[point_index][nxt], rings[point_index + 1][nxt], rings[point_index + 1][side]),
                material,
            )
    if cap:
        start_center = geometry.add_vertex(points[0])
        end_center = geometry.add_vertex(points[-1])
        for side in range(sides):
            nxt = (side + 1) % sides
            geometry.add_face((start_center, rings[0][nxt], rings[0][side]), material_index)
            geometry.add_face((end_center, rings[-1][side], rings[-1][nxt]), material_index)


def append_leaf_blade(
    geometry: GeometryData,
    points: Sequence[Sequence[float]],
    widths: float | Sequence[float],
    *,
    thickness: float = 0.008,
    material_index: int = 0,
) -> None:
    """Append a thin curved, flat solid blade for seagrass.

    A blade is represented by a ribbon with a small physical thickness instead
    of a round tube.  Its two-sided faces remain visible from either side of a
    camera, while the side walls keep geometry labels and shadows well-defined.
    """

    if len(points) < 2:
        return
    if isinstance(widths, (int, float)):
        width_values = [float(widths)] * len(points)
    else:
        width_values = [float(value) for value in widths]
        if len(width_values) != len(points):
            raise ValueError("widths must have one value per blade point")
    front_left: list[int] = []
    front_right: list[int] = []
    back_left: list[int] = []
    back_right: list[int] = []
    for point_index, point in enumerate(points):
        previous = points[max(0, point_index - 1)]
        following = points[min(len(points) - 1, point_index + 1)]
        tangent = _normalize(_sub(following, previous))
        side = _normalize(_cross((0.0, 0.0, 1.0), tangent), (1.0, 0.0, 0.0))
        normal = _normalize(_cross(tangent, side), (0.0, 1.0, 0.0))
        half_side = _mul(side, width_values[point_index] * 0.5)
        half_thickness = _mul(normal, float(thickness) * 0.5)
        front_left.append(geometry.add_vertex(_add(_sub(point, half_thickness), _mul(half_side, -1.0))))
        front_right.append(geometry.add_vertex(_add(_sub(point, half_thickness), half_side)))
        back_left.append(geometry.add_vertex(_add(_add(point, half_thickness), _mul(half_side, -1.0))))
        back_right.append(geometry.add_vertex(_add(_add(point, half_thickness), half_side)))
    for point_index in range(len(points) - 1):
        following = point_index + 1
        # The front and back faces wind toward their exterior normals. Keeping
        # this orientation consistent matters for thin leaves under Cycles.
        geometry.add_face((front_left[point_index], front_right[point_index], front_right[following], front_left[following]), material_index)
        geometry.add_face((back_left[point_index], back_left[following], back_right[following], back_right[point_index]), material_index)
        geometry.add_face((front_left[point_index], front_left[following], back_left[following], back_left[point_index]), material_index)
        geometry.add_face((front_right[point_index], back_right[point_index], back_right[following], front_right[following]), material_index)
    geometry.add_face((front_left[0], back_left[0], back_right[0], front_right[0]), material_index)
    end = len(points) - 1
    geometry.add_face((front_left[end], front_right[end], back_right[end], back_left[end]), material_index)


def append_plate(
    geometry: GeometryData,
    center: Sequence[float],
    radius_x: float,
    radius_y: float,
    thickness: float,
    *,
    segments: int = 20,
    material_index: int = 0,
    rim_material_index: int | None = None,
    phase: float = 0.0,
) -> None:
    """Append a thick, subtly undulating elliptical table-coral plate."""

    segments = max(8, int(segments))
    rim_material_index = material_index if rim_material_index is None else int(rim_material_index)
    top_center = geometry.add_vertex(_add(center, (0.0, 0.0, thickness * 0.5)))
    bottom_center = geometry.add_vertex(_add(center, (0.0, 0.0, -thickness * 0.5)))
    top_ring: list[int] = []
    bottom_ring: list[int] = []
    for segment in range(segments):
        angle = phase + 2.0 * pi * segment / segments
        edge_noise = 1.0 + 0.055 * sin(3.0 * angle + phase) + 0.03 * cos(7.0 * angle - phase)
        x = radius_x * edge_noise * cos(angle)
        y = radius_y * edge_noise * sin(angle)
        top_z = thickness * 0.5 + 0.018 * sin(4.0 * angle + phase)
        top_ring.append(geometry.add_vertex(_add(center, (x, y, top_z))))
        bottom_ring.append(geometry.add_vertex(_add(center, (x, y, -thickness * 0.5))))
    for segment in range(segments):
        nxt = (segment + 1) % segments
        geometry.add_face((top_center, top_ring[segment], top_ring[nxt]), material_index)
        geometry.add_face((bottom_center, bottom_ring[nxt], bottom_ring[segment]), material_index)
        geometry.add_face((top_ring[segment], bottom_ring[segment], bottom_ring[nxt], top_ring[nxt]), rim_material_index)


def append_thick_triangle(
    geometry: GeometryData,
    points: Sequence[Sequence[float]],
    thickness: float,
    *,
    material_index: int = 0,
) -> None:
    """Append a closed triangular prism for fins, leaves, and reef plates."""

    if len(points) != 3:
        raise ValueError("a thick triangle requires three points")
    edge_a = _sub(points[1], points[0])
    edge_b = _sub(points[2], points[0])
    normal = _normalize(_cross(edge_a, edge_b), (0.0, 0.0, 1.0))
    half = _mul(normal, float(thickness) * 0.5)
    front = [geometry.add_vertex(_add(point, half)) for point in points]
    back = [geometry.add_vertex(_sub(point, half)) for point in points]
    geometry.add_face((front[0], front[1], front[2]), material_index)
    geometry.add_face((back[2], back[1], back[0]), material_index)
    geometry.add_face((front[0], back[0], back[1], front[1]), material_index)
    geometry.add_face((front[1], back[1], back[2], front[2]), material_index)
    geometry.add_face((front[2], back[2], back[0], front[0]), material_index)


def append_rock_lump(
    geometry: GeometryData,
    center: Sequence[float],
    scale: Sequence[float],
    *,
    seed_phase: float = 0.0,
    segments: int = 14,
    rings: int = 7,
    material_index: int = 0,
    dark_material_index: int | None = None,
) -> None:
    """Append an irregular, low-poly boulder with faceted natural variation."""

    segments = max(8, int(segments))
    rings = max(4, int(rings))
    bottom = geometry.add_vertex(_add(center, (0.0, 0.0, -float(scale[2]) * 0.72)))
    ring_indices: list[list[int]] = []
    for ring in range(1, rings):
        latitude = -pi / 2.0 + pi * ring / rings
        c_lat = cos(latitude)
        s_lat = sin(latitude)
        current: list[int] = []
        for segment in range(segments):
            longitude = 2.0 * pi * segment / segments
            variation = 1.0 + 0.12 * sin(3.0 * longitude + seed_phase + ring * 0.8) + 0.06 * cos(5.0 * longitude - seed_phase)
            vertical = 0.94 + 0.08 * sin(seed_phase + ring * 1.6)
            current.append(
                geometry.add_vertex(
                    _add(
                        center,
                        (
                            float(scale[0]) * c_lat * cos(longitude) * variation,
                            float(scale[1]) * c_lat * sin(longitude) * variation,
                            float(scale[2]) * s_lat * vertical,
                        ),
                    )
                )
            )
        ring_indices.append(current)
    top = geometry.add_vertex(_add(center, (0.0, 0.0, float(scale[2]) * 0.98)))
    for segment in range(segments):
        nxt = (segment + 1) % segments
        underside = dark_material_index is not None and segment % 5 in (0, 1)
        material = int(dark_material_index if underside else material_index)
        geometry.add_face((bottom, ring_indices[0][nxt], ring_indices[0][segment]), material)
        for ring in range(len(ring_indices) - 1):
            current = ring_indices[ring]
            following = ring_indices[ring + 1]
            shade = dark_material_index is not None and ring == 0 and segment % 4 == 0
            geometry.add_face(
                (current[segment], current[nxt], following[nxt], following[segment]),
                int(dark_material_index if shade else material_index),
            )
        geometry.add_face((ring_indices[-1][segment], ring_indices[-1][nxt], top), material_index)


def _lattice_hash(ix: int, iy: int, phase: float) -> float:
    """Return a deterministic pseudo-random lattice value in ``[0, 1]``.

    Python's process hash is intentionally unsuitable for a dataset seed.  A
    small integer hash keeps terrain continuous between samples while making
    the result independent of Python's hash randomisation and of Blender's
    process state.
    """

    phase_i = int(round(float(phase) * 1000.0))
    value = (
        int(ix) * 374761393
        + int(iy) * 668265263
        + phase_i * 1442695041
        + 1013904223
    ) & 0xFFFFFFFF
    value ^= value >> 13
    value = (value * 1274126177) & 0xFFFFFFFF
    value ^= value >> 16
    return float(value & 0xFFFFFFFF) / 4294967295.0


def _smooth_noise(x: float, y: float, cell_size: float, phase: float = 0.0) -> float:
    """Evaluate smooth, non-periodic 2-D value noise.

    ``cell_size`` is in metres.  The lattice is deliberately much larger than
    the visible sandbed, so expanding the mesh does not repeat the same tile at
    the camera-facing boundary.
    """

    cell = max(float(cell_size), 1.0e-6)
    gx = float(x) / cell
    gy = float(y) / cell
    ix = floor(gx)
    iy = floor(gy)
    fx = gx - ix
    fy = gy - iy
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    v00 = _lattice_hash(ix, iy, phase)
    v10 = _lattice_hash(ix + 1, iy, phase)
    v01 = _lattice_hash(ix, iy + 1, phase)
    v11 = _lattice_hash(ix + 1, iy + 1, phase)
    lower = v00 + (v10 - v00) * sx
    upper = v01 + (v11 - v01) * sx
    return float(lower + (upper - lower) * sy)


def _sand_path_center(y: float, phase: float) -> float:
    """Return a slowly wandering x coordinate for the light sand path."""

    broad = _smooth_noise(3.7, y, 7.5, phase + 4.1) - 0.5
    local = _smooth_noise(-8.4, y + 3.2, 3.1, phase + 8.7) - 0.5
    return float(0.55 * broad + 0.26 * local)


def sandbed_height(x: float, y: float, *, floor_z: float = 0.0, seed_phase: float = 0.0) -> float:
    """Return a seeded, non-periodic low/mid-frequency benthic height.

    The mesh carries only measurable relief.  Sub-millimetre grain remains a
    material normal detail, so the depth label does not accidentally include a
    shader bump.  Domain warping and independent lattice phases make the
    ridges change direction over the outer rings instead of repeating a fixed
    sine tile.
    """

    x_f = float(x)
    y_f = float(y)
    phase = float(seed_phase)

    broad = _smooth_noise(x_f, y_f, 9.0, phase + 0.3) - 0.5
    basin = _smooth_noise(x_f + 16.7, y_f - 11.3, 4.6, phase + 2.7) - 0.5
    low = 0.132 * broad + 0.064 * basin

    # Low-frequency domain warping prevents the mid-frequency field from
    # appearing as parallel rows when the camera looks across the bed.
    warp_x = 1.55 * (_smooth_noise(x_f - 7.2, y_f + 4.3, 5.2, phase + 5.2) - 0.5)
    warp_y = 1.25 * (_smooth_noise(x_f + 5.6, y_f - 8.1, 4.3, phase + 7.8) - 0.5)
    middle = _smooth_noise(x_f + warp_x, y_f + warp_y, 1.55, phase + 10.4) - 0.5
    middle += 0.55 * (_smooth_noise(y_f - 2.1, x_f + 3.8, 0.82, phase + 12.6) - 0.5)
    mid_relief = 0.039 * middle

    # A separate, very shallow field suggests broken sand ripples.  Its
    # direction is locally varied by the broad field rather than fixed to the
    # global X/Y axes, and its amplitude is small enough for a stable depth GT.
    direction = 0.95 * (broad + 0.5) + 0.35 * (basin + 0.5)
    along_ridge = x_f * cos(direction) + y_f * sin(direction)
    ripple_warp = 0.24 * (_smooth_noise(x_f, y_f, 1.9, phase + 15.0) - 0.5)
    ripple = _smooth_noise(along_ridge + ripple_warp, y_f - 0.35 * x_f, 0.42, phase + 16.8) - 0.5
    micro_relief = 0.014 * ripple
    return float(floor_z + low + mid_relief + micro_relief)


def append_sandbed(
    geometry: GeometryData,
    *,
    floor_z: float = 0.0,
    seed_phase: float = 0.0,
    x_min: float = -16.0,
    x_max: float = 16.0,
    y_min: float = -10.0,
    y_max: float = 24.0,
    nx: int = 65,
    ny: int = 69,
    material_index: int = 0,
    path_material_index: int = 1,
) -> None:
    """Append a real undulating sand surface with a curved central sand path."""

    nx = max(8, int(nx))
    ny = max(8, int(ny))

    # Keep the original 0.5 m inner grid intact, then add sparse low-poly
    # outer rings so the camera never sees a finite blue boundary.
    inner_x = [x_min + (x_max - x_min) * column / (nx - 1) for column in range(nx)]
    inner_y = [y_min + (y_max - y_min) * row / (ny - 1) for row in range(ny)]
    x_coords = [-40.0, -32.0, -24.0] + inner_x + [24.0, 32.0, 40.0]
    y_coords = [-50.0, -36.0, -24.0] + inner_y + [32.0, 42.0, 50.0]
    for y in y_coords:
        for x in x_coords:
            geometry.add_vertex((x, y, sandbed_height(x, y, floor_z=floor_z, seed_phase=seed_phase)))

    grid_nx = len(x_coords)
    for row in range(len(y_coords) - 1):
        for column in range(grid_nx - 1):
            a = row * grid_nx + column
            b = a + 1
            c = a + grid_nx + 1
            d = a + grid_nx
            cx = (x_coords[column] + x_coords[column + 1]) * 0.5
            cy = (y_coords[row] + y_coords[row + 1]) * 0.5
            path_x = _sand_path_center(cy, seed_phase)
            in_inner = x_min <= cx <= x_max and y_min <= cy <= y_max
            material = path_material_index if in_inner and abs(cx - path_x) < 0.95 else material_index
            geometry.add_face((a, b, c, d), material)


def make_mesh_object(
    collection: bpy.types.Collection,
    name: str,
    geometry: GeometryData,
    materials: Sequence[bpy.types.Material],
    *,
    semantic_id: int,
    instance_id: int,
    morphology: str,
) -> bpy.types.Object:
    """Create one static mesh object and attach the stable asset contract."""

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(geometry.vertices, [], geometry.faces)
    mesh.update(calc_edges=True)
    for material in materials:
        mesh.materials.append(material)
    for polygon, material_index in zip(mesh.polygons, geometry.material_indices):
        polygon.material_index = max(0, min(int(material_index), len(materials) - 1))
        # Organic silhouettes need smooth normals for the closer camera. Keep
        # occasional hard facets on rock clusters so their crevices retain
        # readable edges while all coral/fish/grass/sand meshes stay smooth.
        polygon.use_smooth = not morphology.startswith("rock") or polygon.index % 4 != 0
    mesh["asset_source"] = "original_programmatic"
    mesh["static_evaluated"] = True
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj["semantic_id"] = int(semantic_id)
    obj["instance_id"] = int(instance_id)
    obj["asset_morphology"] = str(morphology)
    obj["asset_source"] = "original_programmatic"
    obj["generated_seeded"] = True
    return obj


def mesh_triangle_count(objects: Sequence[bpy.types.Object]) -> int:
    """Return evaluated static triangle count without touching modifiers."""

    total = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        obj.data.calc_loop_triangles()
        total += len(obj.data.loop_triangles)
    return int(total)


__all__ = [
    "GeometryData",
    "append_plate",
    "append_rock_lump",
    "append_sandbed",
    "sandbed_height",
    "append_thick_triangle",
    "append_tube",
    "append_leaf_blade",
    "append_uv_sphere",
    "make_mesh_object",
    "mesh_triangle_count",
    "transform_geometry",
]
