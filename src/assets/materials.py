"""Native PBR materials used by the v3 reef assets.

Coral/fish/grass retain editable procedural variation, while the sandbed and
rocks add audited CC0 photo-scanned PBR maps through Blender image nodes.  The
maps are never used as image planes or label sources: only real mesh geometry
participates in depth, semantic and instance exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import bpy  # type: ignore


Color = tuple[float, float, float]
ROOT = Path(__file__).resolve().parents[2]


PBR_SETS = {
    "coral_ground_02": {
        "asset_uid": "polyhaven:coral-ground-02:2k",
        "source_url": "https://polyhaven.com/a/coral_ground_02",
        "license": "CC0",
        "albedo": "assets/source/polyhaven/coral_ground_02/coral_ground_02_diff_2k.jpg",
        "roughness": "assets/source/polyhaven/coral_ground_02/coral_ground_02_rough_2k.jpg",
        "normal": "assets/source/polyhaven/coral_ground_02/coral_ground_02_nor_gl_2k.jpg",
        "height": "assets/source/polyhaven/coral_ground_02/coral_ground_02_disp_2k.jpg",
    },
    "rock_3": {
        "asset_uid": "polyhaven:rock-3:2k",
        "source_url": "https://polyhaven.com/a/rock_3",
        "license": "CC0",
        "albedo": "assets/source/polyhaven/rock_3/rock_3_diff_2k.jpg",
        "roughness": "assets/source/polyhaven/rock_3/rock_3_rough_2k.jpg",
        "normal": "assets/source/polyhaven/rock_3/rock_3_nor_gl_2k.jpg",
        "height": "assets/source/polyhaven/rock_3/rock_3_disp_2k.jpg",
    },
}


def _set_input(node: Any, name: str, value: Any) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def _image(asset: dict[str, str], channel: str, colorspace: str) -> bpy.types.Image:
    """Load an audited local map and retain its project-relative provenance."""

    relative_path = str(asset[channel])
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"v3 PBR source image is missing: {path}")
    image = bpy.data.images.load(str(path.resolve()), check_existing=True)
    image.colorspace_settings.name = colorspace
    image["asset_uid"] = str(asset["asset_uid"])
    image["source_relative_path"] = relative_path
    image["source_url"] = str(asset["source_url"])
    image["license"] = str(asset["license"])
    image["pbr_channel"] = channel
    return image


def _image_node(
    nodes: Any,
    links: Any,
    vector: Any,
    *,
    name: str,
    image: bpy.types.Image,
    location: tuple[float, float],
) -> Any:
    node = nodes.new("ShaderNodeTexImage")
    node.name = name
    node.label = name
    node.image = image
    node.projection = "BOX"
    node.projection_blend = 0.32
    node.location = location
    links.new(vector, node.inputs["Vector"])
    return node


def _photoscanned_pbr(
    name: str,
    asset_key: str,
    *,
    mapping_scale: float,
    tint: Color,
    bump_strength: float,
    bump_distance: float,
) -> bpy.types.Material:
    """Build a box-projected PBR material without relying on generated UVs."""

    asset = PBR_SETS[asset_key]
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = (*tint, 1.0)
    material["asset_source"] = "cc0_imported"
    material["asset_uid"] = str(asset["asset_uid"])
    material["source_url"] = str(asset["source_url"])
    material["license"] = str(asset["license"])
    material["pbr_model"] = "Principled BSDF with CC0 photo-scanned albedo/roughness/normal/height maps"
    material["uses_emission"] = False
    material["texture_coordinate_space"] = "OBJECT_LOCAL_METRES_BOX_PROJECTED"
    material["texture_scale_is_world_stable"] = False
    material["texture_scale_scope"] = "object local metre coordinates; state metadata records all object transforms"
    material["true_displacement"] = False
    material["height_map_role"] = "Bump only; no shader displacement, so the texture alone does not alter depth GT"

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.name = f"{name}_MaterialOutput"
    output.location = (760.0, 40.0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.name = f"{name}_PrincipledPBR"
    bsdf.location = (500.0, 40.0)
    _set_input(bsdf, "Metallic", 0.0)
    _set_input(bsdf, "IOR", 1.42)
    _set_input(bsdf, "Specular IOR Level", 0.28)

    coordinates = nodes.new("ShaderNodeTexCoord")
    coordinates.name = f"{name}_ObjectMeterCoordinates"
    coordinates.location = (-900.0, 40.0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.name = f"{name}_ObjectMeterMapping"
    mapping.location = (-740.0, 40.0)
    _set_input(mapping, "Scale", (float(mapping_scale), float(mapping_scale), float(mapping_scale)))
    links.new(coordinates.outputs["Object"], mapping.inputs["Vector"])

    albedo = _image(asset, "albedo", "sRGB")
    roughness = _image(asset, "roughness", "Non-Color")
    normal = _image(asset, "normal", "Non-Color")
    height = _image(asset, "height", "Non-Color")
    albedo_node = _image_node(nodes, links, mapping.outputs["Vector"], name=f"{name}_Albedo_CC0", image=albedo, location=(-520.0, 210.0))
    roughness_node = _image_node(nodes, links, mapping.outputs["Vector"], name=f"{name}_Roughness_CC0", image=roughness, location=(-520.0, 40.0))
    normal_node = _image_node(nodes, links, mapping.outputs["Vector"], name=f"{name}_Normal_CC0", image=normal, location=(-520.0, -130.0))
    height_node = _image_node(nodes, links, mapping.outputs["Vector"], name=f"{name}_Height_CC0", image=height, location=(-520.0, -300.0))

    tint_node = nodes.new("ShaderNodeRGB")
    tint_node.name = f"{name}_PhysicalTint"
    tint_node.location = (-250.0, 300.0)
    tint_node.outputs["Color"].default_value = (*tint, 1.0)
    albedo_mix = nodes.new("ShaderNodeMixRGB")
    albedo_mix.name = f"{name}_AlbedoTintMix"
    albedo_mix.blend_type = "MULTIPLY"
    _set_input(albedo_mix, "Fac", 0.34)
    albedo_mix.location = (-40.0, 210.0)
    links.new(albedo_node.outputs["Color"], albedo_mix.inputs["Color1"])
    links.new(tint_node.outputs["Color"], albedo_mix.inputs["Color2"])

    # The normal map remains in the material graph for provenance and future
    # UV-ready assets. The height map drives Bump now because the procedural
    # meshes have no tangent-space UV unwrap; object-space box projection
    # avoids seams without misinterpreting a tangent-space normal map.
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.name = f"{name}_NormalMap_CC0"
    normal_map.space = "TANGENT"
    normal_map.location = (-40.0, -130.0)
    links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
    normal_map["preview_usage"] = "provenance_only_until_a_uv_ready_asset_uses_tangent_normals"
    bump = nodes.new("ShaderNodeBump")
    bump.name = f"{name}_HeightBump_CC0"
    bump.location = (210.0, -120.0)
    _set_input(bump, "Strength", float(bump_strength))
    _set_input(bump, "Distance", float(bump_distance))
    links.new(height_node.outputs["Color"], bump.inputs["Height"])

    links.new(albedo_mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(roughness_node.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def _procedural_pbr(
    name: str,
    dark_color: Color,
    light_color: Color,
    *,
    roughness: float,
    noise_scale: float,
    noise_detail: float = 3.0,
    bump_strength: float = 0.2,
    metallic: float = 0.0,
    bump_distance: float = 0.02,
    secondary_noise_scale: float | None = None,
    voronoi_scale: float | None = None,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = (*light_color, 1.0)
    material["asset_source"] = "original_programmatic"
    material["pbr_model"] = "Principled BSDF with multi-frequency procedural Noise/Voronoi/ColorRamp/Bump"
    material["uses_emission"] = False
    # Object coordinates are expressed in local Blender units (metres for the
    # unscaled base assets). Sampled object scale intentionally changes the
    # apparent pattern size, so do not claim this graph is world-scale
    # invariant. The seabed itself is left at unit object scale, which keeps
    # its texture frequency stable when only the outer mesh ring is extended.
    material["texture_coordinate_space"] = "OBJECT_LOCAL_METRES"
    material["texture_scale_is_world_stable"] = False
    material["texture_scale_scope"] = "local_object_units; sampled object scale is part of appearance"

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.name = f"{name}_MaterialOutput"
    output.location = (520.0, 40.0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.name = f"{name}_PrincipledPBR"
    bsdf.location = (240.0, 40.0)
    _set_input(bsdf, "Roughness", float(roughness))
    _set_input(bsdf, "Metallic", float(metallic))
    _set_input(bsdf, "IOR", 1.42)
    _set_input(bsdf, "Specular IOR Level", 0.32)

    # Use object-space coordinates measured in metres.  An unconnected vector
    # socket falls back to Generated coordinates, whose normalisation changes
    # when the seabed bounds are enlarged.
    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.name = f"{name}_ObjectMeterCoordinates"
    texcoord.location = (-780.0, 100.0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.name = f"{name}_ObjectMeterMapping"
    mapping.location = (-650.0, 100.0)
    _set_input(mapping, "Location", (0.0, 0.0, 0.0))
    _set_input(mapping, "Rotation", (0.0, 0.0, 0.0))
    _set_input(mapping, "Scale", (1.0, 1.0, 1.0))
    object_coordinates = texcoord.outputs.get("Object")
    if object_coordinates is not None:
        links.new(object_coordinates, mapping.inputs.get("Vector"))

    texture = nodes.new("ShaderNodeTexNoise")
    texture.name = f"{name}_ProceduralNoise"
    texture.location = (-520.0, 100.0)
    _set_input(texture, "Scale", float(noise_scale))
    _set_input(texture, "Detail", float(noise_detail))
    _set_input(texture, "Roughness", 0.68)
    _set_input(texture, "Distortion", 0.12)
    if object_coordinates is not None:
        links.new(mapping.outputs.get("Vector"), texture.inputs.get("Vector"))

    pattern_output = texture.outputs.get("Fac")
    if secondary_noise_scale is not None:
        secondary = nodes.new("ShaderNodeTexNoise")
        secondary.name = f"{name}_FineNoise"
        secondary.location = (-520.0, -120.0)
        _set_input(secondary, "Scale", float(secondary_noise_scale))
        _set_input(secondary, "Detail", 5.0)
        _set_input(secondary, "Roughness", 0.72)
        if object_coordinates is not None:
            links.new(mapping.outputs.get("Vector"), secondary.inputs.get("Vector"))
        mix = nodes.new("ShaderNodeMixRGB")
        mix.name = f"{name}_MultiFrequencyMix"
        mix.location = (-350.0, 0.0)
        mix.blend_type = "MULTIPLY"
        _set_input(mix, "Fac", 0.52)
        links.new(texture.outputs.get("Fac"), mix.inputs.get("Color1"))
        links.new(secondary.outputs.get("Fac"), mix.inputs.get("Color2"))
        pattern_output = mix.outputs.get("Color")

    if voronoi_scale is not None:
        voronoi = nodes.new("ShaderNodeTexVoronoi")
        voronoi.name = f"{name}_MicroPoreVoronoi"
        voronoi.location = (-510.0, -300.0)
        _set_input(voronoi, "Scale", float(voronoi_scale))
        _set_input(voronoi, "Randomness", 0.38)
        if object_coordinates is not None:
            links.new(mapping.outputs.get("Vector"), voronoi.inputs.get("Vector"))
        mix_voronoi = nodes.new("ShaderNodeMixRGB")
        mix_voronoi.name = f"{name}_PoreMix"
        mix_voronoi.location = (-150.0, -30.0)
        mix_voronoi.blend_type = "MULTIPLY"
        _set_input(mix_voronoi, "Fac", 0.28)
        links.new(pattern_output, mix_voronoi.inputs.get("Color1"))
        links.new(voronoi.outputs.get("Distance"), mix_voronoi.inputs.get("Color2"))
        pattern_output = mix_voronoi.outputs.get("Color")

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = f"{name}_NaturalColorRamp"
    ramp.location = (-180.0, 130.0)
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (*dark_color, 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (*light_color, 1.0)

    bump = nodes.new("ShaderNodeBump")
    bump.name = f"{name}_MicroBump"
    bump.location = (-140.0, -120.0)
    _set_input(bump, "Strength", float(bump_strength))
    _set_input(bump, "Distance", float(bump_distance))

    links.new(pattern_output, ramp.inputs.get("Fac"))
    links.new(ramp.outputs.get("Color"), bsdf.inputs.get("Base Color"))
    links.new(pattern_output, bump.inputs.get("Height"))
    links.new(bump.outputs.get("Normal"), bsdf.inputs.get("Normal"))
    links.new(bsdf.outputs.get("BSDF"), output.inputs.get("Surface"))
    return material


def build_material_library() -> dict[str, bpy.types.Material]:
    """Create or reuse the complete material palette for the ecosystem."""

    # tuple = dark, light, roughness, primary frequency, bump strength,
    # bump distance, secondary frequency, optional Voronoi pore frequency.
    palette = {
        # Frequencies are cycles per metre in Object space.  The geometry
        # carries broad/mid relief; these nodes add stable sub-metre grain and
        # shallow pores without changing depth labels.
        "sand": ((0.27, 0.17, 0.080), (0.63, 0.45, 0.25), 0.96, 2.8, 0.10, 0.006, 8.5, None),
        "sand_path": ((0.34, 0.22, 0.105), (0.72, 0.54, 0.31), 0.97, 2.4, 0.08, 0.005, 7.2, None),
        "rock": ((0.055, 0.070, 0.064), (0.24, 0.28, 0.23), 0.88, 15.0, 0.25, 0.025, 53.0, 8.0),
        "rock_crevice": ((0.012, 0.015, 0.014), (0.055, 0.045, 0.035), 0.98, 34.0, 0.14, 0.018, 91.0, 13.0),
        "coral_teal": ((0.060, 0.19, 0.18), (0.28, 0.50, 0.42), 0.77, 44.0, 0.18, 0.014, 11.0, 44.0),
        "coral_red": ((0.22, 0.060, 0.040), (0.64, 0.27, 0.18), 0.75, 48.0, 0.16, 0.013, 13.0, 48.0),
        "coral_purple": ((0.16, 0.10, 0.18), (0.46, 0.34, 0.50), 0.79, 38.0, 0.17, 0.014, 10.0, 38.0),
        "coral_gold": ((0.25, 0.14, 0.050), (0.64, 0.45, 0.22), 0.76, 42.0, 0.16, 0.013, 12.0, 42.0),
        "coral_groove": ((0.075, 0.060, 0.045), (0.22, 0.18, 0.12), 0.92, 52.0, 0.10, 0.010, 17.0, 52.0),
        "coral_polyp": ((0.30, 0.15, 0.10), (0.76, 0.57, 0.38), 0.72, 50.0, 0.12, 0.010, 18.0, 50.0),
        "coral_tip": ((0.32, 0.28, 0.19), (0.82, 0.75, 0.56), 0.70, 46.0, 0.10, 0.008, 16.0, 46.0),
        "coral_table_ochre": ((0.22, 0.13, 0.060), (0.60, 0.43, 0.23), 0.78, 40.0, 0.14, 0.012, 14.0, 40.0),
        "coral_table_edge": ((0.48, 0.35, 0.20), (0.86, 0.73, 0.51), 0.68, 52.0, 0.12, 0.009, 18.0, 52.0),
        "grass": ((0.018, 0.095, 0.040), (0.12, 0.37, 0.16), 0.84, 26.0, 0.17, 0.012, 72.0, None),
        "grass_tip": ((0.040, 0.16, 0.045), (0.28, 0.56, 0.15), 0.77, 32.0, 0.14, 0.010, 80.0, None),
        "fish_blue": ((0.020, 0.11, 0.22), (0.12, 0.39, 0.62), 0.52, 18.0, 0.10, 0.010, 46.0, None),
        "fish_orange": ((0.27, 0.045, 0.018), (0.78, 0.25, 0.075), 0.50, 20.0, 0.09, 0.010, 50.0, None),
        "fish_yellow": ((0.33, 0.22, 0.015), (0.82, 0.62, 0.080), 0.48, 20.0, 0.09, 0.010, 54.0, None),
        "fish_teal": ((0.015, 0.15, 0.16), (0.10, 0.48, 0.43), 0.51, 22.0, 0.10, 0.010, 52.0, None),
        "fish_stripe": ((0.012, 0.018, 0.022), (0.10, 0.11, 0.11), 0.56, 30.0, 0.07, 0.008, 70.0, None),
        "fish_fin": ((0.12, 0.025, 0.060), (0.56, 0.12, 0.14), 0.53, 22.0, 0.08, 0.009, 52.0, None),
        "fish_fin_gold": ((0.30, 0.11, 0.008), (0.78, 0.40, 0.025), 0.50, 24.0, 0.08, 0.009, 52.0, None),
        "fish_eye": ((0.001, 0.001, 0.001), (0.012, 0.016, 0.017), 0.22, 40.0, 0.04, 0.004, 90.0, None),
        "fish_mouth": ((0.10, 0.004, 0.003), (0.38, 0.015, 0.008), 0.56, 32.0, 0.05, 0.006, 80.0, None),
    }
    library: dict[str, bpy.types.Material] = {}
    for key, (dark, light, roughness, scale, bump, bump_distance, secondary_scale, voronoi_scale) in palette.items():
        library[key] = _procedural_pbr(
            f"ECO_{key}",
            dark,
            light,
            roughness=roughness,
            noise_scale=scale,
            bump_strength=bump,
            bump_distance=bump_distance,
            secondary_noise_scale=secondary_scale,
            voronoi_scale=voronoi_scale,
        )
    # The v3 preview deliberately limits external images to two 2K CC0 PBR
    # sets.  This makes the initial visual comparison meaningful without
    # turning the M2/8 GB machine into an unbounded texture-memory test.
    library["sand"] = _photoscanned_pbr(
        "ECO_sand",
        "coral_ground_02",
        mapping_scale=0.62,
        tint=(0.74, 0.63, 0.48),
        bump_strength=0.24,
        bump_distance=0.028,
    )
    library["sand_path"] = _photoscanned_pbr(
        "ECO_sand_path",
        "coral_ground_02",
        mapping_scale=0.74,
        tint=(0.94, 0.80, 0.58),
        bump_strength=0.18,
        bump_distance=0.020,
    )
    library["rock"] = _photoscanned_pbr(
        "ECO_rock",
        "rock_3",
        mapping_scale=0.90,
        tint=(0.58, 0.67, 0.61),
        bump_strength=0.32,
        bump_distance=0.030,
    )
    return library


__all__ = ["build_material_library"]
