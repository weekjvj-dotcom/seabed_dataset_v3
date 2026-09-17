"""Procedural ecosystem assets for the seabed dataset builder."""


def build_ecosystem(scene, seed, floor_z=0.0):
    """Lazily expose the Blender-backed ecosystem builder."""

    from .ecology import build_ecosystem as _build_ecosystem

    return _build_ecosystem(scene, seed, floor_z)


def prepare_layout(scene, ecosystem):
    """Lazily expose the v2 local-pivot preparation API."""

    from ..randomization import prepare_layout as _prepare_layout

    return _prepare_layout(scene, ecosystem)


def apply_geometry_state(scene, ecosystem, camera, cfg, state_index, attempt_index=0):
    """Lazily expose the v2 per-state camera/object sampler."""

    from ..randomization import apply_geometry_state as _apply_geometry_state

    return _apply_geometry_state(scene, ecosystem, camera, cfg, state_index, attempt_index)

__all__ = ["build_ecosystem", "prepare_layout", "apply_geometry_state"]
