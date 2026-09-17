"""Native Cycles water construction helpers.

The public API is intentionally small so that scene orchestration can keep the
water state behind an explicit handle.  Blender is imported lazily by the
implementation, which also lets the validation tests run with the system
Python interpreter where ``bpy`` is not installed.
"""

from .native import (
    build_water,
    set_water_enabled,
    set_water_preset,
    water_metadata,
)

__all__ = [
    "build_water",
    "set_water_enabled",
    "set_water_preset",
    "water_metadata",
]
