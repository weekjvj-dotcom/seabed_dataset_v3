"""Array-only quality measurements and bounded quality decisions.

The renderer is deliberately absent from this module.  Every public function
accepts arrays already produced by the renderer and either returns JSON-safe
measurements or a bounded decision.  In particular, this module never edits a
colour image and never synthesises a degraded image.

The geometry convention is the one used by the v2 label exporter:

``valid = (semantic > 0) & (instance > 0) & (depth_range > 0)``

Depths are in metres.  The named ranges are ``near`` [0.8, 2.5), ``middle``
[2.5, 6), ``far`` [6, 15), and ``background`` [15, infinity).  Positive
depths below 0.8 m are intentionally kept in ``below_near`` and are never
silently counted as near pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


SEMANTIC_NAMES: dict[int, str] = {
    0: "background",
    1: "sand",
    2: "rock",
    3: "coral",
    4: "grass",
    5: "fish",
}

_BAND_NAMES = ("near", "middle", "far", "background")
_LIGHTING_MODES = {"natural", "artificial", "mixed"}
_SRGB_BREAK = 0.0031308
_DARK_DISPLAY_THRESHOLD = 0.05
_CLIP_DISPLAY_THRESHOLD = 0.995
_MORPHOLOGY_FIELDS = ("morphology", "asset_morphology", "asset_morphologies")


def _array(value: Any, name: str) -> np.ndarray:
    """Convert an input to ndarray while retaining an explanatory error."""

    try:
        result = np.asarray(value)
    except Exception as exc:  # numpy can raise several conversion errors
        raise ValueError(f"{name} must be a numpy array with a numeric dtype") from exc
    if result.dtype.kind not in "biufc":
        raise ValueError(f"{name} must have a numeric dtype, got {result.dtype}")
    return result


def _finite(array: np.ndarray, name: str) -> None:
    """Reject NaN/Inf everywhere, including pixels outside the valid mask."""

    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real finite values, not complex values")
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if not bool(np.all(finite)):
        raise ValueError(f"{name} contains nonfinite values (NaN or Inf)")


def _shape_2d(value: Any, name: str) -> np.ndarray:
    array = _array(value, name)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape HxW; got shape {tuple(array.shape)}")
    _finite(array, name)
    return array


def _shape_rgb(value: Any, name: str) -> np.ndarray:
    array = _array(value, name)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"{name} must have shape HxWx3 (RGB); got shape {tuple(array.shape)}"
        )
    _finite(array, name)
    return array.astype(np.float64, copy=False)


def _shape_same(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{a_name} and {b_name} shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}"
        )


def _integer_labels(value: Any, name: str, *, minimum: int, maximum: int) -> np.ndarray:
    """Validate integer-like labels and return int64 values.

    PNG readers normally provide unsigned integer arrays, but accepting an
    integral float array makes the checker convenient for small numerical
    fixtures while still rejecting fractional IDs and bool labels.
    """

    array = _shape_2d(value, name)
    if array.dtype.kind == "b" or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain integer IDs, not {array.dtype}")
    if array.dtype.kind == "f" and not bool(np.all(array == np.floor(array))):
        raise ValueError(f"{name} contains non-integer label values")
    converted = array.astype(np.int64, copy=False)
    if bool(np.any(converted < minimum)) or bool(np.any(converted > maximum)):
        bad = int(converted[(converted < minimum) | (converted > maximum)].flat[0])
        raise ValueError(
            f"{name} contains illegal ID {bad}; expected integers in {minimum}..{maximum}"
        )
    return converted


def _mask(value: Any, expected_shape: tuple[int, int], name: str = "valid_mask") -> np.ndarray:
    array = _array(value, name)
    if array.ndim != 2 or array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape HxW matching RGB ({expected_shape}); "
            f"got shape {tuple(array.shape)}"
        )
    _finite(array, name)
    # The exporter writes 0/255 and callers may use bool.  Treating any
    # positive finite value as valid also accommodates an in-memory 0/1 mask,
    # without changing the defined mask>0 semantics.
    return array > 0


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _registry_morphologies(item: Mapping[str, Any], label: str) -> list[str]:
    found = False
    values: Any = None
    for field in _MORPHOLOGY_FIELDS:
        if field in item:
            found = True
            values = item[field]
            break
    if not found:
        raise ValueError(
            f"registry[{label!r}] must include morphology or asset_morphology"
        )
    if isinstance(values, str):
        values_list = [values]
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray, str)):
        values_list = list(values)
    else:
        raise ValueError(
            f"registry[{label!r}] morphology must be a non-empty string or list of strings"
        )
    if not values_list or any(not isinstance(value, str) or not value.strip() for value in values_list):
        raise ValueError(
            f"registry[{label!r}] morphology must be a non-empty string or list of strings"
        )
    return sorted({str(value) for value in values_list})


def _registry(registry: Any) -> dict[str, dict[str, Any]]:
    """Validate and normalise the string-indexed instance registry."""

    source = _mapping(registry, "registry")
    result: dict[str, dict[str, Any]] = {}
    for raw_key, raw_item in source.items():
        if not isinstance(raw_key, str):
            raise ValueError(
                f"registry instance key {raw_key!r} must be str(instance_id)"
            )
        try:
            instance_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"registry instance key {raw_key!r} is not an integer ID") from exc
        if str(instance_id) != raw_key or instance_id <= 0 or instance_id > 65535:
            raise ValueError(
                f"registry instance key {raw_key!r} must be canonical str ID in 1..65535"
            )
        item = _mapping(raw_item, f"registry[{raw_key!r}]")
        if "semantic_id" not in item:
            raise ValueError(f"registry[{raw_key!r}] is missing semantic_id")
        semantic_id = _scalar_integer(
            item["semantic_id"],
            f"registry[{raw_key!r}].semantic_id",
            minimum=1,
            maximum=5,
        )
        morphologies = _registry_morphologies(item, raw_key)
        result[raw_key] = {
            "semantic_id": semantic_id,
            "morphologies": morphologies,
            "morphology": morphologies[0] if len(morphologies) == 1 else morphologies,
        }
    return result


def _scalar_integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    integer = int(number)
    if integer < minimum or integer > maximum:
        raise ValueError(f"{name}={integer} is outside {minimum}..{maximum}")
    return integer


def _quality(cfg: Any) -> Mapping[str, Any]:
    if not isinstance(cfg, Mapping) or not isinstance(cfg.get("quality"), Mapping):
        raise ValueError("cfg must contain a quality mapping at cfg['quality']")
    return cfg["quality"]


def _threshold(cfg: Any, key: str) -> float:
    quality = _quality(cfg)
    if key not in quality:
        raise ValueError(f"cfg['quality'] is missing required threshold {key!r}")
    value = quality[key]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"cfg['quality'][{key!r}] must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"cfg['quality'][{key!r}] must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"cfg['quality'][{key!r}] must be a finite number")
    if number < 0:
        raise ValueError(f"cfg['quality'][{key!r}] must be non-negative")
    if key.endswith("fraction_min") or key == "valid_fraction_min":
        if number > 1:
            raise ValueError(f"cfg['quality'][{key!r}] must be in 0..1")
    if key == "clipped_fraction_max" and number > 1:
        raise ValueError(f"cfg['quality'][{key!r}] must be in 0..1")
    return number


def _json_safe(value: Any) -> Any:
    """Convert numpy values to JSON-safe values and reject non-finite numbers."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("quality measurement produced a nonfinite number")
        return number
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"quality measurement contains unsupported value {type(value).__name__}")


def _srgb_oetf(linear: np.ndarray) -> np.ndarray:
    """Apply extended sRGB OETF without clipping HDR values.

    For negative finite input the low branch is retained (12.92*x).  Rendered
    linear RGB is normally non-negative; retaining the value here avoids a
    hidden clamp and still makes the input's invalid brightness visible to the
    metrics.  Values above one are deliberately encoded above one.
    """

    array = np.asarray(linear, dtype=np.float64)
    result = np.empty_like(array)
    low = array <= _SRGB_BREAK
    result[low] = 12.92 * array[low]
    high = ~low
    if np.any(high):
        result[high] = 1.055 * np.power(array[high], 1.0 / 2.4) - 0.055
    return result


def _percentiles(values: np.ndarray) -> tuple[float, float, float]:
    p5, p95 = np.percentile(values, [5.0, 95.0])
    return float(p5), float(p95), float(p95 - p5)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def _rgb_summary(rgb: np.ndarray, roi: np.ndarray) -> dict[str, Any]:
    if not bool(np.any(roi)):
        raise ValueError("RGB quality ROI is empty")
    pixels = rgb[roi]
    luma_linear = _luminance(pixels)
    display_rgb = _srgb_oetf(pixels)
    # Display brightness is the Rec.709 weighting of display-encoded channel
    # values. Applying OETF to already-weighted linear luminance is only
    # equivalent for grey pixels and would bias coloured ROIs.
    display_luma = _luminance(display_rgb)
    display_clip = np.any(display_rgb >= _CLIP_DISPLAY_THRESHOLD, axis=1)
    raw_p5, raw_p95, raw_range = _percentiles(luma_linear)
    display_p5, display_p95, display_range = _percentiles(display_luma)
    return {
        "pixel_count": int(pixels.shape[0]),
        "mean_linear": float(np.mean(luma_linear)),
        "mean_rgb_linear": [float(value) for value in np.mean(pixels, axis=0)],
        "linearP5": raw_p5,
        "linearP95": raw_p95,
        "linear_dynamic_range": raw_range,
        "displayP5": display_p5,
        "displayP95": display_p95,
        "dynamic_range": display_range,
        "display_p5": display_p5,
        "display_p95": display_p95,
        "display_dynamic_range": display_range,
        "clipped_fraction": float(np.mean(display_clip)),
        "fractiondark": float(np.mean(display_luma < _DARK_DISPLAY_THRESHOLD)),
        "fraction_dark": float(np.mean(display_luma < _DARK_DISPLAY_THRESHOLD)),
    }


def _visible_instance_info(
    ids: np.ndarray,
    labels: np.ndarray,
    registry: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[str], list[str], int]:
    visible_ids = sorted({str(int(value)) for value in ids.flat}, key=int)
    counts = {instance_id: int(np.sum(ids == int(instance_id))) for instance_id in visible_ids}
    visible_instances: list[dict[str, Any]] = []
    coral_morphologies: set[str] = set()
    fish_morphologies: set[str] = set()
    visible_fish_count = 0
    for instance_id in visible_ids:
        entry = registry[instance_id]
        semantic_id = int(entry["semantic_id"])
        morphologies = list(entry["morphologies"])
        item = {
            "instance_id": instance_id,
            "semantic_id": semantic_id,
            "morphology": entry["morphology"],
            "morphologies": morphologies,
            "pixel_count": counts[instance_id],
        }
        visible_instances.append(item)
        if semantic_id == 3:
            coral_morphologies.update(morphologies)
        if semantic_id == 5:
            fish_morphologies.update(morphologies)
            visible_fish_count += 1
    return (
        visible_ids,
        visible_instances,
        sorted(coral_morphologies),
        sorted(fish_morphologies),
        visible_fish_count,
    )


def _band_name(depth: np.ndarray) -> np.ndarray:
    names = np.full(depth.shape, "below_near", dtype=object)
    names[(depth >= 0.8) & (depth < 2.5)] = "near"
    names[(depth >= 2.5) & (depth < 6.0)] = "middle"
    names[(depth >= 6.0) & (depth < 15.0)] = "far"
    names[depth >= 15.0] = "background"
    return names


def geometry_metrics(
    depth_range: Any,
    semantic: Any,
    instance: Any,
    registry: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Measure depth bands, labels, visible instances, and morphologies.

    ``cfg`` is accepted as part of the strict public API.  Geometry raw
    metrics do not apply thresholds; thresholds are read by
    :func:`geometry_decision`.
    """

    del cfg  # thresholds belong to the decision function, not raw measuring
    depth = _shape_2d(depth_range, "depth_range")
    sem = _integer_labels(semantic, "semantic", minimum=0, maximum=5)
    inst = _integer_labels(instance, "instance", minimum=0, maximum=65535)
    _shape_same(depth, sem, "depth_range", "semantic")
    _shape_same(depth, inst, "depth_range", "instance")
    registry_map = _registry(registry)

    positive_instances = sorted({str(int(value)) for value in inst[inst > 0].flat}, key=int)
    missing = [instance_id for instance_id in positive_instances if instance_id not in registry_map]
    if missing:
        raise ValueError(
            "instance contains IDs absent from registry: " + ", ".join(missing)
        )

    foreground_without_instance = (sem > 0) & (inst == 0)
    instance_without_foreground = (sem == 0) & (inst > 0)
    if np.any(foreground_without_instance):
        raise ValueError("semantic has foreground pixels with instance_id=0")
    if np.any(instance_without_foreground):
        raise ValueError("instance has positive IDs where semantic_id=0")

    for instance_id in positive_instances:
        pixels = inst == int(instance_id)
        expected_semantic = int(registry_map[instance_id]["semantic_id"])
        observed = np.unique(sem[pixels])
        if observed.size != 1 or int(observed[0]) != expected_semantic:
            observed_values = [int(value) for value in observed]
            raise ValueError(
                f"instance {instance_id} semantic mismatch: registry={expected_semantic}, "
                f"array={observed_values}"
            )

    valid = (sem > 0) & (inst > 0) & (depth > 0)
    total_pixels = int(depth.size)
    valid_count = int(np.sum(valid))
    if valid_count == 0:
        raise ValueError(
            "geometry quality valid ROI is empty: require semantic>0, instance>0, depth>0"
        )

    valid_depth = depth[valid].astype(np.float64, copy=False)
    valid_sem = sem[valid]
    valid_inst = inst[valid]
    names = _band_name(valid_depth)
    all_visible_ids, all_visible_instances, coral_morphologies, fish_morphologies, fish_count = _visible_instance_info(
        valid_inst,
        valid_sem,
        registry_map,
    )
    ecology_visible_ids, ecology_visible_instances, _, _, _ = _visible_instance_info(
        valid_inst[valid_sem >= 2],
        valid_sem[valid_sem >= 2],
        registry_map,
    )

    semantic_counts = {
        str(semantic_id): int(np.sum(valid_sem == semantic_id))
        for semantic_id in sorted(set(int(value) for value in valid_sem.flat))
    }
    instance_counts = {
        instance_id: int(np.sum(valid_inst == int(instance_id)))
        for instance_id in all_visible_ids
    }

    bands: dict[str, dict[str, Any]] = {}
    for band in _BAND_NAMES:
        band_mask = names == band
        band_count = int(np.sum(band_mask))
        band_sem = valid_sem[band_mask]
        band_inst = valid_inst[band_mask]
        ecology_mask = band_sem >= 2
        ecology_count = int(np.sum(ecology_mask))
        ids, visible, corals, fish, band_fish_count = _visible_instance_info(
            band_inst[ecology_mask | (band_sem == 1)],
            band_sem[ecology_mask | (band_sem == 1)],
            registry_map,
        ) if band_count else ([], [], [], [], 0)
        # `_visible_instance_info` above sees all valid labels in the band;
        # the `ecology_mask | sand` expression is intentionally the full band
        # mask and keeps sand instances available for area accounting.
        sand_count = int(np.sum(band_sem == 1))
        band_info = {
            "pixel_count": band_count,
            "count": band_count,
            "pixels": band_count,
            "fraction": float(band_count / total_pixels),
            "fraction_of_valid": float(band_count / valid_count),
            "ecology_pixel_count": ecology_count,
            "ecology_pixels": ecology_count,
            "ecology_fraction": float(ecology_count / total_pixels),
            "ecology_fraction_of_band": float(ecology_count / band_count) if band_count else 0.0,
            "sand_pixel_count": sand_count,
            "sand_fraction": float(sand_count / band_count) if band_count else 0.0,
            "visible_instance_ids": ids,
            "visible_instances": visible,
            "visible_ecology_instance_ids": [
                item["instance_id"] for item in visible if int(item["semantic_id"]) >= 2
            ],
            "visible_ecology_instances": [
                item for item in visible if int(item["semantic_id"]) >= 2
            ],
            "visible_fish_count": band_fish_count,
            "coral_morphologies": corals,
            "fish_morphologies": fish,
        }
        bands[band] = band_info

    below_near_count = int(np.sum(names == "below_near"))
    out_of_range = {
        "pixel_count": below_near_count,
        "fraction": float(below_near_count / total_pixels),
        "fraction_of_valid": float(below_near_count / valid_count),
    }
    background = bands["background"]
    background_sand_count = int(np.sum((names == "background") & (valid_sem == 1)))
    background_band_count = int(background["pixel_count"])
    result = {
        "shape": [int(depth.shape[0]), int(depth.shape[1])],
        "pixel_count": total_pixels,
        "valid_pixel_count": valid_count,
        "invalid_pixel_count": total_pixels - valid_count,
        "valid_fraction": float(valid_count / total_pixels),
        "valid_area_fraction": float(valid_count / total_pixels),
        "bands": bands,
        "range_bands": bands,
        "below_near": out_of_range,
        "out_of_range": out_of_range,
        "semantic_pixel_counts": semantic_counts,
        "semantic_counts": semantic_counts,
        "instance_pixel_counts": instance_counts,
        "band_pixel_counts": {
            band: int(bands[band]["pixel_count"]) for band in _BAND_NAMES
        },
        "band_fractions": {
            band: float(bands[band]["fraction"]) for band in _BAND_NAMES
        },
        "ecology_pixel_counts": {
            band: int(bands[band]["ecology_pixel_count"]) for band in _BAND_NAMES
        },
        "visible_instance_ids": all_visible_ids,
        "visible_instances": all_visible_instances,
        "visible_ecology_instance_ids": ecology_visible_ids,
        "visible_ecology_instances": ecology_visible_instances,
        "visible_ecology_instance_count": len(ecology_visible_ids),
        "visible_fish_count": fish_count,
        "coral_morphologies": coral_morphologies,
        "fish_morphologies": fish_morphologies,
        "sand_pixel_count": int(np.sum(valid_sem == 1)),
        "sand_fraction": float(np.mean(valid_sem == 1)),
        "background_band_pixel_count": background_band_count,
        "background_sand_pixel_count": background_sand_count,
        "background_sand_fraction": (
            float(background_sand_count / background_band_count)
            if background_band_count
            else 0.0
        ),
        "background_sand_fraction_of_valid": float(background_sand_count / valid_count),
    }
    return _json_safe(result)


def _metric(metrics: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in metrics:
            value = metrics[name]
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"quality metric {name!r} must be a finite number")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"quality metric {name!r} must be a finite number") from exc
            if not math.isfinite(number):
                raise ValueError(f"quality metric {name!r} must be a finite number")
            return number
    raise ValueError(f"quality metrics missing required field; expected one of {names!r}")


def _decision(metrics: Mapping[str, Any], reasons: list[str]) -> dict[str, Any]:
    # Keep a JSON-safe copy in the decision for log files.  The raw metric
    # object remains untouched, which is useful to callers retaining it.
    return _json_safe({
        "passed": not reasons,
        "reasons": list(reasons),
        "metrics": dict(metrics),
    })


def _band_metrics(metrics: Mapping[str, Any], band: str) -> Mapping[str, Any]:
    bands = metrics.get("bands", metrics.get("range_bands"))
    if not isinstance(bands, Mapping) or band not in bands or not isinstance(bands[band], Mapping):
        raise ValueError(f"geometry metrics missing band {band!r}")
    return bands[band]


def geometry_decision(metrics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply only configured valid-area, band-area, and ecology gates."""

    if not isinstance(metrics, Mapping):
        raise ValueError("geometry metrics must be a mapping")
    valid_min = _threshold(cfg, "valid_fraction_min")
    fraction_thresholds = {
        "near": _threshold(cfg, "near_fraction_min"),
        "middle": _threshold(cfg, "middle_fraction_min"),
        "far": _threshold(cfg, "far_fraction_min"),
    }
    ecology_min = _threshold(cfg, "band_ecology_pixels_min")
    reasons: list[str] = []
    valid_fraction = _metric(metrics, "valid_fraction")
    if valid_fraction < valid_min:
        reasons.append("valid_fraction_below_min")
    for band in ("near", "middle", "far"):
        band_metrics = _band_metrics(metrics, band)
        # The approved gates describe the composition of the valid target
        # area. Prefer that denominator whenever raw metrics provide it;
        # ``fraction`` remains a compatibility fallback for hand-built metric
        # dictionaries from older callers.
        fraction = _metric(band_metrics, "fraction_of_valid", "fraction")
        ecology_count = _metric(
            band_metrics,
            "ecology_pixel_count",
            "ecology_pixels",
        )
        if fraction < fraction_thresholds[band]:
            reasons.append(f"{band}_fraction_below_min")
        if ecology_count < ecology_min:
            reasons.append(f"{band}_ecology_pixels_below_min")
    return _decision(metrics, reasons)


def rgb_metrics(
    rgb_linear: Any,
    valid_mask: Any,
    semantic: Any = None,
) -> dict[str, Any]:
    """Measure linear luminance and fixed sRGB display statistics.

    Raw statistics retain HDR values.  sRGB OETF values are calculated only in
    local temporary arrays for display statistics; no input array is modified.
    """

    rgb = _shape_rgb(rgb_linear, "rgb_linear")
    roi = _mask(valid_mask, rgb.shape[:2])
    sem: np.ndarray | None = None
    if semantic is not None:
        sem = _integer_labels(semantic, "semantic", minimum=0, maximum=5)
        _shape_same(sem, roi, "semantic", "valid_mask")
    summary = _rgb_summary(rgb, roi)
    total_pixels = int(roi.size)
    valid_count = int(np.sum(roi))
    display_luma = _srgb_oetf(_luminance(rgb[roi]))
    result: dict[str, Any] = {
        "shape": [int(rgb.shape[0]), int(rgb.shape[1]), 3],
        "pixel_count": total_pixels,
        "valid_pixel_count": valid_count,
        "valid_fraction": float(valid_count / total_pixels),
        **summary,
        "display_luma_min": float(np.min(display_luma)),
        "display_luma_max": float(np.max(display_luma)),
        "semantic_stats": {},
    }

    # Report a border summary as a diagnostic.  rgb_decision never changes a
    # threshold for Artificial lighting based on this value.
    border = max(1, min(rgb.shape[0], rgb.shape[1]) // 20)
    border_roi = roi.copy()
    interior = np.zeros_like(border_roi, dtype=bool)
    if rgb.shape[0] > 2 * border and rgb.shape[1] > 2 * border:
        interior[border:-border, border:-border] = True
        border_roi &= ~interior
    if bool(np.any(border_roi)):
        border_summary = _rgb_summary(rgb, border_roi)
        result["edge"] = border_summary
        result["edge_mean_linear"] = border_summary["mean_linear"]
        result["edge_fractiondark"] = border_summary["fractiondark"]
    else:
        result["edge"] = {"pixel_count": 0}

    if sem is not None:
        for semantic_id in sorted(int(value) for value in np.unique(sem[roi])):
            semantic_roi = roi & (sem == semantic_id)
            result["semantic_stats"][str(semantic_id)] = _rgb_summary(rgb, semantic_roi)
    result["per_semantic"] = result["semantic_stats"]
    return _json_safe(result)


def rgb_decision(
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    lighting_mode: str,
) -> dict[str, Any]:
    """Apply fixed global RGB brightness, clipping, and range thresholds."""

    if not isinstance(metrics, Mapping):
        raise ValueError("RGB metrics must be a mapping")
    if not isinstance(lighting_mode, str) or not lighting_mode.strip():
        raise ValueError("lighting_mode must be a non-empty string")
    mode = lighting_mode.strip().lower()
    if mode not in _LIGHTING_MODES:
        raise ValueError(
            f"lighting_mode must be one of {sorted(_LIGHTING_MODES)!r}; got {lighting_mode!r}"
        )
    # Natural/Artificial/Mixed all use the same frozen gates below.
    mean_min = _threshold(cfg, "mean_linear_min")
    clip_max = _threshold(cfg, "clipped_fraction_max")
    dynamic_min = _threshold(cfg, "display_dynamic_range_min")
    reasons: list[str] = []
    mean_linear = _metric(metrics, "mean_linear")
    clipped_fraction = _metric(metrics, "clipped_fraction")
    dynamic_range = _metric(metrics, "dynamic_range", "display_dynamic_range")
    if mean_linear < mean_min:
        reasons.append("mean_linear_below_min")
    if clipped_fraction > clip_max:
        reasons.append("clipped_fraction_above_max")
    if dynamic_range < dynamic_min:
        reasons.append("display_dynamic_range_below_min")
    decision_metrics = dict(metrics)
    decision_metrics["lighting_mode"] = mode
    if mode == "artificial" and isinstance(metrics.get("edge"), Mapping):
        # Diagnostic only.  It is included in the output, while no gate is
        # added and no threshold is lowered for artificial-light dark edges.
        decision_metrics["artificial_dark_edge_reported"] = True
    return _decision(decision_metrics, reasons)


def noise_metrics(
    clear: Any,
    wet_seed_a: Any,
    wet_seed_b: Any,
    valid_mask: Any,
) -> dict[str, Any]:
    """Estimate render noise and water effect from two independent seeds.

    All primary RMS values are calculated over linear RGB channels.  Structure
    contrast and its paired noise estimate use Rec.709 linear luminance so the
    ratio has matching units.  A zero-noise denominator produces ``None`` and
    an explicit ``noise_zero`` flag instead of an infinite JSON value.
    """

    clear_rgb = _shape_rgb(clear, "clear")
    seed_a = _shape_rgb(wet_seed_a, "wet_seed_a")
    seed_b = _shape_rgb(wet_seed_b, "wet_seed_b")
    _shape_same(clear_rgb, seed_a, "clear", "wet_seed_a")
    _shape_same(clear_rgb, seed_b, "clear", "wet_seed_b")
    roi = _mask(valid_mask, clear_rgb.shape[:2])
    if not bool(np.any(roi)):
        raise ValueError("noise quality ROI is empty")

    clear_pixels = clear_rgb[roi]
    a_pixels = seed_a[roi]
    b_pixels = seed_b[roi]
    seed_delta = a_pixels - b_pixels
    seed_difference_rms = float(np.sqrt(np.mean(np.square(seed_delta))))
    noise_estimate = float(seed_difference_rms / math.sqrt(2.0))
    wet_mean = (a_pixels + b_pixels) / 2.0
    effect_delta = wet_mean - clear_pixels
    effect_rms = float(np.sqrt(np.mean(np.square(effect_delta))))

    clear_luma = _luminance(clear_pixels)
    mean_luma = _luminance(wet_mean)
    seed_luma_delta = _luminance(seed_delta)
    luma_noise_estimate = float(np.sqrt(np.mean(np.square(seed_luma_delta))) / math.sqrt(2.0))
    structure_p5, structure_p95, structure_contrast = _percentiles(mean_luma)
    noise_zero = bool(noise_estimate == 0.0)
    luma_noise_zero = bool(luma_noise_estimate == 0.0)
    effect_noise_ratio = None if noise_zero else float(effect_rms / noise_estimate)
    structure_noise_ratio = None if luma_noise_zero else float(structure_contrast / luma_noise_estimate)

    result = {
        "pixel_count": int(clear_pixels.shape[0]),
        "seed_difference_rms": seed_difference_rms,
        "seed_noise_rms": seed_difference_rms,
        "noise_estimate": noise_estimate,
        "noise_rms": noise_estimate,
        "noise_estimate_rms": noise_estimate,
        "effect_rms": effect_rms,
        "effectRMS": effect_rms,
        "water_clear_effect_rms": effect_rms,
        "effect_noise_ratio": effect_noise_ratio,
        "effect_over_noise": effect_noise_ratio,
        "effect_to_noise_ratio": effect_noise_ratio,
        "noise_zero": noise_zero,
        "all_finite": True,
        "wet_mean_linear_luminance": float(np.mean(mean_luma)),
        "wet_mean_luminance_p5": structure_p5,
        "wet_mean_luminance_p95": structure_p95,
        "wet_mean_luminance_dynamic_range": structure_contrast,
        "brightness_p95_minus_p5": structure_contrast,
        "structure_contrast": structure_contrast,
        "structureContrast": structure_contrast,
        "structure_noise": luma_noise_estimate,
        "structureNoise": luma_noise_estimate,
        "structure_noise_ratio": structure_noise_ratio,
        "structure_over_noise": structure_noise_ratio,
        "structure_noise_zero": luma_noise_zero,
        "rms_units": "linear_rgb",
        "structure_units": "linear_rec709_luminance",
        "clear_mean_linear_luminance": float(np.mean(clear_luma)),
        "wet_seed_a_mean_linear_luminance": float(np.mean(_luminance(a_pixels))),
        "wet_seed_b_mean_linear_luminance": float(np.mean(_luminance(b_pixels))),
    }
    return _json_safe(result)


__all__ = [
    "geometry_metrics",
    "rgb_metrics",
    "geometry_decision",
    "rgb_decision",
    "noise_metrics",
]
