"""Small, dependency-free readers and writers for label image formats.

The label exporter uses grayscale PNG files for integer labels and uncompressed
scanline OpenEXR files for float depth.  The helpers in this module deliberately
do not import :mod:`bpy` or :mod:`numpy`, so that an export can be checked from a
normal Python interpreter as well as from Blender's Python.

``write_png(path, pixels, width, height, bit_depth)`` writes one grayscale
channel (PNG colour type 0), with an integer sample depth of 8 or 16 bits.
``read_png(path)`` returns a dictionary containing ``width``, ``height``,
``bit_depth`` and top-to-bottom ``pixels`` rows.

``write_exr(path, pixels, width, height, channel="Y")`` writes one FLOAT
channel with no compression.  ``read_exr(path, channel=None)`` reads that
format and returns the same row convention.  The EXR reader can account for
other uncompressed channel payloads while selecting a FLOAT channel; it does
not decode a requested HALF or UINT channel.

Rows are always represented top-to-bottom in this module.  Pixel values are
never passed through a colour transform.
"""

from __future__ import annotations

import math
import numbers
import struct
import zlib
from pathlib import Path
from typing import Any, Sequence


class FormatError(ValueError):
    """Raised when an image is malformed or uses an unsupported encoding."""


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# OpenEXR's on-disk magic bytes are 76 2f 31 01.  The integer is therefore
# 0x01312f76 when encoded little-endian with struct.pack("<I", ...).
_EXR_MAGIC = 0x01312F76
_EXR_VERSION = 2
_EXR_TILED_FLAG = 0x0200
_EXR_LONG_NAMES_FLAG = 0x0400
_EXR_DEEP_FLAG = 0x0800
_EXR_MULTIPART_FLAG = 0x1000
_EXR_KNOWN_FLAGS = _EXR_TILED_FLAG | _EXR_LONG_NAMES_FLAG | _EXR_DEEP_FLAG | _EXR_MULTIPART_FLAG
_EXR_FLOAT = 2
_EXR_UINT = 0
_EXR_HALF = 1


def _path(path: str | Path) -> Path:
    try:
        return Path(path)
    except TypeError as exc:  # make the public error useful
        raise TypeError("image path must be a str or pathlib.Path") from exc


def _check_dimensions(width: int, height: int) -> tuple[int, int]:
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("image dimensions must be positive integers")
    if not isinstance(width, numbers.Integral) or not isinstance(height, numbers.Integral):
        raise ValueError("image dimensions must be positive integers")
    width_i, height_i = int(width), int(height)
    if width_i <= 0 or height_i <= 0:
        raise ValueError("image dimensions must be positive integers")
    if width_i > 0x7FFFFFFF or height_i > 0x7FFFFFFF:
        raise ValueError("image dimensions exceed the supported 32-bit range")
    return width_i, height_i


def _is_scalar(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (str, bytes, bytearray))


def _png_bit_depth(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("PNG bit_depth must be the integer 8 or 16")
    bit_depth = int(value)
    if bit_depth not in (8, 16):
        raise ValueError("PNG bit_depth must be the integer 8 or 16")
    return bit_depth


def _to_rows(array_or_pixels: Any, width: int, height: int) -> list[list[Any]]:
    """Convert a nested or flat sequence (and an optional ndarray) to rows."""

    # ``tolist`` is intentionally duck-typed.  It keeps numpy optional while
    # accepting numpy arrays inside Blender, where numpy is usually available.
    values = array_or_pixels.tolist() if hasattr(array_or_pixels, "tolist") else array_or_pixels
    if isinstance(values, (bytes, bytearray, str)):
        raise ValueError("pixels must be a nested or flat numeric sequence")
    try:
        outer = list(values)
    except TypeError as exc:
        raise ValueError("pixels must be a nested or flat numeric sequence") from exc

    if len(outer) == height and all(not _is_scalar(row) for row in outer):
        rows: list[list[Any]] = []
        for row in outer:
            if isinstance(row, (bytes, bytearray, str)):
                raise ValueError("each pixel row must be numeric")
            try:
                row_values = list(row)
            except TypeError as exc:
                raise ValueError("each pixel row must be numeric") from exc
            if len(row_values) != width:
                raise ValueError(
                    f"pixel row has length {len(row_values)}, expected {width}"
                )
            rows.append(row_values)
        return rows

    if len(outer) == width * height and all(_is_scalar(value) for value in outer):
        return [outer[row * width : (row + 1) * width] for row in range(height)]

    raise ValueError(
        f"pixels shape does not match width={width}, height={height}; "
        "use a height-by-width nested sequence or a flat sequence"
    )


def _integer_samples(array_or_pixels: Any, width: int, height: int, bit_depth: int) -> list[list[int]]:
    bit_depth = _png_bit_depth(bit_depth)
    rows = _to_rows(array_or_pixels, width, height)
    maximum = (1 << bit_depth) - 1
    output: list[list[int]] = []
    for row in rows:
        converted: list[int] = []
        for value in row:
            if isinstance(value, bool):
                integer = int(value)
            elif isinstance(value, numbers.Integral):
                integer = int(value)
            elif isinstance(value, numbers.Real):
                value_f = float(value)
                if not math.isfinite(value_f) or not value_f.is_integer():
                    raise ValueError("PNG samples must be finite integers")
                integer = int(value_f)
            else:
                raise ValueError("PNG samples must be finite integers")
            if integer < 0 or integer > maximum:
                raise ValueError(f"PNG sample {integer} is outside 0..{maximum}")
            converted.append(integer)
        output.append(converted)
    return output


def _float_samples(array_or_pixels: Any, width: int, height: int) -> list[list[float]]:
    rows = _to_rows(array_or_pixels, width, height)
    output: list[list[float]] = []
    for row in rows:
        converted: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError("EXR samples must be real numbers")
            value_f = float(value)
            if not math.isfinite(value_f):
                raise ValueError("EXR samples must be finite float32 values")
            try:
                packed = struct.pack("<f", value_f)
                value_f32 = struct.unpack("<f", packed)[0]
            except (OverflowError, struct.error) as exc:
                raise ValueError("EXR sample cannot be represented as float32") from exc
            if not math.isfinite(value_f32):
                raise ValueError("EXR sample cannot be represented as finite float32")
            converted.append(value_f32)
        output.append(converted)
    return output


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    if len(kind) != 4:
        raise ValueError("PNG chunk type must have four bytes")
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png(
    path: str | Path,
    array_or_pixels: Any,
    width: int,
    height: int,
    bit_depth: int = 16,
) -> Path:
    """Write a single-channel integer PNG and return ``path``.

    The encoder writes filter type 0 for every row and therefore has no
    dependency on an image library.  Values must already be integer samples;
    silently rounding floating labels would make semantic/instance mistakes
    difficult to detect.
    """

    width_i, height_i = _check_dimensions(width, height)
    bit_depth_i = _png_bit_depth(bit_depth)
    samples = _integer_samples(array_or_pixels, width_i, height_i, bit_depth_i)
    raw_rows = bytearray()
    for row in samples:
        raw_rows.append(0)  # PNG filter: None
        if bit_depth_i == 8:
            raw_rows.extend(row)
        else:
            for value in row:
                raw_rows.extend(struct.pack(">H", value))
    ihdr = struct.pack(">IIBBBBB", width_i, height_i, bit_depth_i, 0, 0, 0, 0)
    encoded = _PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr)
    encoded += _png_chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=6))
    encoded += _png_chunk(b"IEND", b"")
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return output


def _paeth(a: int, b: int, c: int) -> int:
    prediction = a + b - c
    pa, pb, pc = abs(prediction - a), abs(prediction - b), abs(prediction - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_png(raw: bytes, width: int, height: int, bytes_per_pixel: int) -> list[bytes]:
    row_bytes = width * bytes_per_pixel
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        raise FormatError(
            f"PNG decompressed IDAT length is {len(raw)}, expected {expected}"
        )
    rows: list[bytes] = []
    previous = bytes(row_bytes)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        encoded = raw[offset : offset + row_bytes]
        offset += row_bytes
        decoded = bytearray(row_bytes)
        for index, value in enumerate(encoded):
            left = decoded[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            up = previous[index]
            upper_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            elif filter_type == 4:
                predictor = _paeth(left, up, upper_left)
            else:
                raise FormatError(f"PNG uses unsupported filter type {filter_type}")
            decoded[index] = (value + predictor) & 0xFF
        row = bytes(decoded)
        rows.append(row)
        previous = row
    return rows


def read_png(path: str | Path) -> dict[str, Any]:
    """Read a grayscale 8/16-bit PNG into a serializable dictionary."""

    data = _path(path).read_bytes()
    if not data.startswith(_PNG_SIGNATURE):
        raise FormatError("file is not a PNG")
    offset = len(_PNG_SIGNATURE)
    width = height = bit_depth = color_type = None
    compression = filter_method = interlace = None
    idat = bytearray()
    saw_iend = False
    while offset + 8 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        kind = data[offset : offset + 4]
        offset += 4
        end = offset + length
        if end + 4 > len(data):
            raise FormatError("truncated PNG chunk")
        payload = data[offset:end]
        expected_crc = struct.unpack_from(">I", data, end)[0]
        actual_crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise FormatError(f"PNG CRC mismatch in {kind!r} chunk")
        offset = end + 4
        if kind == b"IHDR":
            if width is not None or len(payload) != 13:
                raise FormatError("invalid PNG IHDR")
            width, height, bit_depth, color_type, compression, filter_method, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            if payload:
                raise FormatError("PNG IEND must be empty")
            saw_iend = True
            break
    if width is None or not saw_iend:
        raise FormatError("PNG is missing IHDR or IEND")
    width_i, height_i = _check_dimensions(width, height)
    if bit_depth not in (8, 16) or color_type != 0:
        raise FormatError("only grayscale PNG files with 8 or 16 bits are supported")
    if compression != 0 or filter_method != 0 or interlace != 0:
        raise FormatError("PNG uses unsupported compression, filter, or interlace mode")
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise FormatError("PNG IDAT stream is not valid zlib data") from exc
    bytes_per_sample = bit_depth // 8
    rows_raw = _unfilter_png(raw, width_i, height_i, bytes_per_sample)
    if bit_depth == 8:
        pixels = [list(row) for row in rows_raw]
    else:
        pixels = [
            [struct.unpack_from(">H", row, index * 2)[0] for index in range(width_i)]
            for row in rows_raw
        ]
    return {
        "width": width_i,
        "height": height_i,
        "bit_depth": int(bit_depth),
        "color_type": int(color_type),
        "channels": 1,
        "pixels": pixels,
        # These aliases make independent validation scripts convenient while
        # keeping ``pixels`` as the documented key.
        "array": pixels,
        "data": pixels,
    }


def _exr_attribute(name: str, type_name: str, value: bytes) -> bytes:
    name_bytes = name.encode("ascii") + b"\0"
    type_bytes = type_name.encode("ascii") + b"\0"
    return name_bytes + type_bytes + struct.pack("<I", len(value)) + value


def _exr_channel_list(channels: Sequence[tuple[str, int]]) -> bytes:
    encoded = bytearray()
    for name, pixel_type in channels:
        if not name or "\0" in name:
            raise ValueError("EXR channel names must be non-empty ASCII strings")
        if pixel_type not in (_EXR_HALF, _EXR_FLOAT, _EXR_UINT):
            raise ValueError("unsupported EXR channel pixel type")
        encoded.extend(name.encode("ascii"))
        encoded.append(0)
        encoded.extend(struct.pack("<i", pixel_type))
        encoded.extend(b"\0\0\0\0")  # pLinear and three reserved bytes
        encoded.extend(struct.pack("<ii", 1, 1))
    encoded.append(0)
    return bytes(encoded)


def write_exr(
    path: str | Path,
    array_or_pixels: Any,
    width: int,
    height: int,
    channel: str = "Y",
) -> Path:
    """Write a little-endian, uncompressed, single-channel FLOAT EXR.

    This is the conservative OpenEXR scanline form: all mandatory header
    attributes are present, rows are stored in increasing y order, and each
    depth sample is a 32-bit IEEE-754 float in metres.
    """

    width_i, height_i = _check_dimensions(width, height)
    if not isinstance(channel, str) or not channel or "\0" in channel:
        raise ValueError("EXR channel must be a non-empty string")
    try:
        channel.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("EXR channel must contain ASCII characters") from exc
    samples = _float_samples(array_or_pixels, width_i, height_i)
    header = bytearray()
    header.extend(_exr_attribute("channels", "chlist", _exr_channel_list([(channel, _EXR_FLOAT)])))
    header.extend(_exr_attribute("compression", "compression", b"\0"))
    data_window = struct.pack("<4i", 0, 0, width_i - 1, height_i - 1)
    header.extend(_exr_attribute("dataWindow", "box2i", data_window))
    header.extend(_exr_attribute("displayWindow", "box2i", data_window))
    header.extend(_exr_attribute("lineOrder", "lineOrder", b"\0"))
    header.extend(_exr_attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)))
    header.extend(_exr_attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0)))
    header.extend(_exr_attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)))
    header.extend(_exr_attribute("type", "string", b"scanlineimage"))
    header.append(0)  # end of header attributes

    # Version 2 is a regular scanline image (no tiled/deep/multipart flags).
    prefix = struct.pack("<II", _EXR_MAGIC, _EXR_VERSION)
    offset_table_start = len(prefix) + len(header)
    row_payload_size = width_i * 4
    row_chunk_size = 8 + row_payload_size
    first_chunk = offset_table_start + height_i * 8
    offsets = [first_chunk + row * row_chunk_size for row in range(height_i)]
    output_bytes = bytearray(prefix)
    output_bytes.extend(header)
    for offset in offsets:
        output_bytes.extend(struct.pack("<Q", offset))
    for row_index, row in enumerate(samples):
        payload = struct.pack(f"<{width_i}f", *row)
        output_bytes.extend(struct.pack("<ii", row_index, len(payload)))
        output_bytes.extend(payload)
    output = _path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(output_bytes))
    return output


def _read_cstring(data: bytes, offset: int, label: str) -> tuple[str, int]:
    end = data.find(b"\0", offset)
    if end < 0:
        raise FormatError(f"EXR {label} is missing a terminating NUL")
    try:
        value = data[offset:end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise FormatError(f"EXR {label} is not ASCII") from exc
    return value, end + 1


def _parse_exr_channels(value: bytes) -> list[dict[str, int | str]]:
    channels: list[dict[str, int | str]] = []
    offset = 0
    while offset < len(value):
        if value[offset] == 0:
            if offset != len(value) - 1:
                raise FormatError("EXR chlist has trailing bytes after its terminator")
            return channels
        name, offset = _read_cstring(value, offset, "channel name")
        if offset + 16 > len(value):
            raise FormatError("truncated EXR channel entry")
        pixel_type, p_linear = struct.unpack_from("<iB", value, offset)
        x_sampling, y_sampling = struct.unpack_from("<ii", value, offset + 8)
        offset += 16
        if pixel_type not in (_EXR_HALF, _EXR_FLOAT, _EXR_UINT):
            raise FormatError(f"unsupported EXR channel type {pixel_type}")
        if p_linear not in (0, 1):
            raise FormatError("invalid EXR channel pLinear flag")
        if x_sampling <= 0 or y_sampling <= 0:
            raise FormatError("EXR channel sampling must be positive")
        channels.append(
            {
                "name": name,
                "pixel_type": pixel_type,
                "x_sampling": x_sampling,
                "y_sampling": y_sampling,
            }
        )
    raise FormatError("EXR chlist is missing its terminator")


def read_exr(path: str | Path, channel: str | None = None) -> dict[str, Any]:
    """Read an uncompressed FLOAT scanline EXR channel into top-to-bottom rows.

    Other FLOAT/UINT/HALF channels may be present in the file and are skipped
    when locating the requested channel, but the returned channel itself must
    be FLOAT; HALF and UINT decoding is intentionally unsupported.
    """

    data = _path(path).read_bytes()
    if len(data) < 8 or struct.unpack_from("<I", data, 0)[0] != _EXR_MAGIC:
        raise FormatError("file is not an OpenEXR image")
    version = struct.unpack_from("<I", data, 4)[0]
    version_number = version & 0xFF
    if version_number != _EXR_VERSION:
        raise FormatError(f"unsupported OpenEXR version {version_number}; expected {_EXR_VERSION}")
    unknown_flags = version & ~(_EXR_KNOWN_FLAGS | 0xFF)
    if unknown_flags:
        raise FormatError(f"EXR version contains unknown flags 0x{unknown_flags:x}")
    if version & (_EXR_TILED_FLAG | _EXR_DEEP_FLAG | _EXR_MULTIPART_FLAG):
        raise FormatError("tiled, deep, or multipart EXR is unsupported")
    offset = 8
    attributes: dict[str, tuple[str, bytes]] = {}
    while True:
        if offset >= len(data):
            raise FormatError("EXR header is missing its terminator")
        if data[offset] == 0:
            offset += 1
            break
        name, offset = _read_cstring(data, offset, "attribute name")
        type_name, offset = _read_cstring(data, offset, "attribute type")
        if offset + 4 > len(data):
            raise FormatError("truncated EXR attribute size")
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        end = offset + size
        if end > len(data):
            raise FormatError("truncated EXR attribute value")
        if name in attributes:
            raise FormatError(f"duplicate EXR attribute {name!r}")
        attributes[name] = (type_name, data[offset:end])
        offset = end
    required = ("channels", "compression", "dataWindow", "displayWindow", "lineOrder")
    missing = [name for name in required if name not in attributes]
    if missing:
        raise FormatError(f"EXR header missing {', '.join(missing)}")
    channel_type, channel_value = attributes["channels"]
    if channel_type != "chlist":
        raise FormatError("EXR channels attribute has the wrong type")
    channels = _parse_exr_channels(channel_value)
    if not channels:
        raise FormatError("EXR contains no channels")
    compression_type, compression_value = attributes["compression"]
    if compression_type != "compression" or compression_value != b"\0":
        raise FormatError("only uncompressed EXR scanlines are supported")
    window_type, window_value = attributes["dataWindow"]
    if window_type != "box2i" or len(window_value) != 16:
        raise FormatError("EXR dataWindow has the wrong type or size")
    xmin, ymin, xmax, ymax = struct.unpack("<4i", window_value)
    if xmax < xmin or ymax < ymin:
        raise FormatError("EXR dataWindow is empty")
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    _check_dimensions(width, height)
    line_type, line_value = attributes["lineOrder"]
    if line_type != "lineOrder" or len(line_value) != 1 or line_value[0] not in (0, 1, 2):
        raise FormatError("invalid EXR lineOrder")
    requested = channel if channel is not None else ("Y" if any(c["name"] == "Y" for c in channels) else str(channels[0]["name"]))
    selected_index = next((i for i, item in enumerate(channels) if item["name"] == requested), None)
    if selected_index is None:
        raise FormatError(f"EXR channel {requested!r} is not present")
    selected = channels[selected_index]
    if selected["pixel_type"] != _EXR_FLOAT:
        raise FormatError("requested EXR channel is not FLOAT")
    if any(item["x_sampling"] != 1 or item["y_sampling"] != 1 for item in channels):
        raise FormatError("subsampled EXR channels are unsupported")
    sample_sizes = {_EXR_HALF: 2, _EXR_FLOAT: 4, _EXR_UINT: 4}
    bytes_per_row = sum(width * sample_sizes[int(item["pixel_type"])] for item in channels)
    offset_table_end = offset + height * 8
    if offset_table_end > len(data):
        raise FormatError("EXR is missing scanline offsets")
    row_offsets = [struct.unpack_from("<Q", data, offset + i * 8)[0] for i in range(height)]
    rows: list[list[float] | None] = [None] * height
    for chunk_offset in row_offsets:
        if chunk_offset + 8 > len(data):
            raise FormatError("EXR scanline offset is outside the file")
        y, payload_size = struct.unpack_from("<ii", data, chunk_offset)
        payload_start = chunk_offset + 8
        payload_end = payload_start + payload_size
        if payload_size != bytes_per_row or payload_end > len(data):
            raise FormatError("EXR scanline has an unexpected payload size")
        row_index = y - ymin
        if row_index < 0 or row_index >= height or rows[row_index] is not None:
            raise FormatError("EXR scanlines contain an invalid or duplicate y coordinate")
        channel_offset = sum(width * sample_sizes[int(item["pixel_type"])] for item in channels[:selected_index])
        start = payload_start + channel_offset
        payload = data[start : start + width * 4]
        rows[row_index] = list(struct.unpack(f"<{width}f", payload))
    if any(row is None for row in rows):
        raise FormatError("EXR is missing one or more scanlines")
    pixels = [row for row in rows if row is not None]
    return {
        "width": width,
        "height": height,
        "bit_depth": 32,
        "pixel_type": "FLOAT",
        "channel": requested,
        "channels": [str(item["name"]) for item in channels],
        "pixels": pixels,
        "array": pixels,
        "data": pixels,
        "data_window": [xmin, ymin, xmax, ymax],
    }


# Explicit aliases are useful in validation code that wants to state the
# expected bit depth at the call site.
write_exr32 = write_exr
read_exr32 = read_exr


__all__ = [
    "FormatError",
    "write_png",
    "read_png",
    "write_exr",
    "read_exr",
    "write_exr32",
    "read_exr32",
]
