"""Deterministic image preprocessing and work-plane coordinate mapping."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
AXIS_MAPPINGS = {
    "XY": ("X", "Y", "Z"),
    "XZ": ("X", "Z", "Y"),
    "YZ": ("Y", "Z", "X"),
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def axis_mapping(plane: str) -> dict[str, Any]:
    """Return the frozen v2 axis mapping for one supported work plane."""
    try:
        first, second, offset = AXIS_MAPPINGS[str(plane).upper()]
    except KeyError as exc:
        raise ValueError("工作平面必须是 XY、XZ 或 YZ") from exc
    return {"first_axis": first, "second_axis": second,
            "offset_axis": offset, "image_right_sign": 1,
            "image_up_sign": 1}


def work_plane_payload(plane: str = "XZ", *, offset: float = 0.0,
                       confirmed: bool = False) -> dict[str, Any]:
    if not _finite(offset):
        raise ValueError("工作平面偏移必须是有限数")
    plane = str(plane).upper()
    return {"status": "confirmed" if confirmed else "proposed",
            "plane": plane, "offset": float(offset),
            "axis_mapping": axis_mapping(plane)}


def image_to_model(u: float, v: float, *, width: int, height: int,
                   anchor_uv: tuple[float, float], length_per_pixel: float,
                   plane: str = "XZ", offset: float = 0.0,
                   anchor_xyz: Iterable[float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    """Map normalized derived-image coordinates to SI model coordinates."""
    values = (u, v, length_per_pixel, offset, *anchor_uv, *anchor_xyz)
    if width <= 1 or height <= 1 or not all(_finite(item) for item in values):
        raise ValueError("图像尺寸和坐标参数无效")
    if not 0 <= u <= 1 or not 0 <= v <= 1 or length_per_pixel <= 0:
        raise ValueError("归一化坐标或比例无效")
    anchor = [float(item) for item in anchor_xyz]
    if len(anchor) != 3:
        raise ValueError("anchor_xyz 必须包含三个分量")
    mapping = axis_mapping(plane)
    indexes = {"X": 0, "Y": 1, "Z": 2}
    da = (float(u) - float(anchor_uv[0])) * (width - 1) * float(length_per_pixel)
    db = -(float(v) - float(anchor_uv[1])) * (height - 1) * float(length_per_pixel)
    anchor[indexes[mapping["first_axis"]]] += da
    anchor[indexes[mapping["second_axis"]]] += db
    anchor[indexes[mapping["offset_axis"]]] = float(offset)
    return tuple(anchor)


@dataclass(frozen=True)
class PreprocessResult:
    original_path: str
    derived_path: str
    image_hash: str
    derived_image_hash: str
    raw_width_px: int
    raw_height_px: int
    oriented_width_px: int
    oriented_height_px: int
    width_px: int
    height_px: int
    original_format: str
    exif_orientation: int
    crop: tuple[int, int, int, int]
    rotation_quarters_cw: int

    def source_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        return {
            "original_path": data["original_path"],
            "image_hash": data["image_hash"],
            "raw_width_px": data["raw_width_px"],
            "raw_height_px": data["raw_height_px"],
            "derived_path": data["derived_path"],
            "derived_image_hash": data["derived_image_hash"],
            "width_px": data["width_px"],
            "height_px": data["height_px"],
            "preprocessing": {
                "exif_orientation": data["exif_orientation"],
                "oriented_size_px": [data["oriented_width_px"], data["oriented_height_px"]],
                "crop": list(data["crop"]),
                "rotation_quarters_cw": data["rotation_quarters_cw"],
                "perspective_status": "unconfirmed",
            },
        }

    def oriented_to_derived(self, x: float, y: float) -> tuple[float, float]:
        """Map oriented-source pixel-center coordinates to derived coordinates."""
        left, top, right, bottom = self.crop
        x, y = float(x) - left, float(y) - top
        w, h = right - left, bottom - top
        for _ in range(self.rotation_quarters_cw):
            x, y, w, h = h - 1 - y, x, h, w
        return x, y

    def derived_to_oriented(self, x: float, y: float) -> tuple[float, float]:
        """Invert :meth:`oriented_to_derived` exactly for pixel centers."""
        left, top, right, bottom = self.crop
        sizes = [(right - left, bottom - top)]
        for _ in range(self.rotation_quarters_cw):
            w, h = sizes[-1]
            sizes.append((h, w))
        x, y = float(x), float(y)
        for index in range(self.rotation_quarters_cw, 0, -1):
            old_w, old_h = sizes[index - 1]
            x, y = y, old_h - 1 - x
        return x + left, y + top


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_image(source_path: str | Path, output_dir: str | Path, *,
                     crop: tuple[int, int, int, int] | None = None,
                     rotation_quarters_cw: int = 0) -> PreprocessResult:
    """Apply EXIF orientation, crop, and clockwise quarter-turns without touching the source."""
    source = Path(source_path).resolve()
    destination_dir = Path(output_dir).resolve()
    if not source.is_file():
        raise ValueError(f"图片不存在: {source}")
    rotation = int(rotation_quarters_cw) % 4
    with Image.open(source) as raw:
        image_format = str(raw.format or "").upper()
        if image_format not in SUPPORTED_FORMATS:
            raise ValueError("只支持 PNG、JPEG 和 WebP 图片")
        raw_w, raw_h = raw.size
        if raw_w * raw_h > MAX_IMAGE_PIXELS:
            raise ValueError("图片超过 20 MP 限制")
        orientation = int(raw.getexif().get(274, 1))
        oriented = ImageOps.exif_transpose(raw)
        oriented.load()
    ow, oh = oriented.size
    bounds = crop or (0, 0, ow, oh)
    if (len(bounds) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in bounds)
            or not (0 <= bounds[0] < bounds[2] <= ow)
            or not (0 <= bounds[1] < bounds[3] <= oh)):
        raise ValueError("裁剪范围必须位于应用 EXIF 方向后的图片内")
    derived = oriented.crop(bounds)
    if rotation:
        derived = derived.rotate(-90 * rotation, expand=True)
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(source)
    token = hashlib.sha256(f"{source_hash}:{bounds}:{rotation}".encode()).hexdigest()[:16]
    destination = destination_dir / f"{source.stem}.{token}.png"
    derived.save(destination, format="PNG")
    dw, dh = derived.size
    return PreprocessResult(
        str(source), str(destination), source_hash, _sha256(destination),
        raw_w, raw_h, ow, oh, dw, dh, image_format, orientation,
        tuple(bounds), rotation,
    )
