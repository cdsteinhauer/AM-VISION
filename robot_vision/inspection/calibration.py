from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CalibrationProfile:
    name: str = "default"
    pixels_per_mm_x: float = 2.0
    pixels_per_mm_y: float = 2.0
    source: str = "manual-default"
    pixel_width: float | None = None
    pixel_height: float | None = None
    real_width_mm: float | None = None
    real_height_mm: float | None = None

    @staticmethod
    def from_reference(
        pixel_width: float,
        pixel_height: float,
        real_width_mm: float,
        real_height_mm: float,
        name: str = "default",
    ) -> "CalibrationProfile":
        if real_width_mm <= 0 or real_height_mm <= 0:
            raise ValueError("Real calibration dimensions must be greater than zero")
        if pixel_width <= 0 or pixel_height <= 0:
            raise ValueError("Pixel calibration dimensions must be greater than zero")
        return CalibrationProfile(
            name=name,
            pixels_per_mm_x=pixel_width / real_width_mm,
            pixels_per_mm_y=pixel_height / real_height_mm,
            source="calibration-plate",
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            real_width_mm=real_width_mm,
            real_height_mm=real_height_mm,
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CalibrationProfile":
        return CalibrationProfile(
            name=str(data.get("name", "default")),
            pixels_per_mm_x=float(data.get("pixels_per_mm_x", 2.0)),
            pixels_per_mm_y=float(data.get("pixels_per_mm_y", 2.0)),
            source=str(data.get("source", "manual-default")),
            pixel_width=_optional_float(data.get("pixel_width")),
            pixel_height=_optional_float(data.get("pixel_height")),
            real_width_mm=_optional_float(data.get("real_width_mm")),
            real_height_mm=_optional_float(data.get("real_height_mm")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pixels_per_mm_x": self.pixels_per_mm_x,
            "pixels_per_mm_y": self.pixels_per_mm_y,
            "source": self.source,
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
            "real_width_mm": self.real_width_mm,
            "real_height_mm": self.real_height_mm,
        }

    def width_mm(self, pixels: float) -> float:
        return pixels / self.pixels_per_mm_x

    def height_mm(self, pixels: float) -> float:
        return pixels / self.pixels_per_mm_y


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
