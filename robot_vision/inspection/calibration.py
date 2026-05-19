from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DepthReference:
    width: int
    height: int
    plane_coefficients: tuple[float, float, float]
    roi: tuple[float, float, float, float] | None = None
    sample_count: int = 0
    median_mm: float | None = None
    residual_mad_mm: float | None = None

    @staticmethod
    def from_depth(depth: np.ndarray, roi: tuple[float, float, float, float] | None = None) -> "DepthReference":
        if depth is None or depth.ndim != 2:
            raise ValueError("Depth reference requires a single-channel depth frame")

        height, width = depth.shape
        x0, y0, x1, y1 = _roi_bounds(roi, width, height)
        crop = depth[y0:y1 + 1, x0:x1 + 1]
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]

        values_mm = depth_array_to_mm(crop).reshape(-1)
        xs = xx.reshape(-1).astype(np.float64)
        ys = yy.reshape(-1).astype(np.float64)
        valid = np.isfinite(values_mm) & (values_mm > 0)
        values_mm = values_mm[valid].astype(np.float64)
        xs = xs[valid]
        ys = ys[valid]
        if values_mm.size < 25:
            raise ValueError("Not enough valid depth pixels to fit a reference plane")

        median = float(np.median(values_mm))
        mad = float(np.median(np.abs(values_mm - median)))
        if mad > 0:
            keep = np.abs(values_mm - median) <= max(30.0, mad * 4.0)
            if int(np.count_nonzero(keep)) >= 25:
                values_mm = values_mm[keep]
                xs = xs[keep]
                ys = ys[keep]

        coefficients = _fit_plane(xs, ys, values_mm)
        residuals = values_mm - _eval_plane(coefficients, xs, ys)
        residual_mad = float(np.median(np.abs(residuals - np.median(residuals))))
        return DepthReference(
            width=int(width),
            height=int(height),
            plane_coefficients=tuple(float(value) for value in coefficients),
            roi=roi,
            sample_count=int(values_mm.size),
            median_mm=round(median, 3),
            residual_mad_mm=round(residual_mad, 3),
        )

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> "DepthReference | None":
        if not data:
            return None
        coefficients = data.get("plane_coefficients", (0.0, 0.0, 0.0))
        roi = data.get("roi")
        return DepthReference(
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            plane_coefficients=tuple(float(value) for value in coefficients[:3]),  # type: ignore[index]
            roi=tuple(float(value) for value in roi) if roi else None,
            sample_count=int(data.get("sample_count", 0)),
            median_mm=_optional_float(data.get("median_mm")),
            residual_mad_mm=_optional_float(data.get("residual_mad_mm")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "plane_coefficients": list(self.plane_coefficients),
            "roi": list(self.roi) if self.roi is not None else None,
            "sample_count": self.sample_count,
            "median_mm": self.median_mm,
            "residual_mad_mm": self.residual_mad_mm,
        }

    def depth_mm_at(self, x: np.ndarray, y: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if self.width <= 0 or self.height <= 0:
            return np.full(np.asarray(x).shape, np.nan, dtype=np.float64)
        current_height, current_width = shape
        scaled_x = np.asarray(x, dtype=np.float64) * (self.width / max(1, current_width))
        scaled_y = np.asarray(y, dtype=np.float64) * (self.height / max(1, current_height))
        return _eval_plane(self.plane_coefficients, scaled_x, scaled_y)


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
    depth_reference: DepthReference | None = None

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
            depth_reference=DepthReference.from_dict(data.get("depth_reference")),
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
            "depth_reference": self.depth_reference.to_dict() if self.depth_reference is not None else None,
        }

    def width_mm(self, pixels: float) -> float:
        return pixels / self.pixels_per_mm_x

    def height_mm(self, pixels: float) -> float:
        return pixels / self.pixels_per_mm_y


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def depth_value_to_mm(value: float) -> float:
    if value < 20.0:
        return value * 1000.0
    return value


def depth_array_to_mm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values < 20.0, values * 1000.0, values)


def _fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([x, y, np.ones_like(x)])
    coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
    return float(coefficients[0]), float(coefficients[1]), float(coefficients[2])


def _eval_plane(coefficients: tuple[float, float, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b, c = coefficients
    return a * x + b * y + c


def _roi_bounds(
    roi: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if roi is None:
        return (0, 0, max(0, width - 1), max(0, height - 1))
    x = max(0, min(width - 1, int(roi[0] * width)))
    y = max(0, min(height - 1, int(roi[1] * height)))
    w = max(1, min(width - x, int(roi[2] * width)))
    h = max(1, min(height - y, int(roi[3] * height)))
    return (x, y, x + w - 1, y + h - 1)
