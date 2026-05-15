from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray | None = None
    sequence: int = 0
    timestamp: float = 0.0


class CameraProvider(Protocol):
    name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def snapshot(self) -> Frame: ...

    def status(self) -> dict: ...


def encode_png_base64(image: np.ndarray) -> str:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        pil = Image.fromarray(image, mode="L")
    else:
        pil = Image.fromarray(image, mode="RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_jpeg_base64(image: np.ndarray, max_width: int = 640, quality: int = 72) -> str:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        pil = Image.fromarray(image, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(image, mode="RGB")
    if max_width > 0 and pil.width > max_width:
        height = max(1, round(pil.height * (max_width / pil.width)))
        pil = pil.resize((max_width, height), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality, optimize=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def depth_to_display(depth: np.ndarray | None) -> np.ndarray | None:
    if depth is None:
        return None
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    low = float(np.percentile(valid, 2))
    high = float(np.percentile(valid, 98))
    if high <= low:
        high = low + 1.0
    scaled = (np.clip(depth, low, high) - low) / (high - low)
    return (255 - scaled * 255).astype(np.uint8)
