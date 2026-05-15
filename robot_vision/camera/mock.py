from __future__ import annotations

import time
from typing import Any

import numpy as np

from robot_vision.camera.base import Frame
from robot_vision.config import CameraConfig


class MockCamera:
    name = "mock"

    def __init__(self, config: CameraConfig):
        self.config = config
        self.sequence = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def snapshot(self) -> Frame:
        if not self.started:
            self.start()
        self.sequence += 1
        h = self.config.height
        w = self.config.width
        image = np.full((h, w, 3), 238, dtype=np.uint8)
        image[:, :, 0] = 246
        image[:, :, 1] = 246
        image[:, :, 2] = 242

        x0 = int(w * 0.28)
        y0 = int(h * 0.25)
        x1 = int(w * 0.75)
        y1 = int(h * 0.68)
        image[y0:y1, x0:x1] = np.array([54, 58, 62], dtype=np.uint8)
        image[y0 + 12:y1 - 12, x0 + 12:x1 - 12] = np.array([92, 98, 104], dtype=np.uint8)
        image[y0:y0 + 5, x0:x1] = np.array([255, 126, 36], dtype=np.uint8)
        image[y1 - 5:y1, x0:x1] = np.array([255, 126, 36], dtype=np.uint8)
        image[y0:y1, x0:x0 + 5] = np.array([255, 126, 36], dtype=np.uint8)
        image[y0:y1, x1 - 5:x1] = np.array([255, 126, 36], dtype=np.uint8)

        yy, xx = np.mgrid[0:h, 0:w]
        depth = 850 + ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / max(w, h)
        depth[y0:y1, x0:x1] = 525
        return Frame(rgb=image, depth=depth.astype(np.float32), sequence=self.sequence, timestamp=time.time())

    def status(self) -> dict:
        return {
            "provider": self.name,
            "started": self.started,
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
            "sequence": self.sequence,
            "depth": self.config.depth_enabled,
        }

    def get_settings(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "device": "mock",
            "controls": [
                {"name": "brightness", "label": "Brightness", "kind": "int", "min": -64, "max": 64, "step": 1, "default": 0, "value": 0, "options": {}},
                {"name": "contrast", "label": "Contrast", "kind": "int", "min": 0, "max": 95, "step": 1, "default": 32, "value": 32, "options": {}},
            ],
        }

    def apply_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        settings = self.get_settings()
        settings["applied"] = updates
        settings["errors"] = {}
        return settings
