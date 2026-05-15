from __future__ import annotations

import time

import numpy as np


class PresenceTrigger:
    def __init__(self, roi: tuple[float, float, float, float], min_change: float, debounce_s: float):
        self.roi = roi
        self.min_change = min_change
        self.debounce_s = debounce_s
        self.last_roi: np.ndarray | None = None
        self.last_fire = 0.0

    def update(self, image: np.ndarray) -> dict:
        x, y, w, h = self._roi_pixels(image.shape[1], image.shape[0])
        gray = self._gray(image[y:y + h, x:x + w])
        score = 0.0
        fired = False
        now = time.time()
        if self.last_roi is not None and self.last_roi.shape == gray.shape:
            score = float(np.mean(np.abs(gray.astype(np.float32) - self.last_roi.astype(np.float32))))
            if score >= self.min_change and now - self.last_fire >= self.debounce_s:
                fired = True
                self.last_fire = now
        self.last_roi = gray
        return {"fired": fired, "score": round(score, 3), "roi_px": [x, y, w, h]}

    def _roi_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x = max(0, min(width - 1, int(self.roi[0] * width)))
        y = max(0, min(height - 1, int(self.roi[1] * height)))
        w = max(1, min(width - x, int(self.roi[2] * width)))
        h = max(1, min(height - y, int(self.roi[3] * height)))
        return x, y, w, h

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.uint8)
        return (image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114).astype(np.uint8)
