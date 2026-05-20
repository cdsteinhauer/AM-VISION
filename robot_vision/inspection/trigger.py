from __future__ import annotations

import time

import numpy as np


class PresenceTrigger:
    def __init__(self, roi: tuple[float, float, float, float], min_change: float, debounce_s: float):
        self.roi = roi
        self.min_change = min_change
        self.min_depth_change_mm = max(3.0, min_change * 0.25)
        self.debounce_s = debounce_s
        self.last_roi: np.ndarray | None = None
        self.last_depth_roi: np.ndarray | None = None
        self.last_fire = 0.0

    def update(self, image: np.ndarray, depth: np.ndarray | None = None) -> dict:
        x, y, w, h = self._roi_pixels(image.shape[1], image.shape[0])
        gray = self._gray(image[y:y + h, x:x + w])
        rgb_score = 0.0
        depth_score = 0.0
        depth_changed_fraction = 0.0
        fired_source = ""
        now = time.time()
        if self.last_roi is not None and self.last_roi.shape == gray.shape:
            rgb_score = float(np.mean(np.abs(gray.astype(np.float32) - self.last_roi.astype(np.float32))))
            if rgb_score >= self.min_change:
                fired_source = "rgb"
        if depth is not None:
            depth_score, depth_changed_fraction = self._depth_change(depth)
            if not fired_source and depth_score >= self.min_depth_change_mm and depth_changed_fraction >= 0.001:
                fired_source = "depth"

        fired = bool(fired_source) and now - self.last_fire >= self.debounce_s
        if fired:
            self.last_fire = now
        self.last_roi = gray
        return {
            "fired": fired,
            "score": round(max(rgb_score, depth_score), 3),
            "rgb_score": round(rgb_score, 3),
            "depth_score_mm": round(depth_score, 3),
            "depth_changed_fraction": round(depth_changed_fraction, 5),
            "source": fired_source if fired else "",
            "roi_px": [x, y, w, h],
        }

    def _depth_change(self, depth: np.ndarray) -> tuple[float, float]:
        if depth.ndim != 2:
            return 0.0, 0.0
        x, y, w, h = self._roi_pixels(depth.shape[1], depth.shape[0])
        current = _depth_to_mm(depth[y:y + h, x:x + w])
        score = 0.0
        changed_fraction = 0.0
        if self.last_depth_roi is not None and self.last_depth_roi.shape == current.shape:
            valid = (
                np.isfinite(current)
                & (current > 0)
                & np.isfinite(self.last_depth_roi)
                & (self.last_depth_roi > 0)
            )
            if np.any(valid):
                delta = np.abs(current[valid] - self.last_depth_roi[valid])
                score = float(np.percentile(delta, 99.5))
                changed_pixels = int(np.count_nonzero(delta >= self.min_depth_change_mm))
                minimum_pixels = max(8, int(delta.size * 0.001))
                if changed_pixels >= minimum_pixels:
                    changed_fraction = changed_pixels / float(delta.size)
        self.last_depth_roi = current
        return score, changed_fraction

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


def _depth_to_mm(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    return np.where(values < 20.0, values * 1000.0, values)
