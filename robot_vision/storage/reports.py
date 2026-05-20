from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from robot_vision.camera.base import depth_to_display


class ReportStore:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        result: dict[str, Any],
        recipe: dict[str, Any],
        calibration: dict[str, Any],
    ) -> dict[str, Any]:
        report_id = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1000000:06d}"
        folder = self.report_dir / report_id
        folder.mkdir(parents=True, exist_ok=False)

        rgb_path = folder / "rgb.png"
        depth_path = folder / "depth.png"
        overlay_path = folder / "overlay.png"
        result_path = folder / "result.json"

        Image.fromarray(_uint8_rgb(rgb), mode="RGB").save(rgb_path)
        if depth is not None:
            depth_display = depth_to_display(depth)
            if depth_display is not None:
                Image.fromarray(depth_display, mode="L").save(depth_path)
        self._save_overlay(rgb, result, overlay_path)

        payload = {
            "id": report_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "result": result,
            "recipe": recipe,
            "calibration": calibration,
            "files": {
                "rgb": str(rgb_path),
                "depth": str(depth_path) if depth_path.exists() else None,
                "overlay": str(overlay_path),
            },
        }
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def list_reports(self) -> list[dict[str, Any]]:
        reports = []
        for path in sorted(self.report_dir.glob("*/result.json"), reverse=True):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            reports.append({
                "id": payload.get("id", path.parent.name),
                "created_at": payload.get("created_at", ""),
                "passed": payload.get("result", {}).get("passed", False),
                "recipe": payload.get("result", {}).get("recipe", ""),
            })
        return reports

    def load(self, report_id: str) -> dict[str, Any]:
        path = self.report_dir / report_id / "result.json"
        if not path.exists():
            raise FileNotFoundError(f"Report not found: {report_id}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def delete_all(self) -> int:
        deleted = 0
        for path in self.report_dir.iterdir():
            if not path.is_dir():
                continue
            if (path / "result.json").exists():
                shutil.rmtree(path)
                deleted += 1
        return deleted

    def _save_overlay(self, rgb: np.ndarray, result: dict[str, Any], path: Path) -> None:
        image = Image.fromarray(_uint8_rgb(rgb), mode="RGB")
        draw = ImageDraw.Draw(image)
        for tool in result.get("tools", []):
            measurements = tool.get("measurements", {})
            line_a = measurements.get("line_a")
            line_b = measurements.get("line_b")
            outline_corners = measurements.get("outline_corners") or []
            color = (21, 150, 80) if tool.get("passed") else (210, 52, 42)
            if len(outline_corners) >= 4:
                points = [tuple(point) for point in outline_corners]
                draw.line(points + [points[0]], fill=color, width=5)
                if not line_a:
                    label_x = min(point[0] for point in points)
                    label_y = min(point[1] for point in points)
                    draw.text((label_x + 6, max(0, label_y - 18)), tool.get("name", "Tool"), fill=color)
            if line_a and line_b:
                draw.line(tuple(line_a), fill=color, width=5)
                draw.line(tuple(line_b), fill=color, width=5)
                draw.text((line_a[0] + 6, max(0, line_a[1] - 18)), tool.get("name", "Tool"), fill=color)
                continue
            bbox = tool.get("bbox_px")
            if not bbox or len(outline_corners) >= 4:
                continue
            draw.rectangle(tuple(bbox), outline=color, width=5)
            draw.text((bbox[0] + 6, max(0, bbox[1] - 18)), tool.get("name", "Tool"), fill=color)
        image.save(path)


def _uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)
    return image
