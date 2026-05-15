from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_vision.inspection.calibration import CalibrationProfile
from robot_vision.inspection.models import InspectionRecipe, InspectionTool
from robot_vision.config import PROJECT_ROOT


@dataclass
class ToolResult:
    tool_id: str
    name: str
    passed: bool
    measurements: dict[str, float]
    bbox_px: tuple[int, int, int, int] | None
    messages: list[str]


@dataclass(frozen=True)
class RectangleDetection:
    bbox_px: tuple[int, int, int, int]
    width_px: int
    height_px: int
    search_area_px: tuple[int, int, int, int]


class InspectionEngine:
    def inspect(
        self,
        image: np.ndarray,
        recipe: InspectionRecipe,
        calibration: CalibrationProfile,
    ) -> dict[str, Any]:
        results = []
        for tool in recipe.tools:
            if not tool.enabled:
                continue
            if tool.type == "ai_classifier":
                result = self._inspect_ai_classifier(image, tool)
            elif tool.type in {"edge", "edge_1", "edge_2"}:
                result = self._inspect_edge(image, tool, calibration)
            else:
                result = self._inspect_rectangle(image, tool, calibration)
            results.append(result)
        passed = all(item.passed for item in results) if results else False
        return {
            "passed": passed,
            "recipe": recipe.name,
            "tools": [self._result_to_dict(item) for item in results],
        }

    def _inspect_rectangle(
        self,
        image: np.ndarray,
        tool: InspectionTool,
        calibration: CalibrationProfile,
    ) -> ToolResult:
        messages: list[str] = []
        detection = detect_rectangle_in_roi(image, tool.roi)
        if detection is None:
            return ToolResult(tool.id, tool.name, False, {}, None, ["No rectangular part found in detection area"])

        width_mm = calibration.width_mm(detection.width_px)
        height_mm = calibration.height_mm(detection.height_px)

        passed = True
        passed &= _check_range("width", width_mm, tool.min_width_mm, tool.max_width_mm, messages)
        passed &= _check_range("height", height_mm, tool.min_height_mm, tool.max_height_mm, messages)
        if not messages:
            messages.append("Rectangle detected inside search area")
        return ToolResult(
            tool.id,
            tool.name,
            passed,
            {
                "width_px": float(detection.width_px),
                "height_px": float(detection.height_px),
                "width_mm": round(width_mm, 3),
                "height_mm": round(height_mm, 3),
                "search_x_px": float(detection.search_area_px[0]),
                "search_y_px": float(detection.search_area_px[1]),
                "search_width_px": float(detection.search_area_px[2]),
                "search_height_px": float(detection.search_area_px[3]),
            },
            detection.bbox_px,
            messages,
        )

    def _inspect_ai_classifier(self, image: np.ndarray, tool: InspectionTool) -> ToolResult:
        x, y, w, h = _roi_pixels(tool.roi, image.shape[1], image.shape[0])
        roi = image[y:y + h, x:x + w]
        try:
            from robot_vision.training.hf_vision import predict_image

            model_dir = tool.model_dir
            model_path = Path(model_dir)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path
            prediction = predict_image(model_path, roi)
        except Exception as exc:
            return ToolResult(
                tool.id,
                tool.name,
                False,
                {},
                (x, y, x + w, y + h),
                [f"AI classifier failed: {exc}"],
            )
        label = str(prediction["label"]).upper()
        score = float(prediction["score"])
        passed = label == "PASS" and score >= tool.min_confidence
        return ToolResult(
            tool.id,
            tool.name,
            passed,
            {
                "ai_label": 1.0 if label == "PASS" else 0.0,
                "ai_confidence": round(score, 4),
                "min_confidence": tool.min_confidence,
                "scores": prediction["scores"],
            },
            (x, y, x + w, y + h),
            [f"AI classified {label} at {score:.2%}"],
        )

    def _inspect_edge(self, image: np.ndarray, tool: InspectionTool, calibration: CalibrationProfile) -> ToolResult:
        x, y, w, h = _roi_pixels(tool.roi, image.shape[1], image.shape[0])
        roi = _gray(image[y:y + h, x:x + w])
        debug_lines = debug_candidate_lines(roi, tool.line_orientation, tool.min_line_length_ratio) if tool.debug else []
        if tool.type == "edge_1":
            line = detect_single_line(roi, tool.line_orientation, tool.min_edge_score, tool.min_line_length_ratio)
            if line is None:
                return ToolResult(
                    tool.id,
                    tool.name,
                    False,
                    {"debug_lines": _offset_debug_lines(debug_lines, x, y)},
                    (x, y, x + w, y + h),
                    ["No line found in search area"],
                )
            orientation = line["orientation"]
            detected_line = _offset_line(line["line"], x, y)
            length_px = line["length_px"]
            length_mm = calibration.width_mm(length_px) if orientation == "horizontal" else calibration.height_mm(length_px)
            messages: list[str] = [f"{orientation.title()} line found"]
            passed = True
            passed &= _check_range("length", length_mm, tool.min_length_mm, tool.max_length_mm, messages)
            return ToolResult(
                tool.id,
                tool.name,
                passed,
                {
                    "line_score": round(line["score"], 3),
                    "line_length_px": float(length_px),
                    "line_length_mm": round(length_mm, 3),
                    "line_a": [float(value) for value in detected_line],
                    "debug_lines": _offset_debug_lines(debug_lines, x, y),
                },
                _bbox_from_lines(detected_line),
                messages,
            )

        pair = detect_parallel_line_pair(roi, tool.line_orientation, tool.min_edge_score, tool.min_line_length_ratio)
        if pair is None:
            return ToolResult(
                tool.id,
                tool.name,
                False,
                {"debug_lines": _offset_debug_lines(debug_lines, x, y)},
                (x, y, x + w, y + h),
                ["No parallel line pair found in search area"],
            )

        orientation = pair["orientation"]
        line_a = _offset_line(pair["line_a"], x, y)
        line_b = _offset_line(pair["line_b"], x, y)
        length_px = pair["average_length_px"]
        length_mm = calibration.width_mm(length_px) if orientation == "horizontal" else calibration.height_mm(length_px)
        messages: list[str] = [f"Parallel {orientation} lines found"]
        passed = True
        passed &= _check_range("length", length_mm, tool.min_length_mm, tool.max_length_mm, messages)
        return ToolResult(
            tool.id,
            tool.name,
            passed,
            {
                "line_score": round(pair["score"], 3),
                "line_gap_px": float(pair["gap_px"]),
                "line_a_length_px": float(pair["line_a_length_px"]),
                "line_b_length_px": float(pair["line_b_length_px"]),
                "average_length_px": float(length_px),
                "average_length_mm": round(length_mm, 3),
                "line_a": [float(value) for value in line_a],
                "line_b": [float(value) for value in line_b],
                "debug_lines": _offset_debug_lines(debug_lines, x, y),
            },
            _bbox_from_lines(line_a, line_b),
            messages,
        )

    @staticmethod
    def _result_to_dict(result: ToolResult) -> dict[str, Any]:
        return {
            "tool_id": result.tool_id,
            "name": result.name,
            "passed": result.passed,
            "measurements": result.measurements,
            "bbox_px": list(result.bbox_px) if result.bbox_px else None,
            "messages": result.messages,
        }


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    return (image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114).astype(np.uint8)


def detect_single_line(
    gray: np.ndarray,
    orientation: str = "auto",
    min_score: float = 25.0,
    min_line_length_ratio: float = 0.15,
) -> dict[str, Any] | None:
    candidates = []
    if orientation in {"auto", "horizontal"}:
        horizontal = _detect_hough_single_line(gray, axis="horizontal", min_score=min_score, min_line_length_ratio=min_line_length_ratio)
        if horizontal is None:
            horizontal = _detect_axis_single_line(gray, axis="horizontal", min_score=min_score)
        if horizontal:
            candidates.append(horizontal)
    if orientation in {"auto", "vertical"}:
        vertical = _detect_hough_single_line(gray, axis="vertical", min_score=min_score, min_line_length_ratio=min_line_length_ratio)
        if vertical is None:
            vertical = _detect_axis_single_line(gray, axis="vertical", min_score=min_score)
        if vertical:
            candidates.append(vertical)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def detect_parallel_line_pair(
    gray: np.ndarray,
    orientation: str = "auto",
    min_score: float = 25.0,
    min_line_length_ratio: float = 0.15,
) -> dict[str, Any] | None:
    candidates = []
    if orientation in {"auto", "horizontal"}:
        horizontal = _detect_hough_parallel_lines(gray, axis="horizontal", min_score=min_score, min_line_length_ratio=min_line_length_ratio)
        if horizontal is None:
            horizontal = _detect_axis_parallel_lines(gray, axis="horizontal", min_score=min_score)
        if horizontal:
            candidates.append(horizontal)
    if orientation in {"auto", "vertical"}:
        vertical = _detect_hough_parallel_lines(gray, axis="vertical", min_score=min_score, min_line_length_ratio=min_line_length_ratio)
        if vertical is None:
            vertical = _detect_axis_parallel_lines(gray, axis="vertical", min_score=min_score)
        if vertical:
            candidates.append(vertical)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def _detect_hough_single_line(gray: np.ndarray, axis: str, min_score: float, min_line_length_ratio: float) -> dict[str, Any] | None:
    lines = _hough_axis_lines(gray, axis, min_line_length_ratio)
    if not lines:
        return None
    best = max(lines, key=lambda item: item[4])
    if best[4] < min_score:
        return None
    return {
        "orientation": axis,
        "score": float(best[4]),
        "line": tuple(int(value) for value in best[:4]),
        "length_px": float(best[4]),
    }


def _detect_hough_parallel_lines(gray: np.ndarray, axis: str, min_score: float, min_line_length_ratio: float) -> dict[str, Any] | None:
    normalized = _hough_axis_lines(gray, axis, min_line_length_ratio)
    if len(normalized) < 2:
        return None

    best: dict[str, Any] | None = None
    for index, first in enumerate(normalized):
        for second in normalized[index + 1:]:
            gap = abs(_line_position(first, axis) - _line_position(second, axis))
            min_pair_gap = max(8, (gray.shape[0] if axis == "horizontal" else gray.shape[1]) * 0.08)
            if gap < min_pair_gap:
                continue
            overlap = _line_overlap_ratio(first, second, axis)
            if overlap < 0.45:
                continue
            len_a = first[4]
            len_b = second[4]
            length_similarity = min(len_a, len_b) / max(len_a, len_b)
            if length_similarity < 0.55:
                continue
            score = (len_a + len_b) * overlap * length_similarity
            if score < min_score:
                continue
            candidate = {
                "orientation": axis,
                "score": float(score),
                "gap_px": float(gap),
                "line_a": tuple(int(value) for value in first[:4]),
                "line_b": tuple(int(value) for value in second[:4]),
                "line_a_length_px": float(len_a),
                "line_b_length_px": float(len_b),
                "average_length_px": float((len_a + len_b) / 2.0),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best


def _hough_axis_lines(gray: np.ndarray, axis: str, min_line_length_ratio: float = 0.15) -> list[tuple[int, int, int, int, float]]:
    try:
        import cv2
    except Exception:
        return []

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(lower + 30, 1.33 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    min_line_length = max(6, int((gray.shape[1] if axis == "horizontal" else gray.shape[0]) * min_line_length_ratio))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(8, int(min_line_length * 0.25)),
        minLineLength=min_line_length,
        maxLineGap=max(4, int(min_line_length * 0.25)),
    )
    if lines is None:
        return []

    normalized: list[tuple[int, int, int, int, float]] = []
    for raw_line in lines[:, 0, :]:
        x0, y0, x1, y1 = (int(value) for value in raw_line)
        dx = x1 - x0
        dy = y1 - y0
        length = float((dx * dx + dy * dy) ** 0.5)
        if length < min_line_length:
            continue
        if axis == "horizontal":
            if abs(dy) > max(3, abs(dx) * 0.2):
                continue
            if x1 < x0:
                x0, y0, x1, y1 = x1, y1, x0, y0
        else:
            if abs(dx) > max(3, abs(dy) * 0.2):
                continue
            if y1 < y0:
                x0, y0, x1, y1 = x1, y1, x0, y0
        normalized.append((x0, y0, x1, y1, length))
    return normalized


def _detect_axis_single_line(gray: np.ndarray, axis: str, min_score: float) -> dict[str, Any] | None:
    image = gray.astype(np.float32)
    if axis == "horizontal":
        profile = np.abs(np.diff(image, axis=0)).mean(axis=1)
    else:
        profile = np.abs(np.diff(image, axis=1)).mean(axis=0)
    threshold = max(min_score, float(profile.mean() + profile.std()))
    peaks = _profile_peaks(profile, threshold)
    if not peaks:
        return None
    best_peak = max(peaks, key=lambda index: profile[index])
    line = _line_extent(gray, axis, best_peak)
    return {
        "orientation": axis,
        "score": float(profile[best_peak]),
        "line": line,
        "length_px": _line_length(line),
    }


def _line_position(line: tuple[int, int, int, int, float], axis: str) -> float:
    return (line[1] + line[3]) / 2.0 if axis == "horizontal" else (line[0] + line[2]) / 2.0


def _line_overlap_ratio(
    first: tuple[int, int, int, int, float],
    second: tuple[int, int, int, int, float],
    axis: str,
) -> float:
    if axis == "horizontal":
        a0, a1 = sorted((first[0], first[2]))
        b0, b1 = sorted((second[0], second[2]))
    else:
        a0, a1 = sorted((first[1], first[3]))
        b0, b1 = sorted((second[1], second[3]))
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    shorter = max(1, min(a1 - a0 + 1, b1 - b0 + 1))
    return overlap / shorter


def debug_candidate_lines(
    gray: np.ndarray,
    orientation: str = "auto",
    min_line_length_ratio: float = 0.15,
    limit: int = 20,
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    if orientation in {"auto", "horizontal"}:
        lines.extend(_debug_axis_lines(gray, "horizontal", min_line_length_ratio))
    if orientation in {"auto", "vertical"}:
        lines.extend(_debug_axis_lines(gray, "vertical", min_line_length_ratio))
    lines.sort(key=lambda item: item["length_px"], reverse=True)
    return lines[:limit]


def _debug_axis_lines(gray: np.ndarray, axis: str, min_line_length_ratio: float) -> list[dict[str, Any]]:
    return [
        {
            "orientation": axis,
            "line": [float(line[0]), float(line[1]), float(line[2]), float(line[3])],
            "length_px": float(line[4]),
        }
        for line in _hough_axis_lines(gray, axis, min_line_length_ratio)
    ]


def _offset_debug_lines(lines: list[dict[str, Any]], x: int, y: int) -> list[dict[str, Any]]:
    offset_lines = []
    for item in lines:
        x0, y0, x1, y1 = item["line"]
        offset_lines.append({
            "orientation": item["orientation"],
            "line": [x0 + x, y0 + y, x1 + x, y1 + y],
            "length_px": item["length_px"],
        })
    return offset_lines


def _detect_axis_parallel_lines(gray: np.ndarray, axis: str, min_score: float) -> dict[str, Any] | None:
    image = gray.astype(np.float32)
    if axis == "horizontal":
        gradient = np.abs(np.diff(image, axis=0))
        profile = gradient.mean(axis=1)
        line_length_limit = gray.shape[1]
    else:
        gradient = np.abs(np.diff(image, axis=1))
        profile = gradient.mean(axis=0)
        line_length_limit = gray.shape[0]

    threshold = max(min_score, float(profile.mean() + profile.std()))
    peaks = _profile_peaks(profile, threshold)
    if len(peaks) < 2:
        return None

    best: dict[str, Any] | None = None
    for first_index, first in enumerate(peaks):
        for second in peaks[first_index + 1:]:
            gap = abs(second - first)
            if gap < 3:
                continue
            line_a = _line_extent(gray, axis, first)
            line_b = _line_extent(gray, axis, second)
            len_a = _line_length(line_a)
            len_b = _line_length(line_b)
            if len_a < line_length_limit * 0.15 or len_b < line_length_limit * 0.15:
                continue
            length_similarity = min(len_a, len_b) / max(len_a, len_b)
            if length_similarity < 0.5:
                continue
            score = float((profile[first] + profile[second]) * length_similarity)
            candidate = {
                "orientation": axis,
                "score": score,
                "gap_px": gap,
                "line_a": line_a,
                "line_b": line_b,
                "line_a_length_px": len_a,
                "line_b_length_px": len_b,
                "average_length_px": (len_a + len_b) / 2.0,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best


def _profile_peaks(profile: np.ndarray, threshold: float) -> list[int]:
    peaks: list[int] = []
    for index, value in enumerate(profile):
        if value < threshold:
            continue
        left = profile[index - 1] if index > 0 else -1
        right = profile[index + 1] if index < profile.size - 1 else -1
        if value >= left and value >= right:
            if peaks and index - peaks[-1] <= 2:
                if value > profile[peaks[-1]]:
                    peaks[-1] = index
            else:
                peaks.append(index)
    return peaks


def _line_extent(gray: np.ndarray, axis: str, index: int) -> tuple[int, int, int, int]:
    image = gray.astype(np.float32)
    if axis == "horizontal":
        row_a = image[max(0, index), :]
        row_b = image[min(gray.shape[0] - 1, index + 1), :]
        diff = np.abs(row_b - row_a)
        start, end = _span_from_signal(diff)
        y = min(gray.shape[0] - 1, index + 1)
        return (start, y, end, y)
    col_a = image[:, max(0, index)]
    col_b = image[:, min(gray.shape[1] - 1, index + 1)]
    diff = np.abs(col_b - col_a)
    start, end = _span_from_signal(diff)
    x = min(gray.shape[1] - 1, index + 1)
    return (x, start, x, end)


def _span_from_signal(signal: np.ndarray) -> tuple[int, int]:
    threshold = max(10.0, float(signal.mean() + signal.std() * 0.5))
    indices = np.where(signal >= threshold)[0]
    if indices.size == 0:
        return (0, max(0, signal.size - 1))
    return (int(indices.min()), int(indices.max()))


def _line_length(line: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = line
    return float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 + 1)


def _offset_line(line: tuple[int, int, int, int], x: int, y: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = line
    return (x0 + x, y0 + y, x1 + x, y1 + y)


def _bbox_from_lines(*lines: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    values_x = [value for line in lines for value in (line[0], line[2])]
    values_y = [value for line in lines for value in (line[1], line[3])]
    return (min(values_x), min(values_y), max(values_x), max(values_y))


def detect_rectangle_in_roi(
    image: np.ndarray,
    roi: tuple[float, float, float, float],
    min_pixels: int = 20,
) -> RectangleDetection | None:
    """Find a rectangular foreground object inside a normalized search area."""
    x, y, w, h = _roi_pixels(roi, image.shape[1], image.shape[0])
    region = image[y:y + h, x:x + w]
    gray = _gray(region)
    mask = _rectangle_candidate_mask(gray)
    component = _best_rectangle_component(mask, min_pixels)
    if component is None:
        return None

    local_x0, local_y0, local_x1, local_y1 = component
    x0 = local_x0 + x
    x1 = local_x1 + x
    y0 = local_y0 + y
    y1 = local_y1 + y
    width_px = max(1, x1 - x0 + 1)
    height_px = max(1, y1 - y0 + 1)
    return RectangleDetection(
        bbox_px=(x0, y0, x1, y1),
        width_px=width_px,
        height_px=height_px,
        search_area_px=(x, y, w, h),
    )


def _rectangle_candidate_mask(gray: np.ndarray) -> np.ndarray:
    border = max(2, min(gray.shape[:2]) // 20)
    border_pixels = np.concatenate(
        [
            gray[:border, :].reshape(-1),
            gray[-border:, :].reshape(-1),
            gray[:, :border].reshape(-1),
            gray[:, -border:].reshape(-1),
        ]
    ).astype(np.float32)
    background = float(np.median(border_pixels))
    image = gray.astype(np.float32)
    bright = image >= background + 18.0
    dark = image <= background - 18.0
    return _clean_mask(bright) | _clean_mask(dark)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return cleaned.astype(bool)
    except Exception:
        return mask


def _best_rectangle_component(mask: np.ndarray, min_pixels: int) -> tuple[int, int, int, int] | None:
    components = _merge_nearby_components(_connected_components(mask), mask.shape)
    if not components:
        return None

    height, width = mask.shape
    best_score = 0.0
    best_bbox: tuple[int, int, int, int] | None = None
    for x0, y0, x1, y1, area in components:
        if area < min_pixels:
            continue
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        if bbox_w >= width * 0.90 and bbox_h >= height * 0.90:
            continue
        fill = area / max(1, bbox_w * bbox_h)
        if fill < 0.35:
            continue
        coverage = (bbox_w * bbox_h) / max(1, width * height)
        if coverage > 0.75:
            continue
        score = area * max(0.1, fill) / max(1.0, coverage * 2.0)
        if score > best_score:
            best_score = score
            best_bbox = (x0, y0, x1, y1)
    return best_bbox


def _merge_nearby_components(
    components: list[tuple[int, int, int, int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int, int, int, int]]:
    height, width = shape
    gap = max(3, min(width, height) // 12)
    merged: list[tuple[int, int, int, int, int]] = []
    used = [False] * len(components)
    for index, component in enumerate(components):
        if used[index]:
            continue
        used[index] = True
        x0, y0, x1, y1, area = component
        changed = True
        while changed:
            changed = False
            for other_index, other in enumerate(components):
                if used[other_index]:
                    continue
                ox0, oy0, ox1, oy1, other_area = other
                if _boxes_near_or_overlap((x0, y0, x1, y1), (ox0, oy0, ox1, oy1), gap):
                    x0 = min(x0, ox0)
                    y0 = min(y0, oy0)
                    x1 = max(x1, ox1)
                    y1 = max(y1, oy1)
                    area += other_area
                    used[other_index] = True
                    changed = True
        merged.append((x0, y0, x1, y1, area))
    return merged


def _boxes_near_or_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    separated_x = ax1 + gap < bx0 or bx1 + gap < ax0
    separated_y = ay1 + gap < by0 or by1 + gap < ay0
    if separated_x or separated_y:
        return False
    overlap_x = min(ax1, bx1) - max(ax0, bx0) + 1
    overlap_y = min(ay1, by1) - max(ay0, by0) + 1
    min_width = min(ax1 - ax0 + 1, bx1 - bx0 + 1)
    min_height = min(ay1 - ay0 + 1, by1 - by0 + 1)
    return overlap_x >= min_width * 0.35 or overlap_y >= min_height * 0.35


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        return [
            (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] - 1),
                int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] - 1),
                int(stats[label, cv2.CC_STAT_AREA]),
            )
            for label in range(1, count)
        ]
    except Exception:
        return _connected_components_fallback(mask)


def _connected_components_fallback(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    starts = np.argwhere(mask)
    for start_y, start_x in starts:
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_x), int(start_y))]
        visited[start_y, start_x] = True
        x0 = x1 = int(start_x)
        y0 = y1 = int(start_y)
        area = 0
        while stack:
            px, py = stack.pop()
            area += 1
            x0 = min(x0, px)
            x1 = max(x1, px)
            y0 = min(y0, py)
            y1 = max(y1, py)
            for nx in (px - 1, px, px + 1):
                for ny in (py - 1, py, py + 1):
                    if nx == px and ny == py:
                        continue
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))
        components.append((x0, y0, x1, y1, area))
    return components


def _roi_pixels(roi: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x = max(0, min(width - 1, int(roi[0] * width)))
    y = max(0, min(height - 1, int(roi[1] * height)))
    w = max(1, min(width - x, int(roi[2] * width)))
    h = max(1, min(height - y, int(roi[3] * height)))
    return x, y, w, h


def _check_range(
    label: str,
    value: float,
    minimum: float | None,
    maximum: float | None,
    messages: list[str],
) -> bool:
    if minimum is not None and value < minimum:
        messages.append(f"{label} {value:.2f} mm is below minimum {minimum:.2f} mm")
        return False
    if maximum is not None and value > maximum:
        messages.append(f"{label} {value:.2f} mm is above maximum {maximum:.2f} mm")
        return False
    return True
