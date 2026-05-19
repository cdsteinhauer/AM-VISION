from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_vision.inspection.calibration import CalibrationProfile, depth_array_to_mm, depth_value_to_mm
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
    width_px: float
    height_px: float
    search_area_px: tuple[int, int, int, int]
    corners_px: tuple[tuple[int, int], ...] = ()
    angle_deg: float = 0.0
    fill_ratio: float = 0.0
    outline_score: float = 0.0


@dataclass(frozen=True)
class PartOutlineDetection:
    bbox_px: tuple[int, int, int, int]
    corners_px: tuple[tuple[int, int], ...]
    width_px: float
    height_px: float
    long_side_px: float
    short_side_px: float
    angle_deg: float
    fill_ratio: float
    outline_score: float
    search_area_px: tuple[int, int, int, int]


class InspectionEngine:
    def inspect(
        self,
        image: np.ndarray,
        recipe: InspectionRecipe,
        calibration: CalibrationProfile,
        depth: np.ndarray | None = None,
    ) -> dict[str, Any]:
        results = []
        for tool in recipe.tools:
            if not tool.enabled:
                continue
            if tool.type == "ai_classifier":
                result = self._inspect_ai_classifier(image, tool)
            elif tool.type in {"edge", "edge_1", "edge_2"}:
                result = self._inspect_edge(image, tool, calibration, depth)
            else:
                result = self._inspect_rectangle(image, tool, calibration, depth)
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
        depth: np.ndarray | None = None,
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
                "outline_width_mm": round(width_mm, 3),
                "outline_height_mm": round(height_mm, 3),
                "min_width_mm": tool.min_width_mm,
                "max_width_mm": tool.max_width_mm,
                "min_height_mm": tool.min_height_mm,
                "max_height_mm": tool.max_height_mm,
                "search_x_px": float(detection.search_area_px[0]),
                "search_y_px": float(detection.search_area_px[1]),
                "search_width_px": float(detection.search_area_px[2]),
                "search_height_px": float(detection.search_area_px[3]),
                "outline_corners": _corners_to_measurement(detection.corners_px),
                "outline_angle_deg": round(detection.angle_deg, 3),
                "outline_fill_ratio": round(detection.fill_ratio, 4),
                "outline_score": round(detection.outline_score, 3),
                **_depth_height_measurements(
                    depth,
                    detection.bbox_px,
                    detection.search_area_px,
                    image.shape[:2],
                    calibration,
                    detection.corners_px,
                ),
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
        scores = {str(key).upper(): float(value) for key, value in prediction["scores"].items()}
        pass_score = scores.get("PASS", 0.0)
        fail_score = scores.get("FAIL", 0.0)
        if not scores:
            fail_score = 0.0
        else:
            fail_score = max(
                (value for key, value in scores.items() if key != "PASS"),
                default=1.0 - pass_score if pass_score else 0.0,
            )
        margin_ok = (pass_score - fail_score) >= tool.min_pass_margin
        passed = label == "PASS" and pass_score >= tool.min_confidence and margin_ok
        return ToolResult(
            tool.id,
            tool.name,
            passed,
            {
                "ai_label": 1.0 if label == "PASS" else 0.0,
                "ai_label_display": label,
                "ai_confidence": round(score, 4),
                "ai_pass_confidence": round(pass_score, 4),
                "ai_fail_confidence": round(fail_score, 4),
                "ai_pass_margin": round(pass_score - fail_score, 4),
                "min_confidence": tool.min_confidence,
                "min_pass_margin": tool.min_pass_margin,
                "scores": prediction["scores"],
            },
            (x, y, x + w, y + h),
            [
                f"AI classified {label} at {score:.2%} "
                f"(PASS={pass_score:.2%}, FAIL={fail_score:.2%}, "
                f"margin={pass_score - fail_score:.2%}, threshold={tool.min_confidence:.2%}, "
                f"required margin={tool.min_pass_margin:.2%})",
            ],
        )

    def _inspect_edge(
        self,
        image: np.ndarray,
        tool: InspectionTool,
        calibration: CalibrationProfile,
        depth: np.ndarray | None = None,
    ) -> ToolResult:
        x, y, w, h = _roi_pixels(tool.roi, image.shape[1], image.shape[0])
        orientation_request = tool.line_orientation or "auto"
        outline = detect_part_outline_in_roi(image, tool.roi, tool.min_edge_score)
        debug_lines = debug_candidate_lines(
            image[y:y + h, x:x + w],
            orientation_request,
            tool.min_edge_score,
            tool.min_line_length_ratio,
        ) if tool.debug else []
        if outline is None:
            return ToolResult(
                tool.id,
                tool.name,
                False,
                {"debug_lines": _offset_debug_lines(debug_lines, x, y)},
                (x, y, x + w, y + h),
                ["No part outline found in search area"],
            )
        if tool.type == "edge_1":
            line = _outline_single_line(outline, orientation_request, tool.min_line_length_ratio)
            if line is None:
                return ToolResult(
                    tool.id,
                    tool.name,
                    False,
                    {"debug_lines": _offset_debug_lines(debug_lines, x, y)},
                    (x, y, x + w, y + h),
                    ["No outline edge found in search area"],
                )
            orientation = line["orientation"]
            detected_line = line["line"]
            length_px = line["length_px"]
            length_mm = calibration.width_mm(length_px) if orientation == "horizontal" else calibration.height_mm(length_px)
            messages: list[str] = [f"{orientation.title()} outline edge found"]
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
                    "min_length_mm": tool.min_length_mm,
                    "max_length_mm": tool.max_length_mm,
                    "line_a": [float(value) for value in detected_line],
                    "debug_lines": _offset_debug_lines(debug_lines, x, y),
                    **_outline_measurements(outline, calibration),
                    **_depth_height_measurements(
                        depth,
                        outline.bbox_px,
                        outline.search_area_px,
                        image.shape[:2],
                        calibration,
                        outline.corners_px,
                    ),
                },
                outline.bbox_px,
                messages,
            )

        pair = _outline_parallel_line_pair(outline, orientation_request, tool.min_line_length_ratio)
        if pair is None:
            return ToolResult(
                tool.id,
                tool.name,
                False,
                {"debug_lines": _offset_debug_lines(debug_lines, x, y)},
                (x, y, x + w, y + h),
                ["No parallel outline edge pair found in search area"],
            )

        orientation = pair["orientation"]
        line_a = pair["line_a"]
        line_b = pair["line_b"]
        length_px = pair["average_length_px"]
        length_mm = calibration.width_mm(length_px) if orientation == "horizontal" else calibration.height_mm(length_px)
        messages: list[str] = [f"Parallel {orientation} outline edges found"]
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
                "min_length_mm": tool.min_length_mm,
                "max_length_mm": tool.max_length_mm,
                "line_a": [float(value) for value in line_a],
                "line_b": [float(value) for value in line_b],
                "debug_lines": _offset_debug_lines(debug_lines, x, y),
                **_outline_measurements(outline, calibration),
                **_depth_height_measurements(
                    depth,
                    outline.bbox_px,
                    outline.search_area_px,
                    image.shape[:2],
                    calibration,
                    outline.corners_px,
                ),
            },
            outline.bbox_px,
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


def detect_part_outline_in_roi(
    image: np.ndarray,
    roi: tuple[float, float, float, float],
    min_pixels: float = 20.0,
) -> PartOutlineDetection | None:
    x, y, w, h = _roi_pixels(roi, image.shape[1], image.shape[0])
    outline = _detect_part_outline(image[y:y + h, x:x + w], min_pixels)
    if outline is None:
        return None
    return _offset_outline(outline, x, y, (x, y, w, h))


def _detect_part_outline(image: np.ndarray, min_pixels: float = 20.0) -> PartOutlineDetection | None:
    gray = _gray(image)
    if gray.size == 0:
        return None
    best: PartOutlineDetection | None = None
    for mask in _part_candidate_masks(gray):
        outline = _best_part_outline(mask, max(20, int(min_pixels)))
        if outline is None:
            continue
        if best is None or outline.outline_score > best.outline_score:
            best = outline
    return best


def _part_candidate_masks(gray: np.ndarray) -> list[np.ndarray]:
    border = max(2, min(gray.shape[:2]) // 10)
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
    bright_threshold = background + 18.0
    dark_threshold = background - 18.0
    try:
        import cv2

        blurred = cv2.GaussianBlur(gray.astype(np.uint8), (3, 3), 0)
        otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_threshold = max(bright_threshold, float(otsu_threshold))
        dark_threshold = min(dark_threshold, float(otsu_threshold))
    except Exception:
        pass
    bright = image >= bright_threshold
    dark = image <= dark_threshold
    return [_clean_mask(bright), _clean_mask(dark)]


def _best_part_outline(mask: np.ndarray, min_pixels: int) -> PartOutlineDetection | None:
    height, width = mask.shape
    components = _merge_nearby_components(_connected_components(mask), mask.shape)
    if not components:
        return None

    minimum_area = max(min_pixels, int(width * height * 0.005))
    roi_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    roi_diag = max(1.0, float((width * width + height * height) ** 0.5))
    best: PartOutlineDetection | None = None
    best_score = 0.0
    for x0, y0, x1, y1, area in components:
        if area < minimum_area:
            continue
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        if bbox_w >= width * 0.95 and bbox_h >= height * 0.95:
            continue
        coverage = (bbox_w * bbox_h) / max(1, width * height)
        if coverage > 0.85:
            continue
        fill = area / max(1, bbox_w * bbox_h)
        if fill < 0.20:
            continue
        center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
        center_factor = 1.0 - min(0.75, float(np.linalg.norm(center - roi_center)) / roi_diag)
        outline = _fit_outline_from_mask(mask, (x0, y0, x1, y1), fill, float(area * max(0.1, fill) * center_factor))
        if outline is None:
            continue
        score = outline.outline_score
        if score > best_score:
            best_score = score
            best = outline
    return best


def _fit_outline_from_mask(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    fill_ratio: float,
    score: float,
) -> PartOutlineDetection | None:
    x0, y0, x1, y1 = bbox
    component_mask = mask[y0:y1 + 1, x0:x1 + 1]
    points_yx = np.argwhere(component_mask)
    if points_yx.size == 0:
        return None
    try:
        import cv2

        points_xy = points_yx[:, ::-1].astype(np.float32)
        points_xy[:, 0] += x0
        points_xy[:, 1] += y0
        rect = cv2.minAreaRect(points_xy)
        corners = tuple(
            (int(round(point[0])), int(round(point[1])))
            for point in cv2.boxPoints(rect)
        )
    except Exception:
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    corners = _order_corners(corners)
    sides = _outline_sides(corners)
    side_lengths = sorted((side["length_px"] for side in sides), reverse=True)
    long_side = float((side_lengths[0] + side_lengths[1]) / 2.0)
    short_side = float((side_lengths[2] + side_lengths[3]) / 2.0)
    horizontal = [side for side in sides if side["orientation"] == "horizontal"]
    vertical = [side for side in sides if side["orientation"] == "vertical"]
    width_px = _average_side_length(horizontal) if horizontal else long_side
    height_px = _average_side_length(vertical) if vertical else short_side
    bbox_px = _bbox_from_corners(corners)
    angle_deg = _outline_angle_deg(sides)
    return PartOutlineDetection(
        bbox_px=bbox_px,
        corners_px=corners,
        width_px=float(width_px),
        height_px=float(height_px),
        long_side_px=long_side,
        short_side_px=short_side,
        angle_deg=angle_deg,
        fill_ratio=float(fill_ratio),
        outline_score=float(score),
        search_area_px=(0, 0, mask.shape[1], mask.shape[0]),
    )


def _offset_outline(
    outline: PartOutlineDetection,
    x: int,
    y: int,
    search_area_px: tuple[int, int, int, int],
) -> PartOutlineDetection:
    corners = tuple((px + x, py + y) for px, py in outline.corners_px)
    bbox = (outline.bbox_px[0] + x, outline.bbox_px[1] + y, outline.bbox_px[2] + x, outline.bbox_px[3] + y)
    return PartOutlineDetection(
        bbox_px=bbox,
        corners_px=corners,
        width_px=outline.width_px,
        height_px=outline.height_px,
        long_side_px=outline.long_side_px,
        short_side_px=outline.short_side_px,
        angle_deg=outline.angle_deg,
        fill_ratio=outline.fill_ratio,
        outline_score=outline.outline_score,
        search_area_px=search_area_px,
    )


def _order_corners(corners: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    center_x = sum(point[0] for point in corners) / len(corners)
    center_y = sum(point[1] for point in corners) / len(corners)
    return tuple(sorted(corners, key=lambda point: np.arctan2(point[1] - center_y, point[0] - center_x)))


def _outline_sides(corners: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    sides: list[dict[str, Any]] = []
    for index, start in enumerate(corners):
        end = corners[(index + 1) % len(corners)]
        line = (start[0], start[1], end[0], end[1])
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        sides.append({
            "orientation": "horizontal" if abs(dx) >= abs(dy) else "vertical",
            "line": line,
            "length_px": _line_length(line),
            "axis_error": abs(dy) if abs(dx) >= abs(dy) else abs(dx),
        })
    return sides


def _average_side_length(sides: list[dict[str, Any]]) -> float:
    if not sides:
        return 0.0
    return float(sum(side["length_px"] for side in sides) / len(sides))


def _outline_angle_deg(sides: list[dict[str, Any]]) -> float:
    longest = max(sides, key=lambda side: side["length_px"])
    x0, y0, x1, y1 = longest["line"]
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


def _bbox_from_corners(corners: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _outline_side_pairs(outline: PartOutlineDetection, orientation: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    sides = _outline_sides(outline.corners_px)
    pairs = [(sides[0], sides[2]), (sides[1], sides[3])]
    if orientation == "auto":
        return sorted(pairs, key=lambda pair: _average_side_length([pair[0], pair[1]]), reverse=True)
    return sorted(
        pairs,
        key=lambda pair: (
            pair[0]["orientation"] != orientation and pair[1]["orientation"] != orientation,
            pair[0]["axis_error"] + pair[1]["axis_error"],
            -_average_side_length([pair[0], pair[1]]),
        ),
    )


def _outline_single_line(
    outline: PartOutlineDetection,
    orientation: str,
    min_line_length_ratio: float,
) -> dict[str, Any] | None:
    pairs = _outline_side_pairs(outline, orientation)
    if not pairs:
        return None
    side = max(pairs[0], key=lambda item: item["length_px"])
    limit = (outline.search_area_px[2] if side["orientation"] == "horizontal" else outline.search_area_px[3]) * min_line_length_ratio
    if side["length_px"] < max(1.0, limit):
        return None
    return {
        "orientation": side["orientation"],
        "score": outline.outline_score,
        "line": side["line"],
        "length_px": float(side["length_px"]),
    }


def _outline_parallel_line_pair(
    outline: PartOutlineDetection,
    orientation: str,
    min_line_length_ratio: float,
) -> dict[str, Any] | None:
    if outline.short_side_px < max(4.0, min(outline.search_area_px[2], outline.search_area_px[3]) * 0.05):
        return None
    pairs = _outline_side_pairs(outline, orientation)
    if not pairs:
        return None
    first, second = pairs[0]
    pair_orientation = first["orientation"] if first["length_px"] >= second["length_px"] else second["orientation"]
    line_length_limit = (outline.search_area_px[2] if pair_orientation == "horizontal" else outline.search_area_px[3]) * min_line_length_ratio
    if first["length_px"] < line_length_limit or second["length_px"] < line_length_limit:
        return None
    gap = _line_midpoint_distance(first["line"], second["line"])
    average_length = (first["length_px"] + second["length_px"]) / 2.0
    return {
        "orientation": pair_orientation,
        "score": outline.outline_score,
        "gap_px": float(gap),
        "line_a": first["line"],
        "line_b": second["line"],
        "line_a_length_px": float(first["length_px"]),
        "line_b_length_px": float(second["length_px"]),
        "average_length_px": float(average_length),
    }


def _line_midpoint_distance(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax = (first[0] + first[2]) / 2.0
    ay = (first[1] + first[3]) / 2.0
    bx = (second[0] + second[2]) / 2.0
    by = (second[1] + second[3]) / 2.0
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _outline_measurements(outline: PartOutlineDetection, calibration: CalibrationProfile | None = None) -> dict[str, Any]:
    measurements: dict[str, Any] = {
        "outline_width_px": round(outline.width_px, 3),
        "outline_height_px": round(outline.height_px, 3),
        "outline_long_side_px": round(outline.long_side_px, 3),
        "outline_short_side_px": round(outline.short_side_px, 3),
        "outline_angle_deg": round(outline.angle_deg, 3),
        "outline_fill_ratio": round(outline.fill_ratio, 4),
        "outline_score": round(outline.outline_score, 3),
        "outline_corners": _corners_to_measurement(outline.corners_px),
    }
    if calibration is not None:
        measurements.update({
            "outline_width_mm": round(calibration.width_mm(outline.width_px), 3),
            "outline_height_mm": round(calibration.height_mm(outline.height_px), 3),
            "outline_long_side_mm": round(calibration.width_mm(outline.long_side_px), 3),
            "outline_short_side_mm": round(calibration.height_mm(outline.short_side_px), 3),
        })
    return measurements


def _depth_height_measurements(
    depth: np.ndarray | None,
    bbox_px: tuple[int, int, int, int] | None,
    search_area_px: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
    calibration: CalibrationProfile,
    object_corners_px: tuple[tuple[int, int], ...] = (),
) -> dict[str, Any]:
    if depth is None or bbox_px is None or search_area_px is None:
        return {}
    if depth.ndim != 2:
        return {}
    image_height, image_width = image_shape
    bbox = _scale_box_to_depth(bbox_px, image_width, image_height, depth.shape[1], depth.shape[0])
    search = _scale_search_to_depth(search_area_px, image_width, image_height, depth.shape[1], depth.shape[0])

    object_mask = _object_depth_mask(depth.shape, bbox, object_corners_px, image_width, image_height)
    object_values = depth[object_mask]
    object_depth = _median_valid_depth(object_values)
    if object_depth is None:
        return {}

    reference = calibration.depth_reference
    if reference is not None:
        ys, xs = np.nonzero(object_mask)
        observed_mm = _valid_depth_mm(depth[ys, xs])
        if observed_mm.size == 0:
            return {}
        valid_positions = np.isfinite(depth[ys, xs]) & (depth[ys, xs] > 0)
        xs = xs[valid_positions]
        ys = ys[valid_positions]
        reference_mm_values = reference.depth_mm_at(xs, ys, depth.shape)
        valid_reference = np.isfinite(reference_mm_values) & (reference_mm_values > 0)
        if not np.any(valid_reference):
            return {}
        reference_mm_values = reference_mm_values[valid_reference]
        observed_mm = observed_mm[valid_reference]
        height_values = reference_mm_values - observed_mm
        height_values = height_values[np.isfinite(height_values)]
        if height_values.size == 0:
            return {}
        object_mm = float(np.median(observed_mm))
        reference_mm = float(np.median(reference_mm_values))
        height_mm = max(0.0, float(np.median(height_values)))
        return {
            "depth_object_mm": round(object_mm, 3),
            "depth_top_mm": round(object_mm, 3),
            "depth_background_mm": round(reference_mm, 3),
            "depth_reference_mm": round(reference_mm, 3),
            "depth_height_mm": round(height_mm, 3),
            "depth_valid_px": int(height_values.size),
            "depth_method": "reference_plane",
            "depth_reference_residual_mad_mm": reference.residual_mad_mm,
        }

    background_depth = _background_depth_around_box(depth, bbox, search)
    if background_depth is None:
        return {}
    object_mm = _depth_value_to_mm(object_depth)
    background_mm = _depth_value_to_mm(background_depth)
    height_mm = max(0.0, background_mm - object_mm)
    return {
        "depth_object_mm": round(object_mm, 3),
        "depth_top_mm": round(object_mm, 3),
        "depth_background_mm": round(background_mm, 3),
        "depth_height_mm": round(height_mm, 3),
        "depth_valid_px": int(_valid_depth_mm(object_values).size),
        "depth_method": "local_background_ring",
    }


def _scale_box_to_depth(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    depth_width: int,
    depth_height: int,
) -> tuple[int, int, int, int]:
    sx = depth_width / max(1, image_width)
    sy = depth_height / max(1, image_height)
    x0, y0, x1, y1 = bbox
    return _clamp_depth_box(
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        depth_width,
        depth_height,
    )


def _scale_search_to_depth(
    search: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    depth_width: int,
    depth_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = search
    return _scale_box_to_depth((x, y, x + w - 1, y + h - 1), image_width, image_height, depth_width, depth_height)


def _clamp_depth_box(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, min(x0, x1)))
    top = max(0, min(height - 1, min(y0, y1)))
    right = max(left, min(width - 1, max(x0, x1)))
    bottom = max(top, min(height - 1, max(y0, y1)))
    return (left, top, right, bottom)


def _crop_depth(depth: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return depth[y0:y1 + 1, x0:x1 + 1]


def _object_depth_mask(
    depth_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    corners_px: tuple[tuple[int, int], ...],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    mask = np.zeros(depth_shape, dtype=bool)
    depth_height, depth_width = depth_shape
    if corners_px:
        sx = depth_width / max(1, image_width)
        sy = depth_height / max(1, image_height)
        points = np.array(
            [[int(round(x * sx)), int(round(y * sy))] for x, y in corners_px],
            dtype=np.int32,
        )
        points[:, 0] = np.clip(points[:, 0], 0, depth_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, depth_height - 1)
        try:
            import cv2

            polygon_mask = np.zeros(depth_shape, dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [points], 1)
            mask = polygon_mask.astype(bool)
        except Exception:
            x0, y0, x1, y1 = bbox
            mask[y0:y1 + 1, x0:x1 + 1] = True
    else:
        x0, y0, x1, y1 = bbox
        mask[y0:y1 + 1, x0:x1 + 1] = True

    x0, y0, x1, y1 = bbox
    bounded = np.zeros_like(mask)
    bounded[y0:y1 + 1, x0:x1 + 1] = mask[y0:y1 + 1, x0:x1 + 1]
    if int(np.count_nonzero(bounded)) < 9:
        return bounded
    try:
        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(bounded.astype(np.uint8), kernel, iterations=1).astype(bool)
        if int(np.count_nonzero(eroded)) >= 9:
            return eroded
    except Exception:
        pass
    return bounded


def _background_depth_around_box(
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    search: tuple[int, int, int, int],
) -> float | None:
    search_crop = _crop_depth(depth, search)
    if search_crop.size == 0:
        return None
    sx0, sy0, sx1, sy1 = search
    bx0, by0, bx1, by1 = bbox
    rel_x0 = max(0, bx0 - sx0)
    rel_y0 = max(0, by0 - sy0)
    rel_x1 = min(search_crop.shape[1] - 1, bx1 - sx0)
    rel_y1 = min(search_crop.shape[0] - 1, by1 - sy0)
    mask = np.ones(search_crop.shape, dtype=bool)
    pad = max(3, min(search_crop.shape) // 20)
    mask[
        max(0, rel_y0 - pad):min(search_crop.shape[0], rel_y1 + pad + 1),
        max(0, rel_x0 - pad):min(search_crop.shape[1], rel_x1 + pad + 1),
    ] = False
    background = _median_valid_depth(search_crop[mask])
    if background is not None:
        return background
    return _median_valid_depth(_search_border_depth(search_crop))


def _search_border_depth(search_crop: np.ndarray) -> np.ndarray:
    border = max(1, min(search_crop.shape) // 8)
    return np.concatenate(
        [
            search_crop[:border, :].reshape(-1),
            search_crop[-border:, :].reshape(-1),
            search_crop[:, :border].reshape(-1),
            search_crop[:, -border:].reshape(-1),
        ]
    )


def _median_valid_depth(values: np.ndarray) -> float | None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = flat[np.isfinite(flat) & (flat > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _valid_depth_mm(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = flat[np.isfinite(flat) & (flat > 0)]
    if valid.size == 0:
        return np.array([], dtype=np.float64)
    return depth_array_to_mm(valid)


def _depth_value_to_mm(value: float) -> float:
    # ROS 32FC1 depth is commonly meters; 16UC1 depth and mock depth are millimeters.
    return depth_value_to_mm(value)


def _corners_to_measurement(corners: tuple[tuple[int, int], ...]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in corners]


def detect_single_line(
    gray: np.ndarray,
    orientation: str = "auto",
    min_score: float = 25.0,
    min_line_length_ratio: float = 0.15,
) -> dict[str, Any] | None:
    orientation = orientation or "auto"
    outline = _detect_part_outline(gray, min_score)
    if outline is not None:
        line = _outline_single_line(outline, orientation, min_line_length_ratio)
        if line is not None:
            return line
    candidates = []
    if orientation in {"auto", "horizontal"}:
        hough_line = _detect_hough_single_line(gray, axis="horizontal", min_score=min_score, min_line_length_ratio=min_line_length_ratio)
        profile_line = _detect_axis_single_line(gray, axis="horizontal", min_score=max(8.0, min_score * 0.65))
        if hough_line:
            candidates.append(hough_line)
        if profile_line:
            candidates.append(profile_line)
    if orientation in {"auto", "vertical"}:
        hough_line = _detect_hough_single_line(gray, axis="vertical", min_score=min_score, min_line_length_ratio=min_line_length_ratio)
        profile_line = _detect_axis_single_line(gray, axis="vertical", min_score=max(8.0, min_score * 0.65))
        if hough_line:
            candidates.append(hough_line)
        if profile_line:
            candidates.append(profile_line)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def detect_parallel_line_pair(
    gray: np.ndarray,
    orientation: str = "auto",
    min_score: float = 25.0,
    min_line_length_ratio: float = 0.15,
) -> dict[str, Any] | None:
    orientation = orientation or "auto"
    outline = _detect_part_outline(gray, min_score)
    if outline is not None:
        pair = _outline_parallel_line_pair(outline, orientation, min_line_length_ratio)
        if pair is not None:
            return pair
    candidates = []
    if orientation in {"auto", "horizontal"}:
        hough_lines = _detect_hough_parallel_lines(
            gray,
            axis="horizontal",
            min_score=min_score,
            min_line_length_ratio=min_line_length_ratio,
        )
        profile_lines = _detect_axis_parallel_lines(
            gray,
            axis="horizontal",
            min_score=max(12.0, min_score * 0.65),
        )
        if hough_lines:
            candidates.append(hough_lines)
        if profile_lines:
            candidates.append(profile_lines)
    if orientation in {"auto", "vertical"}:
        hough_lines = _detect_hough_parallel_lines(
            gray,
            axis="vertical",
            min_score=min_score,
            min_line_length_ratio=min_line_length_ratio,
        )
        profile_lines = _detect_axis_parallel_lines(
            gray,
            axis="vertical",
            min_score=max(12.0, min_score * 0.65),
        )
        if hough_lines:
            candidates.append(hough_lines)
        if profile_lines:
            candidates.append(profile_lines)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def _detect_hough_single_line(gray: np.ndarray, axis: str, min_score: float, min_line_length_ratio: float) -> dict[str, Any] | None:
    lines = _hough_axis_lines(gray, axis, min_line_length_ratio, min_score)
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
    normalized = _hough_axis_lines(gray, axis, min_line_length_ratio, min_score)
    if len(normalized) < 2:
        return None

    best: dict[str, Any] | None = None
    for index, first in enumerate(normalized):
        for second in normalized[index + 1:]:
            gap = abs(_line_position(first, axis) - _line_position(second, axis))
            min_pair_gap = max(6, (gray.shape[0] if axis == "horizontal" else gray.shape[1]) * 0.03)
            if gap < min_pair_gap:
                continue
            overlap = _line_overlap_ratio(first, second, axis)
            if overlap < 0.3:
                continue
            len_a = first[4]
            len_b = second[4]
            length_similarity = min(len_a, len_b) / max(len_a, len_b)
            if length_similarity < 0.45:
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


def _hough_axis_lines(
    gray: np.ndarray,
    axis: str,
    min_line_length_ratio: float = 0.15,
    min_score: float = 25.0,
) -> list[tuple[int, int, int, int, float]]:
    try:
        import cv2
    except Exception:
        return []

    blurred = cv2.GaussianBlur(_enhance_gray(gray), (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.55 * median))
    upper = int(min(255, max(lower + 20, 1.35 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    min_line_length = max(6, int((gray.shape[1] if axis == "horizontal" else gray.shape[0]) * min_line_length_ratio))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(12, int(min_line_length * 0.18)),
        minLineLength=min_line_length,
        maxLineGap=max(3, int(min_line_length * 0.35)),
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
            if abs(dy) > max(4, abs(dx) * 0.35):
                continue
            if x1 < x0:
                x0, y0, x1, y1 = x1, y1, x0, y0
        else:
            if abs(dx) > max(4, abs(dy) * 0.35):
                continue
            if y1 < y0:
                x0, y0, x1, y1 = x1, y1, x0, y0
        if length >= max(1.0, min_score):
            normalized.append((x0, y0, x1, y1, length))
    return normalized


def _enhance_gray(gray: np.ndarray) -> np.ndarray:
    try:
        import cv2

        equalized = cv2.equalizeHist(gray.astype(np.uint8))
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        return clahe.apply(equalized)
    except Exception:
        return gray.astype(np.uint8)


def _detect_axis_single_line(gray: np.ndarray, axis: str, min_score: float) -> dict[str, Any] | None:
    image = gray.astype(np.float32)
    if axis == "horizontal":
        profile = np.abs(np.diff(image, axis=0)).mean(axis=1)
    else:
        profile = np.abs(np.diff(image, axis=1)).mean(axis=0)
    profile = _smooth_profile(profile)
    threshold = max(min_score, float(profile.mean() + profile.std() * 0.75))
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
    min_score: float = 25.0,
    min_line_length_ratio: float = 0.15,
    limit: int = 20,
) -> list[dict[str, Any]]:
    orientation = orientation or "auto"
    gray = _gray(gray)
    outline = _detect_part_outline(gray, min_score)
    if outline is not None:
        outline_lines = _debug_outline_lines(outline, orientation, min_line_length_ratio)
        if outline_lines:
            return outline_lines[:limit]
    lines: list[dict[str, Any]] = []
    if orientation in {"auto", "horizontal"}:
        lines.extend(_debug_axis_lines(gray, "horizontal", min_score, min_line_length_ratio))
    if orientation in {"auto", "vertical"}:
        lines.extend(_debug_axis_lines(gray, "vertical", min_score, min_line_length_ratio))
    lines.sort(key=lambda item: item["length_px"], reverse=True)
    return lines[:limit]


def debug_candidate_lines_in_roi(
    image: np.ndarray,
    roi: tuple[float, float, float, float],
    orientation: str = "auto",
    min_score: float = 25.0,
    min_line_length_ratio: float = 0.15,
    limit: int = 20,
) -> dict[str, Any]:
    x, y, w, h = _roi_pixels(roi, image.shape[1], image.shape[0])
    lines = debug_candidate_lines(
        image[y:y + h, x:x + w],
        orientation=orientation,
        min_score=min_score,
        min_line_length_ratio=min_line_length_ratio,
        limit=limit,
    )
    return {
        "search_area_px": [x, y, w, h],
        "lines": _offset_debug_lines(lines, x, y),
    }


def _debug_axis_lines(
    gray: np.ndarray,
    axis: str,
    min_score: float,
    min_line_length_ratio: float,
) -> list[dict[str, Any]]:
    return [
        {
            "orientation": axis,
            "line": [float(line[0]), float(line[1]), float(line[2]), float(line[3])],
            "length_px": float(line[4]),
        }
        for line in _hough_axis_lines(gray, axis, min_line_length_ratio, min_score)
    ]


def _debug_outline_lines(
    outline: PartOutlineDetection,
    orientation: str,
    min_line_length_ratio: float,
) -> list[dict[str, Any]]:
    lines = []
    for side in _outline_sides(outline.corners_px):
        if orientation != "auto" and side["orientation"] != orientation:
            continue
        limit = (outline.search_area_px[2] if side["orientation"] == "horizontal" else outline.search_area_px[3]) * min_line_length_ratio
        if side["length_px"] < max(1.0, limit):
            continue
        lines.append({
            "orientation": side["orientation"],
            "line": [float(value) for value in side["line"]],
            "length_px": float(side["length_px"]),
        })
    lines.sort(key=lambda item: item["length_px"], reverse=True)
    return lines


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


def _smooth_profile(profile: np.ndarray, radius: int = 2) -> np.ndarray:
    if profile.size <= 1:
        return profile.astype(np.float32)
    kernel_size = max(1, radius * 2 + 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    padded = np.pad(profile.astype(np.float32), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _span_from_signal(signal: np.ndarray) -> tuple[int, int]:
    if signal.size == 0:
        return (0, 0)
    values = signal.astype(np.float32)
    values = _smooth_profile(values)
    peak = int(np.argmax(values))
    threshold = max(8.0, float(values.mean() + values.std() * 0.65), float(values[peak] * 0.22))
    indices = np.where(values >= threshold)[0]
    if indices.size == 0:
        return (0, max(0, signal.size - 1))
    nearest = indices[np.argmin(np.abs(indices - peak))]
    left = int(nearest)
    right = int(nearest)
    while left > 0 and values[left - 1] >= threshold * 0.4:
        left -= 1
    while right < values.size - 1 and values[right + 1] >= threshold * 0.4:
        right += 1
    return (left, right)


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
    outline = detect_part_outline_in_roi(image, roi, min_pixels)
    if outline is not None:
        return RectangleDetection(
            bbox_px=outline.bbox_px,
            width_px=outline.width_px,
            height_px=outline.height_px,
            search_area_px=outline.search_area_px,
            corners_px=outline.corners_px,
            angle_deg=outline.angle_deg,
            fill_ratio=outline.fill_ratio,
            outline_score=outline.outline_score,
        )

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
