from pathlib import Path

import numpy as np

from robot_vision.camera.mock import MockCamera
from robot_vision.config import CameraConfig
from robot_vision.inspection.calibration import CalibrationProfile, DepthReference
from robot_vision.inspection.engine import (
    InspectionEngine,
    debug_candidate_lines,
    detect_circle_in_roi,
    detect_depth_rectangle_in_roi,
    detect_parallel_line_pair,
    detect_part_outline_in_roi,
    detect_rectangle_in_roi,
    detect_single_line,
)
from robot_vision.inspection.models import InspectionRecipe, InspectionTool


def test_default_recipe_passes_on_mock_rectangle():
    frame = MockCamera(CameraConfig(width=640, height=360)).snapshot()
    result = InspectionEngine().inspect(frame.rgb, InspectionRecipe.default(), CalibrationProfile(), frame.depth)
    assert result["passed"] is True
    tool = result["tools"][0]
    assert tool["measurements"]["width_px"] > 200
    assert tool["measurements"]["height_px"] > 120
    assert tool["measurements"]["width_mm"] > 100
    assert tool["measurements"]["min_width_mm"] == 20.0
    assert tool["measurements"]["max_width_mm"] == 1000.0
    assert tool["measurements"]["depth_height_mm"] > 250
    assert tool["bbox_px"] is not None


def test_rectangle_detection_measures_object_inside_search_area():
    frame = MockCamera(CameraConfig(width=640, height=360)).snapshot()
    detection = detect_rectangle_in_roi(frame.rgb, (0.1, 0.1, 0.8, 0.8))

    assert detection is not None
    assert detection.search_area_px == (64, 36, 512, 288)
    assert detection.width_px < detection.search_area_px[2]
    assert detection.height_px < detection.search_area_px[3]


def test_rectangle_detection_finds_contrasting_rectangle_not_search_area():
    image = np.full((120, 160, 3), 45, dtype=np.uint8)
    image[42:78, 58:102] = 180

    detection = detect_rectangle_in_roi(image, (0.1, 0.1, 0.8, 0.8))

    assert detection is not None
    assert detection.search_area_px == (16, 12, 128, 96)
    assert detection.bbox_px == (58, 42, 101, 77)
    assert detection.width_px == 44
    assert detection.height_px == 36


def test_rectangle_detection_finds_small_bright_rectangle_on_textured_background():
    rng = np.random.default_rng(7)
    noise = rng.integers(35, 85, size=(140, 180, 1), dtype=np.uint8)
    image = np.repeat(noise, 3, axis=2)
    image[55:87, 72:122] = 190

    detection = detect_rectangle_in_roi(image, (0.25, 0.25, 0.55, 0.55))

    assert detection is not None
    assert 71 <= detection.bbox_px[0] <= 72
    assert detection.bbox_px[1:] == (55, 121, 86)
    assert 50 <= detection.width_px <= 51
    assert detection.height_px == 32


def test_rectangle_detection_uses_best_local_foreground_component():
    image = np.full((120, 160, 3), 55, dtype=np.uint8)
    image[40:63, 58:112] = 185
    image[67:86, 58:112] = 25

    detection = detect_rectangle_in_roi(image, (0.25, 0.2, 0.55, 0.65))

    assert detection is not None
    assert detection.bbox_px == (58, 40, 111, 62)
    assert detection.width_px == 54
    assert detection.height_px == 23


def test_part_outline_detects_bright_part_at_training_sample_scale():
    rng = np.random.default_rng(9)
    noise = rng.integers(45, 80, size=(120, 160, 1), dtype=np.uint8)
    image = np.repeat(noise, 3, axis=2)
    image[52:78, 70:120] = 205

    outline = detect_part_outline_in_roi(image, (0.35, 0.35, 0.5, 0.4))

    assert outline is not None
    assert outline.bbox_px == (70, 52, 119, 77)
    assert 49 <= outline.long_side_px <= 51
    assert 25 <= outline.short_side_px <= 27


def test_part_outline_detects_dark_part_on_bright_background():
    image = np.full((120, 160, 3), 220, dtype=np.uint8)
    image[52:78, 70:120] = 40

    outline = detect_part_outline_in_roi(image, (0.35, 0.35, 0.5, 0.4))

    assert outline is not None
    assert outline.bbox_px == (70, 52, 119, 77)
    assert 49 <= outline.long_side_px <= 51
    assert 25 <= outline.short_side_px <= 27


def test_part_outline_handles_slight_rotation():
    try:
        import cv2
    except Exception:
        return
    image = np.full((140, 180, 3), 50, dtype=np.uint8)
    box = cv2.boxPoints(((90, 70), (52, 26), 8))
    cv2.fillPoly(image, [box.astype(np.int32)], (205, 205, 205))

    outline = detect_part_outline_in_roi(image, (0.25, 0.25, 0.6, 0.6))

    assert outline is not None
    assert outline.width_px == outline.long_side_px
    assert outline.height_px == outline.short_side_px
    assert 49 <= outline.long_side_px <= 55
    assert 24 <= outline.short_side_px <= 30
    assert abs(outline.angle_deg) >= 3


def test_rectangle_measurement_uses_oriented_sides_for_rotated_part():
    try:
        import cv2
    except Exception:
        return
    image = np.full((160, 180, 3), 50, dtype=np.uint8)
    box = cv2.boxPoints(((90, 80), (56, 28), 42))
    cv2.fillPoly(image, [box.astype(np.int32)], (205, 205, 205))
    calibration = CalibrationProfile(pixels_per_mm_x=1.0, pixels_per_mm_y=1.0)
    tool = InspectionTool.from_dict({
        "id": "rotated",
        "name": "Rotated rectangle",
        "type": "rectangle",
        "roi": [0.15, 0.15, 0.7, 0.7],
    })

    result = InspectionEngine()._inspect_rectangle(image, tool, calibration)

    assert result.passed is True
    assert 53 <= result.measurements["width_mm"] <= 59
    assert 25 <= result.measurements["height_mm"] <= 31
    assert result.measurements["width_mm"] > result.measurements["height_mm"]


def test_circle_detection_measures_round_part_inside_search_area():
    try:
        import cv2
    except Exception:
        return
    image = np.full((140, 180, 3), 45, dtype=np.uint8)
    cv2.circle(image, (92, 72), 24, (210, 210, 210), -1)

    detection = detect_circle_in_roi(image, (0.25, 0.2, 0.55, 0.65))

    assert detection is not None
    assert 90 <= detection.center_px[0] <= 94
    assert 70 <= detection.center_px[1] <= 74
    assert 23 <= detection.radius_px <= 25


def test_circle_tool_reports_diameter_measurement():
    try:
        import cv2
    except Exception:
        return
    image = np.full((140, 180, 3), 45, dtype=np.uint8)
    cv2.circle(image, (92, 72), 24, (210, 210, 210), -1)
    calibration = CalibrationProfile(pixels_per_mm_x=2.0, pixels_per_mm_y=2.0)
    tool = InspectionTool.from_dict({
        "id": "circle",
        "name": "Circle check",
        "type": "circle",
        "roi": [0.25, 0.2, 0.55, 0.65],
        "min_diameter_mm": 20,
        "max_diameter_mm": 30,
    })

    result = InspectionEngine()._inspect_circle(image, tool, calibration)

    assert result.passed is True
    assert result.measurements["diameter_mm"] >= 23
    assert result.measurements["diameter_mm"] <= 25
    assert result.measurements["min_diameter_mm"] == 20.0
    assert result.measurements["max_diameter_mm"] == 30.0


def test_parallel_line_pair_detection_reports_average_length():
    image = np.full((80, 120), 40, dtype=np.uint8)
    image[25:27, 20:101] = 210
    image[45:47, 22:103] = 210

    pair = detect_parallel_line_pair(image, "horizontal", min_score=20)

    assert pair is not None
    assert pair["orientation"] == "horizontal"
    assert 79 <= pair["average_length_px"] <= 83
    assert pair["gap_px"] >= 18


def test_single_line_detection_reports_length():
    image = np.full((80, 120), 40, dtype=np.uint8)
    image[35:37, 20:101] = 210

    line = detect_single_line(image, "horizontal", min_score=20)

    assert line is not None
    assert line["orientation"] == "horizontal"
    assert line["length_px"] >= 75


def test_edge_tools_measure_outline_sides():
    image = np.full((120, 160, 3), 45, dtype=np.uint8)
    image[52:78, 70:120] = 205
    depth = np.full((120, 160), 900, dtype=np.float32)
    depth[52:78, 70:120] = 600
    calibration = CalibrationProfile(pixels_per_mm_x=1.0, pixels_per_mm_y=1.0)
    tool = InspectionTool.from_dict({
        "id": "edge",
        "name": "Outline edge",
        "type": "edge_2",
        "roi": [0.35, 0.35, 0.5, 0.4],
        "line_orientation": "auto",
        "min_edge_score": 1,
        "min_line_length_ratio": 0.05,
    })

    result = InspectionEngine()._inspect_edge(image, tool, calibration, depth)

    assert result.passed is True
    assert result.measurements["average_length_px"] == 50.0
    assert result.measurements["average_length_mm"] == 50.0
    assert result.measurements["line_gap_px"] == 25.0
    assert result.measurements["min_length_mm"] is None
    assert result.measurements["max_length_mm"] is None
    assert result.measurements["outline_width_mm"] == 50.0
    assert result.measurements["outline_height_mm"] == 26.0
    assert result.measurements["depth_object_mm"] == 600.0
    assert result.measurements["depth_background_mm"] == 900.0
    assert result.measurements["depth_height_mm"] == 300.0
    assert result.measurements["outline_corners"]


def test_depth_height_uses_reference_plane_when_calibrated():
    image = np.full((120, 160, 3), 45, dtype=np.uint8)
    image[52:78, 70:120] = 205
    yy, xx = np.mgrid[0:120, 0:160]
    reference_depth = 900.0 + xx * 0.25 + yy * 0.1
    depth = reference_depth.astype(np.float32)
    depth[52:78, 70:120] -= 50.0
    calibration = CalibrationProfile(
        pixels_per_mm_x=1.0,
        pixels_per_mm_y=1.0,
        depth_reference=DepthReference.from_depth(reference_depth.astype(np.float32)),
    )
    tool = InspectionTool.from_dict({
        "id": "edge",
        "name": "Outline edge",
        "type": "edge_2",
        "roi": [0.35, 0.35, 0.5, 0.4],
        "line_orientation": "auto",
        "min_edge_score": 1,
        "min_line_length_ratio": 0.05,
    })

    result = InspectionEngine()._inspect_edge(image, tool, calibration, depth)

    assert result.measurements["depth_method"] == "reference_plane"
    assert result.measurements["depth_height_mm"] == 50.0
    assert result.measurements["depth_reference_mm"] > result.measurements["depth_top_mm"]


def test_depth_rectangle_detection_finds_part_when_rgb_is_flat():
    image_shape = (120, 160)
    yy, xx = np.mgrid[0:120, 0:160]
    reference_depth = 900.0 + xx * 0.2 + yy * 0.1
    depth = reference_depth.astype(np.float32)
    depth[52:78, 70:120] -= 45.0
    calibration = CalibrationProfile(
        pixels_per_mm_x=1.0,
        pixels_per_mm_y=1.0,
        depth_reference=DepthReference.from_depth(reference_depth.astype(np.float32)),
    )

    detection = detect_depth_rectangle_in_roi(depth, (0.35, 0.35, 0.5, 0.4), image_shape, calibration)

    assert detection is not None
    assert detection.bbox_px == (70, 52, 119, 77)
    assert detection.width_px == 50.0
    assert detection.height_px == 26.0


def test_rectangle_tool_uses_depth_detection_when_rgb_has_no_outline():
    image = np.full((120, 160, 3), 80, dtype=np.uint8)
    yy, xx = np.mgrid[0:120, 0:160]
    reference_depth = 900.0 + xx * 0.2 + yy * 0.1
    depth = reference_depth.astype(np.float32)
    depth[52:78, 70:120] -= 45.0
    calibration = CalibrationProfile(
        pixels_per_mm_x=1.0,
        pixels_per_mm_y=1.0,
        depth_reference=DepthReference.from_depth(reference_depth.astype(np.float32)),
    )
    tool = InspectionTool.from_dict({
        "id": "depth_rect",
        "name": "Depth rectangle",
        "type": "rectangle",
        "roi": [0.35, 0.35, 0.5, 0.4],
    })

    result = InspectionEngine()._inspect_rectangle(image, tool, calibration, depth)

    assert result.passed is True
    assert result.measurements["detection_method"] == "depth_reference"
    assert result.measurements["depth_method"] == "reference_plane"
    assert result.measurements["depth_height_mm"] == 45.0
    assert result.bbox_px == (70, 52, 119, 77)


def test_depth_rectangle_detection_uses_local_depth_when_reference_is_missing():
    image_shape = (120, 160)
    depth = np.full(image_shape, 900.0, dtype=np.float32)
    depth[52:78, 70:120] -= 45.0
    calibration = CalibrationProfile(pixels_per_mm_x=1.0, pixels_per_mm_y=1.0)

    detection = detect_depth_rectangle_in_roi(depth, (0.35, 0.35, 0.5, 0.4), image_shape, calibration)

    assert detection is not None
    assert detection.bbox_px == (70, 52, 119, 77)
    assert detection.width_px == 50.0
    assert detection.height_px == 26.0


def test_rectangle_tool_prefers_local_depth_when_rgb_spans_search_area():
    image = np.full((120, 160, 3), 80, dtype=np.uint8)
    image[62:76, 56:136] = 205
    depth = np.full((120, 160), 900.0, dtype=np.float32)
    depth[52:78, 70:120] -= 45.0
    calibration = CalibrationProfile(pixels_per_mm_x=1.0, pixels_per_mm_y=1.0)
    tool = InspectionTool.from_dict({
        "id": "depth_rect",
        "name": "Depth rectangle",
        "type": "rectangle",
        "roi": [0.35, 0.35, 0.5, 0.4],
    })

    result = InspectionEngine()._inspect_rectangle(image, tool, calibration, depth)

    assert result.passed is True
    assert result.measurements["detection_method"] == "depth_local"
    assert result.measurements["depth_method"] == "local_background_ring"
    assert result.measurements["depth_height_mm"] == 45.0
    assert result.bbox_px == (70, 52, 119, 77)


def test_depth_detection_offset_moves_overlay_without_moving_height_sample():
    image = np.full((120, 160, 3), 80, dtype=np.uint8)
    yy, xx = np.mgrid[0:120, 0:160]
    reference_depth = 900.0 + xx * 0.2 + yy * 0.1
    depth = reference_depth.astype(np.float32)
    depth[52:78, 70:120] -= 45.0
    calibration = CalibrationProfile(
        pixels_per_mm_x=1.0,
        pixels_per_mm_y=1.0,
        depth_reference=DepthReference.from_depth(reference_depth.astype(np.float32)),
        depth_rgb_offset_x_px=-10.0,
        depth_rgb_offset_y_px=7.0,
    )
    tool = InspectionTool.from_dict({
        "id": "depth_rect",
        "name": "Depth rectangle",
        "type": "rectangle",
        "roi": [0.35, 0.35, 0.5, 0.4],
    })

    result = InspectionEngine()._inspect_rectangle(image, tool, calibration, depth)

    assert result.bbox_px == (60, 59, 109, 84)
    assert result.measurements["depth_height_mm"] == 45.0
    assert result.measurements["depth_rgb_offset_x_px"] == -10.0
    assert result.measurements["depth_rgb_offset_y_px"] == 7.0


def test_rectangle_tool_prefers_rgb_overlay_when_rgb_can_see_part():
    image = np.full((120, 160, 3), 80, dtype=np.uint8)
    image[52:78, 70:120] = 205
    yy, xx = np.mgrid[0:120, 0:160]
    reference_depth = 900.0 + xx * 0.2 + yy * 0.1
    depth = reference_depth.astype(np.float32)
    depth[52:78, 70:120] -= 45.0
    calibration = CalibrationProfile(
        pixels_per_mm_x=1.0,
        pixels_per_mm_y=1.0,
        depth_reference=DepthReference.from_depth(reference_depth.astype(np.float32)),
        depth_rgb_offset_x_px=-10.0,
        depth_rgb_offset_y_px=7.0,
    )
    tool = InspectionTool.from_dict({
        "id": "hybrid_rect",
        "name": "Hybrid rectangle",
        "type": "rectangle",
        "roi": [0.35, 0.35, 0.5, 0.4],
    })

    result = InspectionEngine()._inspect_rectangle(image, tool, calibration, depth)

    assert result.measurements["detection_method"] == "rgb"
    assert result.bbox_px == (70, 52, 119, 77)
    assert result.measurements["depth_height_mm"] == 45.0
    assert result.measurements["depth_rgb_offset_x_px"] == -10.0
    assert result.measurements["depth_rgb_offset_y_px"] == 7.0


def test_edge_tool_uses_depth_outline_when_rgb_has_no_outline():
    image = np.full((120, 160, 3), 80, dtype=np.uint8)
    yy, xx = np.mgrid[0:120, 0:160]
    reference_depth = 900.0 + xx * 0.2 + yy * 0.1
    depth = reference_depth.astype(np.float32)
    depth[52:78, 70:120] -= 45.0
    calibration = CalibrationProfile(
        pixels_per_mm_x=1.0,
        pixels_per_mm_y=1.0,
        depth_reference=DepthReference.from_depth(reference_depth.astype(np.float32)),
    )
    tool = InspectionTool.from_dict({
        "id": "depth_edge",
        "name": "Depth edge",
        "type": "edge_2",
        "roi": [0.35, 0.35, 0.5, 0.4],
        "min_edge_score": 1,
        "min_line_length_ratio": 0.05,
    })

    result = InspectionEngine()._inspect_edge(image, tool, calibration, depth)

    assert result.passed is True
    assert result.measurements["detection_method"] == "depth_reference"
    assert result.measurements["depth_height_mm"] == 45.0
    assert result.measurements["average_length_mm"] == 50.0


def test_debug_candidate_lines_returns_hough_candidates():
    image = np.full((80, 120), 40, dtype=np.uint8)
    image[35:37, 20:101] = 210

    lines = debug_candidate_lines(image, "horizontal", min_line_length_ratio=0.1)

    assert lines
    assert lines[0]["orientation"] == "horizontal"
    assert "line" in lines[0]


def test_legacy_edge_type_loads_as_two_edge_tool():
    tool = InspectionTool.from_dict({"id": "edge", "type": "edge"})

    assert tool.type == "edge_2"


def test_ai_classifier_tool_round_trip():
    tool = InspectionTool.from_dict({
        "id": "ai",
        "type": "ai_classifier",
        "model_dir": "data/models/custom",
        "min_confidence": 0.7,
    })

    assert tool.type == "ai_classifier"
    assert tool.model_dir == "data/models/custom"
    assert tool.min_confidence == 0.7


def test_ai_classifier_confidence_accepts_percent_values():
    tool = InspectionTool.from_dict({"id": "ai", "type": "ai_classifier", "min_confidence": 55})

    assert tool.min_confidence == 0.55


def test_ai_classifier_min_pass_margin_accepts_percent_values():
    tool = InspectionTool.from_dict({"id": "ai", "type": "ai_classifier", "min_pass_margin": 15})

    assert tool.min_pass_margin == 0.15


def test_ai_classifier_fails_when_pass_margin_not_met(monkeypatch):
    frame = MockCamera(CameraConfig(width=320, height=180)).snapshot()

    def fake_predict(_model_dir, _image):
        return {"label": "PASS", "score": 0.68, "scores": {"PASS": 0.68, "FAIL": 0.62, "OTHER": 0.2}}

    monkeypatch.setattr("robot_vision.training.hf_vision.predict_image", fake_predict)
    tool = InspectionTool.from_dict({
        "id": "ai",
        "name": "AI check",
        "type": "ai_classifier",
        "min_confidence": 50,
        "min_pass_margin": 10,
    })
    result = InspectionEngine()._inspect_ai_classifier(frame.rgb, tool)

    assert result.passed is False
    assert result.measurements["min_pass_margin"] == 0.1
    assert result.measurements["ai_pass_margin"] < result.measurements["min_pass_margin"]


def test_ai_classifier_passes_when_pass_margin_met(monkeypatch):
    frame = MockCamera(CameraConfig(width=320, height=180)).snapshot()

    def fake_predict(_model_dir, _image):
        return {"label": "PASS", "score": 0.88, "scores": {"PASS": 0.88, "FAIL": 0.20, "OTHER": 0.12}}

    monkeypatch.setattr("robot_vision.training.hf_vision.predict_image", fake_predict)
    tool = InspectionTool.from_dict({
        "id": "ai",
        "name": "AI check",
        "type": "ai_classifier",
        "min_confidence": 50,
        "min_pass_margin": 10,
    })
    result = InspectionEngine()._inspect_ai_classifier(frame.rgb, tool)

    assert result.passed is True
    assert result.measurements["ai_pass_margin"] >= result.measurements["min_pass_margin"]


def test_tool_live_lines_round_trip():
    tool = InspectionTool.from_dict({"id": "edge", "type": "edge_1", "live_lines": True})

    assert tool.live_lines is True
    assert tool.to_dict()["live_lines"] is True


def test_training_samples_outline_scale_when_available():
    try:
        from PIL import Image
    except Exception:
        return
    sample_paths = sorted(Path("data/training/1x2inch/PASS").glob("*.png"))
    if not sample_paths:
        return
    roi = (0.6387, 0.2, 0.0994, 0.1509)

    outlines = []
    for path in sample_paths:
        image = np.array(Image.open(path).convert("RGB"))
        outline = detect_part_outline_in_roi(image, roi, min_pixels=1)
        if outline is not None:
            outlines.append(outline)

    assert len(outlines) == len(sample_paths)
    assert all(45 <= outline.long_side_px <= 55 for outline in outlines)
    assert all(22 <= outline.short_side_px <= 30 for outline in outlines)
