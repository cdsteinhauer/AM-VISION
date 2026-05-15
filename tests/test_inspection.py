import numpy as np

from robot_vision.camera.mock import MockCamera
from robot_vision.config import CameraConfig
from robot_vision.inspection.calibration import CalibrationProfile
from robot_vision.inspection.engine import InspectionEngine, debug_candidate_lines, detect_parallel_line_pair, detect_rectangle_in_roi, detect_single_line
from robot_vision.inspection.models import InspectionRecipe, InspectionTool


def test_default_recipe_passes_on_mock_rectangle():
    frame = MockCamera(CameraConfig(width=640, height=360)).snapshot()
    result = InspectionEngine().inspect(frame.rgb, InspectionRecipe.default(), CalibrationProfile())
    assert result["passed"] is True
    tool = result["tools"][0]
    assert tool["measurements"]["width_px"] > 200
    assert tool["measurements"]["height_px"] > 120
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


def test_rectangle_detection_merges_split_top_and_bottom_edges():
    image = np.full((120, 160, 3), 55, dtype=np.uint8)
    image[40:63, 58:112] = 185
    image[67:86, 58:112] = 25

    detection = detect_rectangle_in_roi(image, (0.25, 0.2, 0.55, 0.65))

    assert detection is not None
    assert detection.bbox_px == (58, 40, 111, 85)
    assert detection.width_px == 54
    assert detection.height_px == 46


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
