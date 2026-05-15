import pytest

from robot_vision.inspection.calibration import CalibrationProfile


def test_calibration_converts_pixels_to_mm():
    profile = CalibrationProfile.from_reference(800, 500, 400, 250)
    assert profile.pixels_per_mm_x == 2
    assert profile.pixels_per_mm_y == 2
    assert profile.pixel_width == 800
    assert profile.pixel_height == 500
    assert profile.real_width_mm == 400
    assert profile.real_height_mm == 250
    assert profile.width_mm(100) == 50
    assert profile.height_mm(80) == 40


def test_calibration_rejects_zero_dimensions():
    with pytest.raises(ValueError):
        CalibrationProfile.from_reference(800, 500, 0, 250)
