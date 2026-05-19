import pytest
import numpy as np

from robot_vision.inspection.calibration import CalibrationProfile, DepthReference


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


def test_depth_reference_fits_sloped_depth_plane():
    yy, xx = np.mgrid[0:60, 0:80]
    depth = 875.0 + xx * 0.4 + yy * 0.2

    reference = DepthReference.from_depth(depth.astype(np.float32), roi=(0.1, 0.1, 0.8, 0.8))
    predicted = reference.depth_mm_at(np.array([20]), np.array([30]), depth.shape)

    assert reference.sample_count > 1000
    assert reference.residual_mad_mm == pytest.approx(0.0)
    assert predicted[0] == pytest.approx(depth[30, 20], abs=0.01)


def test_calibration_profile_round_trips_depth_reference():
    reference = DepthReference(
        width=80,
        height=60,
        plane_coefficients=(0.4, 0.2, 875.0),
        roi=(0.1, 0.1, 0.8, 0.8),
        sample_count=100,
        median_mm=900.0,
        residual_mad_mm=0.5,
    )
    profile = CalibrationProfile(depth_reference=reference)

    loaded = CalibrationProfile.from_dict(profile.to_dict())

    assert loaded.depth_reference is not None
    assert loaded.depth_reference.plane_coefficients == (0.4, 0.2, 875.0)
    assert loaded.depth_reference.roi == (0.1, 0.1, 0.8, 0.8)
