from robot_vision.camera.mock import MockCamera
from robot_vision.config import CameraConfig


def test_mock_camera_returns_rgb_and_depth():
    camera = MockCamera(CameraConfig(width=640, height=360))
    frame = camera.snapshot()
    assert frame.rgb.shape == (360, 640, 3)
    assert frame.depth is not None
    assert frame.depth.shape == (360, 640)
    assert camera.status()["sequence"] == 1
