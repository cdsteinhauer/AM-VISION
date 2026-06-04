from robot_vision.camera.mock import MockCamera
from robot_vision.camera.opencv import VideoDevice, select_camera_device_index
from robot_vision.config import CameraConfig


def test_mock_camera_returns_rgb_and_depth():
    camera = MockCamera(CameraConfig(width=640, height=360))
    frame = camera.snapshot()
    assert frame.rgb.shape == (360, 640, 3)
    assert frame.depth is not None
    assert frame.depth.shape == (360, 640)
    assert camera.status()["sequence"] == 1


def test_auto_select_prefers_astra_device(monkeypatch):
    monkeypatch.setattr(
        "robot_vision.camera.opencv.list_video_devices",
        lambda: [
            VideoDevice(index=0, path="/dev/video0", label="USB Global Shutter Camera"),
            VideoDevice(index=2, path="/dev/video2", label="Orbbec Astra RGB Camera"),
        ],
    )

    assert select_camera_device_index("astra") == 2


def test_auto_select_prefers_global_shutter_device(monkeypatch):
    monkeypatch.setattr(
        "robot_vision.camera.opencv.list_video_devices",
        lambda: [
            VideoDevice(index=0, path="/dev/video0", label="Orbbec Astra RGB Camera"),
            VideoDevice(index=4, path="/dev/video4", label="USB Global Shutter Camera"),
        ],
    )

    assert select_camera_device_index("global_shutter") == 4


def test_auto_select_keeps_explicit_device():
    assert select_camera_device_index("astra", preferred_index=7) == 7
