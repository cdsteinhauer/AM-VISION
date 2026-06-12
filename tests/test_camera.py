from robot_vision.camera.mock import MockCamera
from robot_vision.camera.astra_hybrid import AstraHybridCamera
from robot_vision.camera.manager import create_camera
from robot_vision.camera.orbbec import OrbbecCamera
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


def test_astra_provider_uses_hybrid_camera():
    camera = create_camera(CameraConfig(provider="astra", device_index=0))

    assert isinstance(camera, AstraHybridCamera)


def test_orbbec_camera_returns_rgb_and_depth(monkeypatch):
    monkeypatch.setattr("robot_vision.camera.orbbec._load_sdk", lambda: FakeOrbbecSdk)

    camera = OrbbecCamera(CameraConfig(provider="orbbec", width=4, height=3, device_index=-1))
    try:
        frame = camera.snapshot()
    finally:
        camera.stop()

    assert frame.rgb.shape == (3, 4, 3)
    assert frame.rgb.dtype == "uint8"
    assert frame.depth is not None
    assert frame.depth.shape == (3, 4)
    assert frame.depth.dtype == "float32"
    assert frame.depth[0, 0] == 1000
    assert camera.status()["backend"] == "orbbec_sdk"


class FakeOrbbecSdk:
    Context = None
    Pipeline = None
    Config = None
    AlignFilter = None

    class OBFormat:
        RGB = "RGB"
        BGR = "BGR"
        YUYV = "YUYV"
        UYVY = "UYVY"
        MJPG = "MJPG"
        I420 = "I420"
        NV12 = "NV12"
        NV21 = "NV21"

    class OBSensorType:
        COLOR_SENSOR = "color"
        DEPTH_SENSOR = "depth"

    class OBStreamType:
        COLOR_STREAM = "color"

    class OBFrameAggregateOutputMode:
        FULL_FRAME_REQUIRE = "full"


class FakeDeviceInfo:
    def get_name(self):
        return "Femto Bolt"

    def get_serial_number(self):
        return "CL8E36300BN"

    def get_connection_type(self):
        return "USB3.1"


class FakeDevice:
    def get_device_info(self):
        return FakeDeviceInfo()


class FakeDeviceList:
    def get_count(self):
        return 1

    def __getitem__(self, index):
        return FakeDevice()


class FakeContext:
    def query_devices(self):
        return FakeDeviceList()


class FakeConfig:
    def enable_stream(self, profile):
        pass

    def set_frame_aggregate_output_mode(self, mode):
        pass


class FakeProfiles:
    def get_video_stream_profile(self, width, height, color_format, fps):
        return object()

    def get_default_video_stream_profile(self):
        return object()


class FakePipeline:
    def __init__(self):
        self.stopped = False

    def get_stream_profile_list(self, sensor):
        return FakeProfiles()

    def enable_frame_sync(self):
        pass

    def start(self, config):
        pass

    def stop(self):
        self.stopped = True

    def wait_for_frames(self, timeout_ms):
        return FakeFrameSet()


class FakeAlignFilter:
    def __init__(self, align_to_stream):
        pass

    def process(self, frames):
        return frames


class FakeFrameSet:
    def get_color_frame(self):
        return FakeColorFrame()

    def get_depth_frame(self):
        return FakeDepthFrame()


class FakeColorFrame:
    def get_width(self):
        return 4

    def get_height(self):
        return 3

    def get_format(self):
        return FakeOrbbecSdk.OBFormat.RGB

    def get_data(self):
        import numpy as np

        return np.arange(36, dtype=np.uint8)


class FakeDepthFrame:
    def get_width(self):
        return 4

    def get_height(self):
        return 3

    def get_depth_scale(self):
        return 1.0

    def get_data(self):
        import numpy as np

        return (np.ones((3, 4), dtype=np.uint16) * 1000).tobytes()


FakeOrbbecSdk.Context = FakeContext
FakeOrbbecSdk.Pipeline = FakePipeline
FakeOrbbecSdk.Config = FakeConfig
FakeOrbbecSdk.AlignFilter = FakeAlignFilter
