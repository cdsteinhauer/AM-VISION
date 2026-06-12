from __future__ import annotations

from robot_vision.camera.mock import MockCamera
from robot_vision.camera.astra_hybrid import AstraHybridCamera
from robot_vision.camera.opencv import OpenCVCamera
from robot_vision.camera.orbbec import OrbbecCamera
from robot_vision.camera.ros_astra import RosAstraCamera
from robot_vision.config import CameraConfig


def create_camera(config: CameraConfig):
    provider = config.provider.lower().strip()
    if provider == "mock":
        return MockCamera(config)
    if provider == "opencv":
        return OpenCVCamera(config)
    if provider == "orbbec":
        return OrbbecCamera(config)
    if provider in {"ros_astra", "astra_ros", "ros"}:
        return RosAstraCamera(config)
    if provider in {"astra", "astra_plus_pro", "astra_hybrid", "hybrid_astra", "rgbd_astra"}:
        return AstraHybridCamera(config)
    raise ValueError(f"Unsupported camera provider: {config.provider}")
