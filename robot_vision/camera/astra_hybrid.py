from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from robot_vision.camera.base import Frame
from robot_vision.camera.opencv import OpenCVCamera
from robot_vision.camera.ros_astra import _image_msg_to_depth
from robot_vision.config import CameraConfig


class AstraHybridCamera:
    name = "astra_hybrid"

    def __init__(self, config: CameraConfig):
        self.config = config
        self.rgb_camera = OpenCVCamera(config)
        self.sequence = 0
        self.started = False
        self._lock = threading.Lock()
        self._depth: np.ndarray | None = None
        self._depth_stamp = 0.0
        self._last_depth_accept = 0.0
        self._depth_min_interval = 0.2
        self._node = None
        self._spin_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.started:
            return
        self.rgb_camera.start()

        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init(args=None)

        parent = self

        class DepthNode(Node):
            def __init__(self):
                super().__init__("robot_vision_astra_depth_subscriber")
                self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, 10)

            def on_depth(self, msg):
                now = time.time()
                if now - parent._last_depth_accept < parent._depth_min_interval:
                    return
                parent._last_depth_accept = now
                parent._set_depth(_image_msg_to_depth(msg))

        self._node = DepthNode()
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()
        self.started = True

    def stop(self) -> None:
        self.rgb_camera.stop()
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        self.started = False

    def snapshot(self) -> Frame:
        if not self.started:
            self.start()
        rgb_frame = self.rgb_camera.snapshot()

        deadline = time.time() + 3.0
        depth = None
        depth_stamp = 0.0
        while time.time() < deadline:
            with self._lock:
                if self._depth is not None:
                    depth = self._depth.copy()
                    depth_stamp = self._depth_stamp
                    break
            time.sleep(0.03)

        self.sequence += 1
        return Frame(
            rgb=rgb_frame.rgb,
            depth=depth,
            sequence=self.sequence,
            timestamp=max(rgb_frame.timestamp, depth_stamp),
        )

    def status(self) -> dict:
        rgb_status = self.rgb_camera.status()
        with self._lock:
            has_depth = self._depth is not None
            depth_shape = list(self._depth.shape) if self._depth is not None else None
        return {
            "provider": self.name,
            "started": self.started,
            "sequence": self.sequence,
            "rgb": rgb_status,
            "depth": has_depth,
            "depth_shape": depth_shape,
            "rgb_source": "/dev/video0 via OpenCV/V4L2",
            "depth_topic": "/camera/depth/image_raw",
        }

    def get_settings(self) -> dict[str, Any]:
        return self.rgb_camera.get_settings()

    def apply_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        return self.rgb_camera.apply_settings(updates)

    def _set_depth(self, depth: np.ndarray) -> None:
        with self._lock:
            self._depth = depth
            self._depth_stamp = time.time()
