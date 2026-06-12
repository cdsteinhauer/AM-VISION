from __future__ import annotations

import threading
import time

import numpy as np

from robot_vision.camera.base import Frame
from robot_vision.config import CameraConfig


class RosAstraCamera:
    name = "ros_astra"

    def __init__(self, config: CameraConfig):
        self.config = config
        self.sequence = 0
        self.started = False
        self._lock = threading.Lock()
        self._color: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._color_stamp = 0.0
        self._depth_stamp = 0.0
        self._last_depth_accept = 0.0
        self._depth_min_interval = 0.2
        self._stop_event = threading.Event()
        self._node = None
        self._spin_thread: threading.Thread | None = None
        self._rclpy = None

    def start(self) -> None:
        if self.started:
            return
        self._stop_event.clear()
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)

        parent = self

        class CameraNode(Node):
            def __init__(self):
                super().__init__("robot_vision_astra_subscriber")
                self.create_subscription(Image, "/camera/color/image_raw", self.on_color, 10)
                self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, 10)

            def on_color(self, msg):
                parent._set_color(_image_msg_to_rgb(msg))

            def on_depth(self, msg):
                now = time.time()
                if now - parent._last_depth_accept < parent._depth_min_interval:
                    return
                parent._last_depth_accept = now
                parent._set_depth(_image_msg_to_depth(msg))

        self._node = CameraNode()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        self.started = True

    def stop(self) -> None:
        self._stop_event.set()
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None
        self.started = False
        with self._lock:
            self._color = None
            self._depth = None
            self._color_stamp = 0.0
            self._depth_stamp = 0.0

    def snapshot(self) -> Frame:
        if not self.started:
            self.start()

        deadline = time.time() + 6.0
        while time.time() < deadline:
            with self._lock:
                color = None if self._color is None else self._color.copy()
                depth = None if self._depth is None else self._depth.copy()
                color_stamp = self._color_stamp
                depth_stamp = self._depth_stamp
            if color is not None:
                self.sequence += 1
                return Frame(rgb=color, depth=depth, sequence=self.sequence, timestamp=max(color_stamp, depth_stamp))
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for Astra ROS color frame on /camera/color/image_raw")

    def status(self) -> dict:
        with self._lock:
            has_color = self._color is not None
            has_depth = self._depth is not None
            color_shape = list(self._color.shape) if self._color is not None else None
            depth_shape = list(self._depth.shape) if self._depth is not None else None
        return {
            "provider": self.name,
            "started": self.started,
            "sequence": self.sequence,
            "depth": has_depth,
            "color": has_color,
            "color_shape": color_shape,
            "depth_shape": depth_shape,
            "color_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/depth/image_raw",
        }

    def _set_color(self, image: np.ndarray) -> None:
        with self._lock:
            self._color = image
            self._color_stamp = time.time()

    def _set_depth(self, depth: np.ndarray) -> None:
        with self._lock:
            self._depth = depth
            self._depth_stamp = time.time()

    def _spin_loop(self) -> None:
        while not self._stop_event.is_set() and self._node is not None and self._rclpy is not None:
            try:
                self._rclpy.spin_once(self._node, timeout_sec=0.1)
            except Exception:
                if not self._stop_event.is_set():
                    raise
                break


def _image_msg_to_rgb(msg) -> np.ndarray:
    encoding = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if encoding in {"rgb8", "bgr8"}:
        image = data.reshape((msg.height, msg.width, 3))
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        return image.copy()
    if encoding in {"mono8", "8uc1"}:
        gray = data.reshape((msg.height, msg.width))
        return np.stack([gray, gray, gray], axis=2).copy()
    if encoding in {"yuyv", "yuyv422", "yuv422"}:
        import cv2

        yuyv = data.reshape((msg.height, msg.width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
    raise RuntimeError(f"Unsupported ROS color image encoding: {msg.encoding}")


def _image_msg_to_depth(msg) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in {"16uc1", "mono16"}:
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width)).astype(np.float32)
    if encoding == "32fc1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width)).copy()
    if encoding in {"8uc1", "mono8"}:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width)).astype(np.float32)
    raise RuntimeError(f"Unsupported ROS depth image encoding: {msg.encoding}")
