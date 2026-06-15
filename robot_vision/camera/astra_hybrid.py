from __future__ import annotations

import os
import shutil
import subprocess
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
        self._depth_start_wait_s = 1.0
        self._depth_stale_s = 1.0
        self._stop_event = threading.Event()
        self._node = None
        self._spin_thread: threading.Thread | None = None
        self._rclpy = None

    def start(self) -> None:
        if self.started:
            return
        self._stop_event.clear()
        self.rgb_camera.start()
        _ensure_astra_ros_driver()

        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
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
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        self.started = True

    def stop(self) -> None:
        self._stop_event.set()
        self.rgb_camera.stop()
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None
        self.started = False
        with self._lock:
            self._depth = None
            self._depth_stamp = 0.0

    def snapshot(self) -> Frame:
        if not self.started:
            self.start()
        rgb_frame = self.rgb_camera.snapshot()

        depth = None
        depth_stamp = 0.0
        deadline = time.time() + self._depth_start_wait_s
        while True:
            with self._lock:
                if self._depth is not None and time.time() - self._depth_stamp <= self._depth_stale_s:
                    depth = self._depth.copy()
                    depth_stamp = self._depth_stamp
                    break
            if self.sequence > 0 or time.time() >= deadline:
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

    def _spin_loop(self) -> None:
        while not self._stop_event.is_set() and self._node is not None and self._rclpy is not None:
            try:
                self._rclpy.spin_once(self._node, timeout_sec=0.1)
            except Exception:
                if not self._stop_event.is_set():
                    raise
                break

    def _set_depth(self, depth: np.ndarray) -> None:
        with self._lock:
            self._depth = depth
            self._depth_stamp = time.time()


def _ensure_astra_ros_driver() -> None:
    if os.name == "nt" or shutil.which("ros2") is None:
        return
    if _astra_ros_driver_running():
        return
    astra_ws = os.environ.get("ASTRA_WS", "/home/csteinhauer/astra_ws")
    command = (
        "set -e; "
        "source /opt/ros/humble/setup.bash; "
        f"if [ -f '{astra_ws}/install/setup.bash' ]; then source '{astra_ws}/install/setup.bash'; fi; "
        "exec ros2 launch astra_camera astra.launch.py enable_color:=false enable_colored_point_cloud:=false"
    )
    log_path = "/home/csteinhauer/robot_vision/data/logs/astra_camera.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "ab") as log_handle:
        subprocess.Popen(
            ["bash", "-lc", command],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    _wait_for_astra_depth_topic(timeout_s=5.0)


def _astra_ros_driver_running() -> bool:
    result = subprocess.run(
        ["bash", "-lc", "ps -ef | grep -E 'ros2 launch astra_camera|astra_camera_container' | grep -v grep"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=2.0,
        check=False,
    )
    return result.returncode == 0


def _wait_for_astra_depth_topic(timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = subprocess.run(
            ["bash", "-lc", "source /opt/ros/humble/setup.bash && timeout 1 ros2 topic echo /camera/depth/image_raw --once --field header >/dev/null 2>&1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(0.25)
