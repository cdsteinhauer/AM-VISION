from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from robot_vision.camera.base import Frame
from robot_vision.config import CameraConfig


MIN_DEPTH_MM = 20.0
MAX_DEPTH_MM = 10000.0


class OrbbecCamera:
    name = "orbbec"

    def __init__(self, config: CameraConfig):
        self.config = config
        self.sequence = 0
        self.pipeline: Any = None
        self._lock = threading.RLock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_rgb: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_timestamp = 0.0
        self._started_at: float | None = None
        self._last_error: str | None = None
        self._frame_count = 0
        self._fps = 0.0
        self._device_info: dict[str, Any] | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._latest_rgb = None
            self._latest_depth = None
            self._latest_timestamp = 0.0
            self._last_error = None
            self._frame_count = 0
            self._fps = 0.0
            self._started_at = time.time()
            self._thread = threading.Thread(target=self._reader_loop, name="orbbec-camera", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        pipeline = None
        with self._lock:
            pipeline = self.pipeline
            self.pipeline = None
            self._frame_ready.notify_all()
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"Pipeline stop warning: {exc}"
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._latest_rgb = None
            self._latest_depth = None

    def snapshot(self) -> Frame:
        if self._thread is None or not self._thread.is_alive():
            self.start()
        deadline = time.time() + 12.0
        with self._lock:
            while time.time() < deadline:
                if self._latest_rgb is not None:
                    self.sequence += 1
                    return Frame(
                        rgb=self._latest_rgb.copy(),
                        depth=self._latest_depth.copy() if self._latest_depth is not None else None,
                        sequence=self.sequence,
                        timestamp=self._latest_timestamp,
                    )
                if self._last_error and self._thread is not None and not self._thread.is_alive():
                    raise RuntimeError(self._last_error)
                self._frame_ready.wait(timeout=0.1)
            raise RuntimeError(self._last_error or "Timed out waiting for Orbbec Femto frame")

    def status(self) -> dict[str, Any]:
        with self._lock:
            started = self._thread is not None and self._thread.is_alive() and self.pipeline is not None
            return {
                "provider": self.name,
                "started": started,
                "device": self._device_info,
                "width": self.config.width,
                "height": self.config.height,
                "fps": round(self._fps, 1) if self._fps else self.config.fps,
                "sequence": self.sequence,
                "frames": self._frame_count,
                "depth": self.config.depth_enabled,
                "backend": "orbbec_sdk",
                "latest_frame_age_s": round(max(0.0, time.time() - self._latest_timestamp), 3) if self._latest_timestamp else None,
                "error": self._last_error,
            }

    def get_settings(self) -> dict[str, Any]:
        return {
            "backend": "orbbec_sdk",
            "device": self._device_info or "Orbbec SDK",
            "controls": [],
        }

    def apply_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        settings = self.get_settings()
        settings["applied"] = {}
        settings["errors"] = {name: "Orbbec SDK camera controls are not implemented yet" for name in updates}
        return settings

    def _reader_loop(self) -> None:
        pipeline = None
        try:
            sdk = _load_sdk()
            self._device_info = _first_device_info(sdk)

            pipeline = sdk.Pipeline()
            config = sdk.Config()

            color_profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            config.enable_stream(color_profile)

            depth_profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

            align_filter = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
            try:
                pipeline.enable_frame_sync()
            except Exception:
                pass

            pipeline.start(config)
            with self._lock:
                self.pipeline = pipeline
                self._last_error = None
                self._frame_ready.notify_all()

            last_fps_time = time.time()
            frames_since_fps = 0
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(1000)
                if not frames:
                    continue
                aligned = align_filter.process(frames)
                if aligned:
                    frames = aligned
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                rgb = _color_frame_to_rgb(color_frame, sdk.OBFormat)
                if rgb is None:
                    continue
                depth = _depth_frame_to_mm(depth_frame)
                if depth.shape[:2] != rgb.shape[:2]:
                    depth = _resize_depth_to_rgb(depth, rgb.shape[1], rgb.shape[0])

                now = time.time()
                frames_since_fps += 1
                elapsed = now - last_fps_time
                with self._lock:
                    self._latest_rgb = rgb
                    self._latest_depth = depth
                    self._latest_timestamp = now
                    self._frame_count += 1
                    if elapsed >= 1.0:
                        self._fps = frames_since_fps / elapsed
                        frames_since_fps = 0
                        last_fps_time = now
                    self._frame_ready.notify_all()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._frame_ready.notify_all()
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            with self._lock:
                if self.pipeline is pipeline:
                    self.pipeline = None
                self._frame_ready.notify_all()


def _load_sdk():
    try:
        import pyorbbecsdk
    except Exception as exc:
        raise RuntimeError("pyorbbecsdk is not importable. Install pyorbbecsdk2 on the Jetson.") from exc
    return pyorbbecsdk


def _first_device_info(sdk) -> dict[str, Any] | None:
    try:
        context = sdk.Context()
        device_list = context.query_devices()
        if device_list.get_count() <= 0:
            return None
        device = device_list[0]
        info = device.get_device_info()
    except Exception as exc:
        return {"error": str(exc)}
    if info is None:
        return None
    return {
        "name": _safe_call(info, "get_name"),
        "serial": _safe_call(info, "get_serial_number"),
        "vid": _safe_call(info, "get_vid"),
        "pid": _safe_call(info, "get_pid"),
        "connection": _safe_call(info, "get_connection_type"),
        "firmware": _safe_call(info, "get_firmware_version"),
        "hardware": _safe_call(info, "get_hardware_version"),
        "min_sdk": _safe_call(info, "get_supported_min_sdk_version"),
    }


def _safe_call(obj: Any, method: str) -> Any:
    try:
        return getattr(obj, method)()
    except Exception:
        return None


def _color_frame_to_rgb(frame: Any, ob_format) -> np.ndarray | None:
    width = int(frame.get_width())
    height = int(frame.get_height())
    color_format = frame.get_format()

    if color_format == ob_format.RGB:
        data = np.asanyarray(frame.get_data())
        return np.resize(data, (height, width, 3)).astype(np.uint8, copy=False).copy()
    import cv2

    if color_format == ob_format.BGR:
        data = np.asanyarray(frame.get_data())
        bgr = np.resize(data, (height, width, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if color_format == ob_format.YUYV:
        data = np.asanyarray(frame.get_data())
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUYV)
    if color_format == ob_format.UYVY:
        data = np.asanyarray(frame.get_data())
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)
    if color_format == ob_format.MJPG:
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    if color_format == ob_format.I420:
        data = np.asanyarray(frame.get_data())
        image = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_I420)
    if color_format == ob_format.NV12:
        data = np.asanyarray(frame.get_data())
        image = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_NV12)
    if color_format == ob_format.NV21:
        data = np.asanyarray(frame.get_data())
        image = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_NV21)
    return None


def _depth_frame_to_mm(frame: Any) -> np.ndarray:
    width = int(frame.get_width())
    height = int(frame.get_height())
    scale = float(frame.get_depth_scale())
    depth_raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((height, width))
    depth_mm = depth_raw.astype(np.float32) * scale
    return np.where((depth_mm > MIN_DEPTH_MM) & (depth_mm < MAX_DEPTH_MM), depth_mm, 0).astype(np.float32)


def _resize_depth_to_rgb(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    y_index = np.clip(np.round(np.linspace(0, depth.shape[0] - 1, height)).astype(int), 0, depth.shape[0] - 1)
    x_index = np.clip(np.round(np.linspace(0, depth.shape[1] - 1, width)).astype(int), 0, depth.shape[1] - 1)
    return depth[np.ix_(y_index, x_index)].astype(np.float32)
