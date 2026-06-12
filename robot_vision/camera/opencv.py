from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_vision.camera.base import Frame
from robot_vision.config import CameraConfig


@dataclass(frozen=True)
class CameraControl:
    name: str
    label: str
    kind: str
    minimum: int | None = None
    maximum: int | None = None
    step: int = 1
    default: int | None = None
    options: dict[int, str] | None = None


@dataclass(frozen=True)
class VideoDevice:
    index: int
    path: str
    label: str


CONTROL_LABELS = {
    "brightness": "Brightness",
    "contrast": "Contrast",
    "saturation": "Saturation",
    "hue": "Hue",
    "white_balance_automatic": "Auto White Balance",
    "gamma": "Gamma",
    "power_line_frequency": "Power Line Frequency",
    "white_balance_temperature": "White Balance Temp",
    "sharpness": "Sharpness",
    "backlight_compensation": "Backlight",
}


DEFAULT_CONTROLS = {
    "brightness": CameraControl("brightness", "Brightness", "int", -64, 64, 1, 0),
    "contrast": CameraControl("contrast", "Contrast", "int", 0, 95, 1, 32),
    "saturation": CameraControl("saturation", "Saturation", "int", 0, 128, 1, 110),
    "hue": CameraControl("hue", "Hue", "int", -2000, 2000, 1, 0),
    "white_balance_automatic": CameraControl("white_balance_automatic", "Auto White Balance", "bool", 0, 1, 1, 1),
    "gamma": CameraControl("gamma", "Gamma", "int", 100, 300, 1, 100),
    "power_line_frequency": CameraControl(
        "power_line_frequency",
        "Power Line Frequency",
        "menu",
        0,
        2,
        1,
        1,
        {0: "Disabled", 1: "50 Hz", 2: "60 Hz"},
    ),
    "white_balance_temperature": CameraControl("white_balance_temperature", "White Balance Temp", "int", 2800, 6500, 1, 4600),
    "sharpness": CameraControl("sharpness", "Sharpness", "int", 1, 7, 1, 2),
    "backlight_compensation": CameraControl("backlight_compensation", "Backlight", "int", 0, 3, 1, 1),
}


class OpenCVCamera:
    name = "opencv"

    def __init__(self, config: CameraConfig):
        self.config = config
        self.sequence = 0
        self.capture = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_bgr: np.ndarray | None = None
        self._latest_timestamp = 0.0
        self._read_error: str | None = None
        self._read_failures = 0
        self._reconnecting = False
        self._reconnects = 0
        self._open_label: str | None = None

    def start(self) -> None:
        import cv2

        if self.capture is not None:
            return
        capture, label = self._open_capture()
        self._configure_capture(capture)
        self.capture = capture
        self._open_label = label
        self._stop_event.clear()
        self._read_error = None
        self._read_failures = 0
        self._reconnecting = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            capture = self.capture
            self.capture = None
        if capture is not None:
            capture.release()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            self._latest_bgr = None

    def snapshot(self) -> Frame:
        if self.capture is None:
            self.start()
        deadline = time.time() + 2.0
        minimum_timestamp = time.time() - 1.0
        bgr = None
        timestamp = 0.0
        read_error = None
        while time.time() < deadline:
            with self._lock:
                if self._latest_bgr is not None and self._latest_timestamp >= minimum_timestamp:
                    bgr = self._latest_bgr.copy()
                    timestamp = self._latest_timestamp
                    break
                read_error = self._read_error
            time.sleep(0.01)
        if bgr is None:
            raise RuntimeError(read_error or "Timed out waiting for OpenCV camera frame")
        self.sequence += 1
        rgb = bgr[:, :, ::-1].copy()
        return Frame(rgb=rgb, depth=None, sequence=self.sequence, timestamp=timestamp or time.time())

    def status(self) -> dict:
        return {
            "provider": self.name,
            "started": self.capture is not None,
            "device_index": self.config.device_index,
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
            "sequence": self.sequence,
            "depth": False,
            "backend": "v4l2",
            "latest_frame_age_s": round(max(0.0, time.time() - self._latest_timestamp), 3) if self._latest_timestamp else None,
            "read_failures": self._read_failures,
            "read_error": self._read_error,
            "reconnecting": self._reconnecting,
            "reconnects": self._reconnects,
            "open_label": self._open_label,
        }

    def get_settings(self) -> dict[str, Any]:
        controls = self._v4l2_controls()
        if not controls:
            controls = DEFAULT_CONTROLS
        values = self._v4l2_values(controls)
        return {
            "backend": "v4l2" if self._has_v4l2_ctl() else "opencv",
            "device": self._device_path(),
            "controls": [
                {
                    "name": control.name,
                    "label": control.label,
                    "kind": control.kind,
                    "min": control.minimum,
                    "max": control.maximum,
                    "step": control.step,
                    "default": control.default,
                    "value": values.get(control.name, control.default),
                    "options": control.options or {},
                }
                for control in controls.values()
            ],
        }

    def apply_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        if self.capture is None:
            self.start()
        controls = self._v4l2_controls() or DEFAULT_CONTROLS
        applied: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for name, raw_value in updates.items():
            control = controls.get(name)
            if control is None:
                errors[name] = "Unsupported camera control"
                continue
            try:
                value = _coerce_control_value(control, raw_value)
            except ValueError as exc:
                errors[name] = str(exc)
                continue
            if self._set_v4l2_control(name, value):
                applied[name] = value
            else:
                errors[name] = "Camera rejected setting"
        settings = self.get_settings()
        settings["applied"] = applied
        settings["errors"] = errors
        return settings

    def _reader_loop(self) -> None:
        assert self.capture is not None
        while not self._stop_event.is_set():
            with self._lock:
                capture = self.capture
            if capture is None:
                self._reconnect_capture()
                continue
            ok, bgr = capture.read()
            if not ok:
                with self._lock:
                    self._read_failures += 1
                    if self._read_failures >= 20:
                        self._read_error = "OpenCV camera read failed; reconnecting"
                if self._read_failures >= 20:
                    self._reconnect_capture()
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest_bgr = bgr
                self._latest_timestamp = time.time()
                self._read_error = None
                self._read_failures = 0

    def _device_path(self) -> str:
        return f"/dev/video{self.config.device_index}"

    def _open_capture(self):
        import cv2

        device_path = self._device_path()
        if not Path(device_path).exists():
            raise RuntimeError(f"OpenCV camera device does not exist: {device_path}")
        errors = []
        for target, backend, label in (
            (device_path, cv2.CAP_V4L2, f"{device_path} via V4L2"),
            (self.config.device_index, cv2.CAP_V4L2, f"index {self.config.device_index} via V4L2"),
            (self.config.device_index, cv2.CAP_ANY, f"index {self.config.device_index} via any backend"),
        ):
            capture = cv2.VideoCapture(target, backend)
            if capture.isOpened():
                return capture, label
            capture.release()
            errors.append(label)
        raise RuntimeError(f"OpenCV could not open camera {device_path}. Tried: {', '.join(errors)}")

    def _configure_capture(self, capture) -> None:
        import cv2

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if self.config.exposure >= 0:
            capture.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
        if self.config.gain >= 0:
            capture.set(cv2.CAP_PROP_GAIN, self.config.gain)

    def _reconnect_capture(self) -> None:
        with self._lock:
            if self._reconnecting or self._stop_event.is_set():
                return
            self._reconnecting = True
            old_capture = self.capture
            self.capture = None
            self._latest_bgr = None
        if old_capture is not None:
            old_capture.release()
        time.sleep(0.2)
        try:
            capture, label = self._open_capture()
            self._configure_capture(capture)
        except Exception as exc:
            with self._lock:
                self._read_error = f"OpenCV camera reconnect failed: {exc}"
                self._reconnecting = False
            return
        with self._lock:
            if self._stop_event.is_set():
                capture.release()
                self._reconnecting = False
                return
            self.capture = capture
            self._open_label = label
            self._read_error = None
            self._read_failures = 0
            self._reconnecting = False
            self._reconnects += 1

    @staticmethod
    def _has_v4l2_ctl() -> bool:
        return shutil.which("v4l2-ctl") is not None

    def _run_v4l2(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["v4l2-ctl", "-d", self._device_path(), *args],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )

    def _v4l2_controls(self) -> dict[str, CameraControl]:
        if not self._has_v4l2_ctl():
            return {}
        result = self._run_v4l2("--list-ctrls-menus")
        if result.returncode != 0:
            return {}
        controls: dict[str, CameraControl] = {}
        current_name: str | None = None
        menu_options: dict[int, str] = {}
        line_re = re.compile(
            r"^\s*(?P<name>[A-Za-z0-9_]+)\s+0x[0-9a-fA-F]+\s+\((?P<kind>[^)]+)\)\s+:\s+(?P<body>.*)$"
        )
        menu_re = re.compile(r"^\s*(?P<value>-?\d+):\s+(?P<label>.+)$")
        for line in result.stdout.splitlines():
            match = line_re.match(line)
            if match:
                if current_name and menu_options and current_name in controls:
                    old = controls[current_name]
                    controls[current_name] = CameraControl(
                        old.name, old.label, old.kind, old.minimum, old.maximum, old.step, old.default, dict(menu_options)
                    )
                    menu_options = {}
                name = match.group("name")
                if name not in CONTROL_LABELS:
                    current_name = None
                    continue
                body = match.group("body")
                kind = match.group("kind")
                minimum = _extract_int(body, "min")
                maximum = _extract_int(body, "max")
                step = _extract_int(body, "step") or 1
                default = _extract_int(body, "default")
                controls[name] = CameraControl(name, CONTROL_LABELS[name], kind, minimum, maximum, step, default)
                current_name = name
                continue
            menu_match = menu_re.match(line)
            if menu_match and current_name:
                menu_options[int(menu_match.group("value"))] = menu_match.group("label").strip()
        if current_name and menu_options and current_name in controls:
            old = controls[current_name]
            controls[current_name] = CameraControl(old.name, old.label, old.kind, old.minimum, old.maximum, old.step, old.default, dict(menu_options))
        return controls

    def _v4l2_values(self, controls: dict[str, CameraControl]) -> dict[str, int | None]:
        values: dict[str, int | None] = {name: control.default for name, control in controls.items()}
        if not self._has_v4l2_ctl():
            return values
        result = self._run_v4l2("--get-ctrl", ",".join(controls.keys()))
        if result.returncode != 0:
            return values
        for line in result.stdout.splitlines():
            if ":" not in line:
                continue
            name, raw_value = line.split(":", 1)
            name = name.strip()
            if name in values:
                values[name] = int(raw_value.strip())
        return values

    def _set_v4l2_control(self, name: str, value: int) -> bool:
        if not self._has_v4l2_ctl():
            return False
        result = self._run_v4l2("--set-ctrl", f"{name}={value}")
        return result.returncode == 0


def _extract_int(text: str, key: str) -> int | None:
    match = re.search(rf"{key}=(-?\d+)", text)
    return int(match.group(1)) if match else None


def _coerce_control_value(control: CameraControl, value: Any) -> int:
    if control.kind == "bool":
        coerced = 1 if value in {True, "true", "True", "1", 1} else 0
    else:
        coerced = int(float(value))
    if control.minimum is not None and coerced < control.minimum:
        raise ValueError(f"Value below minimum {control.minimum}")
    if control.maximum is not None and coerced > control.maximum:
        raise ValueError(f"Value above maximum {control.maximum}")
    return coerced


def select_camera_device_index(camera_kind: str, preferred_index: int | None = None) -> int:
    if preferred_index is not None and preferred_index >= 0:
        return preferred_index
    devices = list_video_devices()
    if not devices:
        raise RuntimeError("No /dev/video camera devices were found")
    scored = sorted(
        devices,
        key=lambda device: (_device_score(camera_kind, device), -device.index),
        reverse=True,
    )
    return scored[0].index


def list_video_devices() -> list[VideoDevice]:
    devices = _list_v4l2_devices()
    if devices:
        return devices
    found = []
    for path in sorted(Path("/dev").glob("video*"), key=lambda item: _video_index(item.name) or 999):
        index = _video_index(path.name)
        if index is not None:
            found.append(VideoDevice(index=index, path=str(path), label=path.name))
    return found


def _list_v4l2_devices() -> list[VideoDevice]:
    if shutil.which("v4l2-ctl") is None:
        return []
    result = subprocess.run(
        ["v4l2-ctl", "--list-devices"],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )
    if result.returncode != 0:
        return []
    devices: list[VideoDevice] = []
    label = ""
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            label = line.rstrip(":")
            continue
        path = line.strip()
        index = _video_index(Path(path).name)
        if index is not None:
            devices.append(VideoDevice(index=index, path=path, label=label))
    return devices


def _video_index(name: str) -> int | None:
    match = re.fullmatch(r"video(\d+)", name)
    return int(match.group(1)) if match else None


def _device_score(camera_kind: str, device: VideoDevice) -> int:
    label = device.label.lower()
    astra_terms = ("astra", "orbbec", "gemini", "depth")
    global_terms = ("global", "shutter", "arducam", "flir", "basler", "imx", "machine vision")
    if camera_kind == "astra":
        if any(term in label for term in astra_terms):
            return 100
        if any(term in label for term in global_terms):
            return 10
        return 50
    if any(term in label for term in global_terms):
        return 100
    if any(term in label for term in astra_terms):
        return 10
    return 60
