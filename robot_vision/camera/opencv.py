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

    def start(self) -> None:
        import cv2

        if self.capture is not None:
            return
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
                self.capture = capture
                break
            capture.release()
            errors.append(label)
        if self.capture is None:
            raise RuntimeError(
                f"OpenCV could not open camera {device_path}. Tried: {', '.join(errors)}"
            )
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if self.config.exposure >= 0:
            self.capture.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
        if self.config.gain >= 0:
            self.capture.set(cv2.CAP_PROP_GAIN, self.config.gain)
        self._stop_event.clear()
        self._read_error = None
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        with self._lock:
            self._latest_bgr = None

    def snapshot(self) -> Frame:
        if self.capture is None:
            self.start()
        deadline = time.time() + 2.0
        bgr = None
        timestamp = 0.0
        while time.time() < deadline:
            with self._lock:
                if self._latest_bgr is not None:
                    bgr = self._latest_bgr.copy()
                    timestamp = self._latest_timestamp
                    break
                read_error = self._read_error
            if read_error:
                raise RuntimeError(read_error)
            time.sleep(0.01)
        if bgr is None:
            raise RuntimeError("Timed out waiting for OpenCV camera frame")
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
                break
            ok, bgr = capture.read()
            if not ok:
                with self._lock:
                    self._read_error = "OpenCV camera read failed"
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest_bgr = bgr
                self._latest_timestamp = time.time()
                self._read_error = None

    def _device_path(self) -> str:
        return f"/dev/video{self.config.device_index}"

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
