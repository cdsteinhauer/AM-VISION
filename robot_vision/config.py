from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CameraConfig:
    provider: str = "mock"
    width: int = 1280
    height: int = 720
    fps: int = 15
    device_index: int = 0
    auto_exposure: bool = True
    exposure: int = -1
    gain: int = -1
    depth_enabled: bool = True


@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    ros_domain_id: str = "77"
    data_dir: Path = PROJECT_ROOT / "data"
    camera: CameraConfig = CameraConfig()
    auto_trigger_roi: tuple[float, float, float, float] = (0.25, 0.25, 0.5, 0.5)
    auto_trigger_min_change: float = 12.0
    auto_trigger_debounce_s: float = 2.0


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "app.yaml"
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    config_root = path.parent.parent if path.parent.name == "config" else path.parent
    raw = _read_yaml(path)
    camera_raw = raw.get("camera", {}) or {}
    camera = CameraConfig(
        provider=str(camera_raw.get("provider", "mock")),
        width=int(camera_raw.get("width", 1280)),
        height=int(camera_raw.get("height", 720)),
        fps=int(camera_raw.get("fps", 15)),
        device_index=int(camera_raw.get("device_index", 0)),
        auto_exposure=bool(camera_raw.get("auto_exposure", True)),
        exposure=int(camera_raw.get("exposure", -1)),
        gain=int(camera_raw.get("gain", -1)),
        depth_enabled=bool(camera_raw.get("depth_enabled", True)),
    )

    trigger_raw = raw.get("auto_trigger", {}) or {}
    roi = trigger_raw.get("roi", [0.25, 0.25, 0.5, 0.5])
    if len(roi) != 4:
        raise ValueError("auto_trigger.roi must be [x, y, width, height] normalized 0..1 values")

    data_dir = Path(raw.get("data_dir", PROJECT_ROOT / "data"))
    if not data_dir.is_absolute():
        data_dir = config_root / data_dir

    return AppConfig(
        host=str(raw.get("host", "0.0.0.0")),
        port=int(raw.get("port", 8080)),
        ros_domain_id=str(os.environ.get("ROS_DOMAIN_ID", raw.get("ros_domain_id", "77"))),
        data_dir=data_dir,
        camera=camera,
        auto_trigger_roi=tuple(float(v) for v in roi),
        auto_trigger_min_change=float(trigger_raw.get("min_change", 12.0)),
        auto_trigger_debounce_s=float(trigger_raw.get("debounce_s", 2.0)),
    )
