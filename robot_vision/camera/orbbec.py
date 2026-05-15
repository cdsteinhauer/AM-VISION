from __future__ import annotations

from robot_vision.camera.mock import MockCamera
from robot_vision.config import CameraConfig


class OrbbecCamera(MockCamera):
    name = "orbbec"

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._error = (
            "Real Orbbec/Astra capture is not enabled in this build. Install the vendor SDK "
            "on the Jetson, then replace robot_vision.camera.orbbec.OrbbecCamera with the "
            "SDK-backed pipeline. Mock mode remains available for UI and inspection testing."
        )

    def start(self) -> None:
        try:
            import pyorbbecsdk  # noqa: F401
        except Exception as exc:
            raise RuntimeError(self._error) from exc
        raise RuntimeError(
            "pyorbbecsdk is importable, but the SDK-backed stream adapter still needs to be wired "
            "for this specific Jetson/Astra install."
        )
