from __future__ import annotations

import json
import re
from pathlib import Path

from robot_vision.inspection.calibration import CalibrationProfile


class CalibrationStore:
    def __init__(self, calibration_dir: Path):
        self.calibration_dir = calibration_dir
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

    def load(self, name: str = "default") -> CalibrationProfile:
        path = self._path(name)
        if not path.exists():
            profile = CalibrationProfile(name=name)
            self.save(profile)
            return profile
        with path.open("r", encoding="utf-8") as handle:
            return CalibrationProfile.from_dict(json.load(handle))

    def save(self, profile: CalibrationProfile) -> Path:
        path = self._path(profile.name)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(profile.to_dict(), handle, indent=2)
        return path

    def _path(self, name: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip() or "default")
        return self.calibration_dir / f"{safe}.json"
