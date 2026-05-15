from __future__ import annotations

from pathlib import Path


class DataPaths:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.recipes = data_dir / "recipes"
        self.reports = data_dir / "reports"
        self.calibration = data_dir / "calibration"
        self.training = data_dir / "training"
        self.models = data_dir / "models"

    def ensure(self) -> None:
        self.recipes.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.calibration.mkdir(parents=True, exist_ok=True)
        self.training.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
