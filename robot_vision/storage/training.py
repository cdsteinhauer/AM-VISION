from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class TrainingCaptureStore:
    def __init__(self, training_dir: Path):
        self.training_dir = training_dir
        self.training_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[dict[str, Any]]:
        datasets = []
        for folder in sorted(self.training_dir.iterdir()):
            if not folder.is_dir():
                continue
            counts = self.count_samples(folder.name)
            datasets.append({"name": folder.name, "counts": counts, "total": sum(counts.values())})
        return datasets

    def count_samples(self, dataset: str) -> dict[str, int]:
        folder = self._dataset_path(dataset)
        return {
            "PASS": len(list((folder / "PASS").glob("*.png"))) if (folder / "PASS").exists() else 0,
            "FAIL": len(list((folder / "FAIL").glob("*.png"))) if (folder / "FAIL").exists() else 0,
        }

    def save_sample(self, dataset: str, label: str, image: np.ndarray) -> dict[str, Any]:
        normalized_label = _normalize_label(label)
        folder = self._dataset_path(dataset) / normalized_label
        folder.mkdir(parents=True, exist_ok=True)
        sample_id = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1000000:06d}"
        path = folder / f"{sample_id}.png"
        Image.fromarray(_uint8_rgb(image), mode="RGB").save(path)
        payload = {
            "dataset": self._safe_name(dataset),
            "label": normalized_label,
            "sample_id": sample_id,
            "path": str(path),
            "counts": self.count_samples(dataset),
        }
        with (path.with_suffix(".json")).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def delete_dataset(self, dataset: str) -> bool:
        import shutil

        folder = self._dataset_path(dataset)
        if not folder.exists():
            return False
        shutil.rmtree(folder)
        return True

    def _dataset_path(self, dataset: str) -> Path:
        return self.training_dir / self._safe_name(dataset)

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip() or "default")


def _normalize_label(label: str) -> str:
    value = label.upper().strip()
    if value not in {"PASS", "FAIL"}:
        raise ValueError("label must be PASS or FAIL")
    return value


def _uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)
    return image

