from __future__ import annotations

import json
import random
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LABELS = {False: "FAIL", True: "PASS"}
LABEL_TO_ID = {"FAIL": 0, "PASS": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
TRAINING_DEPENDENCIES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "accelerate": "accelerate",
}


@dataclass(frozen=True)
class ReportSample:
    image_path: Path
    label: int
    report_id: str
    recipe: str


_MODEL_CACHE: dict[str, tuple[Any, Any, Any]] = {}


def collect_report_samples(report_dir: str | Path, recipe: str | None = None) -> list[ReportSample]:
    root = Path(report_dir)
    samples: list[ReportSample] = []
    for result_path in sorted(root.glob("*/result.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        result = payload.get("result", {})
        sample_recipe = str(result.get("recipe", ""))
        if recipe and sample_recipe != recipe:
            continue

        passed = bool(result.get("passed", False))
        image_path = _resolve_image_path(payload, result_path.parent)
        if image_path is None or not image_path.exists():
            continue

        samples.append(
            ReportSample(
                image_path=image_path,
                label=LABEL_TO_ID[LABELS[passed]],
                report_id=str(payload.get("id", result_path.parent.name)),
                recipe=sample_recipe,
            )
        )
    return samples


def collect_capture_samples(dataset_dir: str | Path, recipe: str | None = None) -> list[ReportSample]:
    root = Path(dataset_dir)
    samples: list[ReportSample] = []
    for label_name, label_id in LABEL_TO_ID.items():
        label_dir = root / label_name
        if not label_dir.exists():
            continue
        for image_path in sorted(label_dir.glob("*.png")):
            samples.append(
                ReportSample(
                    image_path=image_path,
                    label=label_id,
                    report_id=image_path.stem,
                    recipe=recipe or root.name,
                )
            )
    return samples


def split_samples(
    samples: list[ReportSample],
    validation_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[list[ReportSample], list[ReportSample]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be >= 0 and < 1")
    rng = random.Random(seed)
    by_label: dict[int, list[ReportSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    train: list[ReportSample] = []
    validation: list[ReportSample] = []
    for label_samples in by_label.values():
        shuffled = list(label_samples)
        rng.shuffle(shuffled)
        validation_count = int(round(len(shuffled) * validation_fraction))
        if validation_fraction > 0 and len(shuffled) > 1:
            validation_count = max(1, min(len(shuffled) - 1, validation_count))
        validation.extend(shuffled[:validation_count])
        train.extend(shuffled[validation_count:])

    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def train_from_reports(
    report_dir: str | Path,
    output_dir: str | Path,
    model_checkpoint: str = "microsoft/resnet-18",
    recipe: str | None = None,
    validation_fraction: float = 0.2,
    seed: int = 7,
    epochs: float = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
) -> dict[str, Any]:
    samples = collect_report_samples(report_dir, recipe=recipe)
    return train_from_samples(
        samples=samples,
        output_dir=output_dir,
        model_checkpoint=model_checkpoint,
        recipe=recipe,
        validation_fraction=validation_fraction,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def train_from_capture_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_checkpoint: str = "microsoft/resnet-18",
    validation_fraction: float = 0.2,
    seed: int = 7,
    epochs: float = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    samples = collect_capture_samples(dataset_path, recipe=dataset_path.name)
    return train_from_samples(
        samples=samples,
        output_dir=output_dir,
        model_checkpoint=model_checkpoint,
        recipe=dataset_path.name,
        validation_fraction=validation_fraction,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def train_from_samples(
    samples: list[ReportSample],
    output_dir: str | Path,
    model_checkpoint: str = "microsoft/resnet-18",
    recipe: str | None = None,
    validation_fraction: float = 0.2,
    seed: int = 7,
    epochs: float = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
) -> dict[str, Any]:
    dependency_status = check_training_dependencies()
    if dependency_status["missing"]:
        missing = ", ".join(dependency_status["missing"])
        raise RuntimeError(
            f"Hugging Face training dependencies are missing: {missing}. "
            f"Install with: {dependency_status['install_command']}"
        )

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face training dependencies are missing. Install with: "
            'python -m pip install -e ".[train]"'
        ) from exc

    _validate_training_samples(samples)
    train_samples, validation_samples = split_samples(samples, validation_fraction, seed)
    if not train_samples:
        raise ValueError("No training samples remain after validation split")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    train_dataset = ReportImageDataset(train_samples, image_processor, torch)
    validation_dataset = ReportImageDataset(validation_samples, image_processor, torch) if validation_samples else None

    training_args = _training_args(
        TrainingArguments,
        output_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        evaluate=validation_dataset is not None,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "compute_metrics": _compute_metrics if validation_dataset is not None else None,
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=image_processor)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=image_processor)

    trainer.train()
    metrics = trainer.evaluate() if validation_dataset is not None else {}
    trainer.save_model(output_path)
    image_processor.save_pretrained(output_path)

    manifest = {
        "model_checkpoint": model_checkpoint,
        "output_dir": str(output_path),
        "recipe": recipe,
        "labels": ID_TO_LABEL,
        "train_count": len(train_samples),
        "validation_count": len(validation_samples),
        "metrics": metrics,
        "samples": [
            {"report_id": sample.report_id, "image_path": str(sample.image_path), "label": ID_TO_LABEL[sample.label]}
            for sample in samples
        ],
    }
    with (output_path / "robot_vision_training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def predict_image(model_dir: str | Path, image: np.ndarray | Image.Image) -> dict[str, Any]:
    dependency_status = check_training_dependencies()
    if dependency_status["missing"]:
        missing = ", ".join(dependency_status["missing"])
        raise RuntimeError(f"Hugging Face inference dependencies are missing: {missing}")

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model_path = str(Path(model_dir))
    if model_path not in _MODEL_CACHE:
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
        model.eval()
        _MODEL_CACHE[model_path] = (processor, model, torch)
    processor, model, torch_module = _MODEL_CACHE[model_path]

    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(_uint8_rgb(image), mode="RGB")
    else:
        pil_image = image.convert("RGB")
    encoded = processor(images=pil_image, return_tensors="pt")
    with torch_module.no_grad():
        outputs = model(**encoded)
        probabilities = torch_module.softmax(outputs.logits, dim=1)[0]
    scores = {model.config.id2label[index]: float(score) for index, score in enumerate(probabilities)}
    label = max(scores, key=scores.get)
    return {"label": label, "score": scores[label], "scores": scores}


def check_training_dependencies() -> dict[str, Any]:
    missing = [
        package_name
        for module_name, package_name in TRAINING_DEPENDENCIES.items()
        if find_spec(module_name) is None
    ]
    return {
        "ok": not missing,
        "missing": missing,
        "install_command": 'python3 -m pip install --user -e ".[train]"',
    }


class ReportImageDataset:
    def __init__(self, samples: list[ReportSample], image_processor: Any, torch_module: Any):
        self.samples = samples
        self.image_processor = image_processor
        self.torch = torch_module

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        encoded = self.image_processor(images=image, return_tensors="pt")
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = self.torch.tensor(sample.label, dtype=self.torch.long)
        return item


def _resolve_image_path(payload: dict[str, Any], report_folder: Path) -> Path | None:
    files = payload.get("files", {})
    raw_path = files.get("rgb")
    if raw_path:
        path = Path(raw_path)
        if path.exists():
            return path
    fallback = report_folder / "rgb.png"
    return fallback if fallback.exists() else None


def _uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)
    return image


def _validate_training_samples(samples: list[ReportSample]) -> None:
    if len(samples) < 2:
        raise ValueError("At least two report samples are required for training")
    labels = {sample.label for sample in samples}
    if labels != set(ID_TO_LABEL):
        found = ", ".join(ID_TO_LABEL[label] for label in sorted(labels)) or "none"
        raise ValueError(f"Training requires both PASS and FAIL samples; found: {found}")


def _training_args(
    training_arguments_cls: Any,
    output_path: Path,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    seed: int,
    evaluate: bool,
) -> Any:
    base_args = {
        "output_dir": str(output_path),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_strategy": "epoch",
        "logging_steps": 10,
        "remove_unused_columns": False,
        "seed": seed,
        "report_to": "none",
    }
    if evaluate:
        base_args["eval_strategy"] = "epoch"
    else:
        base_args["eval_strategy"] = "no"
    try:
        return training_arguments_cls(**base_args)
    except TypeError:
        base_args["evaluation_strategy"] = base_args.pop("eval_strategy")
        return training_arguments_cls(**base_args)


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": float(np.mean(predictions == labels))}
