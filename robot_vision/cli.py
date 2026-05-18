from __future__ import annotations

import argparse
import json
import sys

from robot_vision.camera import create_camera
from robot_vision.config import load_config
from robot_vision.inspection.calibration import CalibrationProfile
from robot_vision.inspection.engine import InspectionEngine
from robot_vision.inspection.models import InspectionRecipe


def web() -> None:
    import uvicorn

    from robot_vision.web.app import create_app

    parser = argparse.ArgumentParser(description="Run the Robot Vision browser service.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Force mock camera provider.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mock:
        config = _replace_camera_provider(config, "mock")
    app = create_app(config)
    uvicorn.run(app, host=args.host or config.host, port=args.port or config.port)


def camera_check() -> None:
    parser = argparse.ArgumentParser(description="Check camera capture path.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mock:
        config = _replace_camera_provider(config, "mock")
    camera = create_camera(config.camera)
    frame = camera.snapshot()
    print(json.dumps({
        "ok": True,
        "status": camera.status(),
        "shape": list(frame.rgb.shape),
        "depth": frame.depth is not None,
        "sequence": frame.sequence,
    }, indent=2))


def inspect_sample() -> None:
    config = _replace_camera_provider(load_config(), "mock")
    frame = create_camera(config.camera).snapshot()
    result = InspectionEngine().inspect(frame.rgb, InspectionRecipe.default(), CalibrationProfile(), frame.depth)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        sys.exit(1)


def train_vision() -> None:
    from robot_vision.config import PROJECT_ROOT
    from robot_vision.training.hf_vision import train_from_capture_dataset, train_from_reports

    parser = argparse.ArgumentParser(description="Train a Hugging Face PASS/FAIL image classifier from saved reports.")
    parser.add_argument("--source", choices=["reports", "captures"], default="reports")
    parser.add_argument("--reports", default=str(PROJECT_ROOT / "data" / "reports"))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "models" / "pass_fail_classifier"))
    parser.add_argument("--model", default="microsoft/resnet-18")
    parser.add_argument("--recipe", default=None, help="Only train on reports from this recipe name.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.source == "captures":
        if not args.dataset_dir:
            raise SystemExit("--dataset-dir is required for --source captures")
        manifest = train_from_capture_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output,
            model_checkpoint=args.model,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    else:
        manifest = train_from_reports(
            report_dir=args.reports,
            output_dir=args.output,
            model_checkpoint=args.model,
            recipe=args.recipe,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    print(json.dumps(manifest, indent=2))


def _replace_camera_provider(config, provider: str):
    from dataclasses import replace

    return replace(config, camera=replace(config.camera, provider=provider))
