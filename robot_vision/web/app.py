from __future__ import annotations

import json
import os
import subprocess
import sys
import shutil
import time
import yaml
from dataclasses import replace
from pathlib import Path
from threading import Thread
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from robot_vision.camera import create_camera
from robot_vision.camera.base import depth_to_display, encode_jpeg_base64, encode_png_base64
from robot_vision.camera.opencv import list_video_devices, select_camera_device_index
from robot_vision.config import AppConfig, PROJECT_ROOT, load_config
from robot_vision.inspection.calibration import CalibrationProfile, DepthReference
from robot_vision.inspection.engine import InspectionEngine, debug_candidate_lines_in_roi, detect_rectangle_in_roi
from robot_vision.inspection.models import InspectionRecipe, InspectionTool
from robot_vision.inspection.trigger import PresenceTrigger
from robot_vision.storage.calibration import CalibrationStore
from robot_vision.storage.paths import DataPaths
from robot_vision.storage.recipes import RecipeStore
from robot_vision.storage.reports import ReportStore
from robot_vision.storage.training import TrainingCaptureStore
from robot_vision.training.hf_vision import ID_TO_LABEL, check_training_dependencies, collect_capture_samples, collect_report_samples


class InspectRequest(BaseModel):
    recipe_name: str = "default"
    calibration_name: str = "default"
    save_report: bool = True


class RecipePayload(BaseModel):
    recipe: dict[str, Any]


class CalibrationPayload(BaseModel):
    name: str = "default"
    pixel_width: float = Field(gt=0)
    pixel_height: float = Field(gt=0)
    real_width_mm: float = Field(gt=0)
    real_height_mm: float = Field(gt=0)


class CalibrationDetectPayload(BaseModel):
    name: str = "default"
    roi: list[float] = Field(min_length=4, max_length=4)
    real_width_mm: float = Field(gt=0)
    real_height_mm: float = Field(gt=0)


class DepthReferencePayload(BaseModel):
    name: str = "default"
    roi: list[float] = Field(min_length=4, max_length=4)


class DepthAlignmentPayload(BaseModel):
    name: str = "default"
    offset_x_px: float = 0.0
    offset_y_px: float = 0.0


class CameraSettingsPayload(BaseModel):
    settings: dict[str, Any]


class CameraModePayload(BaseModel):
    mode: str = Field(pattern="^(orbbec_femto|astra|global_shutter)$")
    device_index: int = -1


class LineDetectPayload(BaseModel):
    tools: list[dict[str, Any]] = Field(default_factory=list)


class TrainingStartPayload(BaseModel):
    recipe: str | None = None
    dataset: str | None = None
    source: str = "reports"
    model: str = "microsoft/resnet-18"
    output_dir: str = "data/models/pass_fail_classifier"
    validation_fraction: float = Field(default=0.2, ge=0, lt=1)
    epochs: float = Field(default=5, gt=0)
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=5e-5, gt=0)
    seed: int = 7


class TrainingCapturePayload(BaseModel):
    dataset: str = Field(min_length=1)
    label: str = Field(pattern="^(PASS|FAIL|pass|fail)$")


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    paths = DataPaths(cfg.data_dir)
    paths.ensure()

    camera_config = _resolve_auto_camera_config(cfg.camera)
    if _camera_needs_astra_driver(camera_config):
        _ensure_astra_ros_driver(cfg.data_dir)
    camera = create_camera(camera_config)
    camera_lock = Lock()
    recipes = RecipeStore(paths.recipes)
    calibration = CalibrationStore(paths.calibration)
    reports = ReportStore(paths.reports)
    training_captures = TrainingCaptureStore(paths.training)
    engine = InspectionEngine()
    trigger = PresenceTrigger(cfg.auto_trigger_roi, cfg.auto_trigger_min_change, cfg.auto_trigger_debounce_s)
    training_lock = Lock()
    training_status: dict[str, Any] = {
        "state": "idle",
        "message": "No training job has run.",
        "running": False,
        "manifest": None,
        "error": None,
        "log_path": str(cfg.data_dir / "logs" / "hf_training.log"),
        "returncode": None,
    }

    app = FastAPI(title="Robot Vision", version="0.1.0")
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(
            static_dir / "index.html",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.middleware("http")
    async def no_cache_static(request, call_next):
        response: Response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "app": "robot_vision",
            "ros_domain_id": cfg.ros_domain_id,
            "camera_provider": camera_config.provider,
        }

    @app.post("/api/app/restart")
    def restart_app():
        _restart_process_soon()
        return {"restarting": True}

    @app.get("/api/camera/status")
    def camera_status():
        try:
            status = camera.status()
            status["available_devices"] = [
                {"index": device.index, "path": device.path, "label": device.label}
                for device in list_video_devices()
            ]
            return status
        except Exception as exc:
            return {"provider": camera_config.provider, "started": False, "error": str(exc)}

    @app.get("/api/camera/mode")
    def camera_mode():
        return {
            "mode": _camera_mode_from_config(camera_config),
            "provider": camera_config.provider,
            "device_index": camera_config.device_index,
            "depth_enabled": camera_config.depth_enabled,
        }

    @app.post("/api/camera/mode")
    def update_camera_mode(payload: CameraModePayload):
        nonlocal camera, camera_config
        try:
            preferred_index = payload.device_index if payload.device_index >= 0 else None
            if camera_config.provider.lower().strip() != "mock" and payload.mode == _camera_mode_from_config(camera_config) and (
                preferred_index is None or preferred_index == camera_config.device_index
            ):
                return {
                    "mode": _camera_mode_from_config(camera_config),
                    "provider": camera_config.provider,
                    "device_index": camera_config.device_index,
                    "depth_enabled": camera_config.depth_enabled,
                    "restart_required": False,
                }
            if payload.mode == "global_shutter":
                device_index = select_camera_device_index("global_shutter", preferred_index)
                next_config = replace(
                    camera_config,
                    provider="opencv",
                    device_index=device_index,
                    width=1280,
                    height=720,
                    fps=120,
                    auto_exposure=True,
                    exposure=-1,
                    gain=0,
                    depth_enabled=False,
                )
            elif payload.mode == "astra":
                device_index = select_camera_device_index("astra", preferred_index)
                next_config = replace(
                    camera_config,
                    provider="astra_hybrid",
                    device_index=device_index,
                    width=1280,
                    height=720,
                    fps=15,
                    auto_exposure=True,
                    exposure=-1,
                    gain=-1,
                    depth_enabled=True,
                )
            else:
                next_config = replace(
                    camera_config,
                    provider="orbbec",
                    device_index=-1,
                    width=1280,
                    height=720,
                    fps=15,
                    auto_exposure=True,
                    exposure=-1,
                    gain=-1,
                    depth_enabled=True,
                )
            restart_required = _hardware_restart_required(camera_config, next_config)
            with camera_lock:
                if hasattr(camera, "stop"):
                    camera.stop()
                time.sleep(0.25)
                if restart_required:
                    _persist_runtime_camera_config(next_config)
                    if not _camera_needs_astra_driver(next_config):
                        _stop_astra_ros_driver()
                    camera_config = next_config
                    _restart_process_soon()
                    return {
                        "mode": _camera_mode_from_config(camera_config),
                        "provider": camera_config.provider,
                        "device_index": camera_config.device_index,
                        "depth_enabled": camera_config.depth_enabled,
                        "restart_required": True,
                    }
                if _camera_needs_astra_driver(next_config):
                    _ensure_astra_ros_driver(cfg.data_dir)
                else:
                    _stop_astra_ros_driver()
                camera_config = next_config
                if restart_required:
                    _restart_process_soon()
                else:
                    camera = create_camera(camera_config)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "mode": _camera_mode_from_config(camera_config),
            "provider": camera_config.provider,
            "device_index": camera_config.device_index,
            "depth_enabled": camera_config.depth_enabled,
            "restart_required": False,
        }

    @app.get("/api/camera/settings")
    def camera_settings():
        if not hasattr(camera, "get_settings"):
            raise HTTPException(status_code=501, detail="Camera settings are not supported by this provider")
        try:
            return camera.get_settings()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/camera/settings")
    def update_camera_settings(payload: CameraSettingsPayload):
        if not hasattr(camera, "apply_settings"):
            raise HTTPException(status_code=501, detail="Camera settings are not supported by this provider")
        try:
            with camera_lock:
                return camera.apply_settings(payload.settings)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/camera/snapshot")
    def snapshot():
        frame = _snapshot_or_error(camera, camera_lock)
        depth_display = depth_to_display(frame.depth)
        trigger_result = trigger.update(frame.rgb, frame.depth)
        return {
            "sequence": frame.sequence,
            "timestamp": frame.timestamp,
            "rgb_png": encode_png_base64(frame.rgb),
            "depth_png": encode_png_base64(depth_display) if depth_display is not None else None,
            "auto_trigger": trigger_result,
        }

    @app.get("/api/camera/preview")
    def camera_preview(
        view: str = Query(default="rgb", pattern="^(rgb|depth)$"),
        process_trigger: bool = False,
    ):
        frame = _snapshot_or_error(camera, camera_lock)
        trigger_result = trigger.update(frame.rgb, frame.depth) if process_trigger else {"fired": False, "score": 0.0}
        depth_display = depth_to_display(frame.depth) if view == "depth" else None
        return {
            "sequence": frame.sequence,
            "timestamp": frame.timestamp,
            "rgb_jpg": encode_jpeg_base64(frame.rgb),
            "depth_jpg": encode_jpeg_base64(depth_display) if depth_display is not None else None,
            "auto_trigger": trigger_result,
        }

    @app.post("/api/camera/line-detect")
    def camera_line_detect(payload: LineDetectPayload):
        frame = _snapshot_or_error(camera, camera_lock)
        tools = []
        for item in payload.tools:
            tool = InspectionTool.from_dict(item)
            if not tool.enabled or not tool.live_lines or tool.type not in {"edge_1", "edge_2"}:
                continue
            detection = debug_candidate_lines_in_roi(
                frame.rgb,
                tool.roi,
                tool.line_orientation,
                tool.min_edge_score,
                tool.min_line_length_ratio,
                limit=8,
            )
            tools.append({
                "tool_id": tool.id,
                "name": tool.name,
                "type": tool.type,
                "search_area_px": detection["search_area_px"],
                "lines": detection["lines"],
            })
        return {
            "sequence": frame.sequence,
            "timestamp": frame.timestamp,
            "width": int(frame.rgb.shape[1]),
            "height": int(frame.rgb.shape[0]),
            "tools": tools,
        }

    @app.get("/api/recipes")
    def list_recipes():
        return {"recipes": recipes.list_names()}

    @app.get("/api/recipes/{name}")
    def load_recipe(name: str):
        try:
            return {"recipe": recipes.load(name).to_dict()}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/recipes")
    def save_recipe(payload: RecipePayload):
        recipe = InspectionRecipe.from_dict(payload.recipe)
        path = recipes.save(recipe)
        return {"saved": True, "path": str(path), "recipe": recipe.to_dict()}

    @app.delete("/api/recipes/{name}")
    def delete_recipe(name: str):
        try:
            path = recipes.delete(name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"deleted": True, "path": str(path), "recipes": recipes.list_names()}

    @app.post("/api/calibration/run")
    def run_calibration(payload: CalibrationPayload):
        profile = CalibrationProfile.from_reference(
            payload.pixel_width,
            payload.pixel_height,
            payload.real_width_mm,
            payload.real_height_mm,
            name=payload.name,
        )
        path = calibration.save(profile)
        return {"saved": True, "path": str(path), "calibration": profile.to_dict()}

    @app.get("/api/calibration/{name}")
    def load_calibration(name: str):
        try:
            return {"calibration": calibration.load(name).to_dict()}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/calibration/detect")
    def detect_calibration(payload: CalibrationDetectPayload):
        frame = _snapshot_or_error(camera, camera_lock)
        roi = tuple(float(v) for v in payload.roi)
        detection = detect_rectangle_in_roi(frame.rgb, roi)
        if detection is None:
            raise HTTPException(status_code=422, detail="No calibration rectangle found inside detection area")

        profile = CalibrationProfile.from_reference(
            detection.width_px,
            detection.height_px,
            payload.real_width_mm,
            payload.real_height_mm,
            name=payload.name,
        )
        path = calibration.save(profile)
        return {
            "saved": True,
            "path": str(path),
            "calibration": profile.to_dict(),
            "detection": {
                "bbox_px": list(detection.bbox_px),
                "width_px": detection.width_px,
                "height_px": detection.height_px,
                "search_area_px": list(detection.search_area_px),
            },
            "rgb_png": encode_png_base64(frame.rgb),
            "depth_png": encode_png_base64(depth_to_display(frame.depth)) if frame.depth is not None else None,
        }

    @app.post("/api/calibration/depth-reference")
    def capture_depth_reference(payload: DepthReferencePayload):
        frame = _snapshot_or_error(camera, camera_lock)
        if frame.depth is None:
            raise HTTPException(status_code=422, detail="No depth frame available for depth reference")
        roi = tuple(float(v) for v in payload.roi)
        try:
            profile = calibration.load(payload.name)
            profile.depth_reference = DepthReference.from_depth(frame.depth, roi)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        path = calibration.save(profile)
        return {
            "saved": True,
            "path": str(path),
            "calibration": profile.to_dict(),
            "depth_reference": profile.depth_reference.to_dict() if profile.depth_reference is not None else None,
            "rgb_png": encode_png_base64(frame.rgb),
            "depth_png": encode_png_base64(depth_to_display(frame.depth)) if frame.depth is not None else None,
        }

    @app.post("/api/calibration/depth-alignment")
    def save_depth_alignment(payload: DepthAlignmentPayload):
        try:
            profile = calibration.load(payload.name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        profile.depth_rgb_offset_x_px = payload.offset_x_px
        profile.depth_rgb_offset_y_px = payload.offset_y_px
        path = calibration.save(profile)
        return {"saved": True, "path": str(path), "calibration": profile.to_dict()}

    @app.post("/api/inspect")
    def inspect(payload: InspectRequest):
        frame = _snapshot_or_error(camera, camera_lock)
        try:
            recipe = recipes.load(payload.recipe_name)
            profile = calibration.load(payload.calibration_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        result = engine.inspect(frame.rgb, recipe, profile, frame.depth)
        report = None
        if payload.save_report:
            report = reports.save(frame.rgb, frame.depth, result, recipe.to_dict(), profile.to_dict())
        return {
            "sequence": frame.sequence,
            "result": result,
            "report": report,
            "rgb_png": encode_png_base64(frame.rgb),
            "depth_png": encode_png_base64(depth_to_display(frame.depth)) if frame.depth is not None else None,
        }

    @app.get("/api/reports")
    def list_reports():
        return {"reports": reports.list_reports()}

    @app.get("/api/reports/{report_id}")
    def load_report(report_id: str):
        try:
            return reports.load(report_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/reports")
    def delete_reports(confirm: str = ""):
        if confirm != "DELETE":
            raise HTTPException(status_code=400, detail='Pass confirm=DELETE to delete all report samples')
        deleted = reports.delete_all()
        return {"deleted": deleted}

    @app.get("/api/training/samples")
    def training_samples(recipe: str | None = None, dataset: str | None = None, source: str = "reports"):
        if source == "captures":
            if dataset:
                samples = collect_capture_samples(training_captures._dataset_path(dataset))
            else:
                samples = []
                for folder in sorted(paths.training.iterdir()):
                    if not folder.is_dir():
                        continue
                    dataset_samples = collect_capture_samples(folder)
                    samples.extend(dataset_samples)
        else:
            samples = collect_report_samples(paths.reports, recipe=recipe or None)
        counts = {label: 0 for label in ID_TO_LABEL.values()}
        recipes_seen: dict[str, int] = {}
        for sample in samples:
            counts[ID_TO_LABEL[sample.label]] += 1
            recipes_seen[sample.recipe] = recipes_seen.get(sample.recipe, 0) + 1
        return {
            "total": len(samples),
            "counts": counts,
            "recipes": recipes_seen,
            "ready": counts["PASS"] > 0 and counts["FAIL"] > 0,
        }

    @app.get("/api/training/folder")
    def training_folder():
        return {"path": str(paths.training)}

    @app.post("/api/training/open-folder")
    def open_training_folder(dataset: str | None = None):
        target = paths.training
        if dataset:
            target = training_captures._dataset_path(dataset)
        if not target.exists():
            target = paths.training
        _open_path_in_file_manager(target)
        return {"path": str(target)}

    @app.delete("/api/training/captures")
    def delete_training_captures(confirm: str = "", dataset: str | None = None):
        if confirm != "DELETE":
            raise HTTPException(status_code=400, detail='Pass confirm=DELETE to delete training captures')
        if dataset:
            safe_name = training_captures._safe_name(dataset)
            folder = training_captures._dataset_path(safe_name)
            if not folder.exists():
                return {"deleted": 0, "scope": "dataset", "dataset": safe_name, "path": str(folder)}
            deleted = 0
            for file in folder.rglob("*"):
                if file.is_file():
                    deleted += 1
            shutil.rmtree(folder)
            return {"deleted": deleted, "scope": "dataset", "dataset": safe_name, "path": str(folder)}
        total_deleted = 0
        deleted_datasets: list[str] = []
        for folder in sorted(paths.training.iterdir()):
            if not folder.is_dir():
                continue
            for file in folder.rglob("*"):
                if file.is_file():
                    total_deleted += 1
            shutil.rmtree(folder)
            deleted_datasets.append(folder.name)
        return {"deleted": total_deleted, "scope": "all", "datasets": deleted_datasets, "path": str(paths.training)}

    @app.get("/api/training/datasets")
    def list_training_datasets():
        return {"datasets": training_captures.list_datasets()}

    @app.post("/api/training/capture")
    def capture_training_sample(payload: TrainingCapturePayload):
        frame = _snapshot_or_error(camera, camera_lock)
        sample = training_captures.save_sample(payload.dataset, payload.label, frame.rgb)
        return {
            "saved": True,
            "sample": sample,
            "rgb_png": encode_png_base64(frame.rgb),
            "depth_png": encode_png_base64(depth_to_display(frame.depth)) if frame.depth is not None else None,
        }

    @app.get("/api/training/status")
    def get_training_status():
        with training_lock:
            return dict(training_status)

    @app.get("/api/training/dependencies")
    def training_dependencies():
        return check_training_dependencies()

    @app.post("/api/training/start")
    def start_training(payload: TrainingStartPayload):
        with training_lock:
            if training_status["running"]:
                raise HTTPException(status_code=409, detail="Training is already running")
            training_status.update({
                "state": "running",
                "message": "Training started.",
                "running": True,
                "manifest": None,
                "error": None,
            })

        output_dir = Path(payload.output_dir)
        if not output_dir.is_absolute():
            output_dir = cfg.data_dir.parent / output_dir

        log_path = cfg.data_dir / "logs" / "hf_training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "robot_vision.cli",
            "train_vision",
        ]
        # `python -m robot_vision.cli train_vision` is not an argparse entry point, so call function via module below.
        command = [
            sys.executable,
            "-c",
            _training_subprocess_code(),
            "--",
            "--source",
            payload.source,
            "--output",
            str(output_dir),
            "--model",
            payload.model,
            "--validation-fraction",
            str(payload.validation_fraction),
            "--epochs",
            str(payload.epochs),
            "--batch-size",
            str(payload.batch_size),
            "--learning-rate",
            str(payload.learning_rate),
            "--seed",
            str(payload.seed),
        ]
        if payload.source == "captures":
            if not payload.dataset:
                with training_lock:
                    training_status.update({
                        "state": "failed",
                        "message": "Training failed.",
                        "running": False,
                        "error": "dataset is required when source is captures",
                    })
                return dict(training_status)
            command.extend(["--dataset-dir", str(paths.training / payload.dataset)])
        else:
            command.extend(["--reports", str(paths.reports)])
            if payload.recipe:
                command.extend(["--recipe", payload.recipe])

        def run_training_job() -> None:
            try:
                with log_path.open("w", encoding="utf-8") as log_handle:
                    log_handle.write("Command: " + " ".join(command) + "\n\n")
                    log_handle.flush()
                    process = subprocess.run(
                        command,
                        cwd=str(cfg.data_dir.parent),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                manifest = _load_training_manifest(output_dir) if process.returncode == 0 else None
                with training_lock:
                    training_status.update({
                        "state": "completed" if process.returncode == 0 else "failed",
                        "message": "Training completed." if process.returncode == 0 else "Training process exited with an error.",
                        "running": False,
                        "manifest": manifest,
                        "error": None if process.returncode == 0 else _tail_text(log_path),
                        "log_path": str(log_path),
                        "returncode": process.returncode,
                    })
            except Exception as exc:
                with training_lock:
                    training_status.update({
                        "state": "failed",
                        "message": "Training failed.",
                        "running": False,
                        "manifest": None,
                        "error": str(exc),
                        "log_path": str(log_path),
                    })

        Thread(target=run_training_job, daemon=True).start()
        with training_lock:
            return dict(training_status)

    return app


def _snapshot_or_error(camera, camera_lock: Lock):
    try:
        with camera_lock:
            return camera.snapshot()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _resolve_auto_camera_config(camera_config):
    if camera_config.device_index >= 0:
        return camera_config
    provider = camera_config.provider.lower()
    try:
        if provider == "opencv":
            return replace(camera_config, device_index=select_camera_device_index("global_shutter"))
        if provider in {"astra_hybrid", "hybrid_astra", "rgbd_astra"}:
            return replace(camera_config, device_index=select_camera_device_index("astra"))
    except Exception:
        return camera_config
    return camera_config


def _camera_mode_from_config(camera_config) -> str:
    provider = camera_config.provider.lower().strip()
    if provider == "opencv":
        return "global_shutter"
    if provider == "orbbec":
        return "orbbec_femto"
    return "astra"


def _camera_needs_astra_driver(camera_config) -> bool:
    return camera_config.provider.lower().strip() in {"astra", "astra_plus_pro", "astra_hybrid", "hybrid_astra", "rgbd_astra", "ros_astra", "astra_ros", "ros"}


def _hardware_restart_required(current_config, next_config) -> bool:
    current_provider = current_config.provider.lower().strip()
    next_provider = next_config.provider.lower().strip()
    if current_provider == "mock" or next_provider == "mock":
        return False
    return current_provider != next_provider


def _persist_runtime_camera_config(camera_config) -> None:
    path = _runtime_config_path()
    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            raw = loaded
    raw["camera"] = {
        "provider": camera_config.provider,
        "width": camera_config.width,
        "height": camera_config.height,
        "fps": camera_config.fps,
        "device_index": camera_config.device_index,
        "auto_exposure": camera_config.auto_exposure,
        "exposure": camera_config.exposure,
        "gain": camera_config.gain,
        "depth_enabled": camera_config.depth_enabled,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _runtime_config_path() -> Path:
    args = sys.argv[1:]
    if "--config" in args:
        index = args.index("--config")
        if index + 1 < len(args):
            path = Path(args[index + 1])
            return path if path.is_absolute() else (Path.cwd() / path).resolve()
    return PROJECT_ROOT / "config" / "app.yaml"


def _ensure_astra_ros_driver(data_dir: Path) -> None:
    if os.name == "nt" or shutil.which("ros2") is None:
        return
    if _astra_ros_driver_running():
        return
    log_path = data_dir / "logs" / "astra_camera.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    astra_ws = os.environ.get("ASTRA_WS", "/home/csteinhauer/astra_ws")
    command = (
        "set -e; "
        "source /opt/ros/humble/setup.bash; "
        f"if [ -f '{astra_ws}/install/setup.bash' ]; then source '{astra_ws}/install/setup.bash'; fi; "
        "exec ros2 launch astra_camera astra.launch.py enable_color:=false enable_colored_point_cloud:=false"
    )
    with log_path.open("ab") as log_handle:
        subprocess.Popen(
            ["bash", "-lc", command],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    _wait_for_astra_depth_topic(timeout_s=5.0)


def _stop_astra_ros_driver() -> None:
    if os.name == "nt":
        return
    subprocess.run(
        ["bash", "-lc", "ps -ef | awk '/ros2 launch astra_camera|astra_camera_container/ && !/awk/ {print $2}' | xargs -r kill"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5.0,
        check=False,
    )


def _astra_ros_driver_running() -> bool:
    result = subprocess.run(
        ["bash", "-lc", "ps -ef | grep -E 'ros2 launch astra_camera|astra_camera_container' | grep -v grep"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=2.0,
        check=False,
    )
    return result.returncode == 0


def _wait_for_astra_depth_topic(timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = subprocess.run(
            ["bash", "-lc", "source /opt/ros/humble/setup.bash && timeout 1 ros2 topic echo /camera/depth/image_raw --once --field header >/dev/null 2>&1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(0.25)


def _restart_process_soon(delay_s: float = 1.0) -> None:
    def restart() -> None:
        time.sleep(delay_s)
        os.execv(sys.executable, [sys.executable, "-m", "robot_vision", *sys.argv[1:]])

    Thread(target=restart, daemon=True).start()


def _training_subprocess_code() -> str:
    return (
        "import sys;"
        "from robot_vision.cli import train_vision;"
        "sys.argv=['robot-vision-train-vision']+sys.argv[sys.argv.index('--')+1:];"
        "train_vision()"
    )


def _load_training_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "robot_vision_training_manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _tail_text(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _open_path_in_file_manager(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))
        return
    commands: list[list[str]] = [["open", str(path)]] if sys.platform == "darwin" else [
        ["xdg-open", str(path)],
        ["gio", "open", str(path)],
    ]
    for command in commands:
        executable = command[0]
        if not shutil.which(executable):
            continue
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return
        except OSError:
            continue
    raise HTTPException(status_code=500, detail=f"Failed to open path: {path}")
