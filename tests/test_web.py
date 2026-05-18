from fastapi.testclient import TestClient

from robot_vision.config import AppConfig, CameraConfig
from robot_vision.web.app import create_app
from robot_vision.storage.training import TrainingCaptureStore


def _write_capture_samples(root):
    import numpy as np

    store = TrainingCaptureStore(root / "training")
    image = (np.ones((16, 16, 3)) * 255).astype("uint8")
    store.save_sample("part_a", "PASS", image)
    store.save_sample("part_a", "FAIL", image)
    store.save_sample("part_b", "PASS", image)
    return store


def test_health_and_mock_inspection(tmp_path):
    config = AppConfig(data_dir=tmp_path, camera=CameraConfig(width=640, height=360))
    client = TestClient(create_app(config))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ros_domain_id"] == "77"

    preview = client.get("/api/camera/preview?view=rgb")
    assert preview.status_code == 200
    assert preview.json()["rgb_jpg"]

    settings = client.get("/api/camera/settings")
    assert settings.status_code == 200
    assert settings.json()["controls"]

    update_settings = client.post("/api/camera/settings", json={"settings": {"brightness": 5}})
    assert update_settings.status_code == 200
    assert update_settings.json()["applied"]["brightness"] == 5

    processed_preview = client.get("/api/camera/preview?view=rgb&process_trigger=true")
    assert processed_preview.status_code == 200
    assert "auto_trigger" in processed_preview.json()

    line_detect = client.post("/api/camera/line-detect", json={
        "tools": [{
            "id": "live_edge",
            "name": "Live edge",
            "type": "edge_1",
            "enabled": True,
            "live_lines": True,
            "roi": [0.1, 0.1, 0.8, 0.8],
        }]
    })
    assert line_detect.status_code == 200
    assert line_detect.json()["width"] == 640
    assert line_detect.json()["tools"][0]["tool_id"] == "live_edge"

    ignored_ai_line_detect = client.post("/api/camera/line-detect", json={
        "tools": [{
            "id": "ai",
            "name": "AI",
            "type": "ai_classifier",
            "enabled": True,
            "live_lines": True,
            "roi": [0.1, 0.1, 0.8, 0.8],
        }]
    })
    assert ignored_ai_line_detect.status_code == 200
    assert ignored_ai_line_detect.json()["tools"] == []

    inspect = client.post("/api/inspect", json={"recipe_name": "default", "save_report": True})
    assert inspect.status_code == 200
    payload = inspect.json()
    assert payload["result"]["passed"] is True
    assert payload["report"]["id"]

    reports = client.get("/api/reports")
    assert len(reports.json()["reports"]) == 1


def test_training_samples_collect_from_all_capture_datasets(tmp_path):
    _write_capture_samples(tmp_path)
    client = TestClient(create_app(AppConfig(data_dir=tmp_path, camera=CameraConfig(width=640, height=360))))
    response = client.get("/api/training/samples?source=captures")
    data = response.json()
    assert response.status_code == 200
    assert data["total"] == 3


def test_training_samples_collect_from_named_capture_dataset(tmp_path):
    _write_capture_samples(tmp_path)
    client = TestClient(create_app(AppConfig(data_dir=tmp_path, camera=CameraConfig(width=640, height=360))))
    response = client.get("/api/training/samples?source=captures&dataset=part_a")
    data = response.json()
    assert response.status_code == 200
    assert data["total"] == 2


def test_delete_all_captures_trains_dataset_guard_and_scope(tmp_path):
    _write_capture_samples(tmp_path)
    client = TestClient(create_app(AppConfig(data_dir=tmp_path, camera=CameraConfig(width=640, height=360))))
    response = client.delete("/api/training/captures")
    assert response.status_code == 400
    response = client.delete("/api/training/captures?confirm=DELETE")
    assert response.status_code == 200
    assert response.json()["deleted"] == 6
    assert response.json()["scope"] == "all"
