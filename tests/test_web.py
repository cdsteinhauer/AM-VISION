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

    depth_reference = client.post("/api/calibration/depth-reference", json={"roi": [0.0, 0.0, 0.2, 0.2]})
    assert depth_reference.status_code == 200
    assert depth_reference.json()["depth_reference"]["sample_count"] > 0


def test_training_samples_collect_from_all_capture_datasets(tmp_path):
    _write_capture_samples(tmp_path)
    client = TestClient(create_app(AppConfig(data_dir=tmp_path, camera=CameraConfig(width=640, height=360))))
    response = client.get("/api/training/samples?source=captures")
    data = response.json()
    assert response.status_code == 200
    assert data["total"] == 3


def test_camera_mode_supports_orbbec_femto(tmp_path):
    client = TestClient(create_app(AppConfig(data_dir=tmp_path, camera=CameraConfig(width=640, height=360))))

    response = client.post("/api/camera/mode", json={"mode": "orbbec_femto"})
    data = response.json()

    assert response.status_code == 200
    assert data["mode"] == "orbbec_femto"
    assert data["provider"] == "orbbec"
    assert data["device_index"] == -1
    assert data["depth_enabled"] is True

    current = client.get("/api/camera/mode")
    assert current.status_code == 200
    assert current.json()["mode"] == "orbbec_femto"


def test_camera_mode_reapply_current_mode_does_not_restart_camera(tmp_path, monkeypatch):
    class FakeCamera:
        stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    fake_camera = FakeCamera()
    monkeypatch.setattr("robot_vision.web.app.create_camera", lambda config: fake_camera)
    client = TestClient(create_app(AppConfig(data_dir=tmp_path, camera=CameraConfig(provider="orbbec", device_index=-1))))

    response = client.post("/api/camera/mode", json={"mode": "orbbec_femto"})

    assert response.status_code == 200
    assert response.json()["mode"] == "orbbec_femto"
    assert fake_camera.stop_calls == 0


def test_camera_mode_supports_astra_hybrid(tmp_path, monkeypatch):
    monkeypatch.setattr("robot_vision.web.app.select_camera_device_index", lambda kind, preferred_index=None: 2)
    monkeypatch.setattr("robot_vision.web.app._ensure_astra_ros_driver", lambda data_dir: None)
    client = TestClient(create_app(AppConfig(data_dir=tmp_path, camera=CameraConfig(width=640, height=360))))

    response = client.post("/api/camera/mode", json={"mode": "astra"})
    data = response.json()

    assert response.status_code == 200
    assert data["mode"] == "astra"
    assert data["provider"] == "astra_hybrid"
    assert data["device_index"] == 2
    assert data["depth_enabled"] is True


def test_camera_mode_switch_from_astra_to_orbbec_schedules_restart(tmp_path, monkeypatch):
    class FakeCamera:
        stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    fake_camera = FakeCamera()
    restart_calls = []
    persist_calls = []
    monkeypatch.setattr("robot_vision.web.app.create_camera", lambda config: fake_camera)
    monkeypatch.setattr("robot_vision.web.app._restart_process_soon", lambda: restart_calls.append(True))
    monkeypatch.setattr("robot_vision.web.app._persist_runtime_camera_config", lambda config: persist_calls.append(config.provider))
    client = TestClient(create_app(AppConfig(data_dir=tmp_path, camera=CameraConfig(provider="astra_hybrid", device_index=0))))

    response = client.post("/api/camera/mode", json={"mode": "orbbec_femto"})

    assert response.status_code == 200
    assert response.json()["mode"] == "orbbec_femto"
    assert fake_camera.stop_calls == 1
    assert persist_calls == ["orbbec"]
    assert restart_calls == [True]


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
