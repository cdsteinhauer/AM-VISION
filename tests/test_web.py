from fastapi.testclient import TestClient

from robot_vision.config import AppConfig, CameraConfig
from robot_vision.web.app import create_app


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

    inspect = client.post("/api/inspect", json={"recipe_name": "default", "save_report": True})
    assert inspect.status_code == 200
    payload = inspect.json()
    assert payload["result"]["passed"] is True
    assert payload["report"]["id"]

    reports = client.get("/api/reports")
    assert len(reports.json()["reports"]) == 1
