from robot_vision.camera.mock import MockCamera
from robot_vision.config import CameraConfig
from robot_vision.inspection.calibration import CalibrationProfile
from robot_vision.inspection.engine import InspectionEngine
from robot_vision.inspection.models import InspectionRecipe
from robot_vision.storage.calibration import CalibrationStore
from robot_vision.storage.recipes import RecipeStore
from robot_vision.storage.reports import ReportStore


def test_recipe_and_calibration_round_trip(tmp_path):
    recipe_store = RecipeStore(tmp_path / "recipes")
    cal_store = CalibrationStore(tmp_path / "calibration")

    recipe = InspectionRecipe.default()
    recipe.name = "test_recipe"
    recipe_store.save(recipe)
    assert recipe_store.load("test_recipe").name == "test_recipe"
    recipe_store.delete("test_recipe")
    assert "test_recipe" not in recipe_store.list_names()

    profile = CalibrationProfile.from_reference(800, 500, 400, 250)
    cal_store.save(profile)
    assert cal_store.load("default").pixels_per_mm_x == 2


def test_report_store_writes_artifacts(tmp_path):
    frame = MockCamera(CameraConfig(width=640, height=360)).snapshot()
    recipe = InspectionRecipe.default()
    profile = CalibrationProfile()
    result = InspectionEngine().inspect(frame.rgb, recipe, profile)

    payload = ReportStore(tmp_path / "reports").save(
        frame.rgb,
        frame.depth,
        result,
        recipe.to_dict(),
        profile.to_dict(),
    )

    assert payload["id"]
    assert (tmp_path / "reports" / payload["id"] / "rgb.png").exists()
    assert (tmp_path / "reports" / payload["id"] / "overlay.png").exists()
    assert (tmp_path / "reports" / payload["id"] / "result.json").exists()


def test_report_store_deletes_all_report_samples(tmp_path):
    frame = MockCamera(CameraConfig(width=640, height=360)).snapshot()
    recipe = InspectionRecipe.default()
    profile = CalibrationProfile()
    result = InspectionEngine().inspect(frame.rgb, recipe, profile)
    store = ReportStore(tmp_path / "reports")

    first = store.save(frame.rgb, frame.depth, result, recipe.to_dict(), profile.to_dict())
    second = store.save(frame.rgb, frame.depth, result, recipe.to_dict(), profile.to_dict())

    assert store.delete_all() == 2
    assert not (tmp_path / "reports" / first["id"]).exists()
    assert not (tmp_path / "reports" / second["id"]).exists()
