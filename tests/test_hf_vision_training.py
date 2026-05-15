import json

from PIL import Image

from robot_vision.storage.training import TrainingCaptureStore
from robot_vision.training.hf_vision import ID_TO_LABEL, collect_capture_samples, collect_report_samples, split_samples


def test_collect_report_samples_from_saved_reports(tmp_path):
    _write_report(tmp_path, "pass-1", passed=True, recipe="default")
    _write_report(tmp_path, "fail-1", passed=False, recipe="default")
    _write_report(tmp_path, "other-1", passed=True, recipe="other")

    samples = collect_report_samples(tmp_path, recipe="default")

    assert len(samples) == 2
    assert {ID_TO_LABEL[sample.label] for sample in samples} == {"PASS", "FAIL"}
    assert {sample.report_id for sample in samples} == {"pass-1", "fail-1"}


def test_split_samples_keeps_train_and_validation_labels(tmp_path):
    samples = []
    for index in range(4):
        _write_report(tmp_path, f"pass-{index}", passed=True)
        _write_report(tmp_path, f"fail-{index}", passed=False)
    samples = collect_report_samples(tmp_path)

    train, validation = split_samples(samples, validation_fraction=0.25, seed=3)

    assert len(train) == 6
    assert len(validation) == 2
    assert {sample.label for sample in validation} == {0, 1}


def test_collect_capture_samples_from_guided_dataset(tmp_path):
    store = TrainingCaptureStore(tmp_path / "training")
    image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    store.save_sample("part_a", "PASS", _image_array(image))
    store.save_sample("part_a", "FAIL", _image_array(image))

    samples = collect_capture_samples(tmp_path / "training" / "part_a")

    assert len(samples) == 2
    assert {ID_TO_LABEL[sample.label] for sample in samples} == {"PASS", "FAIL"}


def _image_array(image):
    import numpy as np

    return np.asarray(image)


def _write_report(root, report_id, passed, recipe="default"):
    folder = root / report_id
    folder.mkdir()
    image_path = folder / "rgb.png"
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
    payload = {
        "id": report_id,
        "result": {"passed": passed, "recipe": recipe},
        "files": {"rgb": str(image_path)},
    }
    with (folder / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
