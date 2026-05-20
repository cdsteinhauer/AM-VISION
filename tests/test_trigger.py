import numpy as np

from robot_vision.inspection.trigger import PresenceTrigger


def test_presence_trigger_fires_from_depth_change_when_rgb_is_flat():
    trigger = PresenceTrigger((0.0, 0.0, 1.0, 1.0), min_change=12.0, debounce_s=0.0)
    image = np.full((100, 100, 3), 80, dtype=np.uint8)
    baseline = np.full((100, 100), 900.0, dtype=np.float32)
    changed = baseline.copy()
    changed[40:50, 40:50] = 880.0

    first = trigger.update(image, baseline)
    second = trigger.update(image, changed)
    third = trigger.update(image, changed)

    assert first["fired"] is False
    assert second["fired"] is True
    assert second["source"] == "depth"
    assert second["rgb_score"] == 0.0
    assert second["depth_score_mm"] >= 20.0
    assert third["fired"] is False


def test_presence_trigger_keeps_rgb_fallback():
    trigger = PresenceTrigger((0.0, 0.0, 1.0, 1.0), min_change=12.0, debounce_s=0.0)
    baseline = np.full((40, 40, 3), 80, dtype=np.uint8)
    changed = np.full((40, 40, 3), 120, dtype=np.uint8)

    trigger.update(baseline)
    result = trigger.update(changed)

    assert result["fired"] is True
    assert result["source"] == "rgb"
