import numpy as np

from backend.services.analysis import calculate_metrics


def test_metrics_for_simple_square():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 1

    metrics = calculate_metrics(mask, pixel_spacing_mm=0.1)

    assert metrics.lesion_detected is True
    assert metrics.area_px2 > 0
    assert metrics.perimeter_px > 0
    assert metrics.equivalent_diameter_px > 0
    assert 0 <= metrics.circularity <= 1.1
    assert metrics.area_mm2 is not None
    assert metrics.bbox_width_mm is not None


def test_empty_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    metrics = calculate_metrics(mask)

    assert metrics.lesion_detected is False
    assert metrics.area_px2 == 0
