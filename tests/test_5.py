import pytest
import numpy as np
from definition_db5f235c112c4a2881de18fce8d1e38f import generate_calibration_curve

@pytest.mark.parametrize("y_true, y_prob, n_bins, expected_type", [
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 2, tuple),
    ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], 2, tuple),
    ([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9], 2, tuple),
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 5, tuple),
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0, ValueError),
])
def test_generate_calibration_curve(y_true, y_prob, n_bins, expected_type):
    try:
        result = generate_calibration_curve(y_true, y_prob, n_bins)
        assert isinstance(result, expected_type)
    except Exception as e:
        assert isinstance(e, expected_type)
