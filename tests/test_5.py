import pytest
import numpy as np
from definition_47033eaa05b146bbaa89dd4377d31657 import generate_calibration_curve

@pytest.mark.parametrize("y_true, y_pred, n_bins, expected_len", [
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 2, 2),  # Basic test with 2 bins
    ([0, 1, 0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8, 0.3, 0.7], 3, 3), # Test with 3 bins
    ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], 2, 2), # No positive cases
    ([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9], 2, 2), # No negative cases
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 5, 4), # More bins than samples, expect smaller
])
def test_generate_calibration_curve(y_true, y_pred, n_bins, expected_len):
    bins, observed_rates = generate_calibration_curve(y_true, y_pred, n_bins)
    assert len(bins) == expected_len
    assert len(observed_rates) == expected_len

    #Additional check for type
    assert isinstance(bins, np.ndarray)
    assert isinstance(observed_rates, np.ndarray)
