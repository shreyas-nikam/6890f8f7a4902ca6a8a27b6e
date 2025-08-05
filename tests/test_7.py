import pytest
from definition_c186e05dcb2e44c0ad26da60855aa4db import compute_overall_psi
import numpy as np

@pytest.mark.parametrize("expected, actual, expected_result", [
    ([0.5, 0.5], [0.5, 0.5], 0.0),  # Same distribution
    ([0.2, 0.8], [0.8, 0.2], 1.0986),  # Different distribution
    ([0.0, 1.0], [1.0, 0.0], np.inf),  # One distribution has zero probability
    ([0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4], 0.1484),  # Multiple categories
    ([0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25], 0.1484)
])
def test_compute_overall_psi(expected, actual, expected_result):
    # Catch the RuntimeWarning caused by the log of zero or near zero.
    with pytest.warns(RuntimeWarning):
        if expected_result == np.inf:
            with pytest.raises(RuntimeError):
                compute_overall_psi(expected, actual)
        else:
            actual_result = compute_overall_psi(expected, actual)
            assert np.isclose(actual_result, expected_result, rtol=1e-4)
