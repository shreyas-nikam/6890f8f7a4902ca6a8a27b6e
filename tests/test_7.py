import pytest
import numpy as np
from definition_34270ccd0ce542379fafe9e3b57107df import compute_overall_psi

@pytest.mark.parametrize("expected, actual, expected_psi", [
    ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5], 0.0),  # Identical distributions, PSI should be 0
    ([0.2, 0.3, 0.5], [0.3, 0.5, 0.2], 0.427, ),  # Different distributions
    ([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], 0.924), #Reversed distributions
    ([0.5, 0.5], [0.0, 1.0], np.inf),  # Actual has zero value, PSI should be inf
    ([0.0, 1.0], [0.5, 0.5], np.inf) #Expected has zero value, PSI should be inf
])
def test_compute_overall_psi(expected, actual, expected_psi):
    if np.isinf(expected_psi):
        with pytest.raises(ValueError):
            compute_overall_psi(expected, actual)
    else:
        psi = compute_overall_psi(expected, actual)
        assert round(psi, 3) == round(expected_psi, 3)
