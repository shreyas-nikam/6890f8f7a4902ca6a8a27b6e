import pytest
import numpy as np
from definition_0af63744ee5b4ede9c47688526b60f68 import perform_hosmer_lemeshow_test

@pytest.mark.parametrize("y_true, y_prob, n_groups, expected_type", [
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 2, tuple),  # Basic test case
    ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], 2, tuple),  # All negative class
    ([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9], 2, tuple),  # All positive class
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 5, ValueError), # n_groups > len(y_true)
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 'a'], 2, TypeError), # y_prob not numeric
])
def test_perform_hosmer_lemeshow_test(y_true, y_prob, n_groups, expected_type):
    try:
        result = perform_hosmer_lemeshow_test(y_true, y_prob, n_groups)
        assert isinstance(result, expected_type)
    except Exception as e:
        assert isinstance(e, expected_type)
