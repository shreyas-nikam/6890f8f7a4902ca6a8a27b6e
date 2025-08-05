import pytest
import numpy as np
from definition_d05f8e49f01e466d99fc94604a993994 import calculate_auc_gini

@pytest.mark.parametrize("y_true, y_pred, expected", [
    ([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], (1.0, 1.0)),
    ([1, 0, 0, 1], [0.8, 0.2, 0.3, 0.9], (1.0, 1.0)),
    ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], (1.0, 1.0)),
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], (1.0, 1.0)),
    ([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], (1.0, 1.0)),
    ([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5], (0.5, 0.0)),
    ([1, 0, 1, 0], [0.1, 0.2, 0.3, 0.4], (0.0, -1.0)),
    ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], (np.nan, np.nan)),
    ([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4], (np.nan, np.nan)),
    ([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], (1.0, 1.0)),
])
def test_calculate_auc_gini(y_true, y_pred, expected):
    auc, gini = calculate_auc_gini(y_true, y_pred)
    if np.isnan(expected[0]):
        assert np.isnan(auc)
        assert np.isnan(gini)
    else:
        assert auc == expected[0]
        assert gini == expected[1]

