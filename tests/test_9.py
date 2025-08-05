import pytest
from definition_cf18f75e26714ea8af22b396bd847a4f import calculate_override_rate

@pytest.mark.parametrize("num_overrides, total_applications, expected", [
    (10, 100, 10.0),
    (0, 100, 0.0),
    (50, 50, 100.0),
    (10, 0, float('inf')),  # Handle division by zero
    (15, 75, 20.0)
])
def test_calculate_override_rate(num_overrides, total_applications, expected):
    if expected == float('inf'):
        with pytest.raises(ZeroDivisionError):
            calculate_override_rate(num_overrides, total_applications)
    else:
        assert calculate_override_rate(num_overrides, total_applications) == expected
