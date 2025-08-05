import pytest
from definition_03e6f4e2af03453db53d0db0c32d6f36 import calculate_override_rate

@pytest.mark.parametrize("number_of_overrides, total_number_of_applications, expected", [
    (10, 100, 10.0),
    (0, 100, 0.0),
    (50, 50, 100.0),
    (10, 0, ZeroDivisionError),
    (10.5, 100, TypeError),
])
def test_calculate_override_rate(number_of_overrides, total_number_of_applications, expected):
    try:
        assert calculate_override_rate(number_of_overrides, total_number_of_applications) == expected
    except Exception as e:
        assert isinstance(e, expected)
