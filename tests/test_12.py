import pytest
from definition_3ec5add756374bebbcef6e73b0980a26 import generate_kpi_panel

@pytest.mark.parametrize("number_of_overrides, time_since_last_validation, next_review_due, open_remediation_actions, expected", [
    (10, "2023-01-01", "2024-01-01", ["Action1", "Action2"], {'number_of_overrides': 10, 'time_since_last_validation': '2023-01-01', 'next_review_due': '2024-01-01', 'open_remediation_actions': ["Action1", "Action2"]}),
    (0, "2023-12-01", "2024-03-01", [], {'number_of_overrides': 0, 'time_since_last_validation': '2023-12-01', 'next_review_due': '2024-03-01', 'open_remediation_actions': []}),
    (5, None, None, None, {'number_of_overrides': 5, 'time_since_last_validation': None, 'next_review_due': None, 'open_remediation_actions': None}),
    (-1, "2024-01-15", "2024-04-15", ["Action1"], {'number_of_overrides': -1, 'time_since_last_validation': '2024-01-15', 'next_review_due': '2024-04-15', 'open_remediation_actions': ["Action1"]}),
    (100, "2022-12-31", "2025-12-31", ["Action1", "Action2", "Action3"], {'number_of_overrides': 100, 'time_since_last_validation': '2022-12-31', 'next_review_due': '2025-12-31', 'open_remediation_actions': ["Action1", "Action2", "Action3"]}),
])
def test_generate_kpi_panel(number_of_overrides, time_since_last_validation, next_review_due, open_remediation_actions, expected):
    assert generate_kpi_panel(number_of_overrides, time_since_last_validation, next_review_due, open_remediation_actions) == {'number_of_overrides': number_of_overrides, 'time_since_last_validation': time_since_last_validation, 'next_review_due': next_review_due, 'open_remediation_actions': open_remediation_actions}
