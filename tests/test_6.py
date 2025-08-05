import pytest
import pandas as pd
from definition_147aa61e8dcb4a0cab35e17be8ba3e9c import compute_psi

@pytest.fixture
def sample_data():
    expected = pd.Series([0.2, 0.3, 0.5], index=['A', 'B', 'C'])
    actual = pd.Series([0.25, 0.35, 0.4], index=['A', 'B', 'C'])
    grade_names = ['A', 'B', 'C']
    return expected, actual, grade_names

def test_compute_psi_typical(sample_data):
    expected, actual, grade_names = sample_data
    result = compute_psi(expected, actual, grade_names)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(grade_names)
    assert 'PSI' in result.columns


def test_compute_psi_no_change(sample_data):
    expected, _, grade_names = sample_data
    actual = expected.copy()
    result = compute_psi(expected, actual, grade_names)
    assert all(result['PSI'] == 0)

def test_compute_psi_zero_expected(sample_data):
    expected, actual, grade_names = sample_data
    expected['A'] = 0.0
    with pytest.raises(ValueError):
        compute_psi(expected, actual, grade_names)

def test_compute_psi_zero_actual(sample_data):
    expected, actual, grade_names = sample_data
    actual['A'] = 0.0
    with pytest.raises(ValueError):
        compute_psi(expected, actual, grade_names)

def test_compute_psi_mismatched_grades(sample_data):
    expected, actual, grade_names = sample_data
    actual = pd.Series([0.25, 0.35, 0.4, 0.0], index=['A', 'B', 'C', 'D'])
    grade_names = ['A', 'B', 'C', 'D'] # added D although not present in expected
    with pytest.raises(ValueError):
        compute_psi(expected, actual, grade_names)
