import pytest
import pandas as pd
import numpy as np
from definition_0142a7e5130546a3ac1f623988aa7b7b import compute_psi

@pytest.fixture
def sample_data():
    expected = pd.Series([0.2, 0.3, 0.5], index=['A', 'B', 'C'])
    actual = pd.Series([0.25, 0.25, 0.5], index=['A', 'B', 'C'])
    return expected, actual

def test_compute_psi_valid_input(sample_data):
    expected, actual = sample_data
    psi = compute_psi(expected, actual)
    assert isinstance(psi, pd.Series)
    assert not psi.empty

def test_compute_psi_identical_distributions(sample_data):
    expected, _ = sample_data
    actual = expected.copy()
    psi = compute_psi(expected, actual)
    assert all(psi == 0)

def test_compute_psi_zero_expected_probability(sample_data):
    expected, actual = sample_data
    expected['A'] = 0.0
    with pytest.raises(ValueError):
        compute_psi(expected, actual)

def test_compute_psi_zero_actual_probability(sample_data):
    expected, actual = sample_data
    actual['A'] = 0.0
    psi = compute_psi(expected, actual)
    assert not np.isnan(psi['A'])
    
def test_compute_psi_different_categories(sample_data):
    expected, actual = sample_data
    actual['D'] = 0.1  # Introduce a new category in actual
    with pytest.raises(ValueError):
        compute_psi(expected, actual)
