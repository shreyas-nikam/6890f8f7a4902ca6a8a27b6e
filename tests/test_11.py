import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_5a1fec6788334e9292800add333a9a0c import perform_sensitivity_analysis

@pytest.fixture
def mock_model():
    # Create a mock model that returns a fixed prediction for testing purposes
    model = MagicMock()
    model.predict_proba = MagicMock(return_value=[[0.2, 0.8]] * 5)  # Mock PD predictions
    return model

@pytest.fixture
def sample_data():
    # Create sample data for testing
    return pd.DataFrame({'driver1': [1, 2, 3, 4, 5],
                         'driver2': [6, 7, 8, 9, 10],
                         'driver3': [11, 12, 13, 14, 15],
                         'driver4': [16, 17, 18, 19, 20],
                         'driver5': [21, 22, 23, 24, 25]})


def test_perform_sensitivity_analysis_typical(mock_model, sample_data):
    top_drivers = ['driver1', 'driver2', 'driver3', 'driver4', 'driver5']
    perturbation = 0.05
    result = perform_sensitivity_analysis(mock_model, sample_data, top_drivers, perturbation)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(top_drivers)
    assert 'driver' in result.columns
    assert 'change_in_pd' in result.columns

def test_perform_sensitivity_analysis_empty_top_drivers(mock_model, sample_data):
    top_drivers = []
    perturbation = 0.05
    result = perform_sensitivity_analysis(mock_model, sample_data, top_drivers, perturbation)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_perform_sensitivity_analysis_no_drivers_in_data(mock_model, sample_data):
    top_drivers = ['non_existent_driver']
    perturbation = 0.05
    with pytest.raises(KeyError):
        perform_sensitivity_analysis(mock_model, sample_data, top_drivers, perturbation)

def test_perform_sensitivity_analysis_zero_perturbation(mock_model, sample_data):
    top_drivers = ['driver1', 'driver2']
    perturbation = 0.0
    result = perform_sensitivity_analysis(mock_model, sample_data, top_drivers, perturbation)
    assert isinstance(result, pd.DataFrame)
    assert all(result['change_in_pd'] == 0)

def test_perform_sensitivity_analysis_data_empty(mock_model):
    top_drivers = ['driver1', 'driver2']
    perturbation = 0.05
    empty_data = pd.DataFrame()
    with pytest.raises(KeyError):  # or ValueError, depending on how your implementation handles missing columns
        perform_sensitivity_analysis(mock_model, empty_data, top_drivers, perturbation)
