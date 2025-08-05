import pytest
from definition_3210c5005f6e4c30992c5b4151dd25d6 import track_auc_gini_drift
import pandas as pd
import numpy as np

@pytest.fixture
def historical_data_fixture():
    return pd.DataFrame({'time_period': [], 'auc': [], 'gini': []})

def test_track_auc_gini_drift_empty_data(historical_data_fixture):
    historical_data = historical_data_fixture
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.2, 0.8, 0.1, 0.9])
    time_period = '2024Q1'
    updated_data = track_auc_gini_drift(historical_data, y_true, y_pred, time_period)
    assert isinstance(updated_data, pd.DataFrame)
    assert len(updated_data) == 1

def test_track_auc_gini_drift_existing_data(historical_data_fixture):
    historical_data = historical_data_fixture
    historical_data = pd.DataFrame({'time_period': ['2023Q4'], 'auc': [0.7], 'gini': [0.4]})
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.3, 0.7, 0.2, 0.8])
    time_period = '2024Q1'
    updated_data = track_auc_gini_drift(historical_data, y_true, y_pred, time_period)
    assert isinstance(updated_data, pd.DataFrame)
    assert len(updated_data) == 2

def test_track_auc_gini_drift_perfect_predictions(historical_data_fixture):
    historical_data = historical_data_fixture
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.01, 0.99, 0.02, 0.98])  # Near perfect predictions
    time_period = '2024Q1'
    updated_data = track_auc_gini_drift(historical_data, y_true, y_pred, time_period)
    assert isinstance(updated_data, pd.DataFrame)
    assert len(updated_data) == 1
    assert 0 <= updated_data['auc'][0] <= 1
    assert 0 <= updated_data['gini'][0] <= 1

def test_track_auc_gini_drift_invalid_input(historical_data_fixture):
    historical_data = historical_data_fixture
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1.2, 0.8, 0.1, 0.9])  # Invalid probability
    time_period = '2024Q1'
    with pytest.raises(ValueError): #Check exception raised if probabilities are outside 0 and 1
        track_auc_gini_drift(historical_data, y_true, y_pred, time_period)

def test_track_auc_gini_drift_all_same_class(historical_data_fixture):
    historical_data = historical_data_fixture
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.2, 0.8, 0.1, 0.9])
    time_period = '2024Q1'
    updated_data = track_auc_gini_drift(historical_data, y_true, y_pred, time_period)
    assert isinstance(updated_data, pd.DataFrame)
    assert len(updated_data) == 1
    assert np.isnan(updated_data['auc'][0])
    assert np.isnan(updated_data['gini'][0])

