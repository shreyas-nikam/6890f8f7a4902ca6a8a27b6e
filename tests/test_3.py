import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from definition_7ccd32d641a34854885e6b5313ee25a1 import calculate_auc_gini

@pytest.fixture
def mock_model():
    # Create a mock model with a predict_proba method
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # Example probabilities
    return model

def test_calculate_auc_gini_typical_case(mock_model):
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([1, 0, 1])
    auc, gini = calculate_auc_gini(mock_model, X, y)
    assert 0 <= auc <= 1
    assert -1 <= gini <= 1

def test_calculate_auc_gini_perfect_prediction(mock_model):
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([1, 0, 1])
    mock_model.predict_proba.return_value = np.array([[0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])
    auc, gini = calculate_auc_gini(mock_model, X, y)
    assert auc == 1.0
    assert gini == 1.0

def test_calculate_auc_gini_random_prediction(mock_model):
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([1, 0, 1])
    mock_model.predict_proba.return_value = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    auc, gini = calculate_auc_gini(mock_model, X, y)
    assert auc == 0.5
    assert gini == 0.0

def test_calculate_auc_gini_empty_data():
    model = MagicMock()
    X = pd.DataFrame()
    y = pd.Series([])
    auc, gini = calculate_auc_gini(model, X, y)
    assert np.isnan(auc)
    assert np.isnan(gini)

def test_calculate_auc_gini_single_class():
    model = MagicMock()
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([1, 1, 1])
    auc, gini = calculate_auc_gini(model, X, y)
    assert np.isnan(auc)
    assert np.isnan(gini)
