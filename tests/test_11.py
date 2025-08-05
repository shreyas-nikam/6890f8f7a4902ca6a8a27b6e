import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from definition_c38093652a1d4979b858cf6314103bd4 import perform_sensitivity_analysis


def test_perform_sensitivity_analysis_empty_top_drivers():
    model_mock = MagicMock()
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    top_drivers = []
    perturbation = 0.05
    result = perform_sensitivity_analysis(model_mock, X, top_drivers, perturbation)
    assert result.empty

def test_perform_sensitivity_analysis_one_driver():
    model_mock = MagicMock()
    model_mock.predict.return_value = np.array([0.5, 0.6, 0.7])
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    top_drivers = ['feature1']
    perturbation = 0.05
    result = perform_sensitivity_analysis(model_mock, X, top_drivers, perturbation)
    assert not result.empty
    assert len(result) == 1
    assert result.columns[0] == 'delta_PD'

def test_perform_sensitivity_analysis_multiple_drivers():
    model_mock = MagicMock()
    model_mock.predict.return_value = np.array([0.5, 0.6, 0.7])
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'feature3': [7,8,9]})
    top_drivers = ['feature1', 'feature2']
    perturbation = 0.05
    result = perform_sensitivity_analysis(model_mock, X, top_drivers, perturbation)
    assert not result.empty
    assert len(result) == 2
    assert result.columns[0] == 'delta_PD'

def test_perform_sensitivity_analysis_zero_perturbation():
    model_mock = MagicMock()
    model_mock.predict.return_value = np.array([0.5, 0.6, 0.7])
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    top_drivers = ['feature1']
    perturbation = 0.0
    result = perform_sensitivity_analysis(model_mock, X, top_drivers, perturbation)

    assert not result.empty
    assert len(result) == 1
    assert result.columns[0] == 'delta_PD'
    assert (result['delta_PD'] == 0).all()

def test_perform_sensitivity_analysis_string_feature():
    model_mock = MagicMock()
    model_mock.predict.return_value = np.array([0.5, 0.6, 0.7])
    X = pd.DataFrame({'feature1': ['a', 'b', 'c'], 'feature2': [4, 5, 6]})
    top_drivers = ['feature2']
    perturbation = 0.05
    result = perform_sensitivity_analysis(model_mock, X, top_drivers, perturbation)
    assert not result.empty
    assert len(result) == 1
    assert result.columns[0] == 'delta_PD'
