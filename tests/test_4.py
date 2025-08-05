import pytest
import numpy as np
from definition_69fbe3d3bde8439a8299d4483c55b8e8 import perform_hosmer_lemeshow

def test_hosmer_lemeshow_perfect_calibration():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85])
    groups = 2
    h, p = perform_hosmer_lemeshow(y_true, y_pred, groups)
    assert p > 0.05

def test_hosmer_lemeshow_poor_calibration():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.85, 0.15])
    groups = 2
    h, p = perform_hosmer_lemeshow(y_true, y_pred, groups)
    assert p <= 0.05

def test_hosmer_lemeshow_edge_case_same_predictions():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    groups = 2
    h, p = perform_hosmer_lemeshow(y_true, y_pred, groups)
    assert not np.isnan(p)

def test_hosmer_lemeshow_unequal_groups():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85, 0.5])
    groups = 3
    h, p = perform_hosmer_lemeshow(y_true, y_pred, groups)
    assert not np.isnan(p)

def test_hosmer_lemeshow_small_data():
     y_true = np.array([0, 1])
     y_pred = np.array([0.2, 0.8])
     groups = 2
     h, p = perform_hosmer_lemeshow(y_true, y_pred, groups)
     assert not np.isnan(p)

