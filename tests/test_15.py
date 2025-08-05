import pytest
import logging
from definition_20b4ac8d81f241148228d1efaef8c2d1 import raise_alerts

@pytest.fixture
def log_capture(caplog):
    caplog.set_level(logging.WARNING)
    return caplog

def test_no_alerts(log_capture):
    raise_alerts(0.05, 0.05, 0.05)
    assert not log_capture.records

def test_auc_drop_alert(log_capture):
    raise_alerts(0.15, 0.05, 0.05)
    assert len(log_capture.records) == 1
    assert "AUC drop" in log_capture.text

def test_psi_alert(log_capture):
    raise_alerts(0.05, 0.15, 0.05)
    assert len(log_capture.records) == 1
    assert "PSI" in log_capture.text

def test_override_rate_alert(log_capture):
    raise_alerts(0.05, 0.05, 0.15)
    assert len(log_capture.records) == 1
    assert "Override rate" in log_capture.text

def test_multiple_alerts(log_capture):
    raise_alerts(0.15, 0.15, 0.15)
    assert len(log_capture.records) == 3
    assert "AUC drop" in log_capture.text
    assert "PSI" in log_capture.text
    assert "Override rate" in log_capture.text
