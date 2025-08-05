import pytest
from definition_aa7192252c944794af5ed07a36e4dd09 import alert_on_performance_degradation
import logging

@pytest.fixture(autouse=True)
def enable_logging(caplog):
    caplog.set_level(logging.WARNING)

def test_no_alert(caplog):
    alert_on_performance_degradation(0.05, 0.05, 0.05)
    assert len(caplog.records) == 0

def test_auc_drop_alert(caplog):
    alert_on_performance_degradation(0.15, 0.05, 0.05)
    assert len(caplog.records) == 1
    assert "AUC drop is above threshold" in caplog.text

def test_psi_alert(caplog):
    alert_on_performance_degradation(0.05, 0.15, 0.05)
    assert len(caplog.records) == 1
    assert "PSI is above threshold" in caplog.text

def test_override_rate_alert(caplog):
    alert_on_performance_degradation(0.05, 0.05, 0.15)
    assert len(caplog.records) == 1
    assert "Override rate is above threshold" in caplog.text

def test_multiple_alerts(caplog):
    alert_on_performance_degradation(0.15, 0.15, 0.15)
    assert len(caplog.records) == 3
    assert "AUC drop is above threshold" in caplog.text
    assert "PSI is above threshold" in caplog.text
    assert "Override rate is above threshold" in caplog.text
