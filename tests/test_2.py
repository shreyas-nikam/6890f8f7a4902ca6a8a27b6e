import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_ee86eaf7692443f5aacbf091fc77b05c import apply_preprocessing

def test_apply_preprocessing_empty_dataframe():
    data = pd.DataFrame()
    pipeline = MagicMock()
    result = apply_preprocessing(data, pipeline)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_apply_preprocessing_pipeline_called():
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    pipeline = MagicMock()
    apply_preprocessing(data, pipeline)
    pipeline.transform.assert_called_once()

def test_apply_preprocessing_pipeline_returns_dataframe():
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    pipeline = MagicMock()
    pipeline.transform.return_value = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})
    result = apply_preprocessing(data, pipeline)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]}))

def test_apply_preprocessing_pipeline_raises_exception():
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    pipeline = MagicMock()
    pipeline.transform.side_effect = ValueError("Test Exception")
    with pytest.raises(ValueError, match="Test Exception"):
        apply_preprocessing(data, pipeline)

def test_apply_preprocessing_none_pipeline():
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    with pytest.raises(TypeError):
        apply_preprocessing(data, None)
