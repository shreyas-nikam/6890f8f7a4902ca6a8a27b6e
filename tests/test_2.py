import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_2a3c494c17894ce5b64978145173e143 import apply_preprocessing

def test_apply_preprocessing_empty_dataframe():
    pipeline_mock = MagicMock()
    empty_df = pd.DataFrame()
    result = apply_preprocessing(pipeline_mock, empty_df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_apply_preprocessing_pipeline_called():
    pipeline_mock = MagicMock()
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    apply_preprocessing(pipeline_mock, data)
    pipeline_mock.transform.assert_called_once_with(data)

def test_apply_preprocessing_returns_transformed_dataframe():
    pipeline_mock = MagicMock()
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    transformed_data = pd.DataFrame({'col1': [10, 20], 'col2': [30, 40]})
    pipeline_mock.transform.return_value = transformed_data
    result = apply_preprocessing(pipeline_mock, data)
    pd.testing.assert_frame_equal(result, transformed_data)

def test_apply_preprocessing_pipeline_exception():
    pipeline_mock = MagicMock()
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    pipeline_mock.transform.side_effect = ValueError("Pipeline failed")
    with pytest.raises(ValueError, match="Pipeline failed"):
        apply_preprocessing(pipeline_mock, data)

def test_apply_preprocessing_data_not_dataframe():
   pipeline_mock = MagicMock()
   with pytest.raises(TypeError):
        apply_preprocessing(pipeline_mock, [1,2,3])

