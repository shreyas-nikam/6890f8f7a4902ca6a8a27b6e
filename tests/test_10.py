import pytest
import pandas as pd
from definition_e5d24edb525849179f02050a048f98ca import generate_override_matrix

@pytest.fixture
def sample_overrides_data():
    data = {
        'Grade_Change': [1, -1, 0, 2, -2, 1, 0],
        'Override_Reason': ['Credit_Judgement', 'Data_Error', 'Model_Limitation', 'Credit_Judgement', 'Other', 'Model_Limitation', 'Data_Error']
    }
    return pd.DataFrame(data)

def test_generate_override_matrix_empty_data():
    empty_df = pd.DataFrame({'Grade_Change': [], 'Override_Reason': []})
    result = generate_override_matrix(empty_df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_generate_override_matrix_basic(sample_overrides_data):
    result = generate_override_matrix(sample_overrides_data)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'Credit_Judgement' in result.columns
    assert 1 in result.index

def test_generate_override_matrix_nan_values():
    data = {'Grade_Change': [1, -1, 0, float('nan')], 'Override_Reason': ['Credit_Judgement', 'Data_Error', 'Model_Limitation', 'Other']}
    df = pd.DataFrame(data)
    result = generate_override_matrix(df)
    assert isinstance(result, pd.DataFrame)


def test_generate_override_matrix_single_override_reason(sample_overrides_data):
    data = {'Grade_Change': [1, -1, 0, 2, -2, 1, 0],
            'Override_Reason': ['Credit_Judgement'] * 7}
    df = pd.DataFrame(data)
    result = generate_override_matrix(df)
    assert isinstance(result, pd.DataFrame)
    assert 'Credit_Judgement' in result.columns


def test_generate_override_matrix_integer_grade_change(sample_overrides_data):
    result = generate_override_matrix(sample_overrides_data)
    assert isinstance(result, pd.DataFrame)
    assert all(isinstance(idx, (int)) for idx in result.index)

