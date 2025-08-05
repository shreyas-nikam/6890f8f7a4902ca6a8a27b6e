import pytest
import pandas as pd
from definition_67ec495cb0814796ba5677fa75db8bb5 import generate_override_matrix

@pytest.fixture
def sample_overrides():
    data = {'grade_before': ['A', 'B', 'C', 'B', 'A'],
            'grade_after': ['A', 'C', 'C', 'B', 'B'],
            'reason_code': ['Reason1', 'Reason2', 'Reason1', 'Reason3', 'Reason2']}
    return pd.DataFrame(data)

@pytest.fixture
def grade_levels():
    return ['A', 'B', 'C']

@pytest.fixture
def reason_codes():
    return ['Reason1', 'Reason2', 'Reason3']


def test_generate_override_matrix_typical(sample_overrides, grade_levels, reason_codes):
    result = generate_override_matrix(sample_overrides, grade_levels, reason_codes)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(grade_levels), len(reason_codes))

def test_generate_override_matrix_empty_overrides(grade_levels, reason_codes):
    empty_overrides = pd.DataFrame({'grade_before': [], 'grade_after': [], 'reason_code': []})
    result = generate_override_matrix(empty_overrides, grade_levels, reason_codes)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(grade_levels), len(reason_codes))
    assert (result.values == 0).all()

def test_generate_override_matrix_empty_grade_levels(sample_overrides, reason_codes):
    result = generate_override_matrix(sample_overrides, [], reason_codes)
    assert isinstance(result, pd.DataFrame)
    assert result.empty #DataFrame should be empty if no Grade Level exists

def test_generate_override_matrix_empty_reason_codes(sample_overrides, grade_levels):
    result = generate_override_matrix(sample_overrides, grade_levels, [])
    assert isinstance(result, pd.DataFrame)
    assert result.empty #DataFrame should be empty if no Reason Code exists
    
def test_generate_override_matrix_unseen_grade_and_reason(sample_overrides, grade_levels, reason_codes):
    sample_overrides.loc[len(sample_overrides.index)] = ['D', 'E', 'Reason4'] 
    result = generate_override_matrix(sample_overrides, grade_levels, reason_codes)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (len(grade_levels), len(reason_codes)) #Check that dataframe shape still matches
