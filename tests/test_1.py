import pytest
import pandas as pd
from definition_dd68be6de8d44ac1b77104477acd37a9 import load_data

def test_load_data_valid_csv():
    # Create a dummy CSV file for testing
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    df.to_csv('test.csv', index=False)
    
    loaded_df = load_data('test.csv')
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.shape == (2, 2)
    
def test_load_data_empty_csv():
    # Create an empty CSV file
    with open('empty.csv', 'w') as f:
        pass
    
    loaded_df = load_data('empty.csv')
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.empty
    
def test_load_data_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent.csv')

def test_load_data_corrupted_csv():
    # Create a corrupted CSV file
    with open('corrupted.csv', 'w') as f:
        f.write('col1,col2\n1,2\n3')  # Incomplete row

    with pytest.raises(pd.errors.ParserError):
        load_data('corrupted.csv')

def test_load_data_non_csv_file():
     # Create a dummy text file
    with open('test.txt', 'w') as f:
        f.write("This is not a csv")
    with pytest.raises(pd.errors.ParserError):
        load_data('test.txt')
