import pytest
import pandas as pd
from definition_8006d070bf2545379ec1badb42160cd2 import load_data

def test_load_data_success(tmp_path):
    # Create a dummy CSV file
    file_path = tmp_path / "test.csv"
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df.to_csv(file_path, index=False)

    # Load the data
    loaded_df = load_data(str(file_path))

    # Assert that the data is loaded correctly
    pd.testing.assert_frame_equal(loaded_df, df)

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")

def test_load_data_empty_file(tmp_path):
    # Create an empty CSV file
    file_path = tmp_path / "empty.csv"
    file_path.touch()

    # Load the data
    loaded_df = load_data(str(file_path))

    # Assert that the data is loaded correctly
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.empty

def test_load_data_invalid_csv(tmp_path):
    # Create a CSV file with invalid data
    file_path = tmp_path / "invalid.csv"
    with open(file_path, "w") as f:
        f.write("col1,col2\n1,a\n2,b")

    # Load the data and expect a parser error
    with pytest.raises(pd.errors.ParserError):
        load_data(str(file_path))

def test_load_data_wrong_extension(tmp_path):
    # Create a dummy text file
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("col1,col2\n1,2\n3,4")

    # Try loading the data and expect a general exception or error.
    # This depends on the implementation in load_data if extension type is checked or not.
    with pytest.raises((pd.errors.ParserError,Exception)):
        load_data(str(file_path))
