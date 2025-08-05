import pytest
from definition_94b065cf5e47438ebd680c1976aa48d8 import load_model
import pickle
import os

@pytest.fixture
def dummy_model_file(tmp_path):
    model_path = tmp_path / "dummy_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump("dummy model", f)
    return str(model_path)

def test_load_model_success(dummy_model_file):
    model = load_model(dummy_model_file)
    assert model == "dummy model"

def test_load_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_model.pkl")

def test_load_model_invalid_file_type(tmp_path):
    invalid_file_path = tmp_path / "invalid_file.txt"
    with open(invalid_file_path, "w") as f:
        f.write("Not a pickle file")
    with pytest.raises(Exception): #Catching general exception because pickling error message varies.
        load_model(str(invalid_file_path))

def test_load_model_empty_path():
    with pytest.raises(TypeError):
        load_model(None)
        
def test_load_model_incorrect_permissions(tmp_path):
    model_path = tmp_path / "no_permission_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump("dummy model", f)
    os.chmod(model_path, 0o000)  # Remove all permissions
    try:
        with pytest.raises(PermissionError):
            load_model(str(model_path))
    finally:
        os.chmod(model_path, 0o777) # Restore permissions to avoid affecting other tests/system