import pytest
from definition_5a63da97299f422a9916a4e8f95a79ab import write_model_inventory_record
import yaml
import os

def cleanup(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

@pytest.fixture
def temp_yaml_file():
    filepath = "temp_model_inventory.yaml"
    yield filepath
    cleanup(filepath)

def test_write_model_inventory_record_success(temp_yaml_file):
    data = {"model_name": "Test Model", "version": "1.0"}
    write_model_inventory_record(data, temp_yaml_file)
    with open(temp_yaml_file, "r") as f:
        loaded_data = yaml.safe_load(f)
    assert loaded_data == data

def test_write_model_inventory_record_empty_data(temp_yaml_file):
    data = {}
    write_model_inventory_record(data, temp_yaml_file)
    with open(temp_yaml_file, "r") as f:
        loaded_data = yaml.safe_load(f)
    assert loaded_data == data

def test_write_model_inventory_record_invalid_path():
    data = {"model_name": "Test Model", "version": "1.0"}
    with pytest.raises(Exception):  # Expecting an exception due to invalid path
        write_model_inventory_record(data, "/invalid/path/model.yaml")
    
def test_write_model_inventory_record_complex_data(temp_yaml_file):
    data = {
        "model_name": "Complex Model",
        "version": "2.0",
        "metrics": {"auc": 0.85, "gini": 0.7},
        "features": ["feature1", "feature2"],
    }
    write_model_inventory_record(data, temp_yaml_file)
    with open(temp_yaml_file, "r") as f:
        loaded_data = yaml.safe_load(f)
    assert loaded_data == data

def test_write_model_inventory_record_overwrite_existing_file(temp_yaml_file):
    # Create an initial file
    with open(temp_yaml_file, "w") as f:
        yaml.dump({"initial_data": "old"}, f)

    # Write new data
    data = {"model_name": "New Model"}
    write_model_inventory_record(data, temp_yaml_file)

    # Check if the file has been overwritten
    with open(temp_yaml_file, "r") as f:
        loaded_data = yaml.safe_load(f)
    assert loaded_data == data
