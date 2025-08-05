import pytest
from definition_5df061c8401949f48ade0ab1f1aedb8c import write_model_inventory_record
import yaml
import os

@pytest.fixture
def temp_yaml_file(tmpdir):
    file_path = tmpdir.join("temp.yaml")
    return str(file_path)

def test_write_model_inventory_record_success(temp_yaml_file):
    model_inventory_entry = {"model_name": "test_model", "version": "1.0"}
    write_model_inventory_record(model_inventory_entry, temp_yaml_file)
    with open(temp_yaml_file, "r") as f:
        data = yaml.safe_load(f)
    assert data == model_inventory_entry

def test_write_model_inventory_record_empty_entry(temp_yaml_file):
    model_inventory_entry = {}
    write_model_inventory_record(model_inventory_entry, temp_yaml_file)
    with open(temp_yaml_file, "r") as f:
        data = yaml.safe_load(f)
    assert data == model_inventory_entry

def test_write_model_inventory_record_file_exists_overwrite(temp_yaml_file):
    with open(temp_yaml_file, "w") as f:
        yaml.dump({"initial_data": "test"}, f)
    model_inventory_entry = {"model_name": "test_model", "version": "1.0"}
    write_model_inventory_record(model_inventory_entry, temp_yaml_file)
    with open(temp_yaml_file, "r") as f:
        data = yaml.safe_load(f)
    assert data == model_inventory_entry

def test_write_model_inventory_record_invalid_file_path():
    model_inventory_entry = {"model_name": "test_model", "version": "1.0"}
    with pytest.raises(Exception):
        write_model_inventory_record(model_inventory_entry, "/invalid/path/to/file.yaml")

def test_write_model_inventory_record_complex_data(temp_yaml_file):
    model_inventory_entry = {
        "model_name": "complex_model",
        "version": "2.0",
        "features": ["feature1", "feature2"],
        "parameters": {"param1": 0.1, "param2": 0.2}
    }
    write_model_inventory_record(model_inventory_entry, temp_yaml_file)
    with open(temp_yaml_file, "r") as f:
        data = yaml.safe_load(f)
    assert data == model_inventory_entry
