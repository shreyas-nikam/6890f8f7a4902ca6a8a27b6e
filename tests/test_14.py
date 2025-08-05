import pytest
from definition_401ac66175cc421a9718ae063f71e007 import generate_validation_report
import os
import tempfile
import shutil

def is_pdf_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            return f.read(4) == b'%PDF'
    except FileNotFoundError:
        return False
    except Exception:
        return False


def test_generate_validation_report_creates_pdf():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.pdf")
        data = {"some": "data"}  # Minimal data to allow the function to proceed
        generate_validation_report(data, output_path)
        assert os.path.exists(output_path)
        assert is_pdf_file(output_path)

def test_generate_validation_report_handles_empty_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.pdf")
        data = {}  # Empty Data
        generate_validation_report(data, output_path)
        assert os.path.exists(output_path)
        assert is_pdf_file(output_path)

def test_generate_validation_report_handles_invalid_output_path():
     with pytest.raises(Exception):  # Expect some error due to bad path. Exact error may depend on the implementation.
         data = {"some": "data"}
         generate_validation_report(data, "/invalid/path/report.pdf")

def test_generate_validation_report_with_example_data():
     with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.pdf")
        data = {
            "AUC": 0.8,
            "Gini": 0.6,
            "PSI": 0.05,
            "override_rate": 0.02
        }
        generate_validation_report(data, output_path)
        assert os.path.exists(output_path)
        assert is_pdf_file(output_path)


def test_generate_validation_report_overwrites_existing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.pdf")
        # Create a dummy file
        with open(output_path, "w") as f:
            f.write("This is a dummy file.")

        data = {"some": "data"}
        generate_validation_report(data, output_path)
        assert os.path.exists(output_path)
        assert is_pdf_file(output_path)