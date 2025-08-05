import pickle

def load_model(model_path):
    """Loads a model from a .pkl file."""
    if model_path is None:
        raise TypeError("Model path cannot be None")
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {e}")

import pandas as pd

def load_data(data_path):
    """Loads data from a .csv file.
    Args:
        data_path: Path to the .csv file.
    Returns:
        A pandas DataFrame.
    """
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {data_path}")
    except pd.errors.ParserError:
        raise pd.errors.ParserError(f"Error parsing CSV file: {data_path}")

def apply_preprocessing(data, pipeline):
    """Applies a pre-processing pipeline to the input data."""
    if pipeline is None:
        raise TypeError
    try:
        transformed_data = pipeline.transform(data)
        return transformed_data
    except Exception as e:
        raise e

import numpy as np
from sklearn import metrics

def calculate_auc_gini(y_true, y_pred):
    """Calculates the AUC and Gini coefficient.

    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.

    Returns:
        A tuple containing the AUC and Gini coefficient.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    try:
        auc = metrics.roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
    except ValueError:
        return np.nan, np.nan

    return auc, gini

import numpy as np
from scipy import stats

def perform_hosmer_lemeshow(y_true, y_pred, groups):
    """Performs the Hosmer-Lemeshow test."""

    bins = np.linspace(0, 1, groups + 1)
    indices = np.digitize(y_pred, bins)

    observed = []
    expected = []

    for i in range(1, groups + 1):
        bin_true = y_true[indices == i]
        bin_pred = y_pred[indices == i]

        n = len(bin_true)
        if n > 0:
            observed_positives = np.sum(bin_true)
            expected_positives = np.sum(bin_pred)

            observed.append([observed_positives, n - observed_positives])
            expected.append([expected_positives, n - expected_positives])
        else:
            observed.append([0, 0])
            expected.append([0, 0])

    observed = np.array(observed)
    expected = np.array(expected)

    statistic = 0
    for i in range(groups):
        if expected[i, 0] > 0 and expected[i, 1] > 0:
            statistic += (observed[i, 0] - expected[i, 0])**2 / expected[i, 0]
            statistic += (observed[i, 1] - expected[i, 1])**2 / expected[i, 1]
        else:
            statistic = np.nan
            break


    if np.isnan(statistic):
        p_value = np.nan
    else:
        degrees_of_freedom = groups - 2
        p_value = 1 - stats.chi2.cdf(statistic, degrees_of_freedom)

    return statistic, p_value

import numpy as np

def generate_calibration_curve(y_true, y_pred, n_bins):
    """Generates calibration curve data."""

    bins = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_pred, bins) - 1

    observed_rates = []
    actual_bins = []

    for i in range(n_bins):
        indices = np.where(bin_assignments == i)[0]
        if len(indices) > 0:
            observed_rate = np.mean(np.array(y_true)[indices])
            observed_rates.append(observed_rate)
            actual_bins.append((bins[i] + bins[i+1])/2)

    return np.array(actual_bins), np.array(observed_rates)

import pandas as pd
import numpy as np

def compute_psi(expected, actual):
    """Computes the Population Stability Index (PSI) for each category."""

    if not expected.index.equals(actual.index):
        raise ValueError("Expected and actual distributions must have the same categories.")

    if any(expected == 0):
        raise ValueError("Expected distribution cannot contain zero probabilities.")

    psi = pd.Series(index=expected.index, dtype='float64')
    for category in expected.index:
        expected_prob = expected[category]
        actual_prob = actual[category]

        if actual_prob == 0:
            psi[category] = (actual_prob - expected_prob) * np.log(actual_prob/expected_prob)
        else:
            psi[category] = (actual_prob - expected_prob) * np.log(actual_prob / expected_prob)

    return psi

import numpy as np

def compute_overall_psi(expected, actual):
    """Computes the overall Population Stability Index (PSI)."""

    expected = np.array(expected)
    actual = np.array(actual)

    if np.any(expected == 0) or np.any(actual == 0):
        raise RuntimeError("PSI is infinite when a bin has zero expected or actual count.")

    psi_values = (actual - expected) * np.log(actual / expected)
    overall_psi = np.sum(psi_values)

    return overall_psi

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def track_auc_gini_drift(historical_data, y_true, y_pred, time_period):
    """Tracks AUC/Gini drift over time.
    Args:
        historical_data: Historical AUC/Gini data.
        y_true: True labels.
        y_pred: Predicted probabilities.
        time_period: Time period.
    Returns:
        Updated historical data.
    """

    if not all((y_pred >= 0) & (y_pred <= 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    try:
        auc = roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
    except ValueError:
        auc = np.nan
        gini = np.nan

    new_data = pd.DataFrame({'time_period': [time_period], 'auc': [auc], 'gini': [gini]})
    updated_data = pd.concat([historical_data, new_data], ignore_index=True)

    return updated_data

def calculate_override_rate(number_of_overrides, total_number_of_applications):
                """Calculates the override rate."""
                if not isinstance(number_of_overrides, (int, float)) or not isinstance(total_number_of_applications, (int, float)):
                    raise TypeError("Inputs must be numeric.")
                if total_number_of_applications == 0:
                    raise ZeroDivisionError("Total number of applications cannot be zero.")
                return (number_of_overrides / total_number_of_applications) * 100

import pandas as pd

def generate_override_matrix(overrides_data):
    """Generates the override matrix (heatmap).
    Args:
        overrides_data: DataFrame containing override information.
    Output:
        A pandas DataFrame representing the override matrix.
    """
    if overrides_data.empty:
        return pd.DataFrame()

    override_matrix = pd.crosstab(overrides_data['Grade_Change'], overrides_data['Override_Reason'])
    return override_matrix

import pandas as pd

def perform_sensitivity_analysis(model, data, top_drivers, perturbation):
    """Performs sensitivity analysis by perturbing the top drivers."""

    results = []
    original_predictions = model.predict_proba(data)[:, 1]  # PD predictions

    for driver in top_drivers:
        if driver not in data.columns:
            raise KeyError(f"Driver '{driver}' not found in data.")

        original_values = data[driver].copy()
        data[driver] = data[driver] * (1 + perturbation)
        perturbed_predictions = model.predict_proba(data)[:, 1]
        change_in_pd = perturbed_predictions - original_predictions
        mean_change = change_in_pd.mean()

        results.append({'driver': driver, 'change_in_pd': mean_change})
        data[driver] = original_values  # Restore original values

    return pd.DataFrame(results)

def generate_kpi_panel(number_of_overrides, time_since_last_validation, next_review_due, open_remediation_actions):
    """Generates the tabular KPI panel."""

    return {
        'number_of_overrides': number_of_overrides,
        'time_since_last_validation': time_since_last_validation,
        'next_review_due': next_review_due,
        'open_remediation_actions': open_remediation_actions
    }

import yaml
import os

def write_model_inventory_record(model_inventory_entry, yaml_file_path):
    """Writes model inventory record to a .yaml file."""
    try:
        with open(yaml_file_path, "w") as yaml_file:
            yaml.dump(model_inventory_entry, yaml_file)
    except Exception as e:
        raise e

from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            import os

            def generate_model_validation_report(report_data, output_path):
                """Generates the model validation report in .pdf format."""

                try:
                    c = canvas.Canvas(output_path, pagesize=letter)
                    c.setTitle("Model Validation Report")

                    if report_data:
                        title = report_data.get("title", "Model Validation Report")
                        c.drawString(100, 750, title)

                        sections = report_data.get("sections", [])
                        y_position = 730
                        for section in sections:
                            section_title = section.get("title", "Section")
                            c.drawString(120, y_position, section_title)
                            y_position -= 20
                            content = section.get("content", "")
                            c.drawString(140, y_position, content)
                            y_position -= 30
                    c.save()

                except Exception as e:
                    raise e

import logging

def alert_on_performance_degradation(auc_drop, psi, override_rate):
    """Raises Python logging warnings based on performance degradation thresholds."""
    if auc_drop > 0.1:
        logging.warning("AUC drop is above threshold: %s", auc_drop)
    if psi > 0.1:
        logging.warning("PSI is above threshold: %s", psi)
    if override_rate > 0.1:
        logging.warning("Override rate is above threshold: %s", override_rate)