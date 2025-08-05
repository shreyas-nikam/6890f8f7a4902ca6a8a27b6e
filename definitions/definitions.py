import pickle
import os

def load_model(model_path):
    """Loads a pre-trained model from a .pkl file."""
    if model_path is None:
        raise TypeError("Model path cannot be None")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except PermissionError:
        raise PermissionError(f"Insufficient permissions to read model file at {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {e}")

import pandas as pd

def load_data(data_path):
    """Loads data from a .csv file using pandas."""
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError
    except pd.errors.ParserError:
        raise pd.errors.ParserError
    except Exception as e:
        raise e

import pandas as pd

            def apply_preprocessing(pipeline, data):
                """Applies a pre-processing pipeline to the input data.
                Args:
                    pipeline: The pre-processing pipeline object.
                    data (pandas.DataFrame): The data to be pre-processed.
                Returns:
                    pandas.DataFrame: The pre-processed data.
                """
                if not isinstance(data, pd.DataFrame):
                    raise TypeError("Input data must be a pandas DataFrame.")
                try:
                    transformed_data = pipeline.transform(data)
                    return transformed_data
                except Exception as e:
                    raise e

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc_gini(model, X, y):
    """Calculates the AUC and Gini coefficient for a given model and data."""
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        gini = 2 * auc - 1
        return auc, gini
    except ValueError:
        return np.nan, np.nan
    except AttributeError:
        return np.nan, np.nan

import numpy as np
from scipy import stats

def perform_hosmer_lemeshow_test(y_true, y_prob, n_groups):
    """Performs the Hosmer-Lemeshow test for model calibration.

    Args:
        y_true (array-like): The true labels.
        y_prob (array-like): The predicted probabilities.
        n_groups (int): The number of groups to use.

    Returns:
        tuple: The Hosmer-Lemeshow test statistic and p-value.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if not np.issubdtype(y_prob.dtype, np.number):
        raise TypeError("y_prob must be numeric")

    if n_groups > len(y_true):
        raise ValueError("n_groups cannot be greater than the number of samples")

    bins = np.linspace(0, 1, n_groups + 1)
    indices = np.digitize(y_prob, bins)

    observed = []
    expected = []
    for i in range(1, n_groups + 1):
        group_true = y_true[indices == i]
        group_prob = y_prob[indices == i]
        n_obs = len(group_true)
        if n_obs > 0:
            observed_positives = np.sum(group_true)
            expected_positives = np.sum(group_prob)
            observed.append(observed_positives)
            expected.append(expected_positives)
        else:
            observed.append(0)
            expected.append(0)

    observed = np.array(observed)
    expected = np.array(expected)

    statistic = np.sum((observed - expected)**2 / (expected + 1e-8))
    degrees_of_freedom = n_groups - 2
    p_value = 1 - stats.chi2.cdf(statistic, degrees_of_freedom)

    return statistic, p_value

import numpy as np

def generate_calibration_curve(y_true, y_prob, n_bins):
    """Generates calibration curves."""

    if n_bins <= 0:
        raise ValueError("Number of bins must be greater than zero.")

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    indices = np.argsort(y_prob)
    y_prob_sorted = y_prob[indices]
    y_true_sorted = y_true[indices]

    bin_boundaries = np.linspace(0, len(y_true), n_bins + 1, dtype=int)
    bin_means = np.zeros(n_bins)
    bin_proportions = np.zeros(n_bins)

    for i in range(n_bins):
        start = bin_boundaries[i]
        end = bin_boundaries[i+1]
        bin_true = y_true_sorted[start:end]
        bin_prob = y_prob_sorted[start:end]

        if len(bin_true) > 0:
            bin_means[i] = np.mean(bin_prob)
            bin_proportions[i] = np.mean(bin_true)
        else:
            bin_means[i] = np.nan
            bin_proportions[i] = np.nan

    return bin_means, bin_proportions

import pandas as pd
import numpy as np

def compute_psi(expected, actual, grade_names):
    """Computes the Population Stability Index (PSI) for each rating grade.
    Args:
        expected (pandas.Series): The expected distribution of rating grades.
        actual (pandas.Series): The actual distribution of rating grades.
        grade_names (list): List of rating grade names.
    Output:
        pandas.DataFrame: A DataFrame containing the PSI for each grade.
    Raises:
        ValueError: If expected or actual contains zero values, or if the grade names are mismatched.
    """

    if any(expected <= 0):
        raise ValueError("Expected distribution contains zero values.")
    if any(actual <= 0):
        raise ValueError("Actual distribution contains zero values.")
    if len(expected) != len(actual):
        raise ValueError("Mismatched number of grades between expected and actual.")
    if not all(expected.index == actual.index):
        raise ValueError("Grade names are mismatched between expected and actual distributions.")

    psi_values = []
    for grade in grade_names:
        expected_pct = expected[grade]
        actual_pct = actual[grade]
        psi = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi_values.append(psi)

    result = pd.DataFrame({'Grade': grade_names, 'PSI': psi_values})
    result = result.set_index('Grade')
    return result

import numpy as np

def compute_overall_psi(expected, actual):
    """Computes the overall Population Stability Index (PSI)."""

    expected = np.array(expected)
    actual = np.array(actual)

    if np.any(expected == 0) or np.any(actual == 0):
        raise ValueError("PSI is infinite when a bin has zero expected or actual count.")

    psi_summary = (actual - expected) * np.log(actual / expected)
    psi = np.sum(psi_summary)

    return psi

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def track_auc_gini_drift(model, X_snapshots, y_snapshots, time_periods):
    """Tracks AUC/Gini drift over time."""

    if not (len(X_snapshots) == len(y_snapshots) == len(time_periods)):
        raise ValueError("X_snapshots, y_snapshots, and time_periods must have the same length.")

    results = []
    for i in range(len(X_snapshots)):
        X = X_snapshots[i]
        y = y_snapshots[i]
        time_period = time_periods[i]

        if set(y.unique()) != {0, 1}:
            if len(y.unique()) > 2:
                raise ValueError("Target variable must be binary (0 and 1).")
            else:
                auc = np.nan
                gini = np.nan
                results.append({'time_period': time_period, 'auc': auc, 'gini': gini})
                continue

        try:
            y_pred_proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
            gini = 2 * auc - 1
        except ValueError:
            auc = np.nan
            gini = np.nan

        results.append({'time_period': time_period, 'auc': auc, 'gini': gini})

    return pd.DataFrame(results)

def calculate_override_rate(num_overrides, total_applications):
                """Calculates the override rate."""
                if total_applications == 0:
                    raise ZeroDivisionError("Cannot divide by zero")
                return (num_overrides / total_applications) * 100

import pandas as pd

def generate_override_matrix(overrides, grade_levels, reason_codes):
    """Generates the override matrix (heatmap data)."""

    if not grade_levels or not reason_codes:
        return pd.DataFrame()

    matrix = pd.DataFrame(0, index=grade_levels, columns=reason_codes)

    for _, row in overrides.iterrows():
        grade_before = row['grade_before']
        grade_after = row['grade_after']
        reason_code = row['reason_code']

        if grade_before in grade_levels: #Only update the matrix for valid grades
            if reason_code in reason_codes: #Only update the matrix for valid reason codes
                matrix.loc[grade_before, reason_code] += 1

    return matrix

import pandas as pd
import numpy as np

def perform_sensitivity_analysis(model, X, top_drivers, perturbation):
    """Performs sensitivity analysis by perturbing top model drivers."""
    results = []
    X_original = X.copy()
    original_predictions = model.predict(X_original)

    for driver in top_drivers:
        X_perturbed = X.copy()
        X_perturbed[driver] = X_perturbed[driver] * (1 + perturbation)
        perturbed_predictions = model.predict(X_perturbed)
        delta_pd = perturbed_predictions - original_predictions
        results.append(np.mean(delta_pd))  # Use mean change as the sensitivity metric

    df = pd.DataFrame(results, index=top_drivers, columns=['delta_PD'])
    return df

def generate_kpi_panel(num_overrides, time_since_last_validation, next_review_due, open_remediation_actions):
    """Generates the tabular KPI panel data."""

    kpi_panel_data = {
        "num_overrides": num_overrides,
        "time_since_last_validation": time_since_last_validation,
        "next_review_due": next_review_due,
        "open_remediation_actions": open_remediation_actions,
    }

    return kpi_panel_data

import yaml

def write_model_inventory_record(data, output_path):
    """Writes model inventory record to a .yaml file."""
    try:
        with open(output_path, "w") as f:
            yaml.dump(data, f)
    except Exception as e:
        raise Exception(f"Error writing to YAML file: {e}")

import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


def generate_validation_report(data, output_path):
    """Generates model validation report in .pdf format."""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Model Validation Report", styles['h1']))
        for key, value in data.items():
            story.append(Paragraph(f"{key}: {value}", styles['Normal']))

        doc.build(story)
    except Exception as e:
        raise Exception(f"Error generating PDF report: {e}")

import logging

def raise_alerts(auc_drop, psi, override_rate):
    """Raises Python logging warnings based on defined thresholds."""

    if auc_drop > 0.1:
        logging.warning(f"AUC drop is high: {auc_drop}")
    if psi > 0.1:
        logging.warning(f"PSI is high: {psi}")
    if override_rate > 0.1:
        logging.warning(f"Override rate is high: {override_rate}")