
# Credit Model Governance Dashboard - Jupyter Notebook Specification

## 1. Notebook Overview

**Learning Goals:**

*   Understand the key principles of model risk management, including validation, compliance, and governance.
*   Learn how to independently validate credit risk rating models.
*   Understand how to implement continuous performance monitoring for credit risk models.
*   Learn how to design and interpret governance dashboards for override activity and sensitivity testing.
*   Learn how to apply SR 11-7 principles for effective challenge and control.

**Expected Outcomes:**

Upon completion of this notebook, users will be able to:

*   Independently validate a credit risk rating model.
*   Set up continuous performance monitoring for a credit risk rating model.
*   Design and interpret governance dashboards for override activity and sensitivity testing.
*   Produce a regulator-ready validation report.
*   Register the model in the enterprise model inventory.
*   Formulate remediation plans when tests fail or governance triggers fire.

## 2. Mathematical and Theoretical Foundations

### 2.1. Population Stability Index (PSI)

The Population Stability Index (PSI) measures the change in the distribution of a characteristic (e.g., rating grade) between two samples (e.g., development sample and current portfolio).  It is calculated as follows:

$$ PSI = \sum_{i=1}^{N} (Actual\%_{i} - Expected\%_{i}) * ln(\frac{Actual\%_{i}}{Expected\%_{i}}) $$

where:

*   $N$ is the number of categories (e.g., rating grades).
*   $Actual\%_{i}$ is the percentage of observations in category $i$ in the current portfolio (the 'actual' distribution).
*   $Expected\%_{i}$ is the percentage of observations in category $i$ in the development sample (the 'expected' distribution).
*   $ln$ is the natural logarithm.

Real-world applications: PSI is used to monitor the stability of a model's inputs and outputs over time. A high PSI value indicates a significant shift in the population, which may warrant model recalibration or redevelopment.  Common thresholds for PSI are:

*   PSI < 0.1:  No significant change.
*   0.1 <= PSI < 0.25:  Moderate change.
*   PSI >= 0.25: Significant change, potentially requiring action.

### 2.2. Area Under the Receiver Operating Characteristic Curve (AUC) and Gini Coefficient

The Area Under the Receiver Operating Characteristic Curve (AUC) is a metric that assesses a binary classification model's ability to distinguish between positive and negative classes. The Receiver Operating Characteristic (ROC) curve plots the true positive rate (sensitivity) against the false positive rate (1 - specificity) at various threshold settings. The AUC represents the area under this curve. An AUC of 0.5 indicates random classification, while an AUC of 1.0 indicates perfect classification.

The Gini coefficient is directly related to the AUC:

$$Gini = 2 * AUC - 1$$

A Gini coefficient of 0 indicates random classification, while a Gini coefficient of 1 indicates perfect classification.
AUC and Gini coefficient are measures of model discrimination.

### 2.3. Hosmer-Lemeshow Test

The Hosmer-Lemeshow test is a statistical test for assessing the calibration of a binary classification model. Calibration refers to the agreement between predicted probabilities and observed outcomes. The test groups observations into deciles (or other groups) based on predicted probabilities and compares the observed number of events in each group to the expected number of events.

The Hosmer-Lemeshow test statistic is calculated as:

$$ H = \sum_{g=1}^{G} \frac{(O_{g} - E_{g})^2}{N_{g} * \bar{p}_{g} * (1 - \bar{p}_{g})} $$

where:

*   $G$ is the number of groups (e.g., deciles).
*   $O_{g}$ is the observed number of events in group $g$.
*   $E_{g}$ is the expected number of events in group $g$ based on the predicted probabilities.
*   $N_{g}$ is the number of observations in group $g$.
*   $\bar{p}_{g}$ is the average predicted probability in group $g$.

The test statistic follows a chi-square distribution with $G - 2$ degrees of freedom. A high p-value indicates that the model is well-calibrated.

### 2.4. Sensitivity Analysis

Sensitivity analysis involves perturbing the input variables of a model to assess their impact on the model's output. This helps to understand the model's robustness and identify the key drivers of its predictions.  In this context, we will perturb the top five model drivers by $\pm 5\%$ and observe the change in predicted probability of default (PD).

Let $PD_0$ be the original predicted PD and $PD_i$ be the predicted PD after perturbing the $i^{th}$ driver. The sensitivity can be expressed as:

$$ \Delta PD_i = PD_i - PD_0 $$

A large $\Delta PD_i$ indicates that the model is sensitive to changes in the $i^{th}$ driver.

### 2.5 Override Rate

The override rate is the percentage of model outputs that are manually overridden by experts. It's calculated as follows:

$$ OverrideRate = \frac{NumberOfOverrides}{TotalNumberOfModelApplications} * 100 $$

A high override rate may indicate issues with the model's accuracy, relevance, or applicability.

## 3. Code Requirements

### 3.1. Expected Libraries

*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical computations.
*   `scikit-learn`: For machine learning algorithms and model evaluation.
*   `matplotlib`: For creating visualizations.
*   `seaborn`: For creating statistical graphics.
*   `yaml`: For reading and writing YAML files.
*   `pickle`: For saving and loading models.
*   `logging`: for alerting

### 3.2. Input/Output Expectations

**Inputs:**

*   Pre-trained champion and challenger rating models (e.g., logistic regression, gradient boosted trees) in `.pkl` format (`rating_logreg_v1.pkl`, `rating_gbt_v1.pkl`).
*   Pre-processing pipeline in `.pkl` format (`preprocess_v1.pkl`).
*   Rating grade cutoffs in `.csv` format (`grade_cutoffs_v1.csv`).
*   Out-of-time (OOT) sample in `.csv` format.
*   Quarterly portfolio snapshots in `.csv` format (`snap_YYYYQ.csv`).
*   Override log in `.csv` format (`overrides.csv`).
*   Model inventory record template in `.yaml` format (`model_inventory_entry.yaml`).

**Outputs:**

*   Line plot of quarterly AUC & Gini with red alert band.
*   Grade-by-grade bar-line combo: predicted PD vs. realized default rate with Hosmer–Lemeshow χ² overlay.
*   PSI bar chart by grade and an overall PSI meter.
*   Heat-map: rows = grade change (± notches), cols = override reasons.
*   Tornado plot of Δ-PD by ±5 % shocks to top-five drivers.
*   Tabular KPI panel: number of overrides, time-since-last validation, next review due, open remediation actions.
*   Updated KPI datasets stored under `outputs/rmm_data/`.
*   Model validation report (`rating_model_validation_report_2025.pdf`).

### 3.3. Algorithms or Functions to be Implemented

*   **Data Loading and Preprocessing:**
    *   Function to load models and pipelines from `.pkl` files.
    *   Function to load data from `.csv` files.
    *   Function to apply the pre-processing pipeline to new data.
*   **Model Validation:**
    *   Function to calculate AUC and Gini coefficient on OOT data.
    *   Function to perform Hosmer-Lemeshow test.
    *   Function to generate calibration curves (predicted PD vs. realized default rate).
*   **Continuous Performance Monitoring:**
    *   Function to compute PSI for each rating grade.
    *   Function to compute overall PSI.
    *   Function to track AUC/Gini drift over time.
    *   Function to calculate override rates.
*   **Governance Dashboard:**
    *   Function to generate the override matrix (heatmap).
    *   Function to perform sensitivity analysis (tornado plot).
    *   Function to generate the tabular KPI panel.
*   **Reporting:**
    *   Function to write model inventory record to `.yaml` file.
    *   Function to generate the model validation report in `.pdf` format.
*   **Alerting Logic:**
    *   Function to raise Python logging warnings when:
        *   AUC drop >= 10 percentage points.
        *   PSI > 0.10.
        *   Override rate > 10%.

### 3.4. Visualization

The notebook will generate the following visualizations:

*   **Discrimination trend**: Line plot of quarterly AUC & Gini with red alert band when below approved limit (e.g. Gini < 55 %).
*   **Calibration back-test**: Grade-by-grade bar-line combo: predicted PD vs. realized default rate with Hosmer–Lemeshow χ² overlay.
*   **Population-stability**: PSI bar chart by grade and an overall PSI meter; values >0.25 highlighted.
*   **Override matrix**: Heat-map: rows = grade change (± notches), cols = override reasons.
*   **Sensitivity tornado**: Tornado plot of Δ-PD by ±5 % shocks to top-five drivers to visualise robustness.
*   **Governance dashboard**: Tabular KPI panel: number of overrides, time-since-last validation, next review due, open remediation actions.

## 4. Additional Notes or Instructions

*   **Reproducibility:** Set a fixed random seed (e.g., `numpy.random.seed(42)`) at the beginning of the notebook to ensure reproducibility.
*   **Artifact Handling:** Assume the existence of output data/models from part 1.
*   **Data-logging & Archiving:**  Maintain a YAML data-log recording every transformation for audit purposes. Store all raw & derived data under `outputs/rmm_data/` with timestamped sub-folders.
*   **No Re-estimation:** Validation code runs using frozen artifacts – no re-estimation allowed.
*   **SR 11-7 Alignment:** Map each test to SR 11-7 categories: conceptual soundness, process verification, outcomes analysis.
*   **UAE Central Bank Alignment:** Embed UAE Central Bank governance expectations for annual board-level reporting and remedial action tracking.
*   **Hand-off Checklist:**  The notebook ends with a checklist confirming:
    *   paths of updated KPI datasets and validation report.
    *   any breaches raised.
    *   next-scheduled validation date.
*   **Override Reason Codes:** Define a clear set of override reason codes for the override matrix.
*   **Assumptions:** Assume that the input datasets have consistent schemas. Assume that thresholds such as acceptable Gini, PSI, and override rates are provided by the user or defined in a configuration file.

