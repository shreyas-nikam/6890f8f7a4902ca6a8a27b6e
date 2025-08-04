
# Credit Rating Model Validation Lab

## 1. Notebook Overview

**Learning Goals:**

*   Understand the key steps involved in validating a wholesale credit rating model.
*   Apply statistical tests to assess model discrimination and calibration.
*   Implement a performance monitoring framework to detect model deterioration.
*   Interpret governance dashboards to identify potential model issues.
*   Create regulatory-ready validation reports and model inventory entries.
*   Formulate remediation plans based on SR 11-7 principles.

**Expected Outcomes:**

Upon completion of this lab, users will be able to:

*   Independently validate credit rating models.
*   Identify and address potential model weaknesses.
*   Design and implement model performance monitoring systems.
*   Contribute to effective model governance.
*   Prepare documentation for regulatory review.

## 2. Mathematical and Theoretical Foundations

### 2.1. Area Under the ROC Curve (AUC) and Gini Coefficient

The AUC measures the ability of a model to discriminate between defaulting and non-defaulting obligors. It represents the probability that a randomly chosen defaulting obligor will have a higher predicted probability of default (PD) than a randomly chosen non-defaulting obligor.

$$
AUC = P(\text{PD}_{\text{default}} > \text{PD}_{\text{non-default}})
$$

The Gini coefficient is derived from the AUC and represents the ratio of the area between the ROC curve and the line of no discrimination (the 45-degree line) to the area of the triangle formed by the line of perfect discrimination.

$$
Gini = 2 \times AUC - 1
$$

A Gini coefficient close to 1 indicates strong discriminatory power, while a value close to 0 suggests poor discrimination.

*Real-world application:*  Used to compare the discriminatory power of different models. A higher AUC/Gini indicates a better model.

### 2.2. Hosmer-Lemeshow Test

The Hosmer-Lemeshow test assesses the calibration of a model by comparing the predicted and observed default rates across different groups of obligors. The test statistic follows a chi-squared distribution.
The Hosmer-Lemeshow statistic (H) is calculated as:
$$H = \sum_{g=1}^G \frac{(O_g - E_g)^2}{N_g \pi_g (1 - \pi_g)}$$

Where:
* $G$ is the number of groups (usually deciles of predicted probabilities).
* $O_g$ is the observed number of defaults in group $g$.
* $E_g$ is the expected number of defaults in group $g$ (based on the model's predictions).
* $N_g$ is the total number of observations in group $g$.
* $\pi_g$ is the average predicted probability of default for group $g$.

The null hypothesis of the Hosmer-Lemeshow test is that the model is well-calibrated. A high p-value (typically > 0.05) indicates that the null hypothesis cannot be rejected, suggesting that the model is well-calibrated.

*Real-world application:* The Hosmer-Lemeshow test is used to check the overall model calibration.  If the model is poorly calibrated, the PDs might need to be adjusted.

### 2.3. Population Stability Index (PSI)

The PSI measures the change in the distribution of obligors across rating grades between two time periods (e.g., development and validation samples).

$$
PSI = \sum_{i=1}^{N} (Actual_i - Expected_i) \times ln(\frac{Actual_i}{Expected_i})
$$

Where:
* $N$ is the number of rating grades.
* $Actual_i$ is the percentage of obligors in grade $i$ in the validation sample.
* $Expected_i$ is the percentage of obligors in grade $i$ in the development sample.

PSI values are typically interpreted as follows:

*   PSI < 0.1: Insignificant change
*   0.1 <= PSI < 0.25: Minor change
*   PSI >= 0.25: Significant change

*Real-world application:* PSI indicates if the portfolio has changed significantly, which might require model recalibration or redevelopment.

### 2.4. Override Rate

The override rate is the percentage of model-generated ratings that are manually overridden by credit officers.

$$
\text{Override Rate} = \frac{\text{Number of Overrides}}{\text{Total Number of Ratings}} \times 100\%
$$

A high override rate may indicate model shortcomings or issues with the override policy.

*Real-world application:* Track the number of overrides, time-since-last validation, next review due, and open remediation actions.

## 3. Code Requirements

### 3.1. Expected Libraries

*   **pandas:** For data manipulation and analysis.
*   **numpy:** For numerical computations.
*   **scikit-learn:** For machine learning tasks, including model loading, prediction, and performance evaluation.
*   **matplotlib:** For creating visualizations.
*   **seaborn:** For enhanced statistical data visualization.
*   **pickle:** For loading pre-trained models.
*   **logging:** For generating warnings and alerts.
*   **yaml:** For reading and writing model inventory files.
*   **reportlab:** For generating PDF reports.

### 3.2. Input/Output Expectations

*   **Input:**
    *   Pre-trained models (`rating_logreg_v1.pkl`, `rating_gbt_v1.pkl`).
    *   Preprocessing pipeline (`preprocess_v1.pkl`).
    *   Grade cutoffs (`grade_cutoffs_v1.csv`).
    *   Out-of-time (OOT) sample (CSV).
    *   Quarterly portfolio snapshots (CSV files: `snap_YYYYQ.csv`).
    *   Override log (`overrides.csv`).
    *   Model inventory template (`model_inventory_entry.yaml`).
*   **Output:**
    *   Updated KPI datasets.
    *   Model validation report (PDF).
    *   Model inventory record (YAML).
    *   Visualizations (charts, tables, plots).
    *   Warnings and alerts (logged messages).

### 3.3. Algorithms and Functions

1.  **Model Loading:**
    *   Function to load pickled models and pipelines.
2.  **Prediction Generation:**
    *   Function to generate predictions using loaded models on new data.
3.  **Performance Evaluation:**
    *   Function to calculate AUC and Gini coefficient.
    *   Function to perform Hosmer-Lemeshow test.
    *   Function to calculate PSI.
4.  **Override Analysis:**
    *   Function to calculate override rates and generate override matrices.
5.  **Sensitivity Analysis:**
    *   Function to perform sensitivity testing by perturbing input variables.
6.  **Visualization:**
    *   Functions to generate line plots for AUC/Gini trends.
    *   Function to create bar-line combos for calibration back-testing.
    *   Function to generate PSI bar charts and meters.
    *   Function to create heat-maps for override matrices.
    *   Function to generate tornado plots for sensitivity analysis.
7.  **Report Generation:**
    *   Function to automatically generate a PDF validation report.
8.  **Alerting Logic:**
    *   Functions to raise Python logging warnings based on predefined thresholds for AUC drop, PSI, and override rates.
9.  **Model Inventory Management:**
    *   Function to read and write model inventory records in YAML format.

### 3.4. Visualizations

*   **Discrimination Trend:** Line plot of quarterly AUC & Gini with a red alert band when below approved limit (e.g. Gini < 55 %).
*   **Calibration Back-Test:** Grade-by-grade bar-line combo: predicted PD vs. realized default rate with Hosmer–Lemeshow $\chi^2$ overlay.
*   **Population Stability:** PSI bar chart by grade and an overall PSI meter; values >0.25 highlighted.
*   **Override Matrix:** Heat-map: rows = grade change (± notches), cols = override reasons.
*   **Sensitivity Tornado:** Tornado plot of Δ-PD by ±5 % shocks to top-five drivers to visualize robustness.
*   **Governance Dashboard:** Tabular KPI panel: number of overrides, time-since-last validation, next review due, open remediation actions.

## 4. Additional Notes or Instructions

*   **Assumptions:**
    *   Pre-trained models and datasets are available in the specified formats and locations.
    *   The OOT sample and quarterly snapshots have the same schema as the training data.
    *   Grade cutoffs are provided for mapping PDs to rating grades.
*   **Constraints:**
    *   The validation code should run in a separate virtual environment (env\_rmm.yml) to ensure reproducibility and independence.
    *   No model re-estimation is allowed.
    *   All raw and derived data should be stored under outputs/rmm\_data/ with timestamped sub-folders.
*   **Customization:**
    *   Thresholds for alerts (AUC drop, PSI, override rate) can be adjusted.
    *   The period for calculating KPIs (e.g., monthly, quarterly) can be modified.
    *   The sensitivity analysis parameters (e.g., perturbation size, top drivers) can be customized.
    *   The format of the validation report can be tailored to specific regulatory requirements.
*   **Notebook Flow:**

    1.  **Setup & Artefact Upload Check:** Stop execution if files are missing.
    2.  **Reload Models & Regenerate Predictions:** Reload pre-trained models and generate predictions on out-of-time data.
    3.  **Independent Validation Tests:** Perform conceptual soundness checks, calculate AUC, HL, and conduct back-testing.
    4.  **Monitoring Framework:** Build a monthly KPI dataset, compute PSI, overrides, and AUC trend.
    5.  **Governance Section:**
        *   Write `model_inventory_record.yaml` (model ID, owner, tier, validation dates).
        *   Auto-generate `rating_model_validation_report_2025.pdf` summarizing all tests.
    6.  **Alerting Logic:** Raise Python logging.warning when AUC drop ≥ 10 pp, PSI > 0.10, override-rate > 10 %.
    7.  **Reproducibility & Independence**
        *Validation code runs under a separate virtual-env (env\_rmm.yml) and uses the frozen artifacts – no re-estimation allowed.
        *All raw & derived data stored under outputs/rmm\_data/ with timestamped sub-folders; maintain an immutable audit log.

    8.  **Hand-off:** Notebook ends with a checklist confirming:
        *   Paths of updated KPI datasets and validation report.
        *   Any breaches raised.
        *   Next-scheduled validation date.
