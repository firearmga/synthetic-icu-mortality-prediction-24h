# Synthetic ICU Mortality Prediction Using Machine Learning

A machine-learning project for predicting ICU patient mortality using a synthetic 24-hour ICU dataset containing structured clinical variables and LDA-style topic features.

The project is inspired by:

> Chiu et al. (2022), *Predicting the Mortality of ICU Patients by Topic Model with Machine-Learning Techniques*, Healthcare, 10(6), 1087.

**Important:** This project does not use real MIMIC-III patient data. The dataset is completely synthetic and is intended for educational, research, and machine-learning experimentation.

---

## Project Overview

Early prediction of mortality among ICU patients is an important problem in clinical machine learning.

This project develops and evaluates machine-learning models using a synthetic dataset representing information available during the first 24 hours of ICU care.

The dataset combines:

- Demographic characteristics
- Physiological measurements
- Laboratory measurements
- ICU admission characteristics
- Multiple mortality outcomes
- LDA-style topic probability features mimicking patients' clinical texts

The main objective is to investigate how different machine-learning algorithms perform when predicting ICU mortality from these features.

---

## Dataset

This project uses only the 24-hour synthetic ICU dataset.

The dataset contains:

- **27,809** synthetic ICU patients

and includes structured clinical variables, topic-model features, and five binary mortality outcomes.

The dataset is generated using the accompanying Python script:

```
src/generate_synthetic_icu_data.py
```

The resulting file is:

```
data/synthetic_icu_24h.csv
```

---

## Features

### Structured Clinical Variables

The dataset contains 12 structured variables:

| Feature | Description |
|---|---|
| `x1_gcs` | Glasgow Coma Scale |
| `x2_heart_rate` | Heart rate in beats per minute |
| `x3_sbp` | Systolic blood pressure |
| `x4_temp_c` | Body temperature in °C |
| `x5_bun` | Blood urea nitrogen |
| `x6_wbc` | White blood cell count |
| `x7_potassium` | Serum potassium |
| `x8_sodium` | Serum sodium |
| `x9_bicarbonate` | Serum bicarbonate |
| `x10_admission_type` | ICU admission type |
| `x11_gender` | Gender |
| `x12_age` | Patient age |

### Topic-Model Features

In addition to the structured clinical variables, the dataset contains ten LDA-style topic probability features representing clinical themes.

The 24-hour topic features include:

- `topicB_hydroperitoneum`
- `topicB_urinary_retention`
- `topicB_newborn_resp_distress`
- `topicB_coronary_artery_disease`
- `topicB_pneumothorax_effusion`
- `topicB_nursing_assessment`
- `topicB_endotracheal_intubation`
- `topicB_sepsis`
- `topicB_medical_assessment`
- `topicB_spinal_hematoma`

These variables are generated using a Dirichlet distribution so that each patient's topic probabilities form a topic distribution.

The topic features are synthetic representations and are not extracted from actual clinical notes.

---

## Prediction Targets

The dataset contains five binary mortality outcomes:

- `mortality_in_hospital`
- `mortality_48h`
- `mortality_72h`
- `mortality_30day`
- `mortality_1year`

Each target is encoded as:

```
0 = Survival
1 = Mortality
```

The primary goal of the project is to develop machine-learning models capable of predicting these mortality outcomes using information available from the synthetic 24-hour ICU dataset.

The outcomes can either be modeled individually or compared across prediction horizons.

---

## Machine-Learning Approach

The project evaluates several supervised classification algorithms.

### Models

The planned models include:

- Gradient Boosting
- Logistic Regression
- CART Decision Tree
- Random Forest
- MARS-style model

These models provide a mixture of:

- Linear models
- Single decision trees
- Bagging ensembles
- Boosting ensembles
- Nonlinear spline-based models

### Class Imbalance

Mortality is expected to be less frequent than survival, creating a class-imbalance problem.

To address this, the project investigates techniques such as:

- SMOTE
- Class weighting
- Stratified cross-validation

SMOTE is applied only to training data to prevent information leakage.

The recommended workflow is:

```
Original Dataset
       │
       ▼
Train/Test Split
       │
       ├───────────────┐
       ▼               ▼
 Training Set       Test Set
       │
       ▼
     SMOTE
       │
       ▼
 Model Training
       │
       ▼
 Validation
       │
       └───────────────┐
                       ▼
                   Test Set
                       │
                       ▼
                  Final Metrics
```

The test set remains untouched until final evaluation.

### Cross-Validation

Model development uses stratified k-fold cross-validation to preserve the mortality class distribution across folds.

A typical configuration is:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Using stratified cross-validation is particularly important because the mortality classes are imbalanced.

---

## Evaluation Metrics

Because this is an imbalanced binary classification problem, accuracy alone is not sufficient.

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Sensitivity
- Specificity
- Confusion matrix

The primary metrics of interest are:

- ROC-AUC
- PR-AUC
- Recall / Sensitivity
- F1-score

PR-AUC is especially useful for assessing performance on an imbalanced mortality prediction task.

---

## Feature Experiments

An important part of this project is determining whether the synthetic topic features provide additional predictive value beyond traditional structured clinical variables.

Three feature configurations can be evaluated.

### Experiment 1 — Clinical Features

Use only:

```
x1_gcs
x2_heart_rate
x3_sbp
x4_temp_c
x5_bun
x6_wbc
x7_potassium
x8_sodium
x9_bicarbonate
x10_admission_type
x11_gender
x12_age
```

### Experiment 2 — Topic Features

Use only:

```
topicB_hydroperitoneum
topicB_urinary_retention
topicB_newborn_resp_distress
topicB_coronary_artery_disease
topicB_pneumothorax_effusion
topicB_nursing_assessment
topicB_endotracheal_intubation
topicB_sepsis
topicB_medical_assessment
topicB_spinal_hematoma
```

### Experiment 3 — Combined Features (Clinical Features + Topic Features)

Use:

```
x1_gcs
x2_heart_rate
x3_sbp
x4_temp_c
x5_bun
x6_wbc
x7_potassium
x8_sodium
x9_bicarbonate
x10_admission_type
x11_gender
x12_age
topicB_hydroperitoneum
topicB_urinary_retention
topicB_newborn_resp_distress
topicB_coronary_artery_disease
topicB_pneumothorax_effusion
topicB_nursing_assessment
topicB_endotracheal_intubation
topicB_sepsis
topicB_medical_assessment
topicB_spinal_hematoma
```

This comparison allows the project to investigate whether topic-model information improves mortality prediction over structured clinical variables alone.

---

## Feature Importance and Model Interpretation

The project also examines which variables contribute most strongly to model predictions.

For tree-based models, feature importance can be obtained from:

- CART
- Random Forest
- Gradient Boosting

For Logistic Regression, model coefficients can be examined.

Permutation importance can also be used as a model-independent interpretation method.

SHAP can optionally be used to investigate individual predictions and global feature importance.

Example analysis:

```
                 Feature Importance
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
    Clinical          Topic            Combined
    Features         Features          Features
```

Because the dataset is synthetic, feature importance reflects the synthetic data-generating mechanism and should not be interpreted as clinical evidence.

---

## Project Structure

```
synthetic-icu-mortality-prediction-24h/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_interpretation.ipynb
│
├── src/
│   ├── generate_synthetic_icu_data.py
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── feature_importance/
│
├── models/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/firearmga/synthetic-icu-mortality-prediction-24h.git
cd synthetic-icu-mortality-prediction-24h
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Generate the Dataset

Run:

```bash
python src/generate_synthetic_icu_data.py
```

Since this project uses only the 24-hour dataset, the generator should save:

```
data/synthetic_icu_24h.csv
```

The dataset contains 27,809 synthetic patient records.

---

## Recommended Modeling Pipeline

```
Synthetic 24h ICU Dataset
             │
             ▼
     Exploratory Analysis
             │
             ▼
       Data Validation
             │
             ▼
       Train/Test Split
             │
             ▼
       Preprocessing
             │
             ▼
   Stratified 5-Fold CV
             │
             ▼
     ┌───────┴────────┐
     │                │
 Without SMOTE     With SMOTE
     │                │
     └───────┬────────┘
             ▼
      Model Training
             │
             ▼
   Hyperparameter Tuning
             │
             ▼
      Model Evaluation
             │
             ▼
   Feature Interpretation
             │
             ▼
      Model Comparison
```

---

## Example Model Comparison

The final analysis can summarize model performance in a table such as:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — | — |
| CART | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| Gradient Boosting | — | — | — | — | — | — |
| MARS | — | — | — | — | — | — |

Actual results will be generated after model training.

---

## Reproducibility

A fixed random seed is used for synthetic data generation:

```python
RNG = np.random.default_rng(42)
```

The modeling pipeline should also use:

```python
random_state=42
```

where applicable.

This makes the experiments reproducible.

---

## Limitations

This project has important limitations.

**Synthetic dataset**
The dataset does not represent real ICU patients and contains no real patient information.

**Simplified data-generating process**
The relationships between clinical variables, topic features, and mortality were intentionally designed for machine-learning experimentation.

**No real clinical notes**
The topic variables are simulated using probability distributions and were not obtained by applying LDA to real ICU documentation.

**No clinical validation**
Model performance on this dataset cannot be interpreted as performance on real-world ICU populations.

**No causal interpretation**
Feature importance indicates predictive relationships within the synthetic dataset and does not establish clinical causation.

**No clinical deployment**
The models are intended for educational and research experimentation and should not be used for clinical decision-making.

---

## Reference

Chiu et al. (2022).
*Predicting the Mortality of ICU Patients by Topic Model with Machine-Learning Techniques.*
Healthcare, 10(6), 1087.

This project is inspired by the general structure and methodology of the published work but uses an independently generated synthetic dataset rather than MIMIC-III data.

---

## License

This project is intended for educational and research purposes.



---

## Author

**Sidharth**

GitHub: [https://github.com/firearmga](https://github.com/firearmga)
