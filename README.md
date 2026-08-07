# ICU Mortality Prediction Using Machine Learning

## Project Overview

This project develops and evaluates machine learning models to predict in-hospital mortality for ICU patients using a synthetic clinical dataset. The objective is to identify patients at higher risk of mortality based on demographic information, vital signs, laboratory values, and clinical indicators collected during the first 24 hours of ICU admission.

This project demonstrates a complete machine learning workflow including data exploration, preprocessing, class balancing, model training, evaluation, and performance comparison.

---

## Dataset

The dataset used is:

**synthetic_icu_24h.csv**

The dataset contains synthetic ICU patient records and is intended for educational and research purposes only. It does not contain real patient information.

Example features include:

- Age
- Gender
- Heart rate
- Blood pressure
- Respiratory rate
- Oxygen saturation
- Laboratory measurements
- Clinical indicators

Target variable:

- `mortality_in_hospital`
  - 0 = Survived
  - 1 = Died during hospital stay

---

## Project Workflow

1. Load and inspect the dataset
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Handle missing values
5. Encode categorical variables
6. Train-test split
7. Handle class imbalance (SMOTE if applicable)
8. Train machine learning models
9. Evaluate model performance
10. Compare model results

---

## Models Used

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (optional)

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- imbalanced-learn

---

## Repository Structure
