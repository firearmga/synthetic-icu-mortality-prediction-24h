"""
Preprocessing utilities for the synthetic ICU mortality project.

Provides:
    - load_data(): read the 24h dataset
    - FEATURE_SETS: the 3 feature experiments (clinical / topic / combined)
    - get_feature_target(): split a dataframe into X, y for a given experiment + outcome
    - train_test_split_stratified(): stratified train/test split
    - apply_smote(): oversample the minority class (train data only)
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

OUTCOME_COLS = [
    "mortality_in_hospital",
    "mortality_48h",
    "mortality_72h",
    "mortality_30day",
    "mortality_1year",
]

CLINICAL_FEATURES = [
    "x1_gcs", "x2_heart_rate", "x3_sbp", "x4_temp_c", "x5_bun", "x6_wbc",
    "x7_potassium", "x8_sodium", "x9_bicarbonate", "x10_admission_type",
    "x11_gender", "x12_age",
]

TOPIC_FEATURES = [
    "topicB_hydroperitoneum", "topicB_urinary_retention", "topicB_newborn_resp_distress",
    "topicB_coronary_artery_disease", "topicB_pneumothorax_effusion", "topicB_nursing_assessment",
    "topicB_endotracheal_intubation", "topicB_sepsis", "topicB_medical_assessment",
    "topicB_spinal_hematoma",
]

COMBINED_FEATURES = CLINICAL_FEATURES + TOPIC_FEATURES

FEATURE_SETS = {
    "clinical": CLINICAL_FEATURES,
    "topic": TOPIC_FEATURES,
    "combined": COMBINED_FEATURES,
}


def load_data(path="data/synthetic_icu_24h.csv"):
    """Load the 24h synthetic ICU dataset."""
    df = pd.read_csv(path)
    return df


def get_feature_target(df, feature_set="combined", target="mortality_in_hospital"):
    """
    Returns X, y for a given feature experiment ('clinical', 'topic', 'combined')
    and a given mortality outcome column.
    """
    assert feature_set in FEATURE_SETS, f"feature_set must be one of {list(FEATURE_SETS)}"
    assert target in OUTCOME_COLS, f"target must be one of {OUTCOME_COLS}"

    X = df[FEATURE_SETS[feature_set]].copy()
    y = df[target].copy()
    return X, y


def train_test_split_stratified(X, y, test_size=0.25, random_state=42):
    """Stratified train/test split — preserves the mortality class ratio in both sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def apply_smote(X_train, y_train, random_state=42):
    """
    Oversample the minority (death) class in the TRAINING data only.
    Never apply SMOTE to validation/test data — that would leak synthetic
    samples into evaluation and inflate performance.
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res
