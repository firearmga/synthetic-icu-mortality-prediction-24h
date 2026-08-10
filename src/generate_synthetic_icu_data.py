"""
Synthetic ICU Mortality Dataset Generator (24-hour dataset)
=============================================================
Generates a fully synthetic ICU dataset mimicking the structure of the
dataset used in Chiu et al. (2022), Healthcare 10(6):1087 — but this is
NOT real MIMIC-III data. No real patient information is used anywhere.

Produces:
    data/synthetic_icu_24h.csv   (27,809 synthetic patients)

Run from the project root:
    python src/generate_synthetic_icu_data.py
"""

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_PATIENTS_24H = 27809

TOPIC_NAMES_24H = [
    "hydroperitoneum", "urinary_retention", "newborn_resp_distress",
    "coronary_artery_disease", "pneumothorax_effusion", "nursing_assessment",
    "endotracheal_intubation", "sepsis", "medical_assessment", "spinal_hematoma",
]

OUTCOME_COLS = [
    "mortality_in_hospital",
    "mortality_48h",
    "mortality_72h",
    "mortality_30day",
    "mortality_1year",
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_structured_variables(n):
    """x1-x12 structured clinical variables, distributions loosely based on the paper."""
    df = pd.DataFrame()
    df["x1_gcs"] = np.clip(RNG.normal(13, 2.5, n), 3, 15).round().astype(int)
    df["x2_heart_rate"] = np.clip(RNG.normal(88, 18, n), 40, 180).round(1)
    df["x3_sbp"] = np.clip(RNG.normal(122, 22, n), 60, 220).round(1)
    df["x4_temp_c"] = np.clip(RNG.normal(37.0, 0.8, n), 33, 41).round(2)
    df["x5_bun"] = np.clip(RNG.gamma(shape=3.0, scale=8.0, size=n), 2, 150).round(1)
    df["x6_wbc"] = np.clip(RNG.gamma(shape=4.0, scale=2.7, size=n), 0.5, 60).round(2)
    df["x7_potassium"] = np.clip(RNG.normal(4.1, 0.6, n), 2.0, 8.0).round(2)
    df["x8_sodium"] = np.clip(RNG.normal(138, 4.5, n), 110, 165).round(1)
    df["x9_bicarbonate"] = np.clip(RNG.normal(24, 4.5, n), 5, 40).round(1)
    df["x10_admission_type"] = RNG.choice([0, 1, 2], size=n, p=[0.15, 0.03, 0.82])
    df["x11_gender"] = RNG.choice([0, 1], size=n, p=[0.44, 0.56])
    df["x12_age"] = np.clip(RNG.normal(63, 17, n), 16, 100).round(1)
    return df


def generate_topic_variables(n, prefix, topic_names):
    """LDA-style topic probability scores per patient (Dirichlet draw, sums to 1 per row)."""
    alpha = RNG.uniform(0.3, 1.2, size=len(topic_names))
    samples = RNG.dirichlet(alpha, size=n)
    cols = [f"{prefix}_{name}" for name in topic_names]
    return pd.DataFrame(samples, columns=cols)


def generate_outcomes(struct_df, topic_df):
    """5 correlated binary mortality outcomes, weighted toward GCS, BUN, bicarbonate, age, topics."""
    n = len(struct_df)
    x1 = struct_df["x1_gcs"].values
    x5 = struct_df["x5_bun"].values
    x9 = struct_df["x9_bicarbonate"].values
    x12 = struct_df["x12_age"].values

    z1 = (x1 - x1.mean()) / x1.std()
    z5 = (x5 - x5.mean()) / x5.std()
    z9 = (x9 - x9.mean()) / x9.std()
    z12 = (x12 - x12.mean()) / x12.std()

    topic_cols = topic_df.columns.tolist()
    crit_topic_1 = topic_df[topic_cols[2]].values
    crit_topic_2 = topic_df[topic_cols[0]].values
    z_t1 = (crit_topic_1 - crit_topic_1.mean()) / crit_topic_1.std()
    z_t2 = (crit_topic_2 - crit_topic_2.mean()) / crit_topic_2.std()

    outcomes = {}
    configs = {
        "mortality_in_hospital": dict(w=(-0.55, 0.45, -0.35, 0.55, 0.30, 0.15), b=-2.75, noise=0.55),
        "mortality_48h":         dict(w=(-0.60, 0.55, -0.20, 0.35, 0.20, 0.10), b=-4.85, noise=0.50),
        "mortality_72h":         dict(w=(-0.58, 0.50, -0.25, 0.40, 0.25, 0.12), b=-4.20, noise=0.52),
        "mortality_30day":       dict(w=(-0.50, 0.40, -0.30, 0.50, 0.28, 0.14), b=-2.70, noise=0.60),
        "mortality_1year":       dict(w=(-0.40, 0.30, -0.35, 0.45, 0.35, 0.20), b=-2.65, noise=0.75),
    }

    for name, cfg in configs.items():
        w1, w5, w9, w12, wt1, wt2 = cfg["w"]
        lin = (
            cfg["b"]
            + w1 * z1 + w5 * z5 + w9 * z9 + w12 * z12
            + wt1 * z_t1 + wt2 * z_t2
            + RNG.normal(0, cfg["noise"], n)
        )
        p = sigmoid(lin)
        outcomes[name] = RNG.binomial(1, p)

    return pd.DataFrame(outcomes)


def build_24h_dataset():
    struct_df = generate_structured_variables(N_PATIENTS_24H)
    topic_df = generate_topic_variables(N_PATIENTS_24H, "topicB", TOPIC_NAMES_24H)
    outcome_df = generate_outcomes(struct_df, topic_df)

    patient_id = pd.DataFrame({"patient_id": np.arange(1, N_PATIENTS_24H + 1)})
    full_df = pd.concat([patient_id, struct_df, topic_df, outcome_df], axis=1)
    return full_df


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = build_24h_dataset()
    out_path = os.path.join(data_dir, "synthetic_icu_24h.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Shape: {df.shape}")
    for col in OUTCOME_COLS:
        print(f"  {col}: {df[col].mean() * 100:.2f}% mortality")
