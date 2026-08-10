"""
Evaluation utilities: metrics, confusion matrix, ROC/PR curves, feature
importance, and helpers for saving results into results/{metrics,figures,
feature_importance}/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_proba):
    """
    Full metric set: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
    Sensitivity, Specificity, and the confusion matrix components.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # same as Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def evaluate_model(model, X_test, y_test):
    """Predict with a fitted model and compute the full metric set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_proba), y_pred, y_proba


def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Survived", "Died"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Survived", "Died"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(results_dict, y_test_dict, title, save_path=None):
    """
    results_dict: {model_name: y_proba}
    y_test_dict: {model_name: y_test}  (usually the same y_test repeated)
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_proba in results_dict.items():
        y_test = y_test_dict[name]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(results_dict, y_test_dict, title, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_proba in results_dict.items():
        y_test = y_test_dict[name]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={ap:.3f})")
    base_rate = np.mean(list(y_test_dict.values())[0])
    ax.axhline(base_rate, linestyle="--", color="gray", label=f"Baseline ({base_rate:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importances for tree-based models, or |coef| for linear ones."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
        if len(importances) != len(feature_names):
            return None  # e.g. MARS pipeline expands features via splines
    else:
        return None
    return pd.Series(importances, index=feature_names, name=model_name).sort_values(ascending=False)


def get_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10, random_state=42):
    """Model-agnostic importance — works for any fitted model, including pipelines."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state,
        scoring="roc_auc", n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)


def save_metrics_table(results_summary, path):
    """Save a list-of-dicts metrics summary as a CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results_summary)
    df.to_csv(path, index=False)
    return df
