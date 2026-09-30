"""Measuring how good a detector is (lesson 4 onwards).

"Machine" is the positive class throughout: precision is the share of texts
flagged as machine that really are machine, and recall is the share of
machine texts that get flagged.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support

from ragdetect.data import LABEL_NAMES


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1 and the confusion matrix for P(machine) scores."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def compare(results: dict) -> pd.DataFrame:
    """One row per experiment, metrics as percentages. `results` maps a name to evaluate(...) output."""
    table = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1"]].astype(float)
    return (table * 100).round(2)


def plot_confusion(y_true, proba, title: str = "", threshold: float = 0.5, ax=None):
    """Draw the confusion matrix with rows = true label and columns = predicted label."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=LABEL_NAMES,
        ax=ax, values_format="d", colorbar=False,
    )
    ax.set_title(title)
    return ax
