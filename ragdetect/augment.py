"""Combining the classifier with retrieval (lesson 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ragdetect.evaluation import evaluate


def blend(classifier_proba, retrieval_proba, alpha: float = 0.7) -> np.ndarray:
    """alpha * classifier + (1 - alpha) * retrieval.

    alpha = 1 ignores retrieval completely; alpha = 0 ignores the classifier.
    """
    return alpha * np.asarray(classifier_proba) + (1 - alpha) * np.asarray(retrieval_proba)


def decision_window(alpha: float, threshold: float = 0.5) -> tuple[float, float]:
    """The range of classifier probabilities where retrieval can still change the decision.

    Retrieval scores lie between 0 and 1, so the blend lies between
    alpha * p and alpha * p + (1 - alpha). Outside the returned (low, high)
    range, the classifier's answer wins whatever the neighbours say.
    """
    if alpha <= 0:
        return 0.0, 1.0
    low = (threshold - (1 - alpha)) / alpha
    high = threshold / alpha
    return max(0.0, low), min(1.0, high)


def tune_alpha(y_true, classifier_proba, retrieval_proba, alphas=None, metric: str = "accuracy", threshold: float = 0.5):
    """Try each alpha and return (best alpha, table of scores).

    If several alphas score equally well, the smallest one wins. Always tune
    on a validation set, never on the test set you report.
    """
    if alphas is None:
        alphas = np.round(np.linspace(0, 1, 21), 2)
    rows = []
    for alpha in alphas:
        scores = evaluate(y_true, blend(classifier_proba, retrieval_proba, alpha), threshold)
        rows.append({"alpha": float(alpha), **{k: v for k, v in scores.items() if k != "confusion"}})
    table = pd.DataFrame(rows)
    best = float(table.loc[table[metric].idxmax(), "alpha"])
    return best, table
