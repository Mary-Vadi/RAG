import numpy as np
import pytest

from ragdetect.augment import blend, decision_window, tune_alpha
from ragdetect.evaluation import compare, evaluate, plot_confusion


def test_blend_extremes():
    clf, ret = np.array([0.9, 0.2]), np.array([0.1, 0.8])
    np.testing.assert_allclose(blend(clf, ret, alpha=1.0), clf)
    np.testing.assert_allclose(blend(clf, ret, alpha=0.0), ret)
    np.testing.assert_allclose(blend(clf, ret, alpha=0.7), [0.66, 0.38])


def test_decision_window_for_the_original_alpha():
    low, high = decision_window(0.7)
    assert low == pytest.approx(0.2 / 0.7)
    assert high == pytest.approx(0.5 / 0.7)


def test_decision_window_is_everything_once_retrieval_has_half_the_say():
    assert decision_window(0.5) == (0.0, 1.0)
    assert decision_window(0.0) == (0.0, 1.0)


@pytest.mark.parametrize("alpha", [0.6, 0.7, 0.9])
def test_retrieval_can_flip_the_decision_only_inside_the_window(alpha):
    low, high = decision_window(alpha)
    for p in np.linspace(0, 1, 101):
        outcomes = {blend(p, r, alpha) >= 0.5 for r in (0.0, 1.0)}
        can_flip = len(outcomes) == 2
        if low + 1e-9 < p < high - 1e-9:
            assert can_flip, p
        elif p < low - 1e-9 or p > high + 1e-9:
            assert not can_flip, p


def test_tune_alpha_leans_on_whichever_signal_is_right():
    y = np.array([0, 1] * 50)
    rng = np.random.default_rng(0)
    useless = rng.uniform(size=100)
    best, table = tune_alpha(y, classifier_proba=useless, retrieval_proba=y.astype(float))
    assert best < 0.5
    assert len(table) == 21

    # From alpha = 0.5 upwards a perfect classifier can't be outvoted, and ties go to the smallest alpha.
    best, _ = tune_alpha(y, classifier_proba=y.astype(float), retrieval_proba=useless)
    assert best == 0.5


def test_evaluate_counts():
    y_true = [0, 0, 1, 1, 1]
    proba = [0.1, 0.8, 0.9, 0.7, 0.2]
    scores = evaluate(y_true, proba)
    assert scores["confusion"] == [[1, 1], [1, 2]]
    assert scores["accuracy"] == pytest.approx(3 / 5)
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)


def test_compare_and_plot():
    results = {"a": evaluate([0, 1], [0.1, 0.9]), "b": evaluate([0, 1], [0.9, 0.9])}
    table = compare(results)
    assert list(table.columns) == ["accuracy", "precision", "recall", "f1"]
    assert table.loc["a", "accuracy"] == 100.0
    assert plot_confusion([0, 1], [0.1, 0.9], title="t").get_title() == "t"
