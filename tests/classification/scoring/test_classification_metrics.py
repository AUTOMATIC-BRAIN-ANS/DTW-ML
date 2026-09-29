"""
Unit tests for classification/scoring/classification_metrics.py.
"""

import numpy as np
import pandas as pd
import pytest

from project.classification.config import CLASSES, METRIC_NAMES
from project.classification.scoring.classification_metrics import ClassificationMetrics, get_rates, summarise


class TestClassificationMetrics:
    def test_perfect_prediction(self):
        y = np.array(CLASSES * 3)
        proba = np.tile(np.eye(4), (3, 1))
        metrics = ClassificationMetrics.compute(y, y, proba, CLASSES)
        assert list(metrics) == METRIC_NAMES
        assert all(value == pytest.approx(1.0) for value in metrics.values())

    def test_wrong_prediction(self):
        y = np.array(CLASSES * 3)
        y_pred = np.roll(y, 1)
        proba = np.full((12, 4), 0.25)
        metrics = ClassificationMetrics.compute(y, y_pred, proba, CLASSES)
        assert metrics["BACC"] == metrics["Sens"] == metrics["GM"] == 0.0
        assert metrics["ROC-AUC"] == pytest.approx(0.5) and metrics["PR-AUC"] == pytest.approx(0.25)
        # MCC is in [-1, 1]: below 0 for systematically wrong predictions
        assert metrics["MCC"] < 0
        # each class: TN 6, FP 3, FN 3
        assert metrics["Spec"] == pytest.approx(6 / 9) and metrics["NPV"] == pytest.approx(6 / 9)

    def test_binary(self):
        y = np.array(["BAS", "B6"] * 4)
        proba = np.array([[0.2, 0.8], [0.9, 0.1]] * 4)
        metrics = ClassificationMetrics.compute(y, y, proba, ["B6", "BAS"])
        assert metrics["ROC-AUC"] == 1.0 and metrics["PR-AUC"] == 1.0

    def test_sens_spec_npv_gm(self):
        # B6: TP 2, FN 0, FP 1, TN 1; BAS: TP 1, FN 1, FP 0, TN 2
        y_true = np.array(["BAS", "BAS", "B6", "B6"])
        y_pred = np.array(["BAS", "B6", "B6", "B6"])
        proba = np.array([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
        metrics = ClassificationMetrics.compute(y_true, y_pred, proba, ["B6", "BAS"])
        assert metrics["Sens"] == pytest.approx(0.75) == metrics["Rec"]
        assert metrics["Spec"] == pytest.approx(0.75)
        assert metrics["NPV"] == pytest.approx((1 + 2 / 3) / 2)
        assert metrics["GM"] == pytest.approx(np.sqrt(1 * 0.5))

    def test_compute_predictions(self):
        predictions = pd.DataFrame({"condition": ["B6", "BAS"], "predicted": ["B6", "BAS"],
                                    "proba_B6": [0.9, 0.2], "proba_BAS": [0.1, 0.8]})
        metrics = ClassificationMetrics.compute_predictions(predictions, ["B6", "BAS"])
        assert metrics["BACC"] == 1.0 and metrics["ROC-AUC"] == 1.0


def test_summarise():
    metrics = pd.DataFrame({"run": [0, 1], "BACC": [0.5, 0.7]})
    summary = summarise(metrics)
    assert summary.loc["BACC", "mean"] == pytest.approx(0.6)
    assert summary.loc["BACC", "std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    ci = pd.DataFrame({"se": [0.1], "ci_lower": [0.4], "ci_upper": [0.8]}, index=["BACC"])
    summary = summarise(metrics, ci)
    assert list(summary.columns) == ["mean", "std", "se", "ci_lower", "ci_upper"]
    assert summary.loc["BACC", "ci_upper"] == 0.8


def test_summarise_one_run_has_no_std():
    summary = summarise(pd.DataFrame({"run": [0], "BACC": [0.5]}))
    assert list(summary.columns) == ["mean"]


def test_get_rates():
    confusion = pd.DataFrame([[3, 1], [0, 0]], index=["A", "B"], columns=["A", "B"])
    rates = get_rates(confusion)
    assert rates.loc["A"].tolist() == [0.75, 0.25]
    assert rates.loc["B"].isna().all()
