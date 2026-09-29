"""
Unit tests for classification/scoring/bootstrap.py.
"""

import numpy as np
import pandas as pd
import pytest

from project.classification.config import CLASSES, METRIC_NAMES
from project.classification.scoring.bootstrap import Bootstrap
from project.classification.scoring.classification_metrics import ClassificationMetrics


def make_predictions(n_subjects=20, n_runs=1, accuracy=0.7, seed=0):
    """
    Out-of-fold predictions: every subject has one row per class, a share of them predicted correctly.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for run in range(n_runs):
        for s in range(n_subjects):
            for c in CLASSES:
                predicted = c if rng.random() < accuracy else rng.choice([k for k in CLASSES if k != c])
                proba = rng.dirichlet(np.ones(len(CLASSES)))
                rows.append({"run": run, "id": f"V{s}", "condition": c, "predicted": predicted,
                             **{f"proba_{k}": p for k, p in zip(CLASSES, proba)}})
    return pd.DataFrame(rows)


def test_perfect_predictions_give_degenerate_interval():
    predictions = make_predictions(accuracy=1.0)
    for c in CLASSES:
        predictions[f"proba_{c}"] = (predictions["condition"] == c).astype(float)
    ci = Bootstrap(n_bootstrap=20, n_jobs=1).confidence_intervals(predictions, CLASSES)
    assert list(ci.index) == METRIC_NAMES
    assert (ci[["ci_lower", "ci_upper"]] == 1.0).all().all() and (ci["se"] == 0.0).all()


def test_interval_contains_estimate_and_is_reproducible():
    predictions = make_predictions(n_runs=2)
    bootstrap = Bootstrap(n_bootstrap=200, random_state=1, n_jobs=1)
    ci = bootstrap.confidence_intervals(predictions, CLASSES)
    estimate = np.mean([[ClassificationMetrics.compute_predictions(run_df, CLASSES)[m] for m in METRIC_NAMES]
                        for _, run_df in predictions.groupby("run")], axis=0)
    assert (ci["ci_lower"].to_numpy() <= estimate).all() and (estimate <= ci["ci_upper"].to_numpy()).all()
    assert (ci["ci_lower"] < ci["ci_upper"]).all() and (ci["se"] > 0).all()
    # parallel chunks give the same resamples as a single job
    pd.testing.assert_frame_equal(ci, Bootstrap(n_bootstrap=200, random_state=1, n_jobs=2)
                                  .confidence_intervals(predictions, CLASSES))


def test_resamples_whole_subjects():
    predictions = make_predictions(n_subjects=5)
    runs = [predictions.reset_index(drop=True)]
    rows = [{str(s): np.asarray(i) for s, i in runs[0].groupby("id").indices.items()}]
    subjects = np.array(sorted(predictions["id"].unique()))
    # a resample drawing only subject V0 five times has the metrics of the four rows of V0
    values = Bootstrap._score_resamples(runs, rows, subjects, np.zeros((1, 5), dtype=int), CLASSES)
    expected = ClassificationMetrics.compute_predictions(predictions[predictions["id"] == "V0"], CLASSES)
    assert values[0] == pytest.approx([expected[m] for m in METRIC_NAMES], nan_ok=True)
