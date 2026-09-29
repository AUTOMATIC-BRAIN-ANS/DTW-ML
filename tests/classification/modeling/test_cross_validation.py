"""
Unit tests for classification/modeling/cross_validation.py on a synthetic feature matrix.
"""

import numpy as np
import pandas as pd
import pytest

from project.classification.config import CLASSES, METRIC_NAMES
from project.classification.modeling.cross_validation import CrossValidation


class TestCrossValidation:
    def test_loso_one_subject_per_fold(self, data):
        (run,) = CrossValidation(data).get_runs("LOSO")
        groups = data.get_groups()
        assert len(run) == 12
        for train, test in run:
            assert len(set(groups[test])) == 1 and set(groups[test]).isdisjoint(groups[train])
        assert sorted(np.concatenate([test for _, test in run])) == list(range(48))

    def test_repeated_sgkf_groups_and_repeats(self, data):
        runs = CrossValidation(data, n_splits=3, n_repeats=4).get_runs("RepeatedSGKF")
        groups, y = data.get_groups(), data.get_target()
        assert len(runs) == 4 and all(len(run) == 3 for run in runs)
        for run in runs:
            assert sorted(np.concatenate([test for _, test in run])) == list(range(48))
            for train, test in run:
                assert set(groups[test]).isdisjoint(groups[train])
                assert set(y[test]) == set(CLASSES)
        assert not all(np.array_equal(runs[0][0][1], run[0][1]) for run in runs[1:])

    def test_unknown_validation(self, data):
        with pytest.raises(ValueError):
            CrossValidation(data).get_runs("N/A")

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_evaluate(self, data, n_jobs):
        cv = CrossValidation(data, n_splits=3, n_repeats=2, permutation_repeats=3, n_bootstrap=20, n_jobs=n_jobs)
        result = cv.evaluate("LogReg", "RepeatedSGKF")
        assert (result.n_runs, result.n_folds) == (2, 3)
        assert list(result.metrics.columns) == ["run", *METRIC_NAMES]
        assert (result.metrics["BACC"] > 0.8).all()
        assert len(result.predictions) == 2 * 48
        proba = result.predictions[[f"proba_{c}" for c in CLASSES]].to_numpy()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        assert result.confusion.to_numpy().sum() == 2 * 48
        assert list(result.confusion.index) == list(result.confusion.columns) == CLASSES
        assert list(result.importance.columns) == ["feature", "importance", "mean_b6", "mean_b10", "mean_b15", "mean_bas"]
        assert result.importance.loc[0, "feature"] == "signal"
        assert list(result.summary.index) == METRIC_NAMES
        assert list(result.summary.columns) == ["mean", "std", "se", "ci_lower", "ci_upper"]
        s = result.summary
        assert (s["ci_lower"] <= s["ci_upper"]).all() and s[["std", "se"]].notna().all().all()
        pd.testing.assert_series_equal(s["mean"], result.metrics[METRIC_NAMES].mean(), check_names=False)

    def test_evaluate_logs_steps(self, data, caplog):
        caplog.set_level("INFO", logger="classification")
        CrossValidation(data, n_splits=3, n_repeats=2, permutation_repeats=2, n_bootstrap=10,
                        n_jobs=1).evaluate("LogReg", "RepeatedSGKF")
        for text in ["Cross-validation started: 2 run(s) × 3 folds = 6 fits", "Run 1/2 finished", "Run 2/2 finished",
                     "Cross-validation finished in", "Bootstrap started: 10 resamples", "Bootstrap finished in"]:
            assert text in caplog.text

    def test_evaluate_loso_catboost(self, data):
        result = CrossValidation(data, permutation_repeats=2, n_bootstrap=10).evaluate("CatBoost", "LOSO")
        assert (result.n_runs, result.n_folds) == (1, 12) and len(result.predictions) == 48
        # one run: no std over runs, no NaN
        assert list(result.summary.columns) == ["mean", "se", "ci_lower", "ci_upper"]
        assert result.summary.notna().all().all()
        assert result.importance.loc[0, "feature"] == "signal"

    def test_get_importance(self, data):
        # folds × features (signal, noise): mean over folds, most important first
        importance = CrossValidation(data).get_importance(np.array([[0.1, 0.3], [0.3, 0.1], [0.2, 0.5]]))
        assert list(importance.columns) == ["feature", "importance", "mean_b6", "mean_b10", "mean_b15", "mean_bas"]
        assert list(importance["feature"]) == ["noise", "signal"]
        np.testing.assert_allclose(importance["importance"], [0.3, 0.2])
        # mean of the feature in each class: signal = index of the class (BAS 0, B6 1, B10 2, B15 3) + noise
        signal = importance.set_index("feature").loc["signal"]
        df = data.df
        for c in CLASSES:
            assert signal[f"mean_{c.lower()}"] == pytest.approx(df.loc[df["condition"] == c, "signal"].mean())
        np.testing.assert_allclose(signal[["mean_bas", "mean_b6", "mean_b10", "mean_b15"]].astype(float),
                                   [0, 1, 2, 3], atol=0.1)
