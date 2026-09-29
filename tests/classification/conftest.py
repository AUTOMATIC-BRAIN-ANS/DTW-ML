"""
Shared fixtures of the tests of the classification package: a synthetic feature matrix, cross-validation results and
clean loggers.
"""

import logging

import numpy as np
import pandas as pd
import pytest

CLASSES = ["BAS", "B6", "B10", "B15"]


def make_matrix(n_subjects: int = 12, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic feature matrix: "signal" separates the conditions, "noise" does not.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subjects + 1):
        for k, condition in enumerate(CLASSES):
            rows.append({"id": f"V{s}", "condition": condition, "signal": k + rng.normal(0, 0.1),
                         "noise": rng.normal(), "n_obs": 50})
    return pd.DataFrame(rows)


@pytest.fixture
def matrix_path(tmp_path):
    path = tmp_path / "ABP_RR-CBFV_RR.csv"
    make_matrix().to_csv(path, sep=";", index=False)
    return path


@pytest.fixture
def data(matrix_path):
    from project.classification.modeling.data import FeatureMatrixData
    return FeatureMatrixData(str(matrix_path))


@pytest.fixture
def loso_result(data):
    from project.classification.modeling.cross_validation import CrossValidation
    return CrossValidation(data, permutation_repeats=2, n_bootstrap=20, n_jobs=1).evaluate("LogReg", "LOSO")


@pytest.fixture
def info():
    return {"method": "d-method", "metric": "ABP_RR-CBFV_RR", "model": "LogReg", "validation": "LOSO",
            "n_rows": 48, "n_subjects": 12, "n_features": 2, "classes": CLASSES, "chance_level": 0.25,
            "ci_level": 0.95}


@pytest.fixture
def no_latex(monkeypatch):
    """
    Render test figures without LaTeX (not installed in CI); other modules may have switched text.usetex on globally.
    """
    import matplotlib.pyplot as plt
    monkeypatch.setattr("project.classification.reporting.visualizer.use_latex", lambda: None)
    monkeypatch.setitem(plt.rcParams, "text.usetex", False)


@pytest.fixture(autouse=True)
def reset_classification_loggers():
    """
    Restore the classification loggers after LoggerUtils.setup redirected them, so caplog works and log files are
    closed.
    """
    yield
    for name in ("classification", "classification.run"):
        log = logging.getLogger(name)
        for handler in log.handlers:
            handler.close()
        log.handlers.clear()
        log.propagate = True
