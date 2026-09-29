"""
Unit tests for classification/reporting/reporter.py.
"""

import json
import logging
import re

import numpy as np
import pandas as pd

from project.classification.config import CLASSES, METRIC_NAMES
from project.classification.reporting.reporter import Reporter
from project.utils.logger import LoggerUtils

TIMESTAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[(INFO|WARNING|ERROR)\] ")


def test_save(loso_result, info, tmp_path):
    Reporter.save(loso_result, str(tmp_path / "LOSO"), info)
    metrics = json.loads((tmp_path / "LOSO" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model"] == "LogReg" and metrics["n_runs"] == 1 and metrics["n_folds"] == 12
    assert len(metrics["runs"]) == 1 and list(metrics["summary"]) == METRIC_NAMES
    bacc = metrics["summary"]["BACC"]
    # one run (LOSO): no std over runs; bootstrap SE instead
    assert list(bacc) == ["mean", "se", "ci_lower", "ci_upper"]
    assert bacc["se"] > 0
    assert bacc["mean"] == metrics["runs"][0]["BACC"]
    assert bacc["ci_lower"] <= bacc["mean"] <= bacc["ci_upper"]
    confusion = json.loads((tmp_path / "LOSO" / "confusion_matrix.json").read_text(encoding="utf-8"))
    assert confusion["labels"] == CLASSES and "classes" not in confusion
    assert np.array(confusion["counts"]).sum() == 48
    np.testing.assert_allclose(np.array(confusion["rates"]).sum(axis=1), 1.0)
    importance = pd.read_csv(tmp_path / "LOSO" / "feature_importance.csv", sep=";")
    assert list(importance.columns) == ["feature", "importance", "mean_b6", "mean_b10", "mean_b15", "mean_bas"]


def test_log_results_one_timestamp_per_table(loso_result, tmp_path):
    log_path = tmp_path / "test.log"
    LoggerUtils.setup(logging.getLogger("classification"), str(log_path))
    Reporter.log_results(loso_result, 0.95)
    text = log_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    titles = ["Feature ranking", "Confusion matrix (rows = true", "Confusion matrix (rates",
              "Metrics (1 run: no std over runs; bootstrap SE and 95% CI)"]
    # the timestamp only on the line of each title; the rows of the tables are indented below it
    stamped = [line for line in lines if TIMESTAMP.match(line)]
    assert len(stamped) == len(titles)
    for line, title in zip(stamped, titles, strict=True):
        assert title in line and line.endswith(":")
    assert all(line.startswith("    ") for line in lines if not TIMESTAMP.match(line))
    assert re.search(r"^    BACC +\d\.\d{4}", text, re.MULTILINE)
    assert "NaN" not in text
    # LOSO has one run: no table of runs
    assert "Metrics of each run" not in text


def test_log_data(data, caplog):
    caplog.set_level("INFO", logger="classification")
    Reporter.log_data(data, "m.csv")
    for text in ["Loaded: m.csv", "X: 48 rows × 2 features | y: 48 labels | subjects: 12 | chance level: 0.250",
                 "Rows per class: BAS 12, B6 12, B10 12, B15 12", "Features: signal, noise", "Missing values: none"]:
        assert text in caplog.text
