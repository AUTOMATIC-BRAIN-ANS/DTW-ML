"""
Unit tests for classification/summary/collect_metrics.py on synthetic metrics.json files.
"""

import codecs
import json
import logging

import pandas as pd
import pytest

from project.classification.config import METRIC_NAMES
from project.classification.summary.collect_metrics import MetricsCollector, main

LABELS = ["ROC-AUC", "Balanced accuracy", "F1-score", "Precision", "Recall", "PR-AUC",
          "Matthews correlation coefficient", "Geometric mean", "Specificity", "Sensitivity",
          "Negative predictive value"]


@pytest.fixture(autouse=True)
def reset_summary_logger():
    """
    Restore the module logger after main() redirected it, so caplog works and log files are closed.
    """
    yield
    log = logging.getLogger("summary")
    for handler in log.handlers:
        handler.close()
    log.handlers.clear()
    log.propagate = True


def write_metrics(root, method, metric, model, validation, value):
    """
    metrics.json with the same statistics for every metric (NPV = null, as NaN is saved).
    """
    directory = root / method / metric / model / validation
    directory.mkdir(parents=True)
    summary = {name: {"mean": value, "se": 0.01, "ci_lower": value - 0.02, "ci_upper": value + 0.02}
               for name in METRIC_NAMES}
    summary["NPV"]["mean"] = None
    (directory / "metrics.json").write_text(json.dumps({"model": model, "summary": summary}), encoding="utf-8")


def test_table_labels_and_column_order(tmp_path):
    write_metrics(tmp_path, "d-method", "ABP_SPO-CBFV_SPO", "LogReg", "LOSO", 0.5)
    collector = MetricsCollector(str(tmp_path), ["d-method", "td-method"], ["ABP_SPO-CBFV_SPO", "ABP_RR-CBFV_RR"],
                                 ["LOSO", "RepeatedSGKF"])
    table = collector.get_table("LogReg")
    assert table["metric"].tolist() == LABELS
    # methods, then metric pairs, then validation schemes
    assert list(table.columns[1:]) == [
        "d-method | ABP_SPO-CBFV_SPO | LOSO", "d-method | ABP_SPO-CBFV_SPO | RepeatedSGKF",
        "d-method | ABP_RR-CBFV_RR | LOSO", "d-method | ABP_RR-CBFV_RR | RepeatedSGKF",
        "td-method | ABP_SPO-CBFV_SPO | LOSO", "td-method | ABP_SPO-CBFV_SPO | RepeatedSGKF",
        "td-method | ABP_RR-CBFV_RR | LOSO", "td-method | ABP_RR-CBFV_RR | RepeatedSGKF"]
    column = table["d-method | ABP_SPO-CBFV_SPO | LOSO"]
    assert column.iloc[:-1].tolist() == ["0.500 ± 0.010 [0.480–0.520]"] * 10 and pd.isna(column.iloc[-1])
    # no metrics.json: empty column
    assert table.iloc[:, 2:].isna().all().all()


@pytest.mark.parametrize("stats, text", [
    ({"mean": 0.5942, "se": 0.0236, "ci_lower": 0.5481, "ci_upper": 0.6372}, "0.594 ± 0.024 [0.548–0.637]"),
    # LOSO and RepeatedSGKF: the same format (std over runs is not shown)
    ({"mean": 0.5, "std": 0.1, "se": 0.02, "ci_lower": 0.46, "ci_upper": 0.54}, "0.500 ± 0.020 [0.460–0.540]"),
    ({"mean": 0.5, "se": None, "ci_lower": None, "ci_upper": None}, "0.500"),
    ({"mean": None, "se": 0.01}, None),
])
def test_format_value(stats, text):
    assert MetricsCollector.format_value(stats) == text


def test_default_column_order():
    collector = MetricsCollector("unused")
    assert [MetricsCollector.get_column(*c) for c in collector.configurations[:3]] == [
        "d-method | ABP_SPO-CBFV_SPO | LOSO", "d-method | ABP_SPO-CBFV_SPO | RepeatedSGKF",
        "d-method | ABP_SPP-CBFV_SPP | LOSO"]
    assert collector.configurations[-1] == ("td-method", "ABP_RR-CBFV_RR", "RepeatedSGKF")


def test_unreadable_file(tmp_path, caplog):
    directory = tmp_path / "d-method" / "ABP_SPO-CBFV_SPO" / "LogReg" / "LOSO"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text("{not json", encoding="utf-8")
    assert MetricsCollector(str(tmp_path)).read_values(str(directory / "metrics.json")) is None
    assert "✖" in caplog.text


def test_main_writes_one_table_per_model(tmp_path):
    results, out = tmp_path / "results", tmp_path / "out"
    for model, value in [("LogReg", 0.25), ("SVM-RBF", 0.75)]:
        write_metrics(results, "td-method", "ABP_SPP-CBFV_SPP", model, "RepeatedSGKF", value)
    main(["--input-dir", str(results), "--output-dir", str(out), "--models", "LogReg", "SVM-RBF",
          "--methods", "td-method", "--metrics", "ABP_SPP-CBFV_SPP"])
    for model, value in [("LogReg", 0.25), ("SVM-RBF", 0.75)]:
        table = pd.read_csv(out / f"{model}.csv", sep=";", encoding="utf-8-sig")
        assert list(table.columns) == ["metric", "td-method | ABP_SPP-CBFV_SPP | LOSO",
                                       "td-method | ABP_SPP-CBFV_SPP | RepeatedSGKF"]
        assert table["metric"].tolist() == LABELS
        assert table["td-method | ABP_SPP-CBFV_SPP | LOSO"].isna().all()
        assert table["td-method | ABP_SPP-CBFV_SPP | RepeatedSGKF"].iloc[0] == (
            f"{value:.3f} ± 0.010 [{value - 0.02:.3f}–{value + 0.02:.3f}]")
    # BOM, so that Excel reads "±" and "–" as UTF-8
    assert (out / "LogReg.csv").read_bytes().startswith(codecs.BOM_UTF8)
    log = (out / "summary.log").read_text(encoding="utf-8")
    assert "✔ LogReg: 1/2 configurations" in log and "Missing file" in log and "warnings: 2 | errors: 0" in log
