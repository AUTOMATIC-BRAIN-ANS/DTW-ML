"""
End-to-end tests of classification/classify.py on synthetic feature matrices.
"""

import re
import shutil

import pytest

from project.classification.classify import main

pytestmark = pytest.mark.usefixtures("no_latex")

TIMESTAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[(INFO|WARNING|ERROR)\] ")
METHODS = ["td-method", "d-method"]
MODELS = ["LogReg", "HistGradBoost"]


def test_main_writes_results(matrix_path, tmp_path):
    for method in METHODS:
        (tmp_path / method).mkdir()
        shutil.copy(matrix_path, tmp_path / method / "ABP_RR-CBFV_RR.csv")
    out = tmp_path / "out"
    main(["--input-dir", str(tmp_path), "--output-dir", str(out), "--metrics", "ABP_RR-CBFV_RR", "--models", *MODELS,
          "--n-repeats", "2", "--permutation-repeats", "2", "--n-bootstrap", "10", "--n-jobs", "1"])
    for method in METHODS:
        for model in MODELS:
            model_dir = out / method / "ABP_RR-CBFV_RR" / model
            for validation in ["LOSO", "RepeatedSGKF"]:
                for name in ["metrics.json", "confusion_matrix.json", "feature_importance.csv", "figure.pdf"]:
                    assert (model_dir / validation / name).exists()
            log = (model_dir / f"{model}.log").read_text(encoding="utf-8")
            # every line has a timestamp, except the indented rows of the tables
            assert all(TIMESTAMP.match(line) or line.startswith("    ") for line in log.splitlines())
            # the progress bar stays on the console
            assert "fold/s" not in log and "s/fold" not in log
            for text in [f"Method: {method} | metric: ABP_RR-CBFV_RR | model: {model}",
                         "Loading the feature matrix", "X: 48 rows × 2 features", "Cross-validation started",
                         "Run 2/2 finished", "Bootstrap finished in", "Feature ranking", "Confusion matrix",
                         "Metrics (mean and std over 2 runs; bootstrap SE", "Saved the figure", "✔ RepeatedSGKF completed",
                         "Validations completed: 2/2 in"]:
                assert text in log
            assert "warnings: 0 | errors: 0" in log
    run_log = (out / "classification.log").read_text(encoding="utf-8")
    assert run_log.count("✔") == 8 and "classifiers completed: 4/4" in run_log
    assert re.search(r"BACC \d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\]", run_log)


def test_main_missing_matrix(tmp_path):
    out = tmp_path / "out"
    main(["--input-dir", str(tmp_path), "--output-dir", str(out), "--methods", "d-method",
          "--metrics", "ABP_RR-CBFV_RR", "--models", "LogReg", "--n-jobs", "1"])
    log = (out / "d-method" / "ABP_RR-CBFV_RR" / "LogReg" / "LogReg.log").read_text(encoding="utf-8")
    assert "✖ Loading" in log
    assert "classifiers completed: 0/1" in (out / "classification.log").read_text(encoding="utf-8")
