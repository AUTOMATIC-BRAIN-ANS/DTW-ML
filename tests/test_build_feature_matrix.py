"""
Unit tests for classification/build_feature_matrix.py on synthetic cost files.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, skew

from project.classification.build_feature_matrix import (
    FEATURES,
    CostReader,
    FeatureMatrix,
    Identifiers,
    logger,
    main,
)


@pytest.fixture(autouse=True)
def reset_logger():
    """
    Restore the module logger after main() redirected it, so caplog works and log files are closed.
    """
    yield
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = True


def write(path, content, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode(encoding))
    return str(path)


class TestIdentifiers:
    def test_subject_strips_whitespace(self):
        assert Identifiers.subject(" V10 ") == "V10"

    def test_condition_is_folder_name(self, tmp_path):
        assert Identifiers.condition(str(tmp_path / "B6")) == "B6"
        assert Identifiers.condition(str(tmp_path / "B6") + "/") == "B6"

    def test_natural_key(self):
        assert sorted(["V10", "V1", "V19", "V2"], key=Identifiers.natural_key) == ["V1", "V2", "V10", "V19"]


class TestCostReader:
    @pytest.mark.parametrize("cell, expected", [("0.5", 0.5), ("0,5", 0.5), ("-1e-3", -0.001), ("x", None),
                                                ("nan", None), ("inf", None), ("1,2,3", None)])
    def test_parse_number(self, cell, expected):
        assert CostReader.parse_number(cell) == expected

    @staticmethod
    def read(tmp_path, content):
        """
        Write one cost file (condition BAS, metric A) and read it with CostReader.
        :return: costs of the file ({subject -> costs}), or None if the file was rejected.
        """
        write(tmp_path / "BAS" / "A.csv", content)
        return CostReader(str(tmp_path), ["BAS"], ["A"]).get_costs().get(("BAS", "A"))

    def test_drops_trailing_padding_silently(self, tmp_path, caplog):
        data = self.read(tmp_path, "V1;V2\n1;1\n2;\n;\n")
        assert data["V1"].tolist() == [1.0, 2.0]
        assert data["V2"].tolist() == [1.0]
        assert "WARNING" not in caplog.text

    def test_removes_and_logs_bad_cells_in_the_middle(self, tmp_path, caplog):
        data = self.read(tmp_path, "V1\n1\n\nx\n4\n\n")
        assert data["V1"].tolist() == [1.0, 4.0]
        assert "empty cell in the middle of the column (line 3)" in caplog.text
        assert "non-numeric value 'x' (line 4)" in caplog.text

    def test_empty_column(self, tmp_path):
        assert self.read(tmp_path, "V1;V2\n1;\n")["V2"].size == 0

    def test_matches_subjects_by_name(self, tmp_path):
        data = self.read(tmp_path, "V10;V1;V2\n0,5;1.5;1\n1,5;;2\n2,5;;\n")
        assert list(data) == ["V10", "V1", "V2"]
        assert data["V10"].tolist() == [0.5, 1.5, 2.5]
        assert data["V1"].tolist() == [1.5]
        assert data["V2"].tolist() == [1.0, 2.0]

    def test_with_bom(self, tmp_path):
        assert list(self.read(tmp_path, "﻿V10;V1\n1;2\n")) == ["V10", "V1"]

    def test_skips_empty_header(self, tmp_path, caplog):
        assert list(self.read(tmp_path, "V1;;V2\n1;2;3\n")) == ["V1", "V2"]
        assert "column 2 has an empty header" in caplog.text

    @pytest.mark.parametrize("content, message", [("", "empty file"), (";\n1;2\n", "no subject identifiers"),
                                                  ("V1;V2;V1\n1;2;3\n", "Duplicated subjects in the header: V1")])
    def test_rejects_invalid_file(self, tmp_path, caplog, content, message):
        assert self.read(tmp_path, content) is None
        assert message in caplog.text

    def test_logs_missing_folders_and_files(self, tmp_path, caplog):
        write(tmp_path / "BAS" / "A.csv", "V1\n1\n")
        write(tmp_path / "BAS" / "B.csv", "")
        reader = CostReader(str(tmp_path), ["BAS", "B6"], ["A", "B", "C"])
        assert list(reader.get_costs()) == [("BAS", "A")]
        assert reader.get_subjects() == {"V1"}
        assert "Missing folder" in caplog.text and "B6" in caplog.text
        assert "Missing file" in caplog.text and "C.csv" in caplog.text

    def test_loads_files_once(self, tmp_path):
        write(tmp_path / "BAS" / "A.csv", "V1\n1\n")
        reader = CostReader(str(tmp_path), ["BAS"], ["A"])
        costs = reader.get_costs()
        write(tmp_path / "BAS" / "A.csv", "V2\n1\n")
        assert reader.get_costs() is costs


class TestComputeFeatures:
    def test_values(self):
        x = np.array([1.0, 3.0, 2.0, 6.0, 4.0])
        features, skipped = FeatureMatrix.compute_features(x)
        assert skipped == []
        assert list(features) == FEATURES
        assert features["mean"] == pytest.approx(3.2)
        assert features["median"] == 3.0
        assert features["p10"] == pytest.approx(np.percentile(x, 10))
        assert features["p25"] == 2.0 and features["p75"] == 4.0 and features["iqr"] == 2.0
        assert features["p90"] == pytest.approx(np.percentile(x, 90))
        assert features["std"] == pytest.approx(np.std(x, ddof=0))
        assert features["coef_of_variation"] == pytest.approx(np.std(x) / 3.2)
        assert features["skewness"] == pytest.approx(skew(x, bias=True))
        assert features["kurtosis"] == pytest.approx(kurtosis(x, fisher=True, bias=True))
        assert features["trend"] == pytest.approx(np.polyfit(np.arange(5), x, 1)[0])
        assert features["mean_abs_diff"] == pytest.approx(2.25)
        assert features["autocorr_1"] == pytest.approx(np.corrcoef(x[:-1], x[1:])[0, 1])

    @pytest.mark.parametrize("n, expected_skipped", [
        (0, FEATURES),
        (1, ["skewness", "kurtosis", "trend", "mean_abs_diff", "autocorr_1"]),
        (2, ["skewness", "kurtosis", "autocorr_1"]),
        (3, ["kurtosis"]),
        (4, []),
    ])
    def test_minimum_observations(self, n, expected_skipped):
        features, skipped = FeatureMatrix.compute_features(np.arange(1, n + 1, dtype=float) ** 2)
        assert skipped == expected_skipped
        assert all(math.isnan(features[name]) for name in expected_skipped)
        assert not any(math.isnan(v) for name, v in features.items() if name not in expected_skipped)

    def test_constant_series(self):
        features, _ = FeatureMatrix.compute_features(np.full(5, 2.0))
        assert features["std"] == 0.0 and features["coef_of_variation"] == 0.0
        assert features["trend"] == pytest.approx(0.0)
        assert math.isnan(features["autocorr_1"])

    def test_zero_mean_gives_nan_coef_of_variation(self):
        features, _ = FeatureMatrix.compute_features(np.array([-1.0, 1.0]))
        assert math.isnan(features["coef_of_variation"])


class TestFeatureMatrix:
    @pytest.fixture
    def feature_matrix(self):
        costs = {
            ("BAS", "RR"): {"V10": np.array([1.0, 2.0]), "V2": np.array([3.0])},
            ("BAS", "SPO"): {"V2": np.array([1.0])},
            ("B6", "RR"): {"V1": np.array([5.0])},
        }
        return FeatureMatrix(costs, ["BAS", "B6"])

    def test_columns_and_order(self, feature_matrix):
        matrix = feature_matrix.get_matrix("RR")
        assert list(matrix.columns) == ["id", "condition", *FEATURES, "n_obs"]
        assert matrix[["condition", "id"]].values.tolist() == [["BAS", "V2"], ["BAS", "V10"], ["B6", "V1"]]
        assert matrix["n_obs"].tolist() == [1, 2, 1]
        assert matrix.loc[1, "mean"] == 1.5

    def test_subject_missing_from_metric_keeps_row_with_nan(self, feature_matrix):
        matrix = feature_matrix.get_matrix("SPO")
        assert matrix["id"].tolist() == ["V2", "V10", "V1"]
        assert matrix.loc[0, "mean"] == 1.0
        assert matrix.loc[1:, FEATURES].isna().all().all()
        assert matrix["n_obs"].isna().tolist() == [False, True, True]

    def test_logs_skipped_features(self, feature_matrix, caplog):
        feature_matrix.get_matrix("RR")
        assert "BAS/RR [V2]: only 1 observations" in caplog.text

    def test_log_incomplete_subjects(self, feature_matrix, caplog):
        feature_matrix.log_incomplete_subjects()
        assert "Subject V10 is missing from: BAS/SPO, B6/RR" in caplog.text


class TestMain:
    def test_writes_one_matrix_per_metric_and_log(self, tmp_path):
        metrics = ["ABP_RR-CBFV_RR", "ABP_SPO-CBFV_SPO", "ABP_SPP-CBFV_SPP"]
        for condition in ["BAS", "B6", "B10", "B15"]:
            for metric in metrics:
                write(tmp_path / "in" / condition / f"{metric}.csv", "V2;V1\n0,1;0.2\n0,3;0.4\n0,5;\n")
        out = tmp_path / "out"
        main(["--input-dir", str(tmp_path / "in"), "--output-dir", str(out)])
        for metric in metrics:
            matrix = pd.read_csv(out / f"{metric}.csv", sep=";")
            assert len(matrix) == 8
            assert matrix.loc[0, "id"] == "V1" and matrix.loc[0, "condition"] == "BAS"
            assert matrix.loc[1, "mean"] == pytest.approx(0.3)
        assert "errors: 0" in (out / "feature_matrix.log").read_text(encoding="utf-8")
