"""
Unit tests for classification/modeling/data.py on a synthetic feature matrix.
"""

import numpy as np
import pandas as pd
import pytest

from project.classification.config import CLASSES
from project.classification.modeling.data import FeatureMatrixData


class TestFeatureMatrixData:
    def test_features_target_groups(self, data):
        assert data.features == ["signal", "noise"]
        assert data.classes == CLASSES
        assert len(data.get_features()) == len(data.get_target()) == len(data.get_groups()) == 48
        assert set(data.get_groups()) == {f"V{s}" for s in range(1, 13)}
        assert data.describe() == {"n_rows": 48, "n_subjects": 12, "n_features": 2, "classes": CLASSES,
                                   "chance_level": 0.25}

    def test_drops_rows_without_features(self, matrix_path, tmp_path, caplog):
        df = pd.read_csv(matrix_path, sep=";")
        df.loc[0, ["signal", "noise"]] = np.nan
        path = tmp_path / "m.csv"
        df.to_csv(path, sep=";", index=False)
        data = FeatureMatrixData(str(path))
        assert len(data.df) == 47
        assert "V1 / BAS: no features - row dropped" in caplog.text

    def test_keeps_chosen_classes(self, matrix_path):
        data = FeatureMatrixData(str(matrix_path), ["B6", "BAS"])
        assert data.classes == ["B6", "BAS"]
        assert set(data.get_target()) == {"BAS", "B6"}

    def test_needs_two_classes(self, matrix_path):
        with pytest.raises(ValueError):
            FeatureMatrixData(str(matrix_path), ["BAS"])
