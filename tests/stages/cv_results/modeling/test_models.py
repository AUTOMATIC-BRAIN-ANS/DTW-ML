"""
Unit tests for stages/cv_results/modeling/models.py on a synthetic feature matrix.
"""

import numpy as np
import pytest

from ddtw_ml.stages.cv_results.constants import MODELS
from ddtw_ml.stages.cv_results.modeling.models import ModelFactory
from ddtw_ml.stages.cv_results.scoring.classification_metrics import ClassificationMetrics


class TestModelFactory:
    @pytest.mark.parametrize("name", MODELS)
    def test_models_fit_with_nan(self, data, name):
        X = data.get_features().copy()
        X.iloc[0, 1] = np.nan
        model = ModelFactory.get(name).fit(X, data.get_target())
        assert ClassificationMetrics.predict(model, X).shape == (48,)
        assert model.predict_proba(X).shape == (48, 4)

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            ModelFactory.get("N/A")
