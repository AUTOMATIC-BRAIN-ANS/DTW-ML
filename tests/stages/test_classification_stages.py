"""
Tests of the classification stages (features, cv_results, scores): the feature matrices are built from synthetic
costs; the cross-validation and the summary (tested in tests/stages/cv_results and tests/stages/test_scores.py) are checked
for the arguments they get.
"""

import numpy as np
import pandas as pd

from ddtw_ml.stages import scores as scores_stage
from ddtw_ml.stages.cv_results import CrossValidateClassifiers
from ddtw_ml.stages.cv_results import stage as cv_results_stage
from ddtw_ml.stages.features import BuildFeatureMatrices
from ddtw_ml.stages.scores import CollectClassifierScores


def test_feature_matrices_for_every_method_and_pair(config, make_ctx, settings):
    rng = np.random.default_rng(0)
    for method in settings.analysis["methods"]:
        for condition in settings.analysis["conditions"]:
            directory = config.stage_dir("pairs") / method / condition
            directory.mkdir(parents=True)
            for pair in settings.analysis["pairs"]:
                pd.DataFrame({f"V{v}": rng.uniform(0, 1, 20) for v in range(1, 4)}).to_csv(
                    directory / f"{pair}.csv", sep=";", index=False)
    BuildFeatureMatrices(make_ctx("features")).run()
    for method in settings.analysis["methods"]:
        files = sorted(p.stem for p in (config.stage_dir("features") / method).glob("*.csv"))
        assert files == sorted(settings.analysis["pairs"])
    matrix = pd.read_csv(config.stage_dir("features") / "d-method" / f"{settings.analysis['pairs'][-1]}.csv", sep=";")
    assert len(matrix) == 3 * 4 and set(matrix["condition"]) == set(settings.analysis["conditions"])


def test_feature_matrices_use_the_conditions_of_the_analysis(config, make_ctx, settings):
    settings.analysis["conditions"] = ["B15", "BAS"]
    pair = settings.analysis["pairs"][0]
    for method in settings.analysis["methods"]:
        for condition in ["BAS", "B6", "B10", "B15"]:
            directory = config.stage_dir("pairs") / method / condition
            directory.mkdir(parents=True)
            pd.DataFrame({"V1": [0.1, 0.2], "V2": [0.3, 0.4]}).to_csv(directory / f"{pair}.csv", sep=";", index=False)
    BuildFeatureMatrices(make_ctx("features")).run()
    matrix = pd.read_csv(config.stage_dir("features") / "td-method" / f"{pair}.csv", sep=";")
    assert matrix["condition"].tolist() == ["BAS", "BAS", "B15", "B15"]


def test_cross_validation_and_scores_get_the_stage_directories(config, make_ctx, monkeypatch, settings):
    calls = {}
    monkeypatch.setattr(cv_results_stage, "classify", lambda options: calls.setdefault("classify", options))
    monkeypatch.setattr(scores_stage, "collect", lambda *args: calls.setdefault("collect", args))
    config.stage_dir("features").mkdir(parents=True)
    CrossValidateClassifiers(make_ctx("cv_results")).run()
    CollectClassifierScores(make_ctx("scores")).run()
    options = calls["classify"]
    assert options.input_dir == str(config.stage_dir("features"))
    assert options.output_dir == str(config.stage_dir("cv_results"))
    assert options.methods == settings.analysis["methods"] and options.metrics == settings.analysis["pairs"]
    params = settings.sections["stage"]["cv_results"]
    assert (options.models, options.n_bootstrap, options.ci_level) == (params["models"], 1000, 0.95)
    input_dir, output_dir, methods, metrics, validations, models = calls["collect"]
    assert (input_dir, output_dir) == (str(config.stage_dir("cv_results")), str(config.stage_dir("scores")))
    assert metrics == settings.analysis["pairs"] and methods == ["d-method", "td-method"]
