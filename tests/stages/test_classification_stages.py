"""
Tests of the classification stages (features, cv_results, scores): the feature matrices are built from synthetic
costs; the cross-validation and the summary (tested in tests/core/ml) are checked for the arguments they get.
"""

import numpy as np
import pandas as pd

from ddtw_ml.core.ml import classify, summary
from ddtw_ml.stages import build_feature_matrices, collect_classifier_scores, cross_validate_classifiers


def test_feature_matrices_for_every_method_and_pair(config, make_ctx):
    rng = np.random.default_rng(0)
    for method in config.run["methods"]:
        for condition in config.run["conditions"]:
            directory = config.stage_dir("pairs") / method / condition
            directory.mkdir(parents=True)
            for pair in config.run["pairs"]:
                pd.DataFrame({f"V{v}": rng.uniform(0, 1, 20) for v in range(1, 4)}).to_csv(
                    directory / f"{pair}.csv", sep=";", index=False)
    build_feature_matrices.run(make_ctx("features"))
    for method in config.run["methods"]:
        files = sorted(p.stem for p in (config.stage_dir("features") / method).glob("*.csv"))
        assert files == sorted(config.run["pairs"])
    matrix = pd.read_csv(config.stage_dir("features") / "d-method" / "ABP_RR-CBFV_SPP.csv", sep=";")
    assert len(matrix) == 3 * 4 and set(matrix["condition"]) == set(config.run["conditions"])


def test_cross_validation_and_scores_get_the_stage_directories(config, make_ctx, monkeypatch):
    calls = {}
    monkeypatch.setattr(classify, "main", lambda argv: calls.setdefault("classify", argv))
    monkeypatch.setattr(summary, "main", lambda argv: calls.setdefault("summary", argv))
    config.stage_dir("features").mkdir(parents=True)
    cross_validate_classifiers.run(make_ctx("cv_results"))
    collect_classifier_scores.run(make_ctx("scores"))
    args = classify.parse_arguments(calls["classify"])
    assert args.input_dir == str(config.stage_dir("features"))
    assert args.output_dir == str(config.stage_dir("cv_results"))
    assert args.methods == config.run["methods"] and args.metrics == config.run["pairs"]
    args = summary.parse_arguments(calls["summary"])
    assert (args.input_dir, args.output_dir) == (str(config.stage_dir("cv_results")), str(config.stage_dir("scores")))
    assert args.metrics == config.run["pairs"]
