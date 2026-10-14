"""
Tests of the DTW cost stages (costs, pairs, mean_costs, anova), run in sequence on small synthetic metrics.
"""

import numpy as np
import pandas as pd
import pytest

from ddtw_ml.core.dtw import DTW
from ddtw_ml.stages import average_costs_per_subject, compare_conditions, compute_dtw_costs, group_costs_by_pair

CONDITIONS = ["B6", "B10", "B15", "BAS"]
SUBJECTS = 4
COLUMNS = ["DateTime", "ABP_SPO", "ABP_SPP", "ABP_RR", "CBFV_SPO", "CBFV_SPP", "CBFV_RR"]


@pytest.fixture
def normalised(config):
    rng = np.random.default_rng(0)
    for condition in CONDITIONS:
        directory = config.stage_dir("normalised") / condition
        directory.mkdir(parents=True)
        for v in range(1, SUBJECTS + 1):
            df = pd.DataFrame({c: np.arange(40) if c == "DateTime" else rng.uniform(0, 1, 40) for c in COLUMNS})
            df.to_csv(directory / f"V{v}_{condition}_PP_F_M_PP.csv", sep=";", index=False)
    return config


def test_cost_stages_in_sequence(normalised, make_ctx):
    config = normalised
    compute_dtw_costs.run(make_ctx("costs"))
    costs_file = config.stage_dir("costs") / "td-method" / "B6" / "V1_B6_PP_F_M_PP_DTW.csv"
    costs = pd.read_csv(costs_file, sep=";")
    assert list(costs.columns) == ["Window"] + config.run["pairs"]
    source = pd.read_csv(config.stage_dir("normalised") / "B6" / "V1_B6_PP_F_M_PP.csv", sep=";")
    expected, _, _ = DTW(source["ABP_RR"], source["CBFV_SPP"], var="DDTW").sliding_window_dtw(10, 5, "td-method")
    np.testing.assert_allclose(costs["ABP_RR-CBFV_SPP"], expected)
    assert (config.stage_dir("costs") / "d-method" / "BAS" / "V4_BAS_PP_F_M_PP_DTW.csv").exists()

    group_costs_by_pair.run(make_ctx("pairs"))
    pairs = pd.read_csv(config.stage_dir("pairs") / "td-method" / "B6" / "ABP_RR-CBFV_SPP.csv", sep=";")
    assert sorted(pairs.columns) == [f"V{v}" for v in range(1, SUBJECTS + 1)]
    np.testing.assert_allclose(pairs["V1"], expected)

    average_costs_per_subject.run(make_ctx("mean_costs"))
    means = pd.read_csv(config.stage_dir("mean_costs") / "ABP_RR-CBFV_SPP.csv", sep=";", index_col=0)
    assert sorted(means.columns) == sorted(f"{m}_{c}" for m in ["td-method", "d-method"] for c in CONDITIONS)
    assert means.loc["V1", "td-method_B6"] == pytest.approx(np.mean(expected))

    compare_conditions.run(make_ctx("anova"))
    for pair in config.run["pairs"]:
        for method in ["td-method", "d-method"]:
            anova = pd.read_csv(config.stage_dir("anova") / pair / f"{method}_anova_results.csv", sep=";")
            posthoc = pd.read_csv(config.stage_dir("anova") / pair / f"{method}_posthoc_results.csv", sep=";")
            assert list(anova.columns) == ["Source", "F", "p_unc", "ng2"]
            assert len(posthoc) == 6
