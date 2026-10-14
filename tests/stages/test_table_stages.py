"""
Tests of the stages of the metric medians (tables, metric_anova) on small synthetic paired metrics.
"""

import numpy as np
import pandas as pd
import pytest

from ddtw_ml.stages import compare_metric_medians, tabulate_medians

CONDITIONS = ["B6", "B10", "B15", "BAS"]
COLUMNS = ["ABP_SPO", "ABP_SPP", "ABP_RR", "CBFV_SPO", "CBFV_SPP", "CBFV_RR"]


def write_paired(config):
    """
    Subjects V1-V6 in every condition and V7 only in B6; ABP_SPO grows strongly with the breathing rate.
    """
    rng = np.random.default_rng(0)
    shift = {"BAS": 0, "B6": 10, "B10": 20, "B15": 30}
    for condition in CONDITIONS:
        directory = config.stage_dir("paired") / condition
        directory.mkdir(parents=True)
        for v in range(1, 8 if condition == "B6" else 7):
            df = pd.DataFrame({c: rng.normal(100, 5, 50) for c in COLUMNS})
            df["ABP_SPO"] += shift[condition]
            df.insert(0, "DateTime", range(1, 51))
            df.to_csv(directory / f"V{v}_{condition}_PP_F_M.csv", sep=";", index=False)


def test_tables_and_metric_anova(config, make_ctx):
    write_paired(config)
    tabulate_medians.run(make_ctx("tables"))
    medians = pd.read_csv(config.stage_dir("tables") / "metrics_for_anova.csv", sep=";", index_col=0)
    # only the subjects present in all conditions; baseline first
    assert len(medians) == 6
    assert list(medians.columns[:6]) == ["BAS_ABP_SPO", "BAS_CBFV_SPO", "BAS_ABP_SPP", "BAS_CBFV_SPP", "BAS_ABP_RR",
                                         "BAS_CBFV_RR"]
    v1 = pd.read_csv(config.stage_dir("paired") / "B10" / "V1_B10_PP_F_M.csv", sep=";")
    assert medians.loc[0, "B10_CBFV_SPP"] == pytest.approx(v1["CBFV_SPP"].median(), rel=1e-12)
    table = pd.read_csv(config.stage_dir("tables") / "medians_iqr.csv", sep=";", index_col=0)
    assert list(table.index) == ["BAS", "B6", "B10", "B15"]
    assert "\\text{-}" in table.loc["B6", "ABP_RR"] and "\t" not in table.loc["B6", "ABP_RR"]

    compare_metric_medians.run(make_ctx("metric_anova"))
    out = config.stage_dir("metric_anova")
    assert sorted(p.name for p in out.glob("*_anova.csv")) == sorted(
        f"{s}_{m}_anova.csv" for s in ("ABP", "CBFV") for m in ("SPO", "SPP", "RR"))
    anova = pd.read_csv(out / "ABP_SPO_anova.csv", sep=";", index_col=0)
    assert anova["Pr > F"].iloc[0] < 0.05
    posthoc = pd.read_csv(out / "ABP_SPO_posthoc.csv", sep=";")
    assert len(posthoc) == 6 and posthoc["significant"].all()
