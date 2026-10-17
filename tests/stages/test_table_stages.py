"""
Tests of the stages of the metric medians (tables, metric_anova) on small synthetic paired metrics.
"""

import numpy as np
import pandas as pd
import pingouin as pg
import pytest

from ddtw_ml.stages.metric_anova import CompareMetricMedians
from ddtw_ml.stages.tables import TabulateMedians

CONDITIONS = ["B6", "B10", "B15", "BAS"]
COLUMNS = ["ABP_SPO", "ABP_SPP", "ABP_RR", "CBFV_SPO", "CBFV_SPP", "CBFV_RR"]


def write_paired(config):
    """
    Subjects V1-V6 in every condition and V7 only in B6; ABP_SPO grows strongly with the breathing rate. `aligned`
    keeps V1-V5 (V6 rejected by `normalised` in one condition).
    """
    for condition in CONDITIONS:
        aligned = config.stage_dir("aligned") / condition
        aligned.mkdir(parents=True)
        for v in range(1, 6):
            (aligned / f"V{v}_{condition}_PP_F_M_PP.csv").touch()
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
            # filtered signals: ETCO2 = 30 + subject number + shift / 10, one NaN
            signals = config.stage_dir("filtered") / condition
            signals.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"DateTime": np.arange(4) * 5, "ETCO2": [30 + v + shift[condition] / 10] * 3 + [np.nan]}
                         ).to_csv(signals / f"V{v}_{condition}_PP_F.csv", sep=";", index=False)


def test_tables_and_metric_anova(config, make_ctx):
    write_paired(config)
    TabulateMedians(make_ctx("tables")).run()
    medians = pd.read_csv(config.stage_dir("tables") / "metrics_for_anova.csv", sep=";", index_col=0)
    # only the subjects kept by `aligned`, rows labelled by the subject; baseline first
    assert medians.index.name == "Subject" and list(medians.index) == [f"V{v}" for v in range(1, 6)]
    assert list(medians.columns[:6]) == ["BAS_ABP_SPO", "BAS_CBFV_SPO", "BAS_ABP_SPP", "BAS_CBFV_SPP", "BAS_ABP_RR",
                                         "BAS_CBFV_RR"]
    v3 = pd.read_csv(config.stage_dir("paired") / "B10" / "V3_B10_PP_F_M.csv", sep=";")
    assert medians.loc["V3", "B10_CBFV_SPP"] == pytest.approx(v3["CBFV_SPP"].median(), rel=1e-12)
    # mean EtCO2 of each subject, NaN skipped
    assert medians["B10_ETCO2"].tolist() == pytest.approx([32 + v for v in range(1, 6)])
    assert medians["BAS_ETCO2"].tolist() == pytest.approx([30 + v for v in range(1, 6)])
    table = pd.read_csv(config.stage_dir("tables") / "medians_iqr.csv", sep=";", index_col=0)
    assert list(table.index) == ["BAS", "B6", "B10", "B15"]
    assert "\\text{-}" in table.loc["B6", "ABP_RR"] and "\t" not in table.loc["B6", "ABP_RR"]
    assert table.loc["B10", "ETCO2"].startswith("$35.0$")

    CompareMetricMedians(make_ctx("metric_anova")).run()
    out = config.stage_dir("metric_anova")
    assert sorted(p.name for p in out.glob("*_anova.csv")) == sorted(
        f"{s}_{m}_anova.csv" for s in ("ABP", "CBFV") for m in ("SPO", "SPP", "RR"))
    anova = pd.read_csv(out / "ABP_SPO_anova.csv", sep=";", index_col=0)
    assert anova["Pr > F"].iloc[0] < 0.05
    # sphericity columns agree with pingouin.rm_anova on the same data
    long = medians.reset_index().melt(id_vars="Subject", value_vars=[f"{c}_ABP_SPO" for c in CONDITIONS])
    expected = pg.rm_anova(dv="value", within="variable", subject="Subject", data=long, correction=True)
    for column in ["eps", "p_GG_corr", "W_spher", "p_spher"]:
        assert anova[column].iloc[0] == pytest.approx(expected[column].iloc[0], rel=1e-9)
    assert anova["sphericity"].iloc[0] == expected["sphericity"].iloc[0]
    posthoc = pd.read_csv(out / "ABP_SPO_posthoc.csv", sep=";")
    assert len(posthoc) == 6 and posthoc["significant"].all()
