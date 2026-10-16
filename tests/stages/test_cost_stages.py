"""
Tests of the DTW cost stages (aligned, costs, pairs, mean_costs, anova), run in sequence on small synthetic metrics.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from ddtw_ml.pipeline import PipelineError
from ddtw_ml.pipeline.spoiled import FAILURES_FILE
from ddtw_ml.stages.aligned import AlignSubjects
from ddtw_ml.stages.anova import CompareConditions
from ddtw_ml.stages.costs import ComputeDtwCosts
from ddtw_ml.stages.costs.dtw import DTW
from ddtw_ml.stages.mean_costs import AverageCostsPerSubject
from ddtw_ml.stages.pairs import GroupCostsByPair

CONDITIONS = ["B6", "B10", "B15", "BAS"]
SUBJECTS = 4
COLUMNS = ["DateTime", "ABP_SPO", "ABP_SPP", "ABP_RR", "CBFV_SPO", "CBFV_SPP", "CBFV_RR"]
COST_PAIRS = ["ABP_SPO-CBFV_SPO", "ABP_SPP-CBFV_SPP", "ABP_SPO-CBFV_SPP", "ABP_SPP-CBFV_SPO", "ABP_RR-CBFV_SPO",
              "ABP_RR-CBFV_SPP", "ABP_SPO-CBFV_RR", "ABP_SPP-CBFV_RR", "ABP_RR-CBFV_RR"]


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


def test_align_subjects_keeps_the_subjects_of_all_conditions(normalised, make_ctx):
    config = normalised
    # V2 rejected earlier in B10, V3 in B10 and BAS
    (config.stage_dir("normalised") / "B10" / "V2_B10_PP_F_M_PP.csv").unlink()
    (config.stage_dir("normalised") / "B10" / "V3_B10_PP_F_M_PP.csv").unlink()
    (config.stage_dir("normalised") / "BAS" / "V3_BAS_PP_F_M_PP.csv").unlink()
    ctx = make_ctx("aligned")
    AlignSubjects(ctx).run()
    for condition in CONDITIONS:
        kept = sorted(f.name for f in (config.stage_dir("aligned") / condition).iterdir())
        assert kept == [f"V1_{condition}_PP_F_M_PP.csv", f"V4_{condition}_PP_F_M_PP.csv"]
        src = config.stage_dir("normalised") / condition / kept[0]
        assert (config.stage_dir("aligned") / condition / kept[0]).read_bytes() == src.read_bytes()
    spoiled = config.spoiled_dir() / "B6" / "aligning_data"
    assert sorted(f.name for f in spoiled.iterdir()) == ["V2_B6_PP_F_M_PP.csv", "V3_B6_PP_F_M_PP.csv", FAILURES_FILE]
    assert "subject V3 is missing from B10, BAS" in (spoiled / FAILURES_FILE).read_text(encoding="utf-8")
    assert not (config.spoiled_dir() / "B10" / "aligning_data").exists()
    assert len(ctx.report.rejected) == 2 + 0 + 2 + 1  # B6, B10, B15, BAS
    # the input is left in place
    assert (config.stage_dir("normalised") / "B6" / "V2_B6_PP_F_M_PP.csv").exists()


def test_align_subjects_needs_every_condition(normalised, make_ctx):
    shutil.rmtree(normalised.stage_dir("normalised") / "B15")
    with pytest.raises(PipelineError, match="B15"):
        AlignSubjects(make_ctx("aligned")).run()


def test_cost_stages_in_sequence(normalised, make_ctx, settings):
    config = normalised
    AlignSubjects(make_ctx("aligned")).run()
    ComputeDtwCosts(make_ctx("costs")).run()
    costs_file = config.stage_dir("costs") / "td-method" / "B6" / "V1_B6_PP_F_M_PP_DTW.csv"
    costs = pd.read_csv(costs_file, sep=";")
    # costs computes all 9 pairs, whatever [analysis] pairs is (B-20)
    assert list(costs.columns) == ["Window"] + COST_PAIRS
    # a pair of settings.toml, followed through the stages
    pair = settings.analysis["pairs"][-1]
    abp, cbfv = pair.split("-")
    source = pd.read_csv(config.stage_dir("normalised") / "B6" / "V1_B6_PP_F_M_PP.csv", sep=";")
    expected, _, _ = DTW(source[abp], source[cbfv], var="DDTW").sliding_window_dtw(10, 5, "td-method")
    np.testing.assert_allclose(costs[pair], expected)
    assert (config.stage_dir("costs") / "d-method" / "BAS" / "V4_BAS_PP_F_M_PP_DTW.csv").exists()

    GroupCostsByPair(make_ctx("pairs")).run()
    pairs = pd.read_csv(config.stage_dir("pairs") / "td-method" / "B6" / f"{pair}.csv", sep=";")
    assert sorted(pairs.columns) == [f"V{v}" for v in range(1, SUBJECTS + 1)]
    np.testing.assert_allclose(pairs["V1"], expected)

    AverageCostsPerSubject(make_ctx("mean_costs")).run()
    means = pd.read_csv(config.stage_dir("mean_costs") / f"{pair}.csv", sep=";", index_col=0)
    assert sorted(means.columns) == sorted(f"{m}_{c}" for m in ["td-method", "d-method"] for c in CONDITIONS)
    assert means.loc["V1", "td-method_B6"] == pytest.approx(np.mean(expected))

    CompareConditions(make_ctx("anova")).run()
    for pair in settings.analysis["pairs"]:
        for method in ["td-method", "d-method"]:
            anova = pd.read_csv(config.stage_dir("anova") / pair / f"{method}_anova_results.csv", sep=";")
            posthoc = pd.read_csv(config.stage_dir("anova") / pair / f"{method}_posthoc_results.csv", sep=";")
            assert list(anova.columns) == ["Source", "F", "p_unc", "ng2"]
            assert len(posthoc) == 6


def write_costs(config, condition, subjects):
    directory = config.stage_dir("costs") / "td-method" / condition
    directory.mkdir(parents=True)
    for v in subjects:
        pd.DataFrame({"Window": [0, 5], **{p: [v, v + 0.5] for p in COST_PAIRS}}).to_csv(
            directory / f"V{v}_{condition}_PP_F_M_PP_DTW.csv", sep=";", index=False)
    return directory


def test_group_costs_by_pair_takes_every_subject_from_the_file_names(config, make_ctx, settings):
    directory = write_costs(config, "B6", [2, 100, 101])
    (directory / "V3_B10_PP_F_M_PP_DTW.csv").write_text("Window\n0\n", encoding="utf-8")  # another condition
    GroupCostsByPair(make_ctx("pairs")).run()
    pairs = pd.read_csv(config.stage_dir("pairs") / "td-method" / "B6" / f"{settings.analysis['pairs'][0]}.csv",
                        sep=";")
    assert list(pairs.columns) == ["V100", "V101", "V2"]
    assert pairs["V100"].tolist() == [100, 100.5]
