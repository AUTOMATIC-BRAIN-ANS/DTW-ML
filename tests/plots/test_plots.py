"""
Tests of the plots (cleaning, beats, costs) on small synthetic data, rendered without LaTeX.
"""

import numpy as np
import pandas as pd

from ddtw_ml.plots.beats import PlotBeatMetrics
from ddtw_ml.plots.cleaning import PlotCleaningSteps
from ddtw_ml.plots.costs import PlotCostsPerWindow


def test_plot_cleaning_steps(config, make_ctx, no_latex):
    directory = config.stage_dir("recordings") / "B6"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    t = np.arange(3000) / 200
    pd.DataFrame({"DateTime": t, "ABP": 90 + 15 * np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 1, 3000),
                  "CBFV": 60 + 20 * np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 1, 3000),
                  "BPM": np.full(3000, 6.0)}).to_csv(directory / "V1_B6.csv", sep=";", index=False)
    PlotCleaningSteps(make_ctx("cleaning", "plot")).run()
    out = config.plot_dir("cleaning") / "B6"
    assert sorted(p.name for p in out.iterdir()) == ["V1_B6_ABP.pdf", "V1_B6_CBFV.pdf"]


def test_plot_beat_metrics(config, make_ctx, no_latex):
    rng = np.random.default_rng(0)
    paired = config.stage_dir("paired") / "B10"
    normalised = config.stage_dir("normalised") / "B10"
    paired.mkdir(parents=True)
    normalised.mkdir(parents=True)
    metrics = [f"{s}_{m}" for s in ("ABP", "CBFV") for m in ("SPO", "SPP", "DN", "DPP", "RR")]
    df = pd.DataFrame({m: rng.normal(100, 5, 30) for m in metrics})
    df.insert(0, "DateTime", range(1, 31))
    df.to_csv(paired / "V1_B10_PP_F_M.csv", sep=";", index=False)
    df.drop(columns=[c for c in metrics if c.endswith(("_DN", "_DPP"))]).to_csv(
        normalised / "V1_B10_PP_F_M_PP.csv", sep=";", index=False)
    PlotBeatMetrics(make_ctx("beats", "plot")).run()
    out = config.plot_dir("beats")
    assert sorted(str(p.relative_to(out).as_posix()) for p in out.rglob("*.pdf")) == [
        "normalised/abp/B10/V1_B10_PP_F_M_PP.pdf", "normalised/cbfv/B10/V1_B10_PP_F_M_PP.pdf",
        "paired/abp/B10/V1_B10_PP_F_M.pdf", "paired/cbfv/B10/V1_B10_PP_F_M.pdf"]


def test_plot_costs_per_window(config, make_ctx, no_latex, settings):
    directory = config.stage_dir("costs") / "td-method" / "BAS"
    directory.mkdir(parents=True)
    costs = pd.DataFrame({pair: np.random.default_rng(0).uniform(0, 1, 12) for pair in settings.analysis["pairs"]})
    costs.insert(0, "Window", [[5 * i, 5 * i + 10] for i in range(12)])
    costs.to_csv(directory / "V1_BAS_PP_F_M_PP_DTW.csv", sep=";", index=False)
    PlotCostsPerWindow(make_ctx("costs", "plot")).run()
    assert (config.plot_dir("costs") / "td-method" / "BAS" / "V1_BAS_PP_F_M_PP_DTW.pdf").is_file()
