"""
Tests of the beat stages (beats, paired, normalised) on small synthetic data.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.signal import find_peaks

from ddtw_ml.stages.beats import ComputeBeatMetrics
from ddtw_ml.stages.beats.metrics import FromSignal
from ddtw_ml.stages.normalised import NormaliseBeatMetrics, PreprocessMetrics
from ddtw_ml.stages.paired import PairBeatMetrics

METRICS = ["DateTime", "ABP_SPO", "ABP_SPP", "ABP_DN", "ABP_DPP", "ABP_RR",
           "CBFV_SPO", "CBFV_SPP", "CBFV_DN", "CBFV_DPP", "CBFV_RR"]


@pytest.fixture
def simple_troughs(monkeypatch):
    """
    Troughs as the minima of the signal at least 0.5 s apart, instead of detect_troughs (outside the repository).
    """
    monkeypatch.setattr("ddtw_ml.stages.beats.metrics.detect_peaks_troughs_optimized",
                        lambda s: find_peaks(-np.asarray(s), distance=100)[0])


def test_compute_beat_metrics(config, make_ctx, simple_troughs):
    src = config.stage_dir("filtered") / "B6"
    src.mkdir(parents=True)
    t = np.arange(3000) / 200
    df = pd.DataFrame({"DateTime": t, "ABP": 90 + 15 * np.sin(2 * np.pi * 1.2 * t),
                       "CBFV": 60 + 20 * np.sin(2 * np.pi * 1.2 * t + 0.3)})
    df.to_csv(src / "V1_B6_PP_F.csv", sep=";", index=False)
    ComputeBeatMetrics(make_ctx("beats")).run()
    out_dir = config.stage_dir("beats") / "B6" / "V1_B6_PP_F"
    for name, signal_type in (("ABP", "abp"), ("CBFV", "cbfv")):
        out = pd.read_csv(out_dir / f"{name}.csv", sep=";")
        expected = pd.DataFrame(FromSignal(df["DateTime"], df[name], signal_type).get_metrics())
        pd.testing.assert_frame_equal(out, expected, check_dtype=False)
        assert len(out) > 10


def write_beats(directory, n_abp, n_cbfv):
    directory.mkdir(parents=True)
    rng = np.random.default_rng(n_abp)
    for name, n in (("ABP", n_abp), ("CBFV", n_cbfv)):
        cols = [c for c in METRICS if c == "DateTime" or c.startswith(name + "_")]
        pd.DataFrame({c: rng.normal(size=n) for c in cols}).to_csv(directory / f"{name}.csv", sep=";", index=False)


def test_pair_beat_metrics_cuts_to_the_shorter_signal_and_rejects_short_ones(config, make_ctx):
    write_beats(config.stage_dir("beats") / "B6" / "V1_B6_PP_F", 20, 15)
    write_beats(config.stage_dir("beats") / "B6" / "V2_B6_PP_F", 5, 8)
    ctx = make_ctx("paired")
    PairBeatMetrics(ctx).run()
    out = pd.read_csv(config.stage_dir("paired") / "B6" / "V1_B6_PP_F_M.csv", sep=";")
    assert list(out.columns) == METRICS
    assert out["DateTime"].tolist() == list(range(1, 16))
    assert not (config.stage_dir("paired") / "B6" / "V2_B6_PP_F_M.csv").exists()
    assert (config.spoiled_dir() / "B6" / "V2_B6_PP_F" / "ABP.csv").exists()
    assert ctx.report.processed == 1 and len(ctx.report.rejected) == 1


def test_normalise_beat_metrics(config, make_ctx):
    src = config.stage_dir("paired") / "B10"
    src.mkdir(parents=True)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: (np.arange(200) if c == "DateTime" else rng.normal(100, 10, 200)) for c in METRICS})
    df.to_csv(src / "V1_B10_PP_F_M.csv", sep=";", index=False)
    df.drop(columns=["CBFV_RR"]).to_csv(src / "V2_B10_PP_F_M.csv", sep=";", index=False)
    ctx = make_ctx("normalised")
    NormaliseBeatMetrics(ctx).run()
    out = pd.read_csv(config.stage_dir("normalised") / "B10" / "V1_B10_PP_F_M_PP.csv", sep=";")
    expected = PreprocessMetrics(str(src / "V1_B10_PP_F_M.csv")).get_preprocessed_metrics()
    pd.testing.assert_frame_equal(out, expected, check_dtype=False, check_exact=False, rtol=1e-12)
    assert list(out.columns) == ["DateTime", "ABP_SPO", "ABP_SPP", "ABP_RR", "CBFV_SPO", "CBFV_SPP", "CBFV_RR"]
    # the file without CBFV_RR fails and is copied to the spoiled data
    assert ctx.report.processed == 1 and (config.spoiled_dir() / "B10" / "V2_B10_PP_F_M.csv").exists()
