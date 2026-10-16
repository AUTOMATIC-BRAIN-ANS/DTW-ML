"""
Tests of the beat stages (beats, paired, normalised) on small synthetic data.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import signal as sp_sig
from scipy.signal import find_peaks

from ddtw_ml.stages.beats import ComputeBeatMetrics
from ddtw_ml.stages.beats.metrics import FromSignal
from ddtw_ml.stages.beats.troughs import detect_peaks_troughs_optimized
from ddtw_ml.stages.normalised import NormaliseBeatMetrics, PreprocessMetrics
from ddtw_ml.stages.paired import PairBeatMetrics

METRICS = ["DateTime", "ABP_SPO", "ABP_SPP", "ABP_DN", "ABP_DPP", "ABP_RR",
           "CBFV_SPO", "CBFV_SPP", "CBFV_DN", "CBFV_DPP", "CBFV_RR"]


def original_troughs(signal, max_scale=0):
    """
    The original implementation (C:/Python/ZSSI/detect_troughs.py, N x L matrices), the reference for the troughs.
    """
    N = len(signal)
    signal = np.nan_to_num(signal, nan=np.nanmean(signal))
    d_signal = sp_sig.detrend(signal, type='linear')
    L = math.ceil(max_scale / 2) - 1 if max_scale != 0 else math.ceil(N / 2) - 1
    Mn = np.zeros((N, L), dtype=bool)
    for k in range(1, L + 1):
        center = d_signal[k:-k]
        Mn[np.arange(k, N - k), k - 1] = (center < d_signal[:-2*k]) & (center < d_signal[2*k:])
    dn = np.argmax(np.sum(Mn, axis=0))
    Mn = Mn[:, :dn + 1]
    return np.where(np.sum(Mn == 0, axis=1) == 0)[0]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("max_scale", [0, 40, 401])
def test_troughs_match_the_original_implementation(seed, max_scale):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 4000))
    t = np.arange(n) / 200
    s = (90 + 15 * np.sin(2 * np.pi * rng.uniform(0.8, 2) * t) + 3 * np.sin(2 * np.pi * 0.1 * t)
         + rng.normal(0, rng.uniform(0.01, 3), n))
    s[rng.integers(0, n, 5)] = np.nan
    if seed == 0:
        s = np.round(s)  # equal neighbours and equal trough counts at several scales
    assert np.array_equal(detect_peaks_troughs_optimized(s, max_scale), original_troughs(s, max_scale))


def test_troughs_need_at_least_one_scale():
    with pytest.raises(ValueError):
        detect_peaks_troughs_optimized(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        original_troughs(np.array([1.0, 2.0]))


def test_troughs_are_the_minima_of_a_periodic_signal():
    # 1.2 Hz at 200 Hz: a minimum every 1/1.2 s (~167 samples); minima closer to the ends than about half a period
    # have no neighbour at the larger scales and are not found
    t = np.arange(3000) / 200
    troughs = detect_peaks_troughs_optimized(np.sin(2 * np.pi * 1.2 * t) + 0.01 * t)
    minima = find_peaks(-np.sin(2 * np.pi * 1.2 * t))[0]
    expected = minima[(minima >= 84) & (minima < 3000 - 84)]
    assert len(troughs) == len(expected)
    assert np.abs(troughs - expected).max() <= 1


def test_troughs_ignore_nans():
    s = np.sin(2 * np.pi * 1.2 * np.arange(2000) / 200)
    with_nan = s.copy()
    with_nan[10] = np.nan
    assert list(detect_peaks_troughs_optimized(with_nan)) == list(detect_peaks_troughs_optimized(s))


def test_compute_beat_metrics(config, make_ctx):
    src = config.stage_dir("filtered") / "B6"
    src.mkdir(parents=True)
    t = np.arange(3000) / 200
    df = pd.DataFrame({"DateTime": t * 1000, "ABP": 90 + 15 * np.sin(2 * np.pi * 1.2 * t),
                       "CBFV": 60 + 20 * np.sin(2 * np.pi * 1.2 * t + 0.3)})
    df.to_csv(src / "V1_B6_PP_F.csv", sep=";", index=False)
    ComputeBeatMetrics(make_ctx("beats")).run()
    out_dir = config.stage_dir("beats") / "B6" / "V1_B6_PP_F"
    for name, signal_type in (("ABP", "abp"), ("CBFV", "cbfv")):
        out = pd.read_csv(out_dir / f"{name}.csv", sep=";")
        expected = pd.DataFrame(FromSignal(df["DateTime"], df[name], signal_type).get_metrics())
        pd.testing.assert_frame_equal(out, expected, check_dtype=False)
        assert len(out) > 10
        # RR in ms: one beat every 1 / 1.2 s, i.e. 833 ms, up to one sample (5 ms)
        assert np.abs(out[f"{name}_RR"] - 1000 / 1.2).max() <= 5


def test_metric_positions_and_window_plots_are_in_ms(tmp_path, no_latex, monkeypatch):
    t = np.arange(3000) / 200
    from_signal = FromSignal(t * 1000, 90 + 15 * np.sin(2 * np.pi * 1.2 * t), "abp")
    spp_locs, _ = from_signal.find_spp()
    np.testing.assert_array_equal(from_signal.to_time(spp_locs), np.array(spp_locs) * 5)
    assert np.isnan(from_signal.to_time([np.nan, 2])).tolist() == [True, False]
    monkeypatch.setattr(plt, "show", lambda: None)
    from_signal.plot_window(1, output_path=tmp_path / "window.pdf")
    from_signal.plot_pulsation(1, output_path=tmp_path / "pulsation.svg")
    assert (tmp_path / "window.pdf").exists() and (tmp_path / "pulsation.svg").exists()


def test_find_spo_starts_at_the_first_sample_and_keeps_one_location_per_value():
    t = np.arange(600) / 200
    s = 90 + 15 * np.sin(2 * np.pi * 1.2 * t)
    s[598], s[599] = 0, 0  # the lowest values are at the end: a window starting before 0 would wrap around to them
    from_signal = FromSignal(t * 1000, s, "abp")
    from_signal.wd = np.array([5, 170, 337])
    spo_locs, spo = from_signal.find_spo()
    spp_locs, _ = from_signal.find_spp()
    assert len(spo_locs) == len(spo) == 2
    assert 0 <= spo_locs[0] < spp_locs[0] and spo[0] == s[spo_locs[0]]


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
