"""
Tests of the signal stages (recordings, cleaned, filtered) on small synthetic recordings.
"""

import numpy as np
import pandas as pd
import pytest

from ddtw_ml.signal.artefacts import filter_abp
from ddtw_ml.stages.cleaned import CleanSignals
from ddtw_ml.stages.cleaned.cleaning_chain import PreprocessData
from ddtw_ml.stages.filtered import RemoveBreathing, remove_freq
from ddtw_ml.stages.recordings import ImportRecordings, calculate_cbfv, load_signals

FS = 200


def synthetic_signals(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """
    ABP and CBFV with a heart rhythm (1.2 Hz), a breathing rhythm (0.1 Hz) and noise; BPM about 6; ETCO2 steps by
    breath (one value every 10 s).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    heart, breathing = np.sin(2 * np.pi * 1.2 * t), np.sin(2 * np.pi * 0.1 * t)
    return pd.DataFrame({
        "DateTime": 45000 + t / 86400,
        "ABP": 90 + 15 * heart + 3 * breathing + rng.normal(0, 0.5, n),
        "CBFV": 60 + 20 * heart + 4 * breathing + rng.normal(0, 0.5, n),
        "BPM": 6 + rng.normal(0, 0.1, n),
        "ETCO2": 35 + (t // 10) % 3,
    })


@pytest.mark.parametrize("fvl, fvr, expected", [
    (50, 60, 60), (60, 50, 60),  # both valid: the larger one
    (50, 10, 50), (10, 50, 50),  # one side below 20: the other one, whichever side it is
    (50, np.nan, 50), (np.nan, 50, 50),  # one side missing
    (np.nan, np.nan, np.nan), (10, 15, np.nan),  # no valid side
])
def test_calculate_cbfv_keeps_the_valid_side(fvl, fvr, expected):
    # an index that is not 0..n-1 (B-01: the values were indexed by label)
    df = pd.DataFrame({"fvl": [fvl, 70.0], "fvr": [fvr, 30.0]}, index=[5, 6])
    cbfv = calculate_cbfv(df, "fvl", "fvr")
    assert list(cbfv.index) == [5, 6]
    np.testing.assert_array_equal(cbfv.to_numpy(), [expected, 70.0])


@pytest.mark.parametrize("abp, expected", [
    ([80, 81, 150, 82], [80, 81, np.nan, 82]),  # spike up
    ([80, 81, 20, 82], [80, 81, np.nan, 82]),  # spike down
    # steep slopes of a beat with a rounded peak
    ([80, 95, 110, 125, 135, 138, 136, 124, 112], [80, 95, 110, 125, 135, 138, 136, 124, 112]),
    ([80, 81, 150, 160, 82], [80, 81, 150, 160, 82]),  # two-sample artefact: no single sample is an impulse
    ([150, 80, 81, 82, 150], [150, 80, 81, 82, 150]),  # the first and last sample are not checked
    ([80, np.nan, 150, 85], [80, np.nan, 150, 85]),  # a missing neighbour: not an impulse
])
def test_filter_abp_removes_impulses_only(abp, expected):
    df = pd.DataFrame({"ABP": abp}, index=range(10, 10 + len(abp)))
    out = filter_abp(df, "ABP")
    assert list(out.index) == list(df.index)
    np.testing.assert_array_equal(out.to_numpy(), expected)


def test_import_recordings_numbers_subjects_and_rejects_unknown_formats(config, make_ctx):
    raw = config.raw_dir() / "B6"
    raw.mkdir(parents=True)
    s = synthetic_signals(200)
    pd.DataFrame({"DateTime": s["DateTime"], "fv_r[fv_r]": s["CBFV"], "fv_l[fv_l]": s["CBFV"] + 1,
                  "abp_finger[abp_finger]": s["ABP"], "rr[rr]": s["BPM"],
                  "etco2[etco2]": s["ETCO2"]}).to_csv(raw / "PAC_1.csv", index=False)
    (raw / "PAC_2.csv").write_text("unknown;columns\n1;2\n", encoding="utf-8")
    ctx = make_ctx("recordings")
    ImportRecordings(ctx).run()
    out = pd.read_csv(config.stage_dir("recordings") / "B6" / "V01_B6.csv", sep=";")
    assert list(out.columns) == ["DateTime", "ABP", "CBFV", "BPM", "ETCO2"]
    np.testing.assert_allclose(out["ETCO2"], s["ETCO2"])
    # CBFV is the larger of the two flow velocities
    np.testing.assert_allclose(out["CBFV"], s["CBFV"] + 1)
    # the unknown file keeps its number (V02) and is copied to the spoiled data
    assert not (config.stage_dir("recordings") / "B6" / "V02_B6.csv").exists()
    assert (config.spoiled_dir() / "B6" / "PAC_2.csv").exists()
    assert ctx.report.processed == 1 and len(ctx.report.rejected) == 1


@pytest.mark.parametrize("fv_l, fv_r", [("fv_l[]", "fv_r[]"), ("fv_l", "fv_r")])
def test_load_signals_reads_the_mm_hg_format_with_and_without_brackets(tmp_path, fv_l, fv_r):
    s = synthetic_signals(200)
    pd.DataFrame({"DateTime": s["DateTime"], fv_r: s["CBFV"], fv_l: s["CBFV"] + 1, "abp_finger[mm_Hg]": s["ABP"],
                  "etco2[mm_Hg]": s["ETCO2"], "rr[bpm]": s["BPM"]}).to_csv(
        tmp_path / "PAC_1.csv", sep=";", decimal=",", index=False)
    datetime, abp, cbfv, bpm, etco2 = load_signals(str(tmp_path / "PAC_1.csv"))
    np.testing.assert_allclose(abp, s["ABP"])
    np.testing.assert_allclose(cbfv, s["CBFV"] + 1)
    np.testing.assert_allclose(bpm, s["BPM"])


def test_clean_signals_writes_the_preprocessed_data(config, make_ctx):
    src = config.stage_dir("recordings") / "B10"
    src.mkdir(parents=True)
    synthetic_signals().to_csv(src / "V1_B10.csv", sep=";", index=False)
    CleanSignals(make_ctx("cleaned")).run()
    out = pd.read_csv(config.stage_dir("cleaned") / "B10" / "V1_B10_PP.csv", sep=";")
    expected = PreprocessData(str(src / "V1_B10.csv"), "ABP", "CBFV", "BPM", "ETCO2").get_preprocessed_data()
    pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-12)
    assert list(out.columns) == ["DateTime", "ABP", "CBFV", "BPM", "ETCO2"]


def test_clean_signals_trims_bpm_and_etco2_like_abp_and_cbfv(tmp_path):
    signals = synthetic_signals()
    # a gap of 700 samples at the start of ABP: trimmed, the cleaned signals start at sample 700
    signals.loc[:699, "ABP"] = np.nan
    signals["BPM"] = np.arange(len(signals), dtype=float)
    signals.loc[3000, "ETCO2"] = np.nan
    signals.to_csv(tmp_path / "V1_B10.csv", sep=";", index=False)
    prep = PreprocessData(str(tmp_path / "V1_B10.csv"), "ABP", "CBFV", "BPM", "ETCO2")
    assert prep.get_trim_limits() == (700, len(signals))
    out = prep.get_preprocessed_data()
    # ABP and CBFV set the length; NaN in ETCO2 does not shorten the data
    assert len(out) == len(signals) - 700
    np.testing.assert_array_equal(out["BPM"], np.arange(700, len(signals)))
    np.testing.assert_array_equal(out["ETCO2"], signals["ETCO2"].iloc[700:])


@pytest.mark.parametrize("first, second", [("ABP", "CBFV"), ("CBFV", "ABP")])
def test_preprocess_data_filters_the_abp_column_in_either_position(tmp_path, first, second):
    signals = synthetic_signals(100)
    signals.loc[50, "ABP"] = 300  # impulse artefact
    signals.to_csv(tmp_path / "V1_B10.csv", sep=";", index=False)
    prep = PreprocessData(str(tmp_path / "V1_B10.csv"), first, second, "BPM")
    abp, cbfv = (prep.get_first_signal(), prep.get_second_signal())[::1 if first == "ABP" else -1]
    assert np.isnan(abp.iloc[50]) and abp.drop(abp.index[50]).notna().all()
    np.testing.assert_allclose(cbfv, signals["CBFV"], rtol=1e-12)


def test_remove_breathing_uses_the_paced_rate(config, make_ctx):
    src = config.stage_dir("cleaned") / "B6"
    src.mkdir(parents=True)
    signals = synthetic_signals()
    signals.to_csv(src / "V1_B6_PP.csv", sep=";", index=False)
    RemoveBreathing(make_ctx("filtered")).run()
    out = pd.read_csv(config.stage_dir("filtered") / "B6" / "V1_B6_PP_F.csv", sep=";")
    assert list(out.columns) == ["DateTime", "ABP", "CBFV", "ETCO2"]
    signals = pd.read_csv(src / "V1_B6_PP.csv", sep=";")
    np.testing.assert_allclose(out["ABP"], remove_freq(signals["ABP"], T=200, breaths=6))
    # time of each sample in ms at 200 Hz; ETCO2 unchanged
    np.testing.assert_array_equal(out["DateTime"], np.arange(len(signals)) * 5)
    np.testing.assert_array_equal(out["ETCO2"], signals["ETCO2"])


def test_cleaning_plot_data_shows_the_steps_of_the_chosen_signal(tmp_path):
    synthetic_signals(1000).to_csv(tmp_path / "V1_B10.csv", sep=";", index=False)
    prep = PreprocessData(str(tmp_path / "V1_B10.csv"), "ABP", "CBFV", "BPM")
    for s, normalised in (("first", prep.get_first_signal_normalized()),
                          ("second", prep.get_second_signal_normalized())):
        _, signals, _ = prep.get_data_for_plot(s)
        np.testing.assert_array_equal(signals[3], normalised)
