"""
Tests of the signal stages (recordings, cleaned, filtered) on small synthetic recordings.
"""

import numpy as np
import pandas as pd

from ddtw_ml.stages.cleaned import CleanSignals
from ddtw_ml.stages.cleaned.cleaning_chain import PreprocessData
from ddtw_ml.stages.filtered import RemoveBreathing, remove_freq
from ddtw_ml.stages.recordings import ImportRecordings

FS = 200


def synthetic_signals(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """
    ABP and CBFV with a heart rhythm (1.2 Hz), a breathing rhythm (0.1 Hz) and noise; BPM about 6.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    heart, breathing = np.sin(2 * np.pi * 1.2 * t), np.sin(2 * np.pi * 0.1 * t)
    return pd.DataFrame({
        "DateTime": 45000 + t / 86400,
        "ABP": 90 + 15 * heart + 3 * breathing + rng.normal(0, 0.5, n),
        "CBFV": 60 + 20 * heart + 4 * breathing + rng.normal(0, 0.5, n),
        "BPM": 6 + rng.normal(0, 0.1, n),
    })


def test_import_recordings_numbers_subjects_and_rejects_unknown_formats(config, make_ctx):
    raw = config.raw_dir() / "B6"
    raw.mkdir(parents=True)
    s = synthetic_signals(200)
    pd.DataFrame({"DateTime": s["DateTime"], "fv_r[fv_r]": s["CBFV"], "fv_l[fv_l]": s["CBFV"] + 1,
                  "abp_finger[abp_finger]": s["ABP"], "rr[rr]": s["BPM"]}).to_csv(raw / "PAC_1.csv", index=False)
    (raw / "PAC_2.csv").write_text("unknown;columns\n1;2\n", encoding="utf-8")
    ctx = make_ctx("recordings")
    ImportRecordings(ctx).run()
    out = pd.read_csv(config.stage_dir("recordings") / "B6" / "V1_B6.csv", sep=";")
    assert list(out.columns) == ["DateTime", "ABP", "CBFV", "BPM"]
    # CBFV is the larger of the two flow velocities
    np.testing.assert_allclose(out["CBFV"], s["CBFV"] + 1)
    # the unknown file keeps its number (V2) and is copied to the spoiled data
    assert not (config.stage_dir("recordings") / "B6" / "V2_B6.csv").exists()
    assert (config.spoiled_dir() / "B6" / "PAC_2.csv").exists()
    assert ctx.report.processed == 1 and len(ctx.report.rejected) == 1


def test_clean_signals_writes_the_preprocessed_data(config, make_ctx):
    src = config.stage_dir("recordings") / "B10"
    src.mkdir(parents=True)
    synthetic_signals().to_csv(src / "V1_B10.csv", sep=";", index=False)
    CleanSignals(make_ctx("cleaned")).run()
    out = pd.read_csv(config.stage_dir("cleaned") / "B10" / "V1_B10_PP.csv", sep=";")
    expected = PreprocessData(str(src / "V1_B10.csv"), "ABP", "CBFV", "BPM").get_preprocessed_data()
    pd.testing.assert_frame_equal(out, expected, check_exact=False, rtol=1e-12)


def test_remove_breathing_uses_the_paced_rate(config, make_ctx):
    src = config.stage_dir("cleaned") / "B6"
    src.mkdir(parents=True)
    signals = synthetic_signals()
    signals.to_csv(src / "V1_B6_PP.csv", sep=";", index=False)
    RemoveBreathing(make_ctx("filtered")).run()
    out = pd.read_csv(config.stage_dir("filtered") / "B6" / "V1_B6_PP_F.csv", sep=";")
    assert list(out.columns) == ["DateTime", "ABP", "CBFV"]
    signals = pd.read_csv(src / "V1_B6_PP.csv", sep=";")
    np.testing.assert_allclose(out["ABP"], remove_freq(signals["ABP"], T=200, breaths=6))
