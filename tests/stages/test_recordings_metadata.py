"""
Tests of the stage recordings_metadata on synthetic raw recordings run through the stages recordings and cleaned.
"""

import logging
import shutil

import numpy as np
import pandas as pd
import pytest

from ddtw_ml.stages.cleaned import CleanSignals
from ddtw_ml.stages.recordings import ImportRecordings
from ddtw_ml.stages.recordings_metadata import CollectRecordingMetadata, longest_run, raw_id
from tests.stages.test_signal_stages import FS, synthetic_signals


def write_raw(path, s: pd.DataFrame, fv_r_low: int = 0) -> None:
    """
    Raw file in the fourth format (FV columns without brackets); CBFV is the left side, the right one is 1 lower and
    below 20 in its first fv_r_low samples.
    """
    fv_r = s["CBFV"].to_numpy() - 1
    fv_r[:fv_r_low] = 10
    pd.DataFrame({"DateTime": s["DateTime"], "fv_r": fv_r, "fv_l": s["CBFV"], "abp_finger[mm_Hg]": s["ABP"],
                  "etco2[mm_Hg]": s["ETCO2"], "rr[bpm]": s["BPM"]}).to_csv(path, sep=";", decimal=",", index=False)


@pytest.fixture
def metadata(config, make_ctx, caplog) -> pd.DataFrame:
    raw = config.raw_dir()
    (raw / "B6").mkdir(parents=True)
    (raw / "BAS").mkdir()
    good = synthetic_signals(90 * FS)
    good.loc[1000:1099, "ABP"] = np.nan  # a gap of 0.5 s, interpolated
    write_raw(raw / "B6" / "PAC_1.csv", good, fv_r_low=0)
    fast = synthetic_signals(90 * FS, seed=1)
    fast["BPM"] = 10.0  # breathing at 10/min in B6: rejected by cleaned
    write_raw(raw / "B6" / "PAC_2.csv", fast)
    (raw / "B6" / "PAC_3.csv").write_text("unknown;columns\n1;2\n", encoding="utf-8")
    # BAS: a byte copy of B6/PAC_1 and PAC_3 in the place of PAC_2 (V02)
    shutil.copy(raw / "B6" / "PAC_1.csv", raw / "BAS" / "PAC_1.csv")
    write_raw(raw / "BAS" / "PAC_3.csv", synthetic_signals(90 * FS, seed=2), fv_r_low=0)
    ImportRecordings(make_ctx("recordings")).run()
    CleanSignals(make_ctx("cleaned")).run()
    for key in ("filtered", "beats", "paired", "normalised", "aligned"):
        for condition in ("B6", "BAS"):
            (config.stage_dir(key) / condition).mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        CollectRecordingMetadata(make_ctx("recordings_metadata")).run()
    return pd.read_csv(config.data_root / "2.0" / "recordings_metadata.csv", sep=";")


def test_the_metadata_has_a_row_per_raw_file_with_its_subject(metadata):
    assert list(zip(metadata["condition"], metadata["subject"], metadata["raw_file"], metadata["raw_id"])) == [
        ("B6", "V01", "PAC_1.csv", "PAC_1"), ("B6", "V02", "PAC_2.csv", "PAC_2"), ("B6", "V03", "PAC_3.csv", "PAC_3"),
        ("BAS", "V01", "PAC_1.csv", "PAC_1"), ("BAS", "V02", "PAC_3.csv", "PAC_3")]
    assert metadata["raw_sha256"].iloc[0] == metadata["raw_sha256"].iloc[3]


def test_the_metadata_describes_the_recording_and_its_cleaning(metadata):
    row = metadata.iloc[0]
    assert row["raw_format"] == 4 and row["start"] == "2023-03-15T00:00:00"
    assert row["fs_measured"] == pytest.approx(FS, rel=1e-6)
    assert row["samples"] == 90 * FS and row["duration_s"] == 90
    assert row["ABP_missing_pct"] == pytest.approx(100 * 100 / (90 * FS))
    assert row["CBFV_left_pct"] == 100 and row["CBFV_one_side_pct"] == 0
    assert row["BPM_median"] == pytest.approx(6, abs=0.01) and row["usable_breathing_pct"] == 100
    assert row["kept_s"] == 90 and row["trimmed_start_s"] == 0 and row["trimmed_end_s"] == 0
    assert row["ABP_interpolated_pct"] == pytest.approx(100 * 100 / (90 * FS))
    assert row["ABP_longest_gap_s"] == 0.5 and row["CBFV_longest_gap_s"] == 0
    assert row["last_stage"] == "cleaned" and pd.isna(row["rejected_by"]) and pd.isna(row["reason"])


def test_the_metadata_tells_where_and_why_a_recording_was_rejected(metadata):
    fast, unknown = metadata.iloc[1], metadata.iloc[2]
    assert fast["usable_breathing_pct"] == 0
    assert (fast["last_stage"], fast["rejected_by"]) == ("recordings", "cleaned")
    assert "too little usable signal" in fast["reason"]
    assert pd.isna(unknown["last_stage"]) and unknown["rejected_by"] == "recordings"
    assert "No matching format" in unknown["reason"] and np.isnan(unknown["raw_format"])


def test_the_metadata_warns_about_copies_and_subjects_that_differ_between_conditions(metadata, caplog):
    messages = [r.getMessage() for r in caplog.get_records("setup")]
    assert any("identical raw files: B6/PAC_1.csv, BAS/PAC_1.csv" in m for m in messages)
    assert any("V02 is not the same raw subject in all conditions (B6: PAC_2, BAS: PAC_3)" in m for m in messages)


def test_raw_id_and_longest_run():
    assert raw_id("OCH_12_CLEAN_CB_6.csv") == "OCH_12" and raw_id("PAC_48_B.csv") == "PAC_48"
    assert raw_id("other.csv") == "other"
    assert longest_run(np.array([True, False, True, True, False])) == 2
    assert longest_run(np.array([False, False])) == 0
