"""
Stage `recordings_metadata` (raw, recordings ... aligned -> recordings_metadata.csv): describes every raw recording in one
row, for the methods of the work and to check the data: which raw file is which subject (V01 ...), its checksum and start
time, its length, how much of it was removed or interpolated by the cleaning, how well the paced breathing was kept, the
number of beats, the signal-to-noise ratio of the beats and the last stage the recording reached (with the reason of
its rejection).

The statistics of the cleaning are computed again with the code of the stage cleaned (PreprocessData), on every
recording, also on the ones that the breathing criterion rejects; the results of the pipeline do not change.
The output is one file in the directory of the version, outside the stage directories.
"""

import csv
import hashlib
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ddtw_ml.pipeline import Stage
from ddtw_ml.signal.artefacts import filter_fv
from ddtw_ml.stages.beats import FromSignal
from ddtw_ml.stages.cleaned import PreprocessData, count_usable
from ddtw_ml.stages.recordings import FILE_FORMATS, read_raw, subject_files

# stages a recording goes through and the name of its file (directory for beats) in each: (stage, suffix)
STAGE_FILES = [("recordings", ".csv"), ("cleaned", "_PP.csv"), ("filtered", "_PP_F.csv"), ("beats", "_PP_F"),
               ("paired", "_PP_F_M.csv"), ("normalised", "_PP_F_M_PP.csv"), ("aligned", "_PP_F_M_PP.csv")]
# start of the Excel serial dates (DateTime of the raw recordings)
EXCEL_EPOCH = datetime(1899, 12, 30)
SECONDS_PER_DAY = 24 * 60 * 60
RAW_ID = re.compile(r"^([A-Za-z]+_\d+)")
# columns of recordings_metadata.csv, in order (percentages 0-100, times in seconds)
COLUMNS = ["condition", "subject", "raw_file", "raw_id", "raw_sha256", "raw_format", "start", "fs_measured", "samples",
           "duration_s", "ABP_missing_pct", "CBFV_missing_pct", "BPM_missing_pct", "ETCO2_missing_pct",
           "CBFV_left_pct", "CBFV_one_side_pct", "BPM_median", "BPM_iqr", "usable_breathing_s", "usable_breathing_pct",
           "ABP_impulses", "ABP_outliers_pct", "CBFV_outliers_pct", "trimmed_start_s", "trimmed_end_s", "kept_s",
           "ABP_interpolated_pct", "ABP_longest_gap_s", "CBFV_interpolated_pct", "CBFV_longest_gap_s", "ABP_beats",
           "CBFV_beats", "paired_beats", "ABP_SNR_dB", "CBFV_SNR_dB", "last_stage", "rejected_by", "reason"]
# number of points every beat is resampled to before the SNR is computed
SNR_POINTS = 100


def raw_id(file: str) -> str:
    """
    Function to get the subject of a raw file from its name (OCH_12_CLEAN_CB_6.csv -> OCH_12).
    :param file: name of the raw file.
    :return: subject (the name without the extension if it does not start with letters, "_" and a number).
    """
    match = RAW_ID.match(file)
    return match.group(1) if match else Path(file).stem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longest_run(mask: np.ndarray) -> int:
    """
    Function to get the length of the longest run of True values.
    :param mask: boolean array.
    :return: number of samples of the longest run (0 if there is none).
    """
    padded = np.concatenate(([False], mask, [False])).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[::2]).max()) if len(edges) else 0


def percent(count: float, total: float) -> float:
    return 100 * count / total if total else np.nan


def raw_stats(path: Path) -> dict:
    """
    Function to describe a raw recording: its format and from which side the CBFV is taken.
    :param path: path of the raw file.
    :return: raw_format (number in FILE_FORMATS, from 1), CBFV_left_pct (valid CBFV samples from the left FV) and
        CBFV_one_side_pct (valid CBFV samples with only one valid side).
    """
    df, i = read_raw(path)
    cols = FILE_FORMATS[i]["columns"]
    fvl, fvr = filter_fv(df, cols["fv_l"]).to_numpy(), filter_fv(df, cols["fv_r"]).to_numpy()
    valid_l, valid_r = ~np.isnan(fvl), ~np.isnan(fvr)
    valid = int((valid_l | valid_r).sum())
    left = valid_l & (~valid_r | (fvl >= fvr))
    return {"raw_format": i + 1, "CBFV_left_pct": percent(left.sum(), valid),
            "CBFV_one_side_pct": percent((valid_l ^ valid_r).sum(), valid)}


def recording_stats(path: Path, fs: float, breaths: float | None, tolerance: float) -> dict:
    """
    Function to describe a recording (stage recordings): its start, length, missing data, breathing rate and what the
    cleaning removes and interpolates.
    :param path: path of the recording.
    :param fs: sampling frequency [Hz].
    :param breaths: paced breathing rate of the condition [breaths/min] (None: not paced).
    :param tolerance: allowed difference from the paced rate [breaths/min] (as in the stage cleaned).
    :return: statistics by column name.
    """
    df = pd.read_csv(path, delimiter=";")
    n = len(df)
    time = df["DateTime"].to_numpy(dtype=float)
    usable = count_usable(df["BPM"], breaths, tolerance)
    stats = {
        "start": (EXCEL_EPOCH + timedelta(days=float(time[0]))).isoformat(timespec="seconds") if n else "",
        # from the whole recording: the steps of DateTime are rounded (10 decimals of a day in the raw files)
        "fs_measured": (n - 1) / ((time[-1] - time[0]) * SECONDS_PER_DAY) if n > 1 else np.nan,
        "samples": n,
        "duration_s": n / fs,
        "BPM_median": np.nanmedian(df["BPM"]) if n else np.nan,
        "BPM_iqr": np.subtract(*np.nanpercentile(df["BPM"], [75, 25])) if n else np.nan,
        "usable_breathing_s": usable / fs,
        "usable_breathing_pct": percent(usable, n),
    }
    for col in ("ABP", "CBFV", "BPM", "ETCO2"):
        stats[f"{col}_missing_pct"] = percent(df[col].isna().sum(), n)
    prep = PreprocessData(str(path), first_column="ABP", second_column="CBFV", third_column="BPM",
                          fourth_column="ETCO2")
    after_artefacts = prep.get_first_signal().isna().sum()
    abp, cbfv = prep.remove_outliers()
    start, stop = prep.get_trim_limits()
    stats.update({
        "ABP_impulses": int(after_artefacts - df["ABP"].isna().sum()),
        "ABP_outliers_pct": percent(np.isnan(abp).sum() - after_artefacts, n),
        "CBFV_outliers_pct": percent(np.isnan(cbfv).sum() - df["CBFV"].isna().sum(), n),
        "trimmed_start_s": start / fs,
        "trimmed_end_s": (n - stop) / fs,
        "kept_s": (stop - start) / fs,
    })
    for col, signal in (("ABP", abp), ("CBFV", cbfv)):
        gaps = np.isnan(signal[start:stop])
        stats[f"{col}_interpolated_pct"] = percent(gaps.sum(), stop - start)
        stats[f"{col}_longest_gap_s"] = longest_run(gaps) / fs
    return stats


def beat_snr(signal: np.ndarray, onsets: list, points: int = SNR_POINTS) -> float:
    """
    Function to compute the signal-to-noise ratio of the beats of a recording: every beat (from its SPO to the next SPO)
    is resampled to the given number of points and its mean is subtracted; the template is the pointwise median of the
    beats. Signal = variance of the template, noise = mean variance of the residuals (beat minus template).
    :param signal: signal (ABP or CBFV).
    :param onsets: positions of the systolic phase onsets (SPO; NaN: not found, skipped).
    :param points: number of points of a resampled beat.
    :return: SNR [dB] (NaN if there are fewer than 2 beats).
    """
    locs = [int(loc) for loc in onsets if not np.isnan(loc)]
    grid = np.linspace(0, 1, points)
    beats = []
    for start, end in zip(locs[:-1], locs[1:]):
        if end - start < 2:
            continue
        beat = signal[start:end + 1]
        beat = np.interp(grid, np.linspace(0, 1, len(beat)), beat)
        beats.append(beat - beat.mean())
    if len(beats) < 2:
        return np.nan
    beats_array = np.array(beats)
    template = np.median(beats_array, axis=0)
    noise = np.mean(np.var(beats_array - template, axis=1))
    return float(10 * np.log10(np.var(template) / noise))


def filtered_snr(path: Path) -> dict:
    """
    Function to compute the SNR of the beats of ABP and CBFV of a recording of the stage filtered, with the beats
    detected as in the stage beats (FromSignal).
    :param path: path of the filtered recording.
    :return: ABP_SNR_dB and CBFV_SNR_dB.
    """
    df = pd.read_csv(path, delimiter=";")
    stats = {}
    for col in ("ABP", "CBFV"):
        beats = FromSignal(df["DateTime"], signal=df[col], signal_type=col)
        onsets, _ = beats.find_spo()
        stats[f"{col}_SNR_dB"] = beat_snr(beats.signal, onsets)
    return stats


def read_failures(spoiled: Path) -> dict[str, str]:
    """
    Function to read the reasons of the rejections recorded in the spoiled data (_failures.csv in every directory).
    :param spoiled: directory of the spoiled data of the version.
    :return: normalised path of the rejected input -> reason of its last rejection.
    """
    reasons: dict[str, str] = {}
    if not spoiled.is_dir():
        return reasons
    for log_path in sorted(spoiled.rglob("_failures.csv")):
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                reasons[os.path.normcase(os.path.normpath(row["Source"]))] = row["Reason"]
    return reasons


class CollectRecordingMetadata(Stage):
    PARAMS = {"fs": int, "paced_breaths": dict, "breath_tolerance": float}

    def run(self) -> None:
        ctx = self.ctx
        fs, paced_breaths = ctx.params["fs"], ctx.params["paced_breaths"]
        tolerance = ctx.params["breath_tolerance"]
        raw = ctx.input("raw")
        failures = read_failures(ctx.config.spoiled_dir(ctx.version))
        rows = []
        for condition in sorted(d for d in os.listdir(raw) if (raw / d).is_dir()):
            for subject, file in subject_files(raw / condition):
                name = f"{subject}_{condition}"
                row: dict = {"condition": condition, "subject": subject, "raw_file": file, "raw_id": raw_id(file),
                             "raw_sha256": sha256(raw / condition / file)}
                try:
                    row.update(raw_stats(raw / condition / file))
                except Exception as e:
                    ctx.log.warning(f"{condition}/{file}: raw data not described ({type(e).__name__}: {e})")
                recording = ctx.input("recordings") / condition / f"{name}.csv"
                if recording.is_file():
                    try:
                        row.update(recording_stats(recording, fs, paced_breaths.get(condition), tolerance))
                    except Exception as e:
                        ctx.log.warning(f"{recording.name}: recording not described ({type(e).__name__}: {e})")
                for key in ("ABP", "CBFV"):
                    beats = ctx.input("beats") / condition / f"{name}_PP_F" / f"{key}.csv"
                    row[f"{key}_beats"] = len(pd.read_csv(beats, delimiter=";")) if beats.is_file() else np.nan
                paired = ctx.input("paired") / condition / f"{name}_PP_F_M.csv"
                row["paired_beats"] = len(pd.read_csv(paired, delimiter=";")) if paired.is_file() else np.nan
                filtered = ctx.input("filtered") / condition / f"{name}_PP_F.csv"
                if filtered.is_file():
                    try:
                        row.update(filtered_snr(filtered))
                    except Exception as e:
                        ctx.log.warning(f"{filtered.name}: SNR not computed ({type(e).__name__}: {e})")
                row.update(self.fate(condition, file, name, failures))
                rows.append(row)
                ctx.processed(f"{condition}/{file} -> {subject} ({row['last_stage'] or 'not read'})")
        self.warn_duplicates(rows)
        pd.DataFrame(rows, columns=COLUMNS).to_csv(ctx.output_file(), sep=";", index=False)

    def fate(self, condition: str, file: str, name: str, failures: dict[str, str]) -> dict:
        """
        Method to find the last stage a recording reached and why it did not go further.
        :param condition: condition of the recording.
        :param file: name of the raw file.
        :param name: name of the recording ({subject}_{condition}).
        :param failures: reasons of the rejections (see read_failures()).
        :return: last_stage ("" if the raw file was not read), rejected_by (the next stage, "" if the recording reached
            the last one or the next stage did not record a reason) and reason.
        """
        ctx = self.ctx
        last, source = "", ctx.input("raw") / condition / file
        for i, (stage, suffix) in enumerate(STAGE_FILES):
            path = ctx.input(stage) / condition / f"{name}{suffix}"
            if not path.exists():
                reason = failures.get(os.path.normcase(os.path.normpath(source)), "")
                return {"last_stage": last, "rejected_by": stage if reason else "", "reason": reason}
            last, source = stage, path
        return {"last_stage": last, "rejected_by": "", "reason": ""}

    def warn_duplicates(self, rows: list[dict]) -> None:
        """
        Method to warn about raw files that are byte copies of each other and about subjects (V01 ...) that are
        different raw subjects in different conditions (M-03: the subjects are numbered by the position of the file).
        :param rows: rows of the metadata.
        """
        log = self.ctx.log
        df = pd.DataFrame(rows)
        for digest, group in df.groupby("raw_sha256"):
            if len(group) > 1:
                files = ", ".join(f"{c}/{f}" for c, f in zip(group["condition"], group["raw_file"]))
                log.warning(f"identical raw files: {files}")
        for subject, group in df.groupby("subject"):
            if group["raw_id"].nunique() > 1:
                ids = ", ".join(f"{c}: {i}" for c, i in zip(group["condition"], group["raw_id"]))
                log.warning(f"{subject} is not the same raw subject in all conditions ({ids})")
