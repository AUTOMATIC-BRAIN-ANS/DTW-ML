"""
@author: Radoslaw Plawecki
Stage `recordings` (raw -> recordings): reads the raw recordings of every condition, in one of the four CSV formats
of the measuring equipment (FILE_FORMATS), and writes DateTime, ABP, CBFV = max(FV left, FV right), BPM and ETCO2
per subject: {condition}/V{i:02d}_{condition}.csv (V01, V02, ..., V10, ...).
The subject number i is the position of the raw file in the sorted directory listing, so it does not depend
on the file system (os.listdir is alphabetical on NTFS, arbitrary on ext4); the mapping is written to the log
(and by the stage recordings_metadata to recordings_metadata.csv, see subject_files()).
The two digits keep the subjects in their numerical order when the file names are sorted.
"""

import os
from typing import Any

import numpy as np
import pandas as pd

from ddtw_ml.pipeline import Stage
from ddtw_ml.signal.artefacts import filter_fv

FILE_FORMATS: list[dict[str, Any]] = [
    {
        "delimiter": ";",
        "decimal": ",",
        "columns": {
            "datetime": "DateTime",
            "abp": "abp_cnap[mmHg]",
            "fv_l": "fvl",
            "fv_r": "fvr",
            "bpm":  "rr[rpm]",
            "etco2": "etco2[mmHg]"
        }
    },
    {
        "delimiter": ",",
        "decimal": ".",
        "columns": {
            "datetime": "DateTime",
            "abp": "abp_finger[abp_finger]",
            "fv_l": "fv_l[fv_l]",
            "fv_r": "fv_r[fv_r]",
            "bpm": "rr[rr]",
            "etco2": "etco2[etco2]"
        }
    },
    {
        "delimiter": ";",
        "decimal": ",",
        "columns": {
            "datetime": "DateTime",
            "abp": "abp_finger[mm_Hg]",
            "fv_l": "fv_l[]",
            "fv_r": "fv_r[]",
            "bpm": "rr[bpm]",
            "etco2": "etco2[mm_Hg]"
        }
    },
    {
        "delimiter": ";",
        "decimal": ",",
        "columns": {
            "datetime": "DateTime",
            "abp": "abp_finger[mm_Hg]",
            "fv_l": "fv_l",
            "fv_r": "fv_r",
            "bpm": "rr[bpm]",
            "etco2": "etco2[mm_Hg]"
        }
    }
]


def calculate_cbfv(df, col_fvl, col_fvr):
    """
    Function to calculate the CBFV signal based on the FV from left and right side: the larger of the two, or the one
    that is not NaN (NaN only if both sides are NaN).
    :param df: data in the DataFrame format.
    :param col_fvl: FV from the left side.
    :param col_fvr: FV from the right side.
    :return: CBFV signal (index of the data).
    """
    fvl, fvr = filter_fv(df, col_fvl), filter_fv(df, col_fvr)
    return pd.Series(np.fmax(fvl.to_numpy(dtype=float), fvr.to_numpy(dtype=float)), index=df.index)


def read_raw(file_path) -> tuple[pd.DataFrame, int]:
    """
    Function to read a raw recording in the first of FILE_FORMATS that has all its columns.
    :param file_path: path of the raw CSV file.
    :return: data and the position of its format in FILE_FORMATS.
    :raise ValueError: if no format matches.
    """
    for i, fmt in enumerate(FILE_FORMATS):
        try:
            df = pd.read_csv(file_path, delimiter=fmt["delimiter"], decimal=fmt["decimal"])
        except Exception:
            continue
        if all(col in df.columns for col in fmt["columns"].values()):
            return df, i
    raise ValueError(f"No matching format for file: {file_path}")


def load_signals(file_path):
    df, i = read_raw(file_path)
    cols = FILE_FORMATS[i]["columns"]
    datetime = df[cols["datetime"]]
    abp = df[cols["abp"]]
    cbfv = calculate_cbfv(df, cols["fv_l"], cols["fv_r"])
    bpm = df[cols["bpm"]]
    etco2 = df[cols["etco2"]]
    return datetime, abp, cbfv, bpm, etco2


def subject_files(directory_path) -> list[tuple[str, str]]:
    """
    Function to number the subjects of a condition: the position of the raw file in the sorted directory listing.
    :param directory_path: directory of the raw recordings of a condition.
    :return: (subject, raw file name) in order: [("V01", "OCH_12_CLEAN_CB_6.csv"), ...].
    """
    return [(f"V{i:02d}", file) for i, file in enumerate(sorted(os.listdir(directory_path)), start=1)]


class ImportRecordings(Stage):
    def run(self) -> None:
        ctx = self.ctx
        raw = ctx.input()
        for directory in os.listdir(raw):
            directory_path = raw / directory
            if not directory_path.is_dir():
                continue
            out_dir = ctx.output(directory)
            for subject, file in subject_files(directory_path):
                file_path = directory_path / file
                try:
                    datetime, abp, cbfv, bpm, etco2 = load_signals(str(file_path))
                except Exception as e:
                    ctx.reject(file_path, directory, f"{type(e).__name__}: {e}")
                    continue
                new_df = pd.DataFrame({
                    "DateTime": datetime,
                    "ABP": abp,
                    "CBFV": cbfv,
                    "BPM": bpm,
                    "ETCO2": etco2
                })
                new_df.to_csv(out_dir / f"{subject}_{directory}.csv", sep=";", index=False)
                ctx.processed(f"{directory}/{file} -> {subject}_{directory}.csv")
