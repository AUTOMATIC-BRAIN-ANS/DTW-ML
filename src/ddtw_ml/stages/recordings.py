"""
@author: Radoslaw Plawecki
Stage `recordings` (raw -> recordings): reads the raw recordings of every condition, in one of the three CSV formats
of the measuring equipment (FILE_FORMATS), and writes DateTime, ABP, CBFV = max(FV left, FV right) and BPM per subject:
{condition}/V{i}_{condition}.csv.
The subject number i is the position of the raw file in the sorted directory listing (M-03), so it does not depend
on the file system (os.listdir is alphabetical on NTFS, arbitrary on ext4); the mapping is written to the log.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import Stage
from ddtw_ml.signal.artefacts import filter_fv

FILE_FORMATS = [
    {
        "delimiter": ";",
        "decimal": ",",
        "columns": {
            "datetime": "DateTime",
            "abp": "abp_cnap[mmHg]",
            "fv_l": "fvl",
            "fv_r": "fvr",
            "bpm":  "rr[rpm]"
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
            "bpm": "rr[rr]"
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
            "bpm": "rr[bpm]"
        }
    }
]


def calculate_cbfv(df, col_fvl, col_fvr):
    """
    Function to calculate the CBFV signal based on the FV from left and right side.
    :param df: data in the DataFrame format.
    :param col_fvl: FV from the left side.
    :param col_fvr: FV from the right side.
    :return: CBFV signal.
    """
    fvl, fvr = filter_fv(df, col_fvl), filter_fv(df, col_fvr)
    length = min(len(fvl), len(fvr))
    cbfv = []
    for i in range(0, length):
        if fvl[i] > fvr[i]:
            cbfv.append(fvl[i])
        else:
            cbfv.append(fvr[i])
    return pd.Series(cbfv)


def load_signals(file_path):
    for fmt in FILE_FORMATS:
        try:
            df = pd.read_csv(file_path, delimiter=fmt["delimiter"], decimal=fmt["decimal"])
            cols = fmt["columns"]
            required = cols.values()
            if not all(col in df.columns for col in required):
                continue
            datetime = df[cols["datetime"]]
            abp = df[cols["abp"]]
            cbfv = calculate_cbfv(df, cols["fv_l"], cols["fv_r"])
            bpm = df[cols["bpm"]]
            return datetime, abp, cbfv, bpm
        except Exception:
            continue
    raise ValueError(f"No matching format for file: {file_path}")


class ImportRecordings(Stage):
    def run(self) -> None:
        ctx = self.ctx
        raw = ctx.input()
        for directory in os.listdir(raw):
            directory_path = raw / directory
            if not directory_path.is_dir():
                continue
            out_dir = ctx.output(directory)
            for i, file in enumerate(sorted(os.listdir(directory_path)), start=1):
                file_path = directory_path / file
                try:
                    datetime, abp, cbfv, bpm = load_signals(str(file_path))
                except Exception as e:
                    ctx.reject(file_path, directory, f"{type(e).__name__}: {e}")
                    continue
                new_df = pd.DataFrame({
                    "DateTime": datetime,
                    "ABP": abp,
                    "CBFV": cbfv,
                    "BPM": bpm
                })
                new_df.to_csv(out_dir / f"V{i}_{directory}.csv", sep=";", index=False)
                ctx.processed(f"{directory}/{file} -> V{i}_{directory}.csv")
