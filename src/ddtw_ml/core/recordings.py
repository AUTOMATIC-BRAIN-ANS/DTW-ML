"""
@author: Radoslaw Plawecki
Reading of the raw recordings: three CSV formats of the measuring equipment.
"""

import pandas as pd

from ddtw_ml.core.signal.cbfv import calculate_cbfv

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
