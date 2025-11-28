"""
@author: Radoslaw Plawecki
"""

import pandas as pd
from project.common import calculate_cbfv
import os

FILE_FORMATS = [
    {
        "delimiter": ";",
        "decimal": ",",
        "columns": {
            "datetime": "DateTime",
            "abp": "abp_cnap[mmHg]",
            "fv_l": "fvl",
            "fv_r": "fvr"
        }
    },
    {
        "delimiter": ",",
        "decimal": ".",
        "columns": {
            "datetime": "DateTime",
            "abp": "abp_finger[abp_finger]",
            "fv_l": "fv_l[fv_l]",
            "fv_r": "fv_r[fv_r]"
        }
    },
    {
        "delimiter": ";",
        "decimal": ",",
        "columns": {
            "datetime": "DateTime",
            "abp": "abp_finger[mm_Hg]",
            "fv_l": "fv_l[]",
            "fv_r": "fv_r[]"
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
            return datetime, abp, cbfv
        except Exception:
            continue
    raise ValueError(f"No matching format for file: {file_path}")


def extract_signals():
    print("=== Starting extraction ===")
    base = "C:/Python/ZSSI/data/"
    raw = os.path.join(base, "raw")
    output_base = os.path.join(base, "extracted")
    for directory in os.listdir(raw):
        directory_path = os.path.join(raw, directory)
        if not os.path.isdir(directory_path):
            continue
        print(f"\n=== Directory: {directory} ===")
        out_dir = os.path.join(output_base, directory)
        os.makedirs(out_dir, exist_ok=True)
        for i, file in enumerate(os.listdir(directory_path), start=1):
            file_path = os.path.join(directory_path, file)
            print(f"-> Processing: {file}")
            try:
                datetime, abp, cbfv = load_signals(file_path)
                new_df = pd.DataFrame({
                    "DateTime": datetime,
                    "ABP": abp,
                    "CBFV": cbfv
                })
                out_file = os.path.join(out_dir, f"V{i}_{directory}.csv")
                new_df.to_csv(out_file, sep=";", index=False)
                print("✔ Success")
            except Exception as e:
                print(f"✖ Failed ({e})")
    print("\n=== Extraction completed ===")


extract_signals()
