"""
Stage `recordings` (raw -> recordings): reads the raw recordings of every condition and writes DateTime, ABP,
CBFV = max(FV left, FV right) and BPM per subject: {condition}/V{i}_{condition}.csv.
The subject number i is the position of the raw file in os.listdir (M-03); the mapping is written to the log.
"""

import os

import pandas as pd

from ddtw_ml.core.recordings import load_signals
from ddtw_ml.pipeline import StageContext


def run(ctx: StageContext) -> None:
    raw = ctx.input()
    for directory in os.listdir(raw):
        directory_path = raw / directory
        if not directory_path.is_dir():
            continue
        out_dir = ctx.output(directory)
        for i, file in enumerate(os.listdir(directory_path), start=1):
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
