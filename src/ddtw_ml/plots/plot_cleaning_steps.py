"""
Plot `cleaning` (recordings -> plots): ABP and CBFV of every recording at the steps of cleaning (raw, outliers removed,
interpolated, normalised): {condition}/{file}_{ABP,CBFV}.pdf.
"""

import os

from ddtw_ml.core.signal.cleaning_chain import PreprocessData
from ddtw_ml.pipeline import StageContext


def run(ctx: StageContext) -> None:
    data_path = ctx.input()
    for directory in os.listdir(data_path):
        out_dir = ctx.output(directory)
        for file in os.listdir(data_path / directory):
            file_path = data_path / directory / file
            prep = PreprocessData(str(file_path), first_column='ABP', second_column='CBFV', third_column='BPM')
            for s, name in (("first", "ABP"), ("second", "CBFV")):
                prep.plot_signals(s, output_path=out_dir / f"{file_path.stem}_{name}.pdf", show=False)
            ctx.processed(f"{directory}/{file}")
