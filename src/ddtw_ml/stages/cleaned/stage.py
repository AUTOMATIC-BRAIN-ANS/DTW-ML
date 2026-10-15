"""
Stage `cleaned` (recordings -> cleaned): removes the ABP artefacts and the outliers of ABP and CBFV, trims the gaps,
interpolates and smooths the signals: {condition}/{file}_PP.csv. Files that fail are copied to the spoiled data.
"""

import os

from ddtw_ml.pipeline import Stage
from ddtw_ml.stages.cleaned.cleaning_chain import PreprocessData


class CleanSignals(Stage):
    def run(self) -> None:
        ctx = self.ctx
        base_dir = ctx.input()
        for directory in os.listdir(base_dir):
            dir_path = base_dir / directory
            out_dir = ctx.output(directory)
            for file in os.listdir(dir_path):
                file_path = dir_path / file
                try:
                    prep = PreprocessData(str(file_path), first_column='ABP', second_column='CBFV', third_column='BPM')
                    df = prep.get_preprocessed_data()
                except Exception as e:
                    ctx.reject(file_path, directory, f"{type(e).__name__}: {e}")
                    continue
                df.to_csv(out_dir / f"{file_path.stem}_PP.csv", sep=';', index=False)
                ctx.processed(f"{directory}/{file}")
