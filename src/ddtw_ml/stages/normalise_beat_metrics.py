"""
@author: Radoslaw Plawecki
Stage `normalised` (paired -> normalised): removes the outliers of the SPO, SPP and RR metrics, interpolates them and
normalises them with the Generalized Logistic function: {condition}/{file}_PP.csv. Files that fail are copied to the
spoiled data.
"""

import os

from ddtw_ml.core.beats.normalisation import PreprocessMetrics
from ddtw_ml.pipeline import StageContext


def run(ctx: StageContext) -> None:
    base_dir = ctx.input()
    for directory in os.listdir(base_dir):
        dir_path = base_dir / directory
        out_dir = ctx.output(directory)
        for file in os.listdir(dir_path):
            file_path = dir_path / file
            try:
                df = PreprocessMetrics(str(file_path)).get_preprocessed_metrics()
            except Exception as e:
                ctx.reject(file_path, directory, f"{type(e).__name__}: {e}")
                continue
            df.to_csv(out_dir / f"{file_path.stem}_PP.csv", sep=';', index=False)
            ctx.processed(f"{directory}/{file}")
