"""
@author: Radosław Pławecki
Stage `pairs` (costs -> pairs): DTW costs of one metric pair for all subjects of a method and condition, one column
per subject (V1, V2, ...): {method}/{condition}/{pair}.csv, for every pair of [run] pairs.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import StageContext

cols = ["V" + str(i) for i in range(1, 100)]


def run(ctx: StageContext) -> None:
    data_path = ctx.input()
    for method in os.listdir(data_path):
        for breath in os.listdir(data_path / method):
            files = os.listdir(data_path / method / breath)
            dfs = [(os.path.splitext(file)[0], pd.read_csv(data_path / method / breath / file, delimiter=';'))
                   for file in files]
            out_dir = ctx.output(method, breath)
            for metric in ctx.params["pairs"]:
                output_data = []
                for file_name, df in dfs:
                    for col in cols:
                        if f"{col}_{breath}_PP_F_M_PP_DTW" == file_name:
                            output_data.append({col: df[metric]})
                result_df = pd.concat([pd.DataFrame(v) for d in output_data for v in d.values()], axis=1)
                result_df.columns = [k for d in output_data for k in d.keys()]
                result_df.to_csv(out_dir / f"{metric}.csv", index=False, sep=';')
                ctx.processed(f"{method}/{breath}/{metric}")
