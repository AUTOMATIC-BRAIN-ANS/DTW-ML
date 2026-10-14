"""
@author: Radosław Pławecki
Stage `mean_costs` (pairs -> mean_costs): mean DTW cost of every subject (rows) for every method and condition
(columns {method}_{condition}): {pair}.csv, for every pair of [run] pairs.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import StageContext


def run(ctx: StageContext) -> None:
    data_path = ctx.input()
    out_dir = ctx.output()
    for metric in ctx.params["pairs"]:
        all_means = pd.DataFrame()
        for breath in os.listdir(data_path):
            for method in os.listdir(data_path / breath):
                final_path = data_path / breath / method / f"{metric}.csv"
                data = pd.read_csv(final_path, delimiter=';')
                df = pd.DataFrame(data)
                mean_series = df.mean()
                all_means[f"{breath}_{method}"] = mean_series
        all_means.to_csv(out_dir / f"{metric}.csv", sep=';')
        ctx.processed(metric)
