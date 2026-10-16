"""
@author: Radosław Pławecki
Stage `mean_costs` (pairs -> mean_costs): mean DTW cost of every subject (rows) for every method and condition
(columns {method}_{condition}): {pair}.csv, for every pair of [analysis] pairs. The rows are all subjects of any
method and condition (NaN where a subject is missing), in the order they first appear.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import Stage


class AverageCostsPerSubject(Stage):
    def run(self) -> None:
        ctx = self.ctx
        data_path = ctx.input()
        out_dir = ctx.output()
        for metric in ctx.analysis["pairs"]:
            columns = []
            for breath in os.listdir(data_path):
                for method in os.listdir(data_path / breath):
                    final_path = data_path / breath / method / f"{metric}.csv"
                    df = pd.read_csv(final_path, delimiter=';')
                    columns.append(df.mean().rename(f"{breath}_{method}"))
            all_means = pd.concat(columns, axis=1, sort=False)
            all_means.to_csv(out_dir / f"{metric}.csv", sep=';')
            ctx.processed(metric)
