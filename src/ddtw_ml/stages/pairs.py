"""
@author: Radosław Pławecki
Stage `pairs` (costs -> pairs): DTW costs of one metric pair for all subjects of a method and condition, one column
per subject (V1, V2, ...): {method}/{condition}/{pair}.csv, for every pair of [analysis] pairs.
"""

import os
import re

import pandas as pd

from ddtw_ml.pipeline import Stage


class GroupCostsByPair(Stage):
    def run(self) -> None:
        ctx = self.ctx
        data_path = ctx.input()
        for method in os.listdir(data_path):
            for breath in os.listdir(data_path / method):
                # the subject (V1, V2, ...) is taken from the file name; files of other conditions are skipped
                pattern = re.compile(rf"(V\d+)_{re.escape(breath)}_PP_F_M_PP_DTW")
                subjects = []
                for file in sorted(os.listdir(data_path / method / breath)):
                    match = pattern.fullmatch(os.path.splitext(file)[0])
                    if match:
                        subjects.append((match.group(1), pd.read_csv(data_path / method / breath / file,
                                                                     delimiter=';')))
                if not subjects:
                    ctx.log.warning(f"{method}/{breath}: no cost files of this condition - skipped")
                    continue
                out_dir = ctx.output(method, breath)
                for metric in ctx.analysis["pairs"]:
                    result_df = pd.concat([df[metric].rename(subject) for subject, df in subjects], axis=1)
                    result_df.to_csv(out_dir / f"{metric}.csv", index=False, sep=';')
                    ctx.processed(f"{method}/{breath}/{metric}")
