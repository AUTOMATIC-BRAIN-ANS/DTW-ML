"""
@author: Radosław Pławecki
Stage `tables` (paired -> tables): medians of the beat-to-beat metrics per subject, for the subjects present in all
conditions:
- metrics_for_anova.csv -- one row per subject, columns {condition}_{ABP,CBFV}_{metric} (input of `metric_anova`),
- medians_iqr.csv -- per condition, median and IQR of these medians over the subjects as LaTeX strings.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import Stage


def subject_id(file: str) -> str:
    return file.split('_')[0].split('.')[0]


class TabulateMedians(Stage):
    def run(self) -> None:
        ctx = self.ctx
        input_path = ctx.input()
        # baseline first, then the paced conditions in the order of the config
        conditions = sorted(ctx.analysis["conditions"], key=lambda c: c != "BAS")
        columns = [f"{signal}_{metric}" for metric in ctx.analysis["metrics"] for signal in ("ABP", "CBFV")]
        # sorted: the rows of the conditions are joined by position (M-07), os.listdir order depends on the file system
        files = {c: sorted(os.listdir(input_path / c)) for c in conditions}
        allowed_ids = set.intersection(*({subject_id(f) for f in fs} for fs in files.values()))
        per_condition = []
        for condition in conditions:
            medians = []
            for file in files[condition]:
                if subject_id(file) not in allowed_ids:
                    continue
                df = pd.read_csv(input_path / condition / file, delimiter=';')
                medians.append({col: df[col].median() for col in columns})
            per_condition.append(pd.DataFrame(medians).add_prefix(f"{condition}_"))
        merged = pd.concat(per_condition, axis=1)
        out_dir = ctx.output()
        merged.to_csv(out_dir / "metrics_for_anova.csv", sep=';')
        ctx.processed(f"metrics_for_anova.csv ({len(merged)} subjects)")

        rows = {}
        for condition, df in zip(conditions, per_condition):
            df.columns = columns
            median, q1, q3 = df.median(), df.quantile(0.25), df.quantile(0.75)
            rows[condition] = ("$" + median.round(3).astype(str) + "$ $(" + q1.round(3).astype(str) + r"\text{-}"
                               + q3.round(3).astype(str) + ")$")
        pd.DataFrame(rows).T.to_csv(out_dir / "medians_iqr.csv", sep=';')
        ctx.processed("medians_iqr.csv")
