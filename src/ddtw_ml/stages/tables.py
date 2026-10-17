"""
@author: Radosław Pławecki
Stage `tables` (paired, filtered, aligned -> tables): medians of the beat-to-beat metrics and the mean EtCO2 (filtered
signals, NaN skipped) per subject, for the subjects kept by `aligned` (the same as in the DTW costs):
- metrics_for_anova.csv -- one row per subject (index Subject), columns {condition}_{ABP,CBFV}_{metric} (input of
  `metric_anova`) and {condition}_ETCO2,
- medians_iqr.csv -- per condition, median and IQR of these values over the subjects as LaTeX strings.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import Stage


def subject_id(file: str) -> str:
    return file.split('_')[0].split('.')[0]


class TabulateMedians(Stage):
    def run(self) -> None:
        ctx = self.ctx
        input_path, signals_path = ctx.input("paired"), ctx.input("filtered")
        # baseline first, then the paced conditions in the order of the config
        conditions = sorted(ctx.analysis["conditions"], key=lambda c: c != "BAS")
        columns = [f"{signal}_{metric}" for metric in ctx.analysis["metrics"] for signal in ("ABP", "CBFV")]
        # the subjects of the DTW costs: those kept by `aligned` in every condition
        aligned_path = ctx.input("aligned")
        subjects = sorted(set.intersection(*({subject_id(f) for f in os.listdir(aligned_path / c)}
                                             for c in conditions)))
        per_condition = []
        for condition in conditions:
            files = {subject_id(f): f for f in os.listdir(input_path / condition)}
            medians = {}
            for subject in subjects:
                df = pd.read_csv(input_path / condition / files[subject], delimiter=';')
                signals = pd.read_csv(signals_path / condition / f"{subject}_{condition}_PP_F.csv", delimiter=';')
                medians[subject] = {**{col: df[col].median() for col in columns}, "ETCO2": signals["ETCO2"].mean()}
            per_condition.append(pd.DataFrame.from_dict(medians, orient="index").add_prefix(f"{condition}_"))
        # the conditions are joined by the subject
        merged = pd.concat(per_condition, axis=1)
        merged.index.name = "Subject"
        out_dir = ctx.output()
        merged.to_csv(out_dir / "metrics_for_anova.csv", sep=';')
        ctx.processed(f"metrics_for_anova.csv ({len(merged)} subjects)")

        rows = {}
        for condition, df in zip(conditions, per_condition):
            df.columns = columns + ["ETCO2"]
            median, q1, q3 = df.median(), df.quantile(0.25), df.quantile(0.75)
            rows[condition] = ("$" + median.round(3).astype(str) + "$ $(" + q1.round(3).astype(str) + r"\text{-}"
                               + q3.round(3).astype(str) + ")$")
        pd.DataFrame(rows).T.to_csv(out_dir / "medians_iqr.csv", sep=';')
        ctx.processed("medians_iqr.csv")
