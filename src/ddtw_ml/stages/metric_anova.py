"""
@author: Radosław Pławecki
Stage `metric_anova` (tables -> metric_anova): RM-ANOVA of the per-subject medians of every metric of [analysis]
metrics between the conditions, for ABP and CBFV: {signal}_{metric}_anova.csv; if p < alpha (settings.toml), also the
paired t-tests with the Bonferroni correction: {signal}_{metric}_posthoc.csv.
"""

import itertools

import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from ddtw_ml.pipeline import Stage


def post_hoc_tests(df_wide, conditions):
    pairs = list(itertools.combinations(conditions, 2))
    pvals = []
    results = []

    for c1, c2 in pairs:
        stat, p = ttest_rel(df_wide[c1], df_wide[c2])
        pvals.append(p)
        results.append((c1, c2, stat, p))

    # Bonferroni correction
    reject, pvals_corr, _, _ = multipletests(pvals, method="bonferroni")
    rows = [(c1, c2, stat, p, p_corr, r) for (c1, c2, stat, p), p_corr, r in zip(results, pvals_corr, reject)]
    return pd.DataFrame(rows, columns=["A", "B", "T", "p", "p_corr", "significant"])


class CompareMetricMedians(Stage):
    PARAMS = {"alpha": float}

    def run(self) -> None:
        ctx = self.ctx
        df = pd.read_csv(ctx.input() / "metrics_for_anova.csv", sep=";")
        df.columns = df.columns.str.strip()
        df["Subject"] = range(1, len(df) + 1)
        out_dir = ctx.output()
        # baseline first, then the paced conditions in the order of the config
        conditions = sorted(ctx.analysis["conditions"], key=lambda c: c != "BAS")
        for metric in ctx.analysis["metrics"]:
            for signal in ("ABP", "CBFV"):
                cols = [f"{condition}_{signal}_{metric}" for condition in conditions]
                long = pd.melt(df, id_vars=["Subject"], value_vars=cols, var_name="Breath", value_name=signal)
                long["Breath"] = long["Breath"].str.replace(f"_{signal}", "")
                anova = AnovaRM(long, signal, "Subject", within=["Breath"]).fit()
                anova.anova_table.to_csv(out_dir / f"{signal}_{metric}_anova.csv", sep=";")
                p = anova.anova_table["Pr > F"].iloc[0]
                if p < ctx.params["alpha"]:
                    post_hoc_tests(df[cols], cols).to_csv(out_dir / f"{signal}_{metric}_posthoc.csv", sep=";",
                                                          index=False)
                ctx.processed(f"{signal}_{metric} (p = {p:.3g})")
