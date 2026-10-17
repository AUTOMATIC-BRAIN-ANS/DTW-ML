"""
@author: Radosław Pławecki
Stage `metric_anova` (tables -> metric_anova): RM-ANOVA of the per-subject medians of every metric of [analysis]
metrics between the conditions, for ABP and CBFV: {signal}_{metric}_anova.csv (with the Greenhouse-Geisser corrected p
and Mauchly's test of sphericity); if the uncorrected p < alpha (settings.toml), also the
paired t-tests with the Bonferroni correction: {signal}_{metric}_posthoc.csv.
"""

import itertools

import pandas as pd
import pingouin as pg
from scipy.stats import f as f_dist
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from ddtw_ml.pipeline import Stage


def add_sphericity(table: pd.DataFrame, df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Add the Greenhouse-Geisser correction and Mauchly's test of sphericity to an ANOVA table of AnovaRM, with
    the column names of pingouin.rm_anova.
    :param table: ANOVA table of AnovaRM (one within factor).
    :param df_wide: data of the ANOVA, one column per condition and one row per subject.
    :return: table with the columns eps, p_GG_corr, W_spher, p_spher and sphericity.
    """
    eps = pg.epsilon(df_wide, correction="gg")
    spher = pg.sphericity(df_wide)
    table = table.copy()
    table["eps"] = eps
    table["p_GG_corr"] = f_dist.sf(table["F Value"], eps * table["Num DF"], eps * table["Den DF"])
    table["W_spher"] = spher.W
    table["p_spher"] = spher.pval
    table["sphericity"] = spher.spher
    return table


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
                add_sphericity(anova.anova_table, df[cols]).to_csv(out_dir / f"{signal}_{metric}_anova.csv", sep=";")
                p = anova.anova_table["Pr > F"].iloc[0]
                if p < ctx.params["alpha"]:
                    post_hoc_tests(df[cols], cols).to_csv(out_dir / f"{signal}_{metric}_posthoc.csv", sep=";",
                                                          index=False)
                ctx.processed(f"{signal}_{metric} (p = {p:.3g})")
