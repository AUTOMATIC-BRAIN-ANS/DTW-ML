"""
@author: Radosław Pławecki
"""

import itertools

import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

metrics = ["SPO", "SPP", "RR"]
i = 2

df = pd.read_csv("C:/Python/ZSSI/data/characteristics/metrics_for_anova.csv", sep=";")
df.columns = df.columns.str.strip()
df["Subject"] = range(1, len(df) + 1)


# ---------- FUNCTION FOR POST-HOC ----------
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

    print("\nPost-hoc pairwise comparisons (Bonferroni corrected):")
    for (c1, c2, stat, p), p_corr, r in zip(results, pvals_corr, reject):
        print(f"{c1} vs {c2}: t={stat:.3f}, p={p:.5f}, p_corr={p_corr:.5f}, significant={r}")


# ----- ABP -----
# abp_cols = [f"BAS_ABP", f"B6_ABP", f"B10_ABP", f"B15_ABP"]
abp_cols = [f"BAS_ABP_{metrics[i]}", f"B6_ABP_{metrics[i]}",
             f"B10_ABP_{metrics[i]}", f"B15_ABP_{metrics[i]}"]

abp = pd.melt(df,
              id_vars=["Subject"],
              value_vars=abp_cols,
              var_name="Breath",
              value_name="ABP")

abp["Breath"] = abp["Breath"].str.replace("_ABP", "")

anova_abp = AnovaRM(abp, "ABP", "Subject", within=["Breath"]).fit()

p_abp = anova_abp.anova_table["Pr > F"][0]
print("ABP p-value:", p_abp)

if p_abp < 0.05:
    post_hoc_tests(df[abp_cols], abp_cols)

# ----- CBFV -----
# cbfv_cols = [f"BAS_CBFV", f"B6_CBFV", f"B10_CBFV", f"B15_CBFV"]
cbfv_cols = [f"BAS_CBFV_{metrics[i]}", f"B6_CBFV_{metrics[i]}",
             f"B10_CBFV_{metrics[i]}", f"B15_CBFV_{metrics[i]}"]

cbfv = pd.melt(df,
               id_vars=["Subject"],
               value_vars=cbfv_cols,
               var_name="Breath",
               value_name="CBFV")

cbfv["Breath"] = cbfv["Breath"].str.replace("_CBFV", "")

anova_cbfv = AnovaRM(cbfv, "CBFV", "Subject", within=["Breath"]).fit()

p_cbfv = anova_cbfv.anova_table["Pr > F"][0]
print("CBFV p-value:", p_cbfv)

if p_cbfv < 0.05:
    post_hoc_tests(df[cbfv_cols], cbfv_cols)
