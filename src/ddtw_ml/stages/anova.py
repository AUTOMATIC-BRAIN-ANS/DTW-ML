"""
@author: Radosław Pławecki
Stage `anova` (mean_costs -> anova): RM-ANOVA of the mean DTW costs between the conditions and Bonferroni-corrected
pairwise tests, per metric pair and method: {pair}/{method}_{anova,posthoc}_results.csv. The ANOVA results include
the Greenhouse-Geisser corrected p and Mauchly's test of sphericity.
"""

import os

import pandas as pd
import pingouin as pg

from ddtw_ml.pipeline import Stage

# Greenhouse-Geisser epsilon and p, Mauchly's W and p, whether sphericity holds (alpha 0.05 of pingouin)
SPHERICITY_COLUMNS = ['eps', 'p_GG_corr', 'W_spher', 'p_spher', 'sphericity']


class CompareConditions(Stage):
    def run(self) -> None:
        ctx = self.ctx
        path = ctx.input()
        methods = ctx.analysis["methods"]
        # columns {method}_{condition}, conditions in alphabetical order (B10, B15, B6, BAS)
        cols = [[f"{method}_{condition}" for condition in sorted(ctx.analysis["conditions"])] for method in methods]
        for comb in os.listdir(path):
            comb_no_ext = os.path.splitext(comb)[0]
            output_path = ctx.output(comb_no_ext)
            for index, col in enumerate(cols):
                data = pd.read_csv(path / comb, delimiter=';')
                df = pd.DataFrame(data)[col]
                df['subject'] = df.index + 1
                df_long = df.melt(id_vars='subject', var_name='condition', value_name='score')

                # correction: Greenhouse-Geisser p and Mauchly's test of sphericity
                aov = pg.rm_anova(dv='score', within='condition', subject='subject', data=df_long, detailed=True,
                                  correction=True)

                posthoc = pg.pairwise_tests(
                    dv='score',
                    within='condition',
                    subject='subject',
                    data=df_long,
                    padjust='bonf'
                )

                aov[['Source', 'F', 'p_unc', 'ng2'] + SPHERICITY_COLUMNS].to_csv(
                    output_path / f'{methods[index]}_anova_results.csv', index=False, sep=';')
                posthoc[['A', 'B', 'T', 'p_corr']].to_csv(output_path / f'{methods[index]}_posthoc_results.csv',
                                                          index=False, sep=';')
                ctx.processed(f"{comb_no_ext}/{methods[index]}")
