"""
@author: Radosław Pławecki
"""
import os

import pandas as pd
import pingouin as pg

path = "C:/Python/ZSSI/data/dtw/summary/"
combs = os.listdir(path)
cols_to_keep_td = ["td-method_B10", "td-method_B15", "td-method_B6", "td-method_BAS"]
cols_to_keep_d = ["d-method_B10", "d-method_B15", "d-method_B6", "d-method_BAS"]
cols = [cols_to_keep_td, cols_to_keep_d]
methods = ["td-method", "d-method"]
for comb in combs:
    for index, col in enumerate(cols):
        f_path = os.path.join(path, comb)
        comb_no_ext = os.path.splitext(comb)[0]
        data = pd.read_csv(f_path, delimiter=';')
        df = pd.DataFrame(data)[col]
        df['subject'] = df.index + 1
        df_long = df.melt(id_vars='subject', var_name='condition', value_name='score')

        aov = pg.rm_anova(dv='score', within='condition', subject='subject', data=df_long, detailed=True)
        print(aov[['Source', 'F', 'p-unc', 'ng2']])  # 'p-GG-corr'

        posthoc = pg.pairwise_tests(
            dv='score',
            within='condition',
            subject='subject',
            data=df_long,
            padjust='bonf'
        )

        print(posthoc[['A', 'B', 'T', 'p-corr']])
        output_path = f'C:/Python/ZSSI/data/statistics/{comb_no_ext}'
        os.makedirs(output_path, exist_ok=True)
        aov[['Source', 'F', 'p-unc', 'ng2']].to_csv(os.path.join(output_path, f'{methods[index]}_anova_results.csv'),
                                                    index=False, sep=';')
        posthoc[['A', 'B', 'T', 'p-corr']].to_csv(os.path.join(output_path, f'{methods[index]}_posthoc_results.csv'),
                                                  index=False, sep=';')
