"""
@author: Radosław Pławecki
"""

import pandas as pd
import pingouin as pg

path = "C:/Python/ZSSI/data2/dtw/summary/ABP_SPO-CBFV_SPO.csv"
data = pd.read_csv(path, delimiter=';')
df = pd.DataFrame(data)
columns_to_keep_td = ["B10_td-method", "B15_td-method", "B6_td-method", "BAS_td-method"]
columns_to_keep_d = ["B10_d-method", "B15_d-method", "B6_d-method", "BAS_d-method"]
df = df[columns_to_keep_td]
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
