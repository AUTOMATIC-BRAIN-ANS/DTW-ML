"""
@author: Radosław Pławecki
"""

import pandas as pd
import pingouin as pg

path = "C:/Python/ZSSI/data2/statistics/all-in-one/td-method.csv"
data = pd.read_csv(path, delimiter=';')
df = pd.DataFrame(data)
df['subject'] = df.index + 1
df_long = df.melt(id_vars='subject', var_name='condition', value_name='score')

aov = pg.rm_anova(dv='score', within='condition', subject='subject', data=df_long, detailed=True)
print(aov[['Source', 'F', 'p-unc', 'p-GG-corr', 'ng2']])

posthoc = pg.pairwise_tests(
    dv='score',
    within='condition',
    subject='subject',
    data=df_long,
    padjust='bonf'
)

print(posthoc[['A', 'B', 'T', 'p-corr']])
