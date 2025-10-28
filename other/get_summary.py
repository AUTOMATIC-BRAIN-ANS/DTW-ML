"""
@author: Radosław Pławecki
"""

import os
import pandas as pd

metric = "ABP_SPP-CBFV_SPP"
data_path = "C:/Python/ZSSI/data2/dtw/reorganised"
all_means = pd.DataFrame()
breaths = os.listdir(data_path)
for breath in breaths:
    methods = os.listdir(os.path.join(data_path, breath))
    for method in methods:
        final_path = os.path.join(data_path, breath, method, f"{metric}.csv")
        data = pd.read_csv(final_path, delimiter=';')
        df = pd.DataFrame(data)
        mean_series = df.mean()
        all_means[f"{breath}_{method}"] = mean_series

all_means.to_csv(f"C:/Python/ZSSI/data2/dtw/summary/{metric}.csv", sep=';')
print(all_means)
