"""
@author: Radosław Pławecki
"""

import os
import pandas as pd

cols = ["V" + str(i) for i in range(1, 100)]
output = []
metric = "ABP_RR-CBFV_SPP"

data_path = "C:/Python/ZSSI/data/dtw/dtw"
methods = os.listdir(data_path)
for method in methods:
    breaths = os.listdir(os.path.join(data_path, method))
    for breath in breaths:
        files = os.listdir(os.path.join(data_path, method, breath))
        output_data = []
        for file in files:
            final_path = os.path.join(data_path, method, breath, file)
            file_name = os.path.splitext(file)[0]
            data = pd.read_csv(final_path, delimiter=';')
            df = pd.DataFrame(data)
            for col in cols:
                if f"{col}_{breath}_PP_F_M_PP_DTW" == file_name:
                    output_data.append({col: df[metric]})
        result_df = pd.concat([pd.DataFrame(v) for d in output_data for v in d.values()], axis=1)
        result_df.columns = [k for d in output_data for k in d.keys()]
        output_path = f"C:/Python/ZSSI/data/dtw/reorganised/{method}/{breath}/{metric}.csv"
        result_df.to_csv(output_path, index=False, sep=';')
