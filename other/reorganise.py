"""
@author: Radosław Pławecki
"""

import os
import pandas as pd

cols = ["V" + str(i) for i in range(1, 38)]
output = []
metric = "ABP_RR-CBFV_SPP"

data_path = "/data2/dtw/dtw"
breaths = os.listdir(data_path)
for breath in breaths:
    methods = os.listdir(os.path.join(data_path, breath))
    for method in methods:
        files = os.listdir(os.path.join(data_path, breath, method))
        output_data = []
        for file in files:
            final_path = os.path.join(data_path, breath, method, file)
            file_name = os.path.splitext(file)[0]
            data = pd.read_csv(final_path, delimiter=';')
            df = pd.DataFrame(data)
            for col in cols:
                if f"{col}_{breath}_PP_F_M_PP_DTW" == file_name:
                    output_data.append({col: df[metric]})
        result_df = pd.concat([pd.DataFrame(v) for d in output_data for v in d.values()], axis=1)
        result_df.columns = [k for d in output_data for k in d.keys()]
        output_path = f"C:/Python/ZSSI/data2/dtw/reorganised/{breath}/{method}/{metric}.csv"
        result_df.to_csv(output_path, index=False, sep=';')