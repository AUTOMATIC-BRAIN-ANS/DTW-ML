"""
@author: Radosław Pławecki
"""

import os
import pandas as pd


directory_path = "C:/Python/ZSSI/data/raw"
directories = os.listdir(directory_path)
for directory in directories:
    files = os.listdir(os.path.join(directory_path, directory))
    for file in files:
        if "PAC" in file and "ODD" in file:
            data_path = os.path.join(directory_path, directory, file)
            data = pd.read_csv(data_path)
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = [], [], [], [], [], [], [], [], [], []
            for i in range(data.shape[0]):
                row = str(data.iloc[i, 0])
                parts = str(row).split(".")
                j = 0
                for j in range(len(parts)):
                    if j == 0:
                        col1.append(".".join(parts[j:j + 2]))
                    if j == 2:
                        col2.append(".".join(parts[j:j + 2]))
                    if j == 4:
                        col3.append(".".join(parts[j:j + 2]))
                    if j == 6:
                        col4.append(".".join(parts[j:j + 2]))
                    if j == 8:
                        col5.append(".".join(parts[j:j + 2]))
                    if j == 10:
                        col6.append(".".join(parts[j:j + 2]))
                    if j == 12:
                        col7.append(".".join(parts[j:j + 2]))
                    if j == 14:
                        col8.append(".".join(parts[j:j + 2]))
                    if j == 16:
                        col9.append(".".join(parts[j:j + 2]))
                    if j == 18:
                        col10.append(".".join(parts[j:j + 2]))
                    continue

            length = min(len(col1), len(col2), len(col3), len(col4))

            df = pd.DataFrame({
                "DateTime": col1[:length],
                "fv_r[fv_r]": col2[:length],
                "fv_l[fv_l]": col3[:length],
                "abp_finger[abp_finger]": col4[:length],
            })

            df.to_csv(data_path, sep=';', index=False)
            print(file)
