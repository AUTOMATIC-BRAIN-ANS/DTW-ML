"""
@author: Radoslaw Plawecki
"""

import pandas as pd
import os

method = 'd-method'
data_path = f"C:/Python/ZSSI/data_v2/dtw/reorganised/{method}"
directories = os.listdir(data_path)
for directory in directories:
    print(f"=== {directory} ===")
    directory_path = os.path.join(data_path, directory)
    files = os.listdir(directory_path)
    for file in files:
        print(f"{file} being processed...")
        file_path = os.path.join(directory_path, file)
        data = pd.read_csv(file_path, delimiter=';')
        df = pd.DataFrame(data)
        merged_col = df.iloc[0:].stack().reset_index(drop=True)
        merged_df = merged_col.to_frame(name="costs")
        output_path = f"C:/Python/ZSSI/data_v2/ml-data/{method}/{directory}/{file}.csv"
        merged_df.to_csv(output_path, sep=';', index=False)
        print("Done!")
