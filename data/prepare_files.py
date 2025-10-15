"""
@author: Radoslaw Plawecki
"""

import pandas as pd
from project.common import calculate_cbfv
import os


def extract_signals():
    """
    Function to extract specific signals from a file.
    """
    data_path = "C:/Python/ZSSI/data2/"
    raw_data_path = os.path.join(data_path, "raw")
    directories = os.listdir(raw_data_path)
    for directory in directories:
        directory_path = os.path.join(raw_data_path, directory)
        files = os.listdir(directory_path)
        i = 1
        for file in files:
            print(f"Directory: {directory}, file: {file} being processed...")
            file_path = os.path.join(directory_path, file)
            data = pd.read_csv(file_path, delimiter=';', decimal=',')
            df = pd.DataFrame(data)
            datetime, abp, cbfv = df['DateTime'], df['abp_cnap[mmHg]'], calculate_cbfv(df, 'fvl', 'fvr')
            new_df = pd.DataFrame({
                "DateTime": datetime,
                "ABP": abp,
                "CBFV": cbfv
            })
            output_path = os.path.join(data_path, f"extracted/{directory}/V{i}_{directory}.csv")
            i += 1
            new_df.to_csv(output_path, sep=';', index=False)
            print("Process completed successfully.")


extract_signals()
