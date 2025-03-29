"""
@author: Radoslaw Plawecki
"""

import pandas as pd
from project.common import smooth_data
import os


def extract_signals():
    """
    Function to extract specific signals from a file.
    """
    data_path = "C:/Python/ZSSI/data"
    raw_data_path = os.path.join(data_path, "raw")
    folder_names = os.listdir(raw_data_path)
    for index, folder_name in enumerate(folder_names, start=1):
        volunteer_path = os.path.join(raw_data_path, folder_name)
        csv_files = os.listdir(volunteer_path)
        for csv_file in csv_files:
            breath_char = csv_file[7]
            if breath_char == 'b':
                breath_label = "BAS"
            elif breath_char == '1':
                breath_label = csv_file[7:9]
            else:
                breath_label = breath_char
            filepath = os.path.join(volunteer_path, csv_file)
            df = pd.read_csv(filepath, delimiter=',')
            print(csv_file)
            new_df = pd.DataFrame({
                "DateTime": df['timestamp[s]'],
                "ABP": smooth_data(df['abp[mmHg]']),
                "CBFV": smooth_data(df['fv[cm/s]'])
            })
            output_dir = os.path.join(data_path, "smoothed", "macro", f"V{index}")
            os.makedirs(output_dir, exist_ok=True)
            if breath_label == "BAS":
                output_filename = f"V{index}_{breath_label}.csv"
            else:
                output_filename = f"V{index}_B{breath_label}.csv"
            output_path = os.path.join(output_dir, output_filename)
            new_df.to_csv(output_path, sep=';', index=False)


extract_signals()
