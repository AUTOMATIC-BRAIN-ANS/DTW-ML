"""
@author: Radoslaw Plawecki
"""

from project.common import use_latex
import os
import pandas as pd
from project.metrics.extract_metrics import FromSignal

use_latex()

input_path = "C:/Python/ZSSI/data/filtered"
directories = os.listdir(input_path)
# i = 0
for directory in directories:
    directory_path = os.path.join(input_path, directory)
    os.makedirs(directory_path, exist_ok=True)
    files = os.listdir(directory_path)
    for file in files:
        print(f"Directory: {directory}, file: {file}")
        file_path = os.path.join(directory_path, file)
        data = pd.read_csv(file_path, delimiter=';')
        df = pd.DataFrame(data)
        s = ['DateTime', 'ABP', 'CBFV']
        datetime, abp, cbfv = df[s[0]], df[s[1]], df[s[2]]
        file_directory = os.path.join(directory, os.path.splitext(file)[0])
        from_signal_abp = FromSignal(datetime, signal=abp, signal_type="abp")
        from_signal_cbfv = FromSignal(datetime, signal=cbfv, signal_type="cbfv")
        from_signal_abp.export_to_csv(filename="ABP.csv", directory=file_directory)
        from_signal_cbfv.export_to_csv(filename="CBFV.csv", directory=file_directory)
        """if i == 0:
            from_signal_abp.plot_window(wd_number=2, file="abp_metrics1")
            from_signal_cbfv.plot_window(wd_number=2, file="cbfv_metrics1")
            i += 1"""
