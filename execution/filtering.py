"""
@author: Radoslaw Plawecki
"""

import os

import numpy as np
import pandas as pd

from project.utils.filter_signals import remove_freq

base_dir = "C:/Python/ZSSI/data/preprocessed"
for directory in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, directory)
    for file in os.listdir(dir_path):
        filename = os.path.join(dir_path, file)
        data = pd.read_csv(filename, delimiter=';')
        df = pd.DataFrame(data)
        s = ['DateTime', 'ABP', 'CBFV', 'BPM']
        datetime, abp, cbfv, bpm = df[s[0]], df[s[1]], df[s[2]], df[s[3]]
        if directory == "B6":
            breaths = 6
        elif directory == "B10":
            breaths = 10
        elif directory == "B15":
            breaths = 15
        else:
            breaths = np.mean(bpm)
        abp = remove_freq(abp, T=200, breaths=breaths)
        cbfv = remove_freq(cbfv, T=200, breaths=breaths)
        filtered = {
            "DateTime": datetime,
            "ABP": abp,
            "CBFV": cbfv
        }
        df = pd.DataFrame(filtered)
        file = os.path.splitext(file)[0]
        df.to_csv(f"C:/Python/ZSSI/data/filtered/{directory}/{file}_F.csv", sep=';', index=False)
        print("Data was exported!")
