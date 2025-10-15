"""
@author: Radoslaw Plawecki
"""

import pandas as pd
from project.execution.preprocessing import remove_freq
import os

base_dir = "/data2/preprocessed"
for directory in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, directory)
    if directory == "B6":
        breaths = 6
    elif directory == "B10":
        breaths = 10
    elif directory == "B15":
        breaths = 15
    else:
        breaths = 30
    for file in os.listdir(dir_path):
        filename = os.path.join(dir_path, file)
        print(filename)
        data = pd.read_csv(filename, delimiter=';')
        df = pd.DataFrame(data)
        s = ['DateTime', 'ABP', 'CBFV']
        datetime, abp, cbfv = df[s[0]], df[s[1]], df[s[2]]
        abp = remove_freq(abp, T=200, breaths=breaths)
        cbfv = remove_freq(cbfv, T=200, breaths=breaths)
        data = {
            "DateTime": datetime,
            "ABP": abp,
            "CBFV": cbfv
        }
        df = pd.DataFrame(data)
        file = os.path.splitext(file)[0]
        df.to_csv(f"C:/Python/ZSSI/data2/filtered/{directory}/{file}_F.csv", sep=';', index=False)
        print("Data was exported!")
