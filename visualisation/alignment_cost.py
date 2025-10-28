"""
@author: Radosław Pławecki
"""
import os

from project.common import use_latex
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

use_latex()

path = "C:/Python/ZSSI/data2/dtw/dtw"
breaths = os.listdir(path)
for breath in breaths:
    methods = os.listdir(os.path.join(path, breath))
    for method in methods:
        files = os.listdir(os.path.join(path, breath, method))
        for file in files:
            title = os.path.splitext(file)[0]
            final_path = os.path.join(path, breath, method, file)
            data = pd.read_csv(final_path, delimiter=';')
            df = pd.DataFrame(data)
            cols = [col for col in df.columns if col != "Window"]
            plt.figure(figsize=(12, 6))
            for col in cols:
                datetime = np.linspace(0, len(df[col]), len(df[col]))
                plt.plot(datetime, df[col], label=f"{col}")
                plt.xlim(left=0, right=len(datetime))
            plt.title(title, fontsize=18, pad=15)
            plt.xlabel("Nr okienka", fontsize=14, labelpad=12)
            plt.ylabel("Koszt dopasowania", fontsize=14, labelpad=12)
            plt.grid()
            plt.legend()
            # plt.savefig(f"C:/Python/ZSSI/plots/dtw/cost_per_window/{breath}/{method}/{title}.pdf", format='pdf')
            plt.close()
            # plt.show()
