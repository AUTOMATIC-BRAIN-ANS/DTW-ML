"""
@author: Radosław Pławecki
Plot `costs` (costs -> plots): DTW cost per sliding window of every metric pair, one figure per file:
{method}/{condition}/{file}.pdf.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ddtw_ml.pipeline import StageContext
from ddtw_ml.utils.plotting import use_latex


def run(ctx: StageContext) -> None:
    use_latex()
    path = ctx.input()
    for method in os.listdir(path):
        for breath in os.listdir(path / method):
            out_dir = ctx.output(method, breath)
            for file in os.listdir(path / method / breath):
                title = os.path.splitext(file)[0]
                data = pd.read_csv(path / method / breath / file, delimiter=';')
                df = pd.DataFrame(data)
                cols = [col for col in df.columns if col != "Window"]
                plt.figure(figsize=(12, 6))
                for col in cols:
                    datetime = np.linspace(0, len(df[col]), len(df[col]))
                    plt.plot(datetime, df[col], label=f"{col}")
                    plt.xlim(left=0, right=len(datetime))
                plt.title("Koszt dopasowania w zależności od nr okienka", fontsize=18, pad=15)
                plt.xlabel("Nr okienka", fontsize=14, labelpad=12)
                plt.ylabel("Koszt dopasowania", fontsize=14, labelpad=12)
                plt.grid()
                plt.legend()
                plt.savefig(out_dir / f"{title}.pdf", format='pdf')
                plt.close()
                ctx.processed(f"{method}/{breath}/{file}")
