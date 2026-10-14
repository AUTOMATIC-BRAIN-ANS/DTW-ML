"""
@author: Radoslaw Plawecki
Stage `filtered` (cleaned -> filtered): removes the breathing frequency from ABP and CBFV with the FFT (fs = 200 Hz):
{condition}/{file}_F.csv. The breathing rate is the paced one (B6, B10, B15) or the mean recorded BPM (BAS).
"""

import os

import numpy as np
import pandas as pd

from ddtw_ml.core.signal.breathing_filter import remove_freq
from ddtw_ml.pipeline import StageContext


def run(ctx: StageContext) -> None:
    base_dir = ctx.input()
    for directory in os.listdir(base_dir):
        dir_path = base_dir / directory
        out_dir = ctx.output(directory)
        for file in os.listdir(dir_path):
            filename = dir_path / file
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
            df.to_csv(out_dir / f"{filename.stem}_F.csv", sep=';', index=False)
            ctx.processed(f"{directory}/{file}")
