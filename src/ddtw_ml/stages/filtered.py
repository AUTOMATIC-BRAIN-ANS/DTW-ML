"""
@author: Radoslaw Plawecki
Stage `filtered` (cleaned -> filtered): removes the breathing frequency from ABP and CBFV with the FFT (fs in
settings.toml): {condition}/{file}_F.csv. The breathing rate is the paced one of the condition (paced_breaths in
settings.toml: B6, B10, B15) or, in the other conditions (BAS), the mean recorded BPM.
DateTime is the time of each sample in ms (sample number * 1000 / fs, from 0); ETCO2 is copied unchanged.
"""

import os

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft

from ddtw_ml.pipeline import Stage


def remove_freq(s, T, breaths, thresh=0.75, margin=0.01):
    N = len(s)
    xf, yf = fftfreq(N, d=1/T), fft(s)
    amp = np.abs(yf) / N
    f_breath = np.round(breaths / 60.0, 2)
    lower_bound, upper_bound = f_breath - margin, f_breath + margin
    mask = (np.abs(xf) >= lower_bound) & (np.abs(xf) <= upper_bound) & (amp > thresh)
    yf[mask] = 0
    filtered_s = np.real(ifft(yf))
    return filtered_s


class RemoveBreathing(Stage):
    PARAMS = {"fs": int, "paced_breaths": dict}

    def run(self) -> None:
        ctx = self.ctx
        base_dir = ctx.input()
        for directory in os.listdir(base_dir):
            dir_path = base_dir / directory
            out_dir = ctx.output(directory)
            for file in os.listdir(dir_path):
                filename = dir_path / file
                data = pd.read_csv(filename, delimiter=';')
                df = pd.DataFrame(data)
                s = ['ABP', 'CBFV', 'BPM', 'ETCO2']
                abp, cbfv, bpm, etco2 = df[s[0]], df[s[1]], df[s[2]], df[s[3]]
                fs = ctx.params["fs"]
                paced_breaths = ctx.params["paced_breaths"]
                if directory in paced_breaths:
                    breaths = paced_breaths[directory]
                else:
                    breaths = np.mean(bpm)
                abp = remove_freq(abp, T=fs, breaths=breaths)
                cbfv = remove_freq(cbfv, T=fs, breaths=breaths)
                filtered = {
                    "DateTime": np.arange(len(df)) * 1000 / fs,
                    "ABP": abp,
                    "CBFV": cbfv,
                    "ETCO2": etco2
                }
                df = pd.DataFrame(filtered)
                df.to_csv(out_dir / f"{filename.stem}_F.csv", sep=';', index=False)
                ctx.processed(f"{directory}/{file}")
