"""
@author: Radoslaw Plawecki
"""

from scipy.fft import fft, fftfreq, ifft
import numpy as np


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
