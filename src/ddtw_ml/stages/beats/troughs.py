"""
Troughs of a signal (beat onsets for the stage `beats`) with the multiscale local-extremum method (AMPD): a sample
is a trough when it is lower than both neighbours k samples away for every scale k up to the scale with the most
troughs. The signal is detrended first; NaNs are replaced with the mean. Memory: two N x L boolean matrices,
L = ceil(N / 2) - 1 unless max_scale is given (P-01).
"""

import math

import numpy as np
from scipy import signal as sp_sig


def detect_peaks_troughs_optimized(signal, max_scale=0):
    N = len(signal)

    # Handle NaNs
    mean_sig = np.nanmean(signal)
    signal = np.nan_to_num(signal, nan=mean_sig)

    # Detrend signal
    d_signal = sp_sig.detrend(signal, type='linear')

    # Determine number of scales
    if max_scale != 0:
        L = math.ceil(max_scale / 2) - 1
    else:
        L = math.ceil(N / 2) - 1

    # Initialize peak/trough matrices
    Mx = np.zeros((N, L), dtype=bool)
    Mn = np.zeros((N, L), dtype=bool)

    # Vectorized neighborhood comparison
    for k in range(1, L + 1):
        # Create shifted versions
        center = d_signal[k:-k]
        left = d_signal[:-2*k]
        right = d_signal[2*k:]
        idx = np.arange(k, N - k)

        # Check peak and trough conditions
        Mx[idx, k - 1] = (center > left) & (center > right)
        Mn[idx, k - 1] = (center < left) & (center < right)

    # Select dominant scale range
    dx = np.argmax(np.sum(Mx, axis=0))
    Mx = Mx[:, :dx + 1]

    dn = np.argmax(np.sum(Mn, axis=0))
    Mn = Mn[:, :dn + 1]

    # Count non-zero entries per row and find consensus peaks/troughs
    Zx = np.sum(Mx == 0, axis=1)
    Zn = np.sum(Mn == 0, axis=1)

    peaks = np.where(Zx == 0)[0]
    troughs = np.where(Zn == 0)[0]

    return troughs
