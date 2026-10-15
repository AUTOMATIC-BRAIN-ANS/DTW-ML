"""
Troughs of a signal (beat onsets for the stage `beats`) with the multiscale local-extremum method (AMPD): a sample
is a trough when it is lower than both neighbours k samples away for every scale k up to the scale with the most
troughs. The signal is detrended first; NaNs are replaced with the mean. Scales: 1..L, L = ceil(N / 2) - 1 unless
max_scale is given.
The scales are scanned twice (trough counts, then the troughs up to the dominant scale) instead of storing the N x L
matrices of the original implementation (~1.7 GB for N = 42 340, P-01); the troughs are the same
(tests/stages/test_beat_stages.py compares them with the original). Peaks are not computed (they were never used).
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

    def troughs_at(k):
        # samples lower than both neighbours k samples away (samples k..N-k-1)
        center = d_signal[k:-k]
        return (center < d_signal[:-2*k]) & (center < d_signal[2*k:])

    # Dominant scale: the one with the most troughs (the first of equal ones; an error when L = 0, as before)
    counts = np.array([np.count_nonzero(troughs_at(k)) for k in range(1, L + 1)], dtype=int)
    dn = np.argmax(counts)

    # Troughs at every scale 1..dn+1; a sample closer than k to an end is not a trough at scale k
    is_trough = np.ones(N, dtype=bool)
    for k in range(1, dn + 2):
        at_k = np.zeros(N, dtype=bool)
        at_k[k:N - k] = troughs_at(k)
        is_trough &= at_k

    return np.where(is_trough)[0]
