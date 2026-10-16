"""
@author: Radoslaw Plawecki
Removal of the artefacts of the ABP and FV signals.
"""

import numpy as np
import pandas as pd


def filter_abp(df, col_abp, max_diff=10):
    """
    Function to clear the ABP signal from impulse artefacts: a sample that differs by more than max_diff from both
    neighbours, in the same direction (above both or below both), becomes NaN. Steep but monotonic slopes of the
    beats are kept; the first and last sample are not checked.
    :param df: data in the DataFrame format.
    :param col_abp: column with values of the ABP signal.
    :param max_diff: maximum difference from a neighbour [mmHg].
    :return: cleared signal (index of the data).
    """
    abp = df[col_abp].to_numpy(dtype=float)
    previous, following = np.full_like(abp, np.nan), np.full_like(abp, np.nan)
    previous[1:], following[:-1] = abp[1:] - abp[:-1], abp[:-1] - abp[1:]
    impulse = ((previous > max_diff) & (following > max_diff)) | ((previous < -max_diff) & (following < -max_diff))
    return pd.Series(np.where(impulse, np.nan, abp), index=df.index, name=col_abp)


def filter_fv(df, col_fv, min_value=20):
    """
    Function to clear the FV signal from artefacts (when FV < 20).
    :param df: data in the DataFrame format.
    :param col_fv: column with values of the FV signal.
    :param min_value: minimum value of a range.
    :return: cleared signal.
    """
    fv = df[col_fv].astype(float)
    return fv.mask(fv < min_value)
