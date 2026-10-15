"""
@author: Radoslaw Plawecki
Removal of the artefacts of the ABP and FV signals.
"""

import numpy as np


def filter_abp(df, col_abp, max_diff=10):
    """
    Function to clear the ABP signal from artefacts (when difference between next values of ABP > B10).
    :param df: data in the DataFrame format.
    :param col_abp: column with values of the ABP signal.
    :param max_diff: maximum value of next ABP values.
    :return: cleared signal.
    """
    abp = df[col_abp].copy()
    length = len(abp)
    for i in range(1, length):
        if abs(abp.iloc[i] - abp.iloc[i - 1]) > max_diff:
            for j in range(i, length - 1):
                if abp.iloc[j + 1] < abp.iloc[j]:
                    break
                abp[j] = np.nan
    return abp


def filter_fv(df, col_fv, min_value=20):
    """
    Function to clear the FV signal from artefacts (when FV < 20).
    :param df: data in the DataFrame format.
    :param col_fv: column with values of the FV signal.
    :param min_value: minimum value of a range.
    :return: cleared signal.
    """
    fv = df[col_fv].copy()
    length = len(fv)
    for i in range(0, length):
        if fv[i] < min_value:
            fv[i] = np.nan
    return fv
