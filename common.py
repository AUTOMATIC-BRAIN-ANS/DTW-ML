"""
@author: Radoslaw Plawecki
"""

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from os import path
import pandas as pd
import numpy as np


def use_latex():
    """
    Function to use LaTeX formatting for plots.
    """
    # use LaTeX for text rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({
        'text.latex.preamble': r'\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc}'
    })


def smooth_data(s):
    """
    Function to smooth data using the Savitzky-Golay filter.
    :return: smoothed signal.
    """
    return savgol_filter(s, window_length=20, polyorder=3)


def check_column_existence(df, col):
    """
    Function to check if a column with the given name exists in a file.
    :param df: data as DataFrame object.
    :param col: name of a column to look for.
    :raise KeyError: if a column doesn't exist in a file.
    """
    if not list(df.columns).__contains__(col):
        raise KeyError(f"Column '{col}' doesn't exist in a file!")


def check_path(filepath):
    if not path.exists(filepath):
        raise FileNotFoundError("File not found!")
    if not path.isfile(filepath):
        raise IsADirectoryError("The path exists but is not a file!")
    if path.splitext(filepath)[1] != '.csv':
        raise ValueError("File must be a CSV file!")


def values_in_order(index_list):
    """
    Function to replace the list of rowed indices into an array of tuples (A, B), where A is the number in sequence
    occurring indices, and B is the first element of the given sequence.
    :param index_list: list of rowed indices.
    :return: array of tuples.
    """
    count = 1
    summary = []
    for i in range(len(index_list) - 1, -1, -1):
        if index_list[i] == index_list[i - 1] + 1:
            count += 1
        else:
            if count >= 1:
                last_index = index_list[i]
                summary.append((count, last_index))
            count = 1
    return list(reversed(summary))


def make_blocks(s):
    """
    Function to make divide a signal into blocks.
    :param s: signal.
    :return: signal divided into blocks.
    """
    one_day = 24 * 60 * 60 // 10
    s = [s[0:one_day], s[one_day:one_day * 2], s[one_day * 2:]]
    return s


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


def calculate_cbfv(df, col_fvl, col_fvr):
    """
    Function to calculate the CBFV signal based on the FV from left and right side.
    :param df: data in the DataFrame format.
    :param col_fvl: FV from the left side.
    :param col_fvr: FV from the right side.
    :return: CBFV signal.
    """
    fvl, fvr = filter_fv(df, col_fvl), filter_fv(df, col_fvr)
    length = min(len(fvl), len(fvr))
    cbfv = []
    for i in range(0, length):
        if fvl[i] > fvr[i]:
            cbfv.append(fvl[i])
        else:
            cbfv.append(fvr[i])
    return pd.Series(cbfv)
