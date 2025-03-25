"""
@author: Radoslaw Plawecki
"""

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from os import path
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


def filter_cbfv(df, col_cbfv, min_value=20):
    """
    Function to clear the CBFV signal from artefacts (when CBFV < 20).
    :param df: data in the DataFrame format.
    :param col_cbfv: column with values of the CBFV signal.
    :param min_value: minimum value of a range.
    :return: cleared signal.
    :raise ValueError: if signals have different length.
    """
    cbfv = df[col_cbfv].copy()
    length = len(cbfv)
    for i in range(1, length):
        if cbfv[i] < min_value:
            cbfv[i] = np.nan
    return cbfv
