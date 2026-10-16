"""
@author: Radosław Pławecki
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ddtw_ml.signal.nan_handler import NaNHandler as NaNH
from ddtw_ml.signal.normalisation import NormalizeData


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


class PreprocessingUtils:
    @staticmethod
    def iqr_outlier_removal(s, threshold):
        """
        Method to detect and remove outliers using the IQR method.
        :param s: signal.
        :param threshold: threshold after which the outliers will be removed.
        :return: signal with removed outliers.
        """
        q1 = np.nanpercentile(s, 25)
        q3 = np.nanpercentile(s, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        # copy signal to avoid modifying the original
        s = np.array(s.copy())
        s = s.astype(float)
        for i in range(1, len(s)):
            if s[i] < lower_bound or s[i] > upper_bound:
                s[i] = np.nan
        return s

    @staticmethod
    def interpolate_data(s, method=None, order=None, min_valid=20):
        """
        Method to interpolate signals using a specific method of interpolation; NaNs at both ends are filled with the
        nearest valid value.
        :param s: signal.
        :param method: method of interpolation.
        :param order: if a method is polynomial, then its order.
        :param min_valid: minimum number of valid (not NaN) samples.
        :return: interpolated signals.
        :raise ValueError: if a signal has fewer than min_valid valid samples.
        """
        s = pd.Series(s)
        n_valid = int(s.notna().sum())
        if n_valid < min_valid:
            raise ValueError(f"Signal has too many gaps: {n_valid} valid samples of {len(s)} (at least {min_valid} "
                             f"needed)!")
        return s.interpolate(method=method, order=order, limit_direction='both')

    @staticmethod
    def normalize(s, method, min_value=-1, max_value=1):
        """
        Method to normalize signals using a specific method of normalization.
        :param s: signal.
        :param method: method of normalization.
        :param min_value: a minimum value.
        :param max_value: a maximum value.
        :return: normalized signals.
        """
        return NormalizeData(s).norm(method=method, min_value=min_value, max_value=max_value)

    @staticmethod
    def smooth(s):
        """
        Function to smooth data using the Savitzky-Golay filter.
        :return: smoothed signal.
        """
        return savgol_filter(s, window_length=20, polyorder=3)

    @staticmethod
    def trim(s1, s2):
        """
        Method to find new limits of signals depending on length of gaps.
        :param s1: first signal.
        :param s2: second signal.
        :return: new limits of signals.
        """
        nan_in_s1, nan_in_s2 = NaNH.get_nan_number(s1), NaNH.get_nan_number(s2)
        if nan_in_s1 >= nan_in_s2:
            s = pd.Series(s1)
        else:
            s = pd.Series(s2)
        start, stop = 0, len(s)
        nan_indices = s.index[s.isna()].tolist()
        ordered_nans = values_in_order(nan_indices)
        for count, loc in ordered_nans:
            if count > 600:
                if abs(start - loc) < abs(loc - stop):
                    start = loc + count
                else:
                    stop = loc
        return start, stop
