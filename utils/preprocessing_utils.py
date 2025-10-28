"""
@author: Radosław Pławecki
"""

import numpy as np
import pandas as pd
from project.utils.nan_handler import NaNHandler as NaNH
from project.utils.normalization import NormalizeData
from project.common import values_in_order
from scipy.signal import savgol_filter


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
    def __fill_nans(s, method=None, order=None):
        """
        Method to fill NaNs in signals using a specific method of interpolation.
        :param s: signal.
        :param method: method of interpolation.
        :param order: if a method is polynomial, then its order.
        :return: signals with filled NaNs.
        :raise ValueError: if a signal has too many gaps.
        """
        s = pd.Series(s)
        s = s.interpolate(method=method, order=order)
        if len(s) < 20:
            raise ValueError("Signal has too many gaps!")
        else:
            return s

    @staticmethod
    def interpolate_data(s, method=None, order=None):
        """
        Method to interpolate signals using a specific method of interpolation.
        :param s: signal.
        :param method: method of interpolation.
        :param order: if a method is polynomial, then its order.
        :return: interpolated signals.
        """
        try:
            return PreprocessingUtils.__fill_nans(s, method=method, order=order)
        except ValueError as e:
            print(f"Error occurred: {e}")

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
