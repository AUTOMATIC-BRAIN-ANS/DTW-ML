"""
@author: Radoslaw Plawecki
"""

import pandas as pd
import numpy as np


class NaNHandler:
    @staticmethod
    def replace_zeros_with_nans(s):
        """
        Method to replace zeros from a signal with NaNs.
        :param s: signal.
        :return: signal with zeros replaced with NaNs.
        :raise: TypeError: if type of input is incorrect.
        """
        if isinstance(s, (pd.DataFrame, pd.Series)):
            return s.replace(0, np.nan)
        elif isinstance(s, np.ndarray):
            s = s.astype(float)
            s[s == 0] = np.nan
            return s
        else:
            raise TypeError("Input must be a DataFrame, Series, or numpy array!")

    @staticmethod
    def get_nan_number(s):
        """
        Method to get number of nans in a signal.
        :param s: signal.
        :return: number of nans in a signal.
        """
        return pd.DataFrame(s).isna().sum().sum()
