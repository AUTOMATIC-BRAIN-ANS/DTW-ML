"""
@author: Radoslaw Plawecki
"""

import numpy as np
from project.preprocessing.generalized_logistic import GeneralizedLogistic


class NormalizeData:
    def __init__(self, data):
        """
        Method to initialize params of a class.
        :param data: data to normalise.
        """
        self.data = data

    @staticmethod
    def min_max(data, min_value, max_value):
        """
        Method to perform min-max normalization to a given range on data.
        :param data: data to normalise.
        :param min_value: minimum value after normalization.
        :param max_value: maximum value after normalization.
        :raise ValueError: if range limits are not given or minimum value is greater than maximum.
        :return: normalised data.
        """
        if min_value is None or max_value is None:
            raise ValueError("Specify minimum and maximum value!")
        if min_value > max_value:
            raise ValueError("The minimum value cannot be greater than maximum!")
        return (max_value - min_value) * (data - min(data)) / (max(data) - min(data)) + min_value

    @staticmethod
    def z_score(data):
        """
        Method to perform z-score normalization on data.
        :param data: data to normalise.
        :return: normalised data.
        """
        return (data - np.mean(data)) / np.std(data)

    @staticmethod
    def generalized_logistic(data):
        """
        Method to perform normalization to on data using GL algorithm.
        :param data: data to normalise.
        :return: normalised data.
        """
        return GeneralizedLogistic(data)()

    def normalize(self, method=None, min_value=None, max_value=None):
        """
        Method to normalise data using a given method.
        :param method: normalisation method.
        :param min_value: if min-max normalisation, then minimum value after normalisation.
        :param max_value: if min-max normalisation, then maximum value after normalisation.
        :return: normalised data.
        :raise ValueError: if a given method doesn't exist.
        """
        if method == "min-max":
            return self.min_max(self.data, min_value=min_value, max_value=max_value)
        elif method == "z-score":
            return self.z_score(self.data)
        elif method == "generalized-logistic":
            return self.generalized_logistic(self.data)
        else:
            raise ValueError(f"Allowed methods of normalization are: 'min-max', 'z-score' and 'generalized-logistic'. "
                             f"Got '{method}' instead.")
