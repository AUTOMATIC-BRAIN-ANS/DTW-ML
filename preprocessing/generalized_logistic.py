"""
@author: Radoslaw Plawecki
Sources:
[1] Cao, X. H., Stojkovic, I., & Obradovic, Z. (2016). A robust data scaling algorithm to improve classification
accuracies in biomedical data. BMC Bioinformatics, 17, 359. Available on: https://doi.org/10.1186/s12859-016-1236-x.
Access: 30.10.2024.
"""

import numpy as np
from scipy.optimize import minimize, fminbound


class GeneralizedLogistic:
    def __init__(self, data):
        """
        Method to initialize params of a class.
        :param data: data.
        """
        self.data_sorted = np.sort(data)
        # store the original indices to restore order
        self.original_indices = np.argsort(data)
        # calculate ECDF in a sorted order
        self.ecdf_values_sorted = np.arange(1, len(data) + 1) / len(data)
        self.data = data
        # get ECDF in an original order
        self.ecdf_values = np.zeros_like(self.ecdf_values_sorted)
        self.ecdf_values[self.original_indices] = self.ecdf_values_sorted

    @staticmethod
    def generalized_logistic(x, Q, B, M, nu):
        """
        Method to implement the generalized logistic (GL) function.
        :param x: argument.
        :param Q: parameter Q.
        :param B: parameter B.
        :param M: parameter M.
        :param nu: parameter nu.
        :return: value for an argument.
        """
        return 1 / (1 + Q * np.exp(-B * (x - M))) ** (1 / nu)

    def get_statistics(self):
        """
        Method to get minimum, median and maximum value from data.
        :return: minimum, median and maximum value.
        """
        return np.min(self.data), np.median(self.data), np.max(self.data)

    def find_Q0(self):
        """
        Method to find value of Q0.
        :return: infinity, Q0 or None.
        :raise ValueError, OverflowError: if there is an error in computation.
        """
        def equation(Q0):
            x_min, x_med, x_max = self.get_statistics()
            A = (x_max - x_med) / (x_min - x_med)
            B = np.log2(10)
            if Q0 <= 0 or (1 + Q0) ** B - 1 <= 0:
                return np.inf
            try:
                left_side = 1 / (1 + Q0 * np.exp((np.log((1 + Q0) ** B - 1) - np.log(Q0)) * A))
                right_side = 0.9 ** np.log2(1 + Q0)
                return left_side - right_side
            except (ValueError, OverflowError) as e:
                print(f"Error in computation: {e}!")
                return np.inf
        # use a bounded optimization to find the root in a specific range
        Q0_solution = fminbound(equation, 1e-5, 10)  # bound Q0 to positive range [1e-5, 10]
        return Q0_solution

    def initialize_parameters(self):
        """
        Method to initialize parameters for the GL function.
        :return: initialized parameters Q0, B0, M0, nu0.
        """
        x_min, x_med, x_max = self.get_statistics()
        M0 = x_med
        Q0 = self.find_Q0()
        B0 = (np.log(1 + Q0) ** np.log2(10) - 1) - np.log(Q0) / (x_med - x_min)
        nu0 = np.log2(1 + Q0)
        return Q0, B0, M0, nu0

    def objective(self, params):
        """
        Method to define an objective function to optimize.
        :param params: parameters of the GL function.
        :return: sum of distances between values for ECDF and GL function.
        """
        Q, B, M, nu = params
        gl_values = self.generalized_logistic(self.data_sorted, Q, B, M, nu)
        return np.sum((gl_values - self.ecdf_values_sorted) ** 2)

    def fit_gl_to_ecdf(self):
        """
        Method to fit the GL function to ECDF.
        :return: optimized parameters for the GL function.
        :raise RuntimeError: if optimization is failed.
        """
        initial_params = self.initialize_parameters()
        result = minimize(self.objective, initial_params, method='L-BFGS-B',
                          bounds=[(0.1, 10), (0.1, 10), (None, None), (0.1, 10)])
        if result.success:
            optimized_params = result.x
        else:
            raise RuntimeError("Optimization failed: " + result.message)
        return optimized_params

    def __call__(self):
        """
        Method to call the class.
        :return: normalized data values.
        """
        Q_opt, B_opt, M_opt, nu_opt = self.fit_gl_to_ecdf()
        # calculate generalized logistic values for sorted data
        gl_values_sorted = self.generalized_logistic(self.data_sorted, Q_opt, B_opt, M_opt, nu_opt)
        # restore generalized logistic values to original data order
        gl_values_original_order = np.empty_like(gl_values_sorted)
        gl_values_original_order[self.original_indices] = gl_values_sorted
        return gl_values_original_order

