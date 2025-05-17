"""
@author: Radoslaw Plawecki
Sources:
[1] Kamper, H. (2021). Dynamic time warping 2: Algorithm [Video]. YouTube.
Available on: https://www.youtube.com/watch?v=X6phfLqN5pY&list=PLmZlBIcArwhMJoGk5zpiRlkaHUqy5dLzL&index=3.
Access: 29.B10.2024.
[2] Kamper, H. (2021). Dynamic time warping (DTW) tutorial notebook. GitHub. Available on:
https://github.com/kamperh/lecture_dtw_notebook/blob/main/dtw.ipynb. Access: 29.B10.2024.
"""

from project.common import use_latex
import numpy as np
import matplotlib.pyplot as plt


class DTW:
    __cost_matrix = []
    __matches, __insertions, __deletions = 0, 0, 0
    __path = []

    def __init__(self, x, y):
        """
        Method to initialize params of a class.
        :param x: first signal.
        :param y: second signal.
        """
        self.x = np.array(x)
        self.y = np.array(y)

    def __initialize_matrix(self):
        """
        Method to initialize a matrix.
        :return: initialized matrix.
        """
        x, y = self.x, self.y
        rows, cols = len(x) + 1, len(y) + 1
        matrix = np.zeros([rows, cols])
        matrix[0, 1:], matrix[1:, 0], matrix[0, 0] = np.inf, np.inf, 0
        return matrix

    def fill_matrix(self):
        """
        Method to fill a matrix.
        :return: filled matrix.
        """
        x, y = self.x, self.y
        matrix = self.__initialize_matrix()
        rows, cols = np.shape(matrix)
        for i in range(1, rows):
            for j in range(1, cols):
                distance = abs(x[i - 1] - y[j - 1])
                component = np.min([matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i][j - 1]])
                matrix[i][j] = distance + component
        return matrix

    def __init_global_variables(self, i, j):
        """
        Method to initialize global variables.
        :param i: number of rows in a matrix.
        :param j: number of columns in a matrix.
        :return: initialized global variables.
        """
        self.__cost_matrix = self.fill_matrix()
        self.__matches, self.__insertions, self.__deletions = 0, 0, 0
        self.__path = [(i - 1, j - 1)]

    def traceback(self):
        """
        Method get traceback matrix.
        :return: traceback matrix.
        """
        x, y = self.x, self.y
        i, j = rows, cols = len(x), len(y)
        self.__init_global_variables(i, j)
        traceback_matrix = np.zeros([rows + 1, cols + 1])
        cost_matrix = self.__cost_matrix
        while i > 0 and j > 0:
            score = cost_matrix[i][j]
            distance = abs(x[i - 1] - y[j - 1])
            match, insertion, deletion = [cost_matrix[i - 1][j - 1],
                                          cost_matrix[i - 1][j],
                                          cost_matrix[i][j - 1]]
            if score == distance + match:
                traceback_matrix[i][j] = 1
                self.__matches += 1
                i -= 1
                j -= 1
            elif score == distance + insertion:
                traceback_matrix[i][j] = 1
                self.__insertions += 1
                i -= 1
            else:
                traceback_matrix[i][j] = 1
                self.__deletions += 1
                j -= 1
            self.__path.append((i - 1, j - 1))
        return traceback_matrix

    def get_statistics(self):
        """
        Method to get statistics of DTW.
        :return: number of matches, insertions and deletions.
        """
        self.traceback()
        return self.__matches, self.__insertions, self.__deletions

    def calc_alignment_cost(self, method):
        """
        Method to calculate alignment cost using a given method.
        :param method: method to calculate alignment cost.
        :return: alignment cost.
        """
        if method == "d-method":
            return self.__use_distance_method()
        elif method == "td-method":
            return self.__use_time_distance_method()
        elif method == "c-method":
            return self.__use_cost_method()
        else:
            raise ValueError(f"Allowed methods to calculate alignment cost are: 'd-method', 'td-method' and "
                             f"'c-method'. Got '{method}' instead.")

    def __use_distance_method(self):
        """
        Method to calculate alignment cost using the distance method.
        :return: alignment cost.
        """
        matrix = self.fill_matrix()[1:, 1:]
        n, m = np.shape(matrix)
        return matrix[n - 1][m - 1] / (n + m)

    def __use_time_distance_method(self):
        """
        Method to calculate alignment cost using the time-distance method.
        :return: alignment cost.
        """
        self.traceback()
        matrix = self.__cost_matrix[1:, 1:]
        cost = sum(matrix[n, m] for n, m in self.__path[:-1])
        len_traceback = len(self.__path[:-1])
        return cost / len_traceback

    def __use_cost_method(self):
        """
        Method to calculate alignment cost using the cost method.
        :return: alignment cost.
        """
        matches, insertions, deletions = self.get_statistics()
        len_traceback = matches + insertions + deletions
        return (insertions + deletions) / len_traceback

    def __sliding_window_dtw(self, window_size, step, method):
        """
        Method to implement DTW with sliding window.
        :param window_size: size of a window.
        :param step: step between windows.
        :param method: method to calculate alignment cost.
        :return: list with alignment costs per window and the list with analyzed windows.
        """
        if window_size < 5:
            raise ValueError("Window is not big enough!")
        if step <= 0:
            raise ValueError("Step must have a positive value!")
        x, y = self.x, self.y
        windows = []
        alignment_costs = []
        for i in range(0, max(len(x), len(y)) - window_size + 1, step):
            window = [i, window_size + i]
            dtw = DTW(x[window[0]:window[1]], y[window[0]:window[1]])
            alignment_cost = dtw.calc_alignment_cost(method=method)
            windows.append(window)
            alignment_costs.append(alignment_cost)
        return alignment_costs, windows

    def __perform_dtw_window(self, windows, pos, filename=None):
        """
        Method to perform DTW on a specific window.
        :param windows: list of analyzed windows.
        :param pos: index of a specific window from the list.
        :param filename: name of a file to save plots.
        """
        dtw = DTW(self.x[windows[pos][0]:windows[pos][1]], self.y[windows[pos][0]:windows[pos][1]])
        dtw.traceback()
        dtw.__make_plots(x_signal='x', y_signal='y', filename=filename)

    def __get_min_max_alignment_cost(self, min_max, window_size, step, method):
        """
        Method to get minimum or maximum alignment cost from a list.
        :param min_max: MIN/MAX, whether to look for minimum or maximum value.
        :param window_size: size of a window.
        :param step: step between windows.
        :param method: method to calculate alignment cost.
        :return: list with alignment costs per window and the list with analyzed windows, and position of a window with
                 minimum or maximum alignment cost.
        """
        alignment_costs, windows = self.__sliding_window_dtw(window_size, step, method)
        alignment_cost = 0
        if min_max == "MIN":
            alignment_cost = np.min(alignment_costs)
        elif min_max == "MAX":
            alignment_cost = np.max(alignment_costs)
        return alignment_cost, windows, np.where(alignment_costs == alignment_cost)[0]

    def __get_min_alignment_cost(self, window_size, step, method, filename=None):
        """
        Method to get minimum alignment cost together with the plots.
        :param window_size: size of a window.
        :param step: step between windows.
        :param method: method to calculate alignment cost.
        :param filename: name of a file to save plots.
        :return: minimum alignment cost.
        """
        alignment_cost, windows, positions = self.__get_min_max_alignment_cost(min_max="MIN", window_size=window_size,
                                                                               step=step, method=method)
        for position in positions:
            self.__perform_dtw_window(windows, position, filename)
        return alignment_cost

    def __get_max_alignment_cost(self, window_size, step, method, filename=None):
        """
        Method to get maximum alignment cost together with the plots.
        :param window_size: size of a window.
        :param step: step between windows.
        :param method: method to calculate alignment cost.
        :param filename: name of a file to save plots.
        :return: maximum alignment cost.
        """
        alignment_cost, windows, positions = self.__get_min_max_alignment_cost(min_max="MAX", window_size=window_size,
                                                                               step=step, method=method)
        for position in positions:
            self.__perform_dtw_window(windows, position, filename)
        return alignment_cost

    def __get_mean_alignment_cost(self, window_size, step, method):
        """
        Method to get mean alignment cost.
        :param window_size: size of a window.
        :param step: step between windows.
        :param method: method to calculate alignment cost.
        :return: mean alignment cost.
        """
        alignment_costs, windows = self.__sliding_window_dtw(window_size, step, method)
        return np.mean(alignment_costs)

    def find_alignment_cost(self, method, look_for, window_size=10, step=1, filename=None):
        """
        Method to find minimum, maximum or mean alignment cost.
        :param method: method to calculate alignment cost.
        :param look_for: value to look for (MIN/MAX/MEAN).
        :param window_size: size of a window.
        :param step: step between windows.
        :param filename: name of a file to save plots.
        :return: minimum, maximum or mean alignment cost.
        :raise ValueError: if value for look_for is not an expected one.
        """
        if look_for == "MEAN":
            return self.__get_mean_alignment_cost(window_size, step, method)
        elif look_for == "MIN":
            return self.__get_min_alignment_cost(window_size, step, method, filename)
        elif look_for == "MAX":
            return self.__get_max_alignment_cost(window_size, step, method, filename)
        else:
            raise ValueError(f"Allowed statistics are 'MIN', 'MAX' and 'MEAN'! Got {look_for} instead.")

    def __make_plots(self, x_signal=None, y_signal=None, filename=None):
        """
        Method to make plots visualizing the results.
        :param x_signal: label for the x-signal.
        :param y_signal: label for the y-signal.
        :param filename: name of a file to save plots.
        """
        if filename is None:
            self.plot_signals(x_signal=x_signal, y_signal=y_signal)
            self.plot_alignment()
            self.plot_cost_matrix(x_signal=x_signal, y_signal=y_signal)
        else:
            self.plot_signals(x_signal=x_signal, y_signal=y_signal, filename=f"s_{filename}")
            self.plot_alignment(filename=f"d_{filename}")
            self.plot_cost_matrix(x_signal=x_signal, y_signal=y_signal, filename=f"m_{filename}")

    def plot_signals(self, x_signal=None, y_signal=None, filename=None):
        """
        Method to plot signals in the 1x2 grid.
        :param x_signal: label for the x-signal.
        :param y_signal: label for the y-signal.
        :param filename: name of a file to save a plot.
        """
        use_latex()
        x, y = self.x, self.y
        start = 0
        label_pad = 8
        tx_stop = tx_num = len(x)
        ty_stop = ty_num = len(y)
        tx = np.linspace(start=start, stop=tx_stop, num=tx_num)
        ty = np.linspace(start=start, stop=ty_stop, num=ty_num)
        fig, ax = plt.subplots(2)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.55)
        ax[0].plot(tx, x)
        ax[1].plot(ty, y)
        # add a title and labels to the 1st plot
        ax[0].set_xlabel("$t$ [s]")
        ax[0].set_ylabel(f"${x_signal}$ [a.u.]", labelpad=label_pad)
        ax[0].set_title(f"Zależność ${x_signal}(t)$", )
        ax[0].set_xlim(xmin=0, xmax=tx_stop)
        ax[0].grid()
        # add a title and labels to the 2nd plot
        ax[1].set_xlabel("$t$ [s]", labelpad=label_pad)
        ax[1].set_ylabel(f"${y_signal}$ [a.u.]", labelpad=label_pad)
        ax[1].set_title(f"Zależność ${y_signal}(t)$")
        ax[1].set_xlim(xmin=0, xmax=ty_stop)
        ax[1].grid()
        if filename is not None:
            plt.savefig(f"{filename}.pdf", format='pdf')
        plt.show()

    def plot_cost_matrix(self, x_signal=None, y_signal=None, filename=None):
        """
        Method to plot a cost matrix.
        :param x_signal: label for the x-signal.
        :param y_signal: label for the y-signal.
        :param filename: name of a file to save a plot.
        """
        use_latex()
        label_pad = 8
        c = plt.imshow(self.fill_matrix()[1:, 1:], cmap=plt.get_cmap("Blues"), interpolation="nearest", origin="upper")
        plt.colorbar(c)
        x_path, y_path = zip(*self.__path[:-1])
        plt.plot(y_path, x_path, color="#003A7D", linewidth=1.5)
        plt.title("Macierz kosztów")
        plt.xlabel(f"{x_signal}", labelpad=label_pad)
        plt.ylabel(f"{y_signal}", labelpad=label_pad)
        plt.legend(['Ścieżka dopasowania'])
        if filename is not None:
            plt.savefig(f"{filename}.pdf", format='pdf')
        plt.show()

    def plot_alignment(self, filename=None):
        """
        Method to plot signals with alignment.
        :param filename: name of a file to save a plot.
        """
        use_latex()
        x, y = self.x, self.y
        for x_i, y_j in self.__path[:-1]:
            plt.plot([x_i, y_j], [x[x_i] + 1.5, y[y_j] - 1.5], c="C7")
        plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
        plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
        plt.axis("off")
        if filename is not None:
            plt.savefig(f"{filename}.pdf", format='pdf')
        plt.show()
