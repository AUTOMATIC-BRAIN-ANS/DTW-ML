"""
@author: Radoslaw Plawecki
Sources:
https://medium.com/@nirajan.acharya777/understanding-outlier-removal-using-interquartile-range-iqr-b55b9726363e
"""

from project.common import (use_latex, values_in_order, filter_abp, check_column_existence, check_path,
                            smooth_data)
from project.preprocessing.normalization import NormalizeData
from project.preprocessing.nan_handler import NaNHandler as NaNH
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PreprocessData:
    def __init__(self, filename, first_column, second_column):
        """
        Method to initialize params of a class.
        :param filename: name of a file.
        :param first_column: name of a first column to analyse.
        :param second_column: name of a second column to analyse.
        :raise FileNotFoundError: if a file is not found.
        :raise IsADirectoryError: if a path exists but leads to, for example, directory instead of a file.
        :raise ValueError: if a file is not a CSV file.
        """
        filepath = f"C:/Python/ZSSI/data/extracted/macro/{filename}.csv"
        check_path(filepath)
        self.filename = filename
        data = pd.read_csv(filepath, delimiter=';')
        df = pd.DataFrame(data)
        check_column_existence(df=df, col=first_column)
        check_column_existence(df=df, col=second_column)
        self.first_column, self.second_column = first_column, second_column
        s1, s2 = self.__assign_signals(df=df, first_column=first_column,
                                       second_column=second_column)
        self.first_signal, self.second_signal = s1, s2

    def get_first_signal(self):
        """
        Getter to get a first, initialized signal.
        :return: first signal.
        """
        return self.first_signal

    def get_second_signal(self):
        """
        Getter to get a second, initialized signal.
        :return: second signal.
        """
        return self.second_signal

    @staticmethod
    def __assign_signals(df, first_column, second_column):
        """
        Method to assign signals to variables based on the specified column names, involving looking for artefacts in
        the CBFV signal.
        :param df: data.
        :param first_column: first column name.
        :param second_column: second column name.
        :return: signals to be assigned.
        """
        if second_column == 'CBFV':
            s1, s2 = filter_abp(df, col_abp='ABP'), df[second_column]
        elif first_column == 'CBFV':
            s1, s2 = filter_abp(df, col_abp='ABP'), df[second_column]
        else:
            s1, s2 = df[first_column], df[second_column]
        return s1, s2

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
        for i in range(1, len(s)):
            if s[i] < lower_bound or s[i] > upper_bound:
                s[i] = np.nan
        return s

    def remove_outliers(self, threshold=1.5):
        """
        Method to remove outliers using the method iqr_outlier_removal().
        :param threshold: threshold after which the outliers will be removed.
        :return: signals with removed outliers.
        """
        s1, s2 = self.get_first_signal(), self.get_second_signal()
        return self.iqr_outlier_removal(s1, threshold), self.iqr_outlier_removal(s2, threshold)

    def get_first_signal_outliers_removed(self):
        """
        Getter to get a first signal with removed outliers.
        :return: first signal without outliers.
        """
        return self.remove_outliers()[0]

    def get_second_signal_outliers_removed(self):
        """
        Getter to get a second signal with removed outliers.
        :return: second signal without outliers.
        """
        return self.remove_outliers()[1]

    @staticmethod
    def trim_signals(s1, s2):
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
            if count > 300:
                if abs(start - loc) < abs(loc - stop):
                    start = loc + count
                else:
                    stop = loc
        return start, stop

    @staticmethod
    def __fill_nans(s1, s2, method=None, order=None):
        """
        Method to fill NaNs in signals using a specific method of interpolation.
        :param s1: first signal.
        :param s2: second signal.
        :param method: method of interpolation.
        :param order: if a method is polynomial, then its order.
        :return: signals with filled NaNs.
        :raise ValueError: if a signal has too many gaps.
        """
        ppd = PreprocessData
        start, stop = ppd.trim_signals(s1, s2)
        s1, s2 = pd.Series(s1[start:stop]), pd.Series(s2[start:stop])
        s1, s2 = s1.interpolate(method=method, order=order), s2.interpolate(method=method, order=order)
        if len(s1) < 600 or len(s2) < 600:
            raise ValueError("Signal has too many gaps!")
        else:
            return s1, s2

    def interpolate_signal(self, method=None, order=None):
        """
        Method to interpolate signals using a specific method of interpolation.
        :param method: method of interpolation.
        :param order: if a method is polynomial, then its order.
        :return: interpolated signals.
        """
        s1, s2 = self.get_first_signal_outliers_removed(), self.get_second_signal_outliers_removed()
        try:
            return self.__fill_nans(s1, s2, method=method, order=order)
        except ValueError as e:
            print(f"Error occurred: {e}")

    def get_first_signal_interpolated(self):
        """
        Getter to get a first, interpolated signal.
        :return: first, interpolated signal.
        """
        return self.interpolate_signal(method='linear')[0]

    def get_second_signal_interpolated(self):
        """
        Getter to get a second, interpolated signal.
        :return: second, interpolated signal.
        """
        return self.interpolate_signal(method='linear')[1]

    def normalize_signal(self, method):
        """
        Method to normalize signals using a specific method of normalization.
        :param method: method of normalization.
        :return: normalized signals.
        """
        s1, s2 = self.get_first_signal_interpolated(), self.get_second_signal_interpolated()
        nd1, nd2 = NormalizeData(s1), NormalizeData(s2)
        return (nd1.normalize(method=method, min_value=-1, max_value=1),
                nd2.normalize(method=method, min_value=-1, max_value=1))

    def get_first_signal_normalized(self):
        """
        Getter to get a first, preprocessed signal.
        :return: first, preprocessed signal.
        """
        return self.normalize_signal(method='generalized-logistic')[0]

    def get_second_signal_normalized(self):
        """
        Getter to get a second, preprocessed signal.
        :return: second, preprocessed signal.
        """
        return self.normalize_signal(method='generalized-logistic')[1]

    def smooth_signal(self):
        """
        Method to smooth signals using the Savitzky-Golay filter.
        :return: smoothed signal.
        """
        s1, s2 = self.get_first_signal_interpolated(), self.get_second_signal_interpolated()
        sm1, sm2 = smooth_data(s1), smooth_data(s2)
        return sm1, sm2

    def get_first_signal_smoothed(self):
        """
        Getter to get a first, preprocessed signal.
        :return: first, preprocessed signal.
        """
        return self.smooth_signal()[0]

    def get_second_signal_smoothed(self):
        """
        Getter to get a second, preprocessed signal.
        :return: second, preprocessed signal.
        """
        return self.smooth_signal()[1]

    @staticmethod
    def get_time(s):
        """
        Method to get time series based on a signal.
        :param s: signal.
        :return: time.
        """
        return np.linspace(0, len(s), len(s))

    def get_data_for_plot(self, s=None):
        """
        Method to get data for a plot to compare signals on different steps of preprocessing.
        :param s: signal.
        :return: timeseries, signal and title for each signal.
        """
        if s == 'first':
            signals = [self.get_first_signal(), self.get_first_signal_outliers_removed(),
                       self.get_first_signal_interpolated(), self.get_first_signal_smoothed()]
        elif s == 'second':
            signals = [self.get_second_signal(), self.get_second_signal_outliers_removed(),
                       self.get_second_signal_interpolated(), self.get_second_signal_smoothed()]
        else:
            raise ValueError(f"Alloweds signal are 'first' and 'second'! Got {s} instead.")
        timeseries = [self.get_time(signals[0]), self.get_time(signals[1]),
                      self.get_time(signals[2]), self.get_time(signals[3])]
        titles = ["(a) Sygnał nieprzetworzony", "(b) Sygnał po usunięciu wartości odstających",
                  "(c) Sygnał zinterpolowany", "(d) Sygnał po wygładzeniu"]
        return timeseries, signals, titles

    def plot_signals(self, s=None, filename=None):
        """
        Method to plot a signal on different steps of preprocessing.
        :param s: signal.
        :param filename: name of a file with a plot.
        :return: None, if a signal is incorrect.
        """
        use_latex()
        try:
            timeseries, signals, titles = self.get_data_for_plot(s)
        except ValueError as e:
            print(f"Error occurred: {e}")
            return None
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.65)
        k = 0
        label_pad = 12
        label_fontsize = 12
        for i in range(2):
            for j in range(2):
                ax[i, j].plot(timeseries[k], signals[k])
                ax[i, j].set_title(titles[k], pad=label_pad, fontsize=label_fontsize + 2)
                ax[i, j].set_xlabel("Czas [s]", labelpad=label_pad, fontsize=label_fontsize)
                ax[i, j].set_ylabel("Amplituda [a.u.]", labelpad=label_pad, fontsize=label_fontsize)
                ax[i, j].set_xlim(xmin=0, xmax=max(timeseries[k]))
                ax[i, j].grid()
                k += 1
        if filename is not None:
            plt.savefig(f"C:/Python/ZSSI/plots/preprocessing/{filename}.pdf", format='pdf')
        plt.show()

    def export_preprocessed_data(self):
        """
        Method to export preprocessed data.
        :return: None.
        """
        print(f"File: {self.filename} being processed...")
        datetime, col1, col2 = "DateTime", self.first_column, self.second_column
        s1, s2 = self.get_first_signal_smoothed(), self.get_second_signal_smoothed()
        datetime_values = np.linspace(0, len(s1), len(s1))
        data = {
            datetime: datetime_values,
            col1: s1,
            col2: s2
        }
        df = pd.DataFrame(data)
        df.to_csv(f"C:/Python/ZSSI/data/preprocessed/macro/{self.filename}_PP.csv", sep=';', index=False)
        print("Data was exported!")
