"""
@author: Radoslaw Plawecki
Sources:
https://medium.com/@nirajan.acharya777/understanding-outlier-removal-using-interquartile-range-iqr-b55b9726363e
"""

from project.common import use_latex, filter_abp, check_column_existence, check_path
from project.utils.preprocessing_utils import PreprocessingUtils as prepUtils
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
        filepath = f"C:/Python/ZSSI/data/extracted/{filename}.csv"
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

    def remove_outliers(self, threshold=1.5):
        """
        Method to remove outliers using the method iqr_outlier_removal().
        :param threshold: threshold after which the outliers will be removed.
        :return: signals with removed outliers.
        """
        s1, s2 = self.get_first_signal(), self.get_second_signal()
        return prepUtils.iqr_outlier_removal(s1, threshold), prepUtils.iqr_outlier_removal(s2, threshold)

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

    def trim_signals(self):
        """
        Method to trim signals using the trim() function.
        :return: signals trimmed.
        """
        s1, s2 = self.get_first_signal_outliers_removed(), self.get_second_signal_outliers_removed()
        start, stop = prepUtils.trim(s1, s2)
        return pd.Series(s1[start:stop]), pd.Series(s2[start:stop])

    def get_first_signal_trimmed(self):
        """
        Getter to get a first signal trimmed.
        :return: first signal trimmed.
        """
        return self.trim_signals()[0]

    def get_second_signal_trimmed(self):
        """
        Getter to get a second signal trimmed.
        :return: second signal trimmed.
        """
        return self.trim_signals()[1]

    def interpolate_signal(self, method=None, order=None):
        """
        Method to interpolate signals using a specific method of interpolation.
        :param method: method of interpolation.
        :param order: if a method is polynomial, then its order.
        :return: interpolated signals.
        """
        s1, s2 = self.get_first_signal_trimmed(), self.get_second_signal_trimmed()
        return prepUtils.interpolate_data(s1, method=method, order=order), prepUtils.interpolate_data(s2, method=method, order=order)

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

    def normalize_signal(self, method, min_value=-1, max_value=1):
        """
        Method to normalize signals using a specific method of normalization.
        :param method: method of normalization.
        :param min_value: a minimum value.
        :param max_value: a maximum value.
        :return: normalized signals.
        """
        s1, s2 = self.get_first_signal_interpolated(), self.get_second_signal_interpolated()
        return (prepUtils.normalize(s1, method=method, min_value=min_value, max_value=max_value),
                prepUtils.normalize(s2, method=method, min_value=min_value, max_value=max_value))

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
        return prepUtils.smooth(s1), prepUtils.smooth(s2)

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
                       self.get_first_signal_interpolated(), self.get_second_signal_normalized()]
        elif s == 'second':
            signals = [self.get_second_signal(), self.get_second_signal_outliers_removed(),
                       self.get_second_signal_interpolated(), self.get_second_signal_normalized()]
        else:
            raise ValueError(f"Allowed signal are 'first' and 'second'! Got {s} instead.")
        timeseries = [self.get_time(signals[0]), self.get_time(signals[1]),
                      self.get_time(signals[2]), self.get_time(signals[3])]
        titles = ["(a) Sygnał nieprzetworzony", "(b) Sygnał po usunięciu wartości odstających",
                  "(c) Sygnał zinterpolowany", "(d) Sygnał po normalizacji"]
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
        s1, s2 = self.get_first_signal_normalized(), self.get_second_signal_normalized()
        datetime_values = np.linspace(0, len(s1), len(s1))
        data = {
            datetime: datetime_values,
            col1: s1,
            col2: s2
        }
        df = pd.DataFrame(data)
        df.to_csv(f"C:/Python/ZSSI/data/preprocessed/{self.filename}_PP.csv", sep=';', index=False)
