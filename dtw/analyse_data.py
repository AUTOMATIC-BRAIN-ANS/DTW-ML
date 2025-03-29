"""
@author: Radoslaw Plawecki
"""

from project.common import check_column_existence, make_blocks
from dtw import DTW
import numpy as np
import pandas as pd
from os import listdir


class AnalyseData:
    def __block_analysis(self, directory, method):
        filenames = self.__get_filenames(directory)
        results = np.zeros([len(filenames), 3])
        for i in range(len(filenames)):
            filepath = f"patients/standardized/{directory}/{filenames[i]}"
            x, y = self.__get_data(filepath, x=directory, y='Toxa')
            for j in range(len(x)):
                dtw = DTW(x[j], y[j])
                results[i][j] = dtw.calc_alignment_cost(method=method)
        return results

    def __window_analysis(self, directory, method, window_size, step):
        filenames = self.__get_filenames(directory)
        results = np.zeros([len(filenames), 3])
        for i in range(len(filenames)):
            filepath = f"patients/standardized/{directory}/{filenames[i]}"
            x, y = self.__get_data(filepath, x=directory, y='Toxa')
            for j in range(len(x)):
                dtw = DTW(x[j], y[j])
                results[i][j] = dtw.find_alignment_cost(method=method, look_for="MEAN", window_size=window_size,
                                                        step=step)
        return results

    def analyze(self, directory, method, analysis=None, export_results=False):
        window_size = 6 * 60 * 60 // 10
        step = 1 * 60 * 60 // 10
        if analysis == 'block':
            results = self.__block_analysis(directory=directory, method=method)
        elif analysis == 'window':
            results = self.__window_analysis(directory=directory, method=method, window_size=window_size, step=step)
        else:
            raise ValueError(f"Allowed analysis are 'block' and 'window'! Got '{analysis}' instead.")
        if export_results:
            self.__export_results(analysis=analysis, directory=directory, method=method, results=results)

    @staticmethod
    def __get_filenames(directory):
        filepath = f"patients/standardized/{directory}"
        return listdir(filepath)

    @staticmethod
    def __get_data(filepath, x, y):
        data = pd.read_csv(filepath, delimiter=';')
        df = pd.DataFrame(data)
        check_column_existence(df=df, col=x)
        check_column_existence(df=df, col=y)
        return make_blocks(df[x]), make_blocks(df[y])

    @staticmethod
    def __export_results(analysis, directory, method, results):
        ad = AnalyseData()
        filenames = ad.__get_filenames(directory)
        files, first_block, second_block, third_block = "Plik", "I. blok", "II. blok", "III. blok"
        data = {
            files: [filename[0] for filename in filenames],
            first_block: pd.Series(results[:, 0]),
            second_block: pd.Series(results[:, 1]),
            third_block: pd.Series(results[:, 2])
        }
        df = pd.DataFrame(data)
        df.to_csv(f"patients/results/{analysis}/{directory}_{method}.csv", sep=';', index=False)
        print("Data was exported!")
