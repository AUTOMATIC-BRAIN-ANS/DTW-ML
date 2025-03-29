"""
@author: Radoslaw Plawecki
"""

from project.preprocessing.preprocess_data import PreprocessData
import pandas as pd
from project.common import filter_abp

filename = 'V19/V19_B15_TEST'

prep = PreprocessData(filename, first_column='ABP', second_column='CBFV')
prep.plot_signals(s='second')

abp = prep.get_first_signal()[175:205].reset_index(drop=True)
