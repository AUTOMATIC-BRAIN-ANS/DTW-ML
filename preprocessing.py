"""
@author: Radoslaw Plawecki
"""

from project.preprocessing.preprocess_data import PreprocessData
import os

base_dir = "C:/Python/ZSSI/data/extracted/macro"
# for directory in os.listdir(base_dir):
dir_path = os.path.join(base_dir, "BAS")
for file in os.listdir(dir_path):
    filename = os.path.join("BAS", file)
    filename_without_extension = os.path.splitext(filename)[0]
    prep = PreprocessData(filename_without_extension, first_column='ABP', second_column='CBFV')
    prep.export_preprocessed_data()
