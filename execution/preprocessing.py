"""
@author: Radoslaw Plawecki
"""

from project.data.preprocess_data import PreprocessData
from project.metrics.preprocess_metrics import PreprocessMetrics
import os

# DATA PREPROCESSING
"""base_dir = "C:/Python/ZSSI/data2/extracted"
# for directory in os.listdir(base_dir):
dir_path = os.path.join(base_dir, "B15")
for file in os.listdir(dir_path):
    filename = os.path.join("B15", file)
    filename_without_extension = os.path.splitext(filename)[0]
    print(filename)
    prep = PreprocessData(filename_without_extension, first_column='ABP', second_column='CBFV')
    prep.export_preprocessed_data()"""

# METRICS PREPROCESSING
base_dir = "C:/Python/ZSSI/data2/dtw/raw"
for file in os.listdir(base_dir):
    filename_without_extension = os.path.splitext(file)[0]
    print(filename_without_extension)
    prep = PreprocessMetrics(filename_without_extension)
    print(prep.export_preprocessed_metrics())
