"""
@author: Radoslaw Plawecki
"""

from project.data.preprocess_data import PreprocessData
from project.metrics.preprocess_metrics import PreprocessMetrics
import shutil
import os

# DATA PREPROCESSING
"""print("=== Starting preprocessing ===")
base_dir = "C:/Python/ZSSI/data/extracted"
error_dir = "C:/Python/ZSSI/data/spoiled"
for directory in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, directory)
    print(f"\n=== Directory: {directory} ===")
    for file in os.listdir(dir_path):
        print(f"-> Processing: {file}")
        filename = os.path.join(directory, file)
        filename_without_extension = os.path.splitext(filename)[0]
        try:
            prep = PreprocessData(filename_without_extension, first_column='ABP', second_column='CBFV')
            prep.export_preprocessed_data()
            print("✔ Success")
        except Exception as e:
            print(f"✖ Failed ({e})")
            dst = os.path.join(error_dir, f"{filename_without_extension}.csv")
            file_path = os.path.join(dir_path, file)
            shutil.move(file_path, dst)
            print(f"Moved to {dst}")
print("\n=== Preprocessing completed ===")"""

# METRICS PREPROCESSING
print("=== Starting preprocessing ===")
base_dir = "C:/Python/ZSSI/data/dtw/raw"
error_dir = "C:/Python/ZSSI/data/spoiled"
directories = os.listdir(base_dir)
for directory in directories:
    dir_path = os.path.join(base_dir, directory)
    print(f"\n=== Directory: {directory} ===")
    for file in os.listdir(dir_path):
        filename_without_extension = os.path.splitext(file)[0]
        try:
            prep = PreprocessMetrics(directory=directory, filename=filename_without_extension)
            prep.export_preprocessed_metrics()
            print("✔ Success")
        except Exception as e:
            print(f"✖ Failed ({e})")
            dst = os.path.join(error_dir, directory, f"{filename_without_extension}.csv")
            file_path = os.path.join(dir_path, file)
            shutil.move(file_path, dst)
            print(f"Moved to {dst}")
print("\n=== Preprocessing completed ===")
