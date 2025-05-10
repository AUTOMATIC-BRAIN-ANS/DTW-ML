"""
@author: Radoslaw Plawecki
"""

from project.preprocessing.preprocess_data import PreprocessData
import os

"""
base_dir = "C:/Python/ZSSI/data/preprocessed/macro"

os.makedirs(base_dir, exist_ok=True)

for i in range(1, 35):
    folder_name = f"V{i}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

print("Folders created successfully!")
"""

base_dir = "C:/Python/ZSSI/data/extracted/macro"
for directory in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, directory)
    for file in os.listdir(dir_path):
        filename = os.path.join(directory, file)
        filename_without_extension = os.path.splitext(filename)[0]
        prep = PreprocessData(filename_without_extension, first_column='ABP', second_column='CBFV')
        prep.export_preprocessed_data()
