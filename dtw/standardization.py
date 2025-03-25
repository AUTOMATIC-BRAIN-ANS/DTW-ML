"""
@author: Radoslaw Plawecki
"""

from os import path
import pandas as pd


def standardize(length, directory, filename):
    """
    Function to standardize data to a given length.
    :param length: expected length.
    :param directory: directory, where a file to standardize is located.
    :param filename: filename.
    :return: None.
    """
    filepath = f"patients/preprocessed/{directory}/{filename}.csv"
    if not path.exists(filepath):
        raise FileNotFoundError("File not found!")
    if not path.isfile(filepath):
        raise IsADirectoryError("The path exists but is not a file!")
    if path.splitext(filepath)[1] != '.csv':
        raise ValueError("File must be a CSV file!")
    data = pd.read_csv(filepath, delimiter=';')
    df = pd.DataFrame(data)
    datetime, s1, s2 = df["DateTime"], df[directory], df["Toxa"]
    standardized_data = {
        "DateTime": datetime[0:length],
        directory: s1[0:length],
        "Toxa": s2[0:length]
    }
    df = pd.DataFrame(standardized_data)
    df.to_csv(f"patients/standardized/{directory}/{filename}_S.csv", sep=';', index=False)
    print("Data was standardized!")
