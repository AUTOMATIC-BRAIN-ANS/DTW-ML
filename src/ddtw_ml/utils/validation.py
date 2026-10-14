"""
Checks of the input files and their columns.
"""

from os import path


def check_column_existence(df, col):
    """
    Function to check if a column with the given name exists in a file.
    :param df: data as DataFrame object.
    :param col: name of a column to look for.
    :raise KeyError: if a column doesn't exist in a file.
    """
    if not list(df.columns).__contains__(col):
        raise KeyError(f"Column '{col}' doesn't exist in a file!")


def check_path(filepath):
    if not path.exists(filepath):
        raise FileNotFoundError("File not found!")
    if not path.isfile(filepath):
        raise IsADirectoryError("The path exists but is not a file!")
    if path.splitext(filepath)[1] != '.csv':
        raise ValueError("File must be a CSV file!")
