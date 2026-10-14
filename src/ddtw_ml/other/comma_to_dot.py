"""
@author: Radoslaw Plawecki
Legacy: overwrites every raw file in place, replacing all commas (including the column separators) with dots.
Running it as a script is disabled.
"""

import os

data_path = "C:/Python/ZSSI/data/raw"


def main() -> None:
    directories = os.listdir(data_path)
    for directory in directories:
        directory_path = os.path.join(data_path, directory)
        files = os.listdir(directory_path)
        for file in files:
            print(f"=== Directory: {directory} | File: {file} ===")
            file_path = os.path.join(directory_path, file)
            with open(file_path, 'r') as r:
                lines = r.readlines()
                converted_lines = [line.strip().replace(',', '.') for line in lines]
            with open(file_path, 'w') as w:
                for line in converted_lines:
                    w.write(line + '\n')
            print("Conversion complete.")


if __name__ == "__main__":
    raise SystemExit("Legacy script disabled: it overwrites the raw data in place and breaks the files with comma "
                     "separators.")
