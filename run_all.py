import subprocess
import os

parameters = [3, 4, 5]
base_folders = [f"V{i}" for i in range(1, 35)]
csv_variants = ["B6", "B10", "B15"]

for folder in base_folders:
    for variant in csv_variants:
        csv_filename = f"{folder}_{variant}_PP.csv"
        csv_path = f"{folder}/{csv_filename}"
        for param in parameters:
            print(f"Running: {csv_path} with prominence {param}")
            subprocess.run([
                "python3", "plot_peaks.py",
                "--parameter", str(param),
                "--csv", csv_path
            ])
