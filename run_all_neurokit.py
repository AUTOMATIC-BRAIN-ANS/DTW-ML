import subprocess
import os

base_folders = [f"V{i}" for i in range(1, 35)]
csv_variants = ["B6", "B10", "B15"]

for folder in base_folders:
    for variant in csv_variants:
        csv_filename = f"{folder}_{variant}_PP.csv"
        csv_path = f"{folder}/{csv_filename}"
        print(f"Running: {csv_path}")
        subprocess.run([
            "python3", "plot_peaks_neurokit.py",
            "--csv", csv_path
        ])
