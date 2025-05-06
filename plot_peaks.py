import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parse arguments
parser = argparse.ArgumentParser(description='Detect peaks and save chart.')
parser.add_argument('--parameter', type=float, required=True, help='Prominence value')
parser.add_argument('--csv', type=str, required=True, help='Subpath to CSV file, e.g. V19/V19_B15_PP.csv')
args = parser.parse_args()

# Paths
BASE_DIR = "/home/joachim/Desktop/DTW-ML/OneDrive_1_4-29-2025/preprocessed/macro"
csv_full_path = os.path.join(BASE_DIR, args.csv)
chart_dir = f"charts/parameter_{int(args.parameter)}"
os.makedirs(chart_dir, exist_ok=True)

# Read data
df = pd.read_csv(csv_full_path, delimiter=';')
values = 1000 - df["ABP"].values

# Find peaks
peaks, properties = find_peaks(values[:1000], prominence=args.parameter)
print(f"\n--- {args.csv} | Prominence: {args.parameter} ---")
print("Properties:", properties)
print("Intervals between peaks:")
for i in range(1, len(peaks)):
    print(int(peaks[i] - peaks[i - 1]))

# Plot and save
plt.figure(figsize=(10, 4))
plt.plot(values[:1000], label='Signal')
plt.plot(peaks, values[peaks], "x", label='Peaks')
plt.title(f"Peaks for {args.csv} (prominence={args.parameter})")
plt.xlabel('Sample')
plt.ylabel('Signal')
plt.legend()

filename = args.csv.replace('/', '_').replace('.csv', f'_p{int(args.parameter)}.png')
output_path = os.path.join(chart_dir, filename)
plt.savefig(output_path)
plt.close()

print(f"Saved chart: {output_path}")
