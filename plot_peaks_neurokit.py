import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk

# Parse arguments
parser = argparse.ArgumentParser(description='Detect PPG peaks using NeuroKit2 and save chart.')
parser.add_argument('--csv', type=str, required=True, help='Subpath to CSV file, e.g. V19/V19_B15_PP.csv')
args = parser.parse_args()

# Paths
BASE_DIR = "/home/joachim/Desktop/DTW-ML/OneDrive_1_4-29-2025/preprocessed/macro"
csv_full_path = os.path.join(BASE_DIR, args.csv)
chart_dir = "charts_neurokit"
os.makedirs(chart_dir, exist_ok=True)

# Read and preprocess signal
df = pd.read_csv(csv_full_path, delimiter=';')
signal = 1000 - df["ABP"].values[:1000]
signal_cleaned = nk.ppg_clean(signal, sampling_rate=1000)
peaks = nk.ppg_findpeaks(signal_cleaned, sampling_rate=1000)["PPG_Peaks"]

# Print intervals
print(f"\n--- {args.csv} ---")
print("Intervals between peaks:")
for i in range(1, len(peaks)):
    print(int(peaks[i] - peaks[i - 1]))

# Plot and save
plt.figure(figsize=(10, 4))
plt.plot(signal_cleaned, label='Cleaned Signal')
plt.plot(peaks, signal_cleaned[peaks], "x", label='Peaks')
plt.title(f"Peaks for {args.csv}")
plt.xlabel('Sample')
plt.ylabel('Signal')
plt.legend()

filename = args.csv.replace('/', '_').replace('.csv', '.png')
output_path = os.path.join(chart_dir, filename)
plt.savefig(output_path)
plt.close()

print(f"Saved chart: {output_path}")
