import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from algorithm_file import detect_peaks_troughs
from scipy.signal import find_peaks

BASE_DIR = r"C:\Users\Joachim\Desktop\PNW\DTW-ML\OneDrive_1_10.05.2025\preprocessed\macro"
OUTPUT_DIR = "results"
CHARTS_DIR = "charts"
SAMPLING_RATE = 200
ALLOWED_FOLDERS = ["V1"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


def find_dn(signal):
    signal = np.array(signal)
    if len(signal) < 10:
        return None, None

    systolic_idx = np.argmax(signal)
    search_window = signal[systolic_idx + 1:]

    inv = -search_window
    peaks, _ = find_peaks(inv, distance=5, prominence=0.5)

    if len(peaks) == 0:
        return None, None

    dn_idx_rel = peaks[0]
    dn_idx = systolic_idx + 1 + dn_idx_rel
    dn_value = signal[dn_idx]
    return dn_value, dn_idx


def find_dpp(signal, dn_idx, end_idx):
    if end_idx - dn_idx < 3:
        return None, None

    segment = signal[dn_idx:end_idx]
    dpp_idx_rel = np.argmax(segment)
    dpp_idx = dn_idx + dpp_idx_rel
    dpp_value = signal[dpp_idx]
    return dpp_value, dpp_idx



def convert_array(array):
    array = np.array(array)
    return [element[0] for element in array]

for folder in sorted(os.listdir(BASE_DIR)):
    if not os.path.isdir(os.path.join(BASE_DIR, folder)):
        continue
    if folder not in ALLOWED_FOLDERS:
        continue

    folder_path = os.path.join(BASE_DIR, folder)

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, file)
        print(f"Przetwarzanie: {file_path}")
        df = pd.read_csv(file_path, delimiter=';')
        abp = df["ABP"]
        abp = abp[0:3000]

        # Znajdź minima w całym sygnale
        _, troughs = detect_peaks_troughs(abp)
        troughs = convert_array(troughs)

        start_x_list, end_x_list = [], []
        spo_list, spp_list, dn_list, dpp_list, dpe_list = [], [], [], [], []
        spp_index_list, dn_index_list, dpp_index_list = [], [], []


        for i in range(len(troughs) - 1):
            start, end = int(troughs[i]), int(troughs[i + 1])
            if end > len(abp):
                continue

            window = abp[start:end]
            if len(window) < 3:
                continue
            
            # SPP
            spp = np.max(window)
            spp_idx_rel = np.argmax(window)
            spp_idx = start + spp_idx_rel
            spp_list.append(spp)
            spp_index_list.append(spp_idx)

            # SPO — min value before SPP in a small range
            spo_search_start = max(start, spp_idx - 10)
            spo = np.min(abp[spo_search_start:spp_idx])
            spo_list.append(spo)

            # DN
            dn, dn_rel_idx = find_dn(window)

            if dn is not None:
                dn_idx = start + dn_rel_idx
                dn_list.append(dn)
                dn_index_list.append(dn_idx)
            else:
                dn_list.append(None)
                dn_index_list.append(None)
                continue
            
            dpp, dpp_idx = find_dpp(abp, dn_idx, end-10)
            if dpp is not None:
                dpp_list.append(dpp)
                dpp_index_list.append(dpp_idx)
            else:
                dpp_list.append(None)
                dpp_index_list.append(None)


            # DPE — min near end
            dpe_search_start = max(0, end - 10)
            dpe_search_end = min(len(abp), end + 10)
            dpe = np.min(abp[dpe_search_start:dpe_search_end])
            dpe_list.append(dpe)

            start_x_list.append(start)
            end_x_list.append(end)


        if len(start_x_list) == 0:
            print(f"⚠️ Brak wystarczających okien w pliku {file}, pomijam.")
            continue
        
        result_df = pd.DataFrame({
            "start_x": start_x_list,
            "end_x": end_x_list,
            "SPO": spo_list,
            "SPP": spp_list,
            "DN": dn_list,
            "DPP": dpp_list,
            "DPE": dpe_list
        })

        result_filename = file.replace(".csv", "_results.csv")
        result_df.to_csv(os.path.join(OUTPUT_DIR, result_filename), index=False)

        # Plot only if signal is long enough
        abp_plot = abp[:1000]

        plot_start_x = [s for s in start_x_list if s < 1000]
        plot_start_y = [abp_plot[s] for s in plot_start_x]

        plot_end_x = [e for e in end_x_list if e < 1000]
        plot_end_y = [abp_plot[e] for e in plot_end_x]

        plot_spp_x = [i for i in spp_index_list if i is not None and i < 1000]
        plot_spp_y = [abp_plot[i] for i in plot_spp_x]

        plot_dn_x = [i for i in dn_index_list if i is not None and i < 1000]
        plot_dn_y = [abp_plot[i] for i in plot_dn_x]

        plot_dpp_x = [i for i in dpp_index_list if i is not None and  i < 1000]
        plot_dpp_y = [abp_plot[i] for i in plot_dpp_x]

        print(dpp_index_list)
        print(dpp_list)
        print(plot_dpp_x)
        print(plot_dpp_y)
        plt.figure(figsize=(12, 4))
        plt.plot(abp_plot, label="ABP cleaned")
        plt.scatter(plot_start_x, plot_start_y, color='red', label='Start (min)', marker='o')
        plt.scatter(plot_end_x, plot_end_y, color='blue', label='End (min)', marker='x')
        plt.scatter(plot_spp_x, plot_spp_y, color='green', label='SPP', marker='v')
        plt.scatter(plot_dn_x, plot_dn_y, color='orange', label='DN', marker='^')
        plt.scatter(plot_dpp_x, plot_dpp_y, color='purple', label='DPP', marker='s')


        plt.title(f"{file} - first 1000 samples")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()
        chart_path = os.path.join(CHARTS_DIR, f"{file.replace('.csv', '_chart2.pdf')}")
        plt.savefig(chart_path)
        plt.close()

print("✅ Wszystkie pliki zostały podzielone na okna, przeanalizowane i zapisane.")
