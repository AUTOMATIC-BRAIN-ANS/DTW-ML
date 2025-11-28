"""
@author: Radosław Pławecki
"""

from project.dtw.dtw import DTW
import pandas as pd
import os

print("=== Starting DDTW ===")
data_path = "C:/Python/ZSSI/data/dtw/preprocessed"
breaths = os.listdir(data_path)
for breath in breaths:
    print(f"\n=== Directory: {breath} ===")
    files = os.listdir(os.path.join(data_path, breath))
    j = 1
    for file in files:
        print(f"{file} being processed...")
        try:
            data = pd.read_csv(os.path.join(data_path, breath, file), delimiter=';')
            df = pd.DataFrame(data)
            method = 'd-method'
            (abp_spo, abp_spp, abp_rr,
             cbfv_spo, cbfv_spp, cbfv_rr) = (df["ABP_SPO"], df["ABP_SPP"], df["ABP_RR"],
                                             df["CBFV_SPO"], df["CBFV_SPP"], df["CBFV_RR"])

            pair = [(abp_rr, cbfv_rr)]

            dtw_list = [DTW(x, y, var="DDTW") for x, y in pair]
            vals = []
            matrices = []
            windows = None
            for i, dtw in enumerate(dtw_list):
                cost, wds, matrices = dtw.sliding_window_dtw(window_size=10, step=5, method=method)
                vals.append(cost)
                if i == 0:
                    windows = wds

            for matrix in matrices:
                df = pd.DataFrame(matrix)
                df.to_csv(f"C:/Python/ZSSI/data/ml-data/ABP_RR-CBFV_RR/{breath}/{j}.csv", sep=';',
                          header=False, index=False)
                j += 1
        except Exception as e:
            print(f"✖ Failed ({e})")
        finally:
            print(f"✔ Success")
    print(f"✔ Success. Extracted {j - 1} windows for {breath}.")
print("\n=== DDTW completed ===")
