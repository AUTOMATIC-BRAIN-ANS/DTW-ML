"""
@author: Radosław Plawecki
"""

from project.dtw.dtw import DTW
import pandas as pd
import os

"""data_path = "C:/Python/ZSSI/data2/dtw/preprocessed"
files = os.listdir(data_path)
for file in files:
    data = pd.read_csv(os.path.join(data_path, file), delimiter=';')
    df = pd.DataFrame(data)
    method = 'c-method'
    if "C1" in file:
        abp_spo, abp_spp, abp_rr, cbfv_spo, cbfv_spp = (df["ABP_SPO"], df["ABP_SPP"], df["ABP_RR"],
                                                        df["CBFV_SPO"], df["CBFV_SPP"])

        pairs = [(abp_spo, cbfv_spo), (abp_spp, cbfv_spp), (abp_spo, cbfv_spp), (abp_spp, cbfv_spo), (abp_rr, cbfv_spo),
                 (abp_rr, cbfv_spp)]

        dtw_list = [DTW(x, y, var="DDTW") for x, y in pairs]

        vals = []
        windows = None
        for i, dtw in enumerate(dtw_list):
            cost, wds = dtw.sliding_window_dtw(window_size=10, step=5, method=method)
            vals.append(cost)
            if i == 0:
                windows = wds

        col1, col2, col3, col4, col5, col6, col7 = ("Window", "ABP_SPO-CBFV_SPO", "ABP_SPP-CBFV_SPP", "ABP_SPO-CBFV_SPP",
                                                        "ABP_SPP-CBFV_SPO", "ABP_RR-CBFV_SPO", "ABP_RR-CBFV_SPP")
        data = {
            col1: windows,
            col2: vals[0],
            col3: vals[1],
            col4: vals[2],
            col5: vals[3],
            col6: vals[4],
            col7: vals[5],
        }

        file_no_extension = os.path.splitext(file)[0]

        df = pd.DataFrame(data)
        df.to_csv(f"C:/Python/ZSSI/data2/dtw/dtw/{method}/{file_no_extension}_DTW.csv", sep=';', index=False)
    else:
        abp_dn, abp_dpp, cbfv_dn, cbfv_dpp = df["ABP_DN"], df["ABP_DPP"], df["CBFV_DN"], df["CBFV_DPP"]
        pairs = [(abp_dn, cbfv_dn), (abp_dpp, cbfv_dpp), (abp_dn, cbfv_dpp), (abp_dpp, cbfv_dn)]
        dtw_list = [DTW(x, y, var="DDTW") for x, y in pairs]

        vals = []
        windows = None
        for i, dtw in enumerate(dtw_list):
            cost, wds = dtw.sliding_window_dtw(window_size=10, step=5, method=method)
            vals.append(cost)
            if i == 0:
                windows = wds

        col1, col2, col3, col4, col5 = ("Window", "ABP_DN-CBFV_DN", "ABP_DPP-CBFV_DPP", "ABP_DN-CBFV_DPP",
                                        "ABP_DPP-CBFV_DN")
        data = {
            col1: windows,
            col2: vals[0],
            col3: vals[1],
            col4: vals[2],
            col5: vals[3]
        }

        file_no_extension = os.path.splitext(file)[0]

        df = pd.DataFrame(data)
        df.to_csv(f"C:/Python/ZSSI/data2/dtw/dtw/{method}/{file_no_extension}_DTW.csv", sep=';', index=False)"""

print("=== Starting DDTW ===")
data_path = "C:/Python/ZSSI/data/dtw/preprocessed"
breaths = os.listdir(data_path)
for breath in breaths:
    print(f"\n=== Directory: {breath} ===")
    files = os.listdir(os.path.join(data_path, breath))
    for file in files:
        print(f"{file} being processed...")
        try:
            data = pd.read_csv(os.path.join(data_path, breath, file), delimiter=';')
            df = pd.DataFrame(data)
            method = 'd-method'
            (abp_spo, abp_spp, abp_rr,
             cbfv_spo, cbfv_spp, cbfv_rr) = (df["ABP_SPO"], df["ABP_SPP"], df["ABP_RR"],
                                             df["CBFV_SPO"], df["CBFV_SPP"], df["CBFV_RR"])

            pairs = [(abp_spo, cbfv_spo), (abp_spp, cbfv_spp), (abp_spo, cbfv_spp), (abp_spp, cbfv_spo), (abp_rr, cbfv_spo),
                     (abp_rr, cbfv_spp), (abp_spo, cbfv_rr), (abp_spp, cbfv_rr), (abp_rr, cbfv_rr)]

            dtw_list = [DTW(x, y, var="DDTW") for x, y in pairs]
            vals = []
            windows = None
            for i, dtw in enumerate(dtw_list):
                cost, wds, trs = dtw.sliding_window_dtw(window_size=10, step=5, method=method)
                vals.append(cost)
                if i == 0:
                    windows = wds
            (col1, col2, col3, col4, col5,
             col6, col7, col8, col9, col10) = ("Window", "ABP_SPO-CBFV_SPO", "ABP_SPP-CBFV_SPP", "ABP_SPO-CBFV_SPP",
                                               "ABP_SPP-CBFV_SPO", "ABP_RR-CBFV_SPO", "ABP_RR-CBFV_SPP", "ABP_SPO-CBFV_RR",
                                               "ABP_SPP-CBFV_RR", "ABP_RR-CBFV_RR")
            data = {
                col1: windows,
                col2: vals[0],
                col3: vals[1],
                col4: vals[2],
                col5: vals[3],
                col6: vals[4],
                col7: vals[5],
                col8: vals[6],
                col9: vals[7],
                col10: vals[8]
            }
            file_no_extension = os.path.splitext(file)[0]

            df = pd.DataFrame(data)
            # df.to_csv(f"C:/Python/ZSSI/data/dtw/dtw/{method}/{breath}/{file_no_extension}_DTW.csv", sep=';', index=False)
            print("✔ Success")
        except Exception as e:
            print(f"✖ Failed ({e})")
print("\n=== DDTW completed ===")
