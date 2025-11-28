"""
@author: Radoslaw Plawecki
"""

import os
import shutil
import pandas as pd

# the code to merge spreadsheets with metrics
data_path = "C:/Python/ZSSI/data/metrics"
directories = os.listdir(data_path)
error_dir = "C:/Python/ZSSI/data/spoiled"
for directory in directories:
    directory_path = os.path.join(data_path, directory)
    breaths = os.listdir(directory_path)
    for breath in breaths:
        print(breath)
        breath_path = os.path.join(directory_path, breath)
        files = os.listdir(breath_path)
        abp, cbfv = os.path.join(breath_path, files[0]), os.path.join(breath_path, files[1])
        df_abp = pd.read_csv(abp, delimiter=';')
        df_cbfv = pd.read_csv(cbfv, delimiter=';')
        min_len = min(len(df_abp), len(df_cbfv))
        df_abp = df_abp.drop(columns=["DateTime"])
        df_cbfv = df_cbfv.drop(columns=["DateTime"])
        df_abp = df_abp.iloc[:min_len].reset_index(drop=True)
        df_cbfv = df_cbfv.iloc[:min_len].reset_index(drop=True)
        datetime_col = pd.Series(range(1, min_len + 1), name='DateTime')
        merged_df = pd.concat([datetime_col, df_abp, df_cbfv], axis=1)
        output_path = f"C:/Python/ZSSI/data/dtw/raw/{directory}"
        os.makedirs(output_path, exist_ok=True)
        if len(merged_df) > 10:
            merged_df.to_csv(f"{output_path}/{breath}_M.csv", index=False, sep=';')
            print("✔ Success. Rows:", len(merged_df))
        else:
            print(f"✖ Failed ({breath}) Rows:", len(merged_df))
            dst = os.path.join(error_dir, directory)
            shutil.move(breath_path, dst)
            print(f"Moved to {dst}")


labels = ['DateTime', 'ABP_SPO', 'ABP_SPP', 'ABP_DN', 'ABP_DPP',
          'ABP_RR', 'CBFV_SPO', 'CBFV_SPP', 'CBFV_DN', 'CBFV_DPP']

"""data_path = "C:/Python/ZSSI/data2/metrics"
directories = os.listdir(data_path)
for directory in directories:
    directory_path = os.path.join(data_path, directory)
    breaths = os.listdir(directory_path)
    abp_spo, abp_spp, abp_rr, cbfv_spo, cbfv_spp = [], [], [], [], []
    abp_dn, abp_dpp, cbfv_dn, cbfv_dpp = [], [], [], []
    for breath in breaths:
        breath_path = os.path.join(directory_path, breath)
        file_path = os.path.join(breath_path, "merged.csv")
        data = pd.read_csv(file_path, delimiter=',')
        df = pd.DataFrame(data)
        abp_spo.append(df[labels[1]].reset_index(drop=True))
        abp_spp.append(df[labels[2]].reset_index(drop=True))
        abp_dn.append(df[labels[3]].reset_index(drop=True))
        abp_dpp.append(df[labels[4]].reset_index(drop=True))
        abp_rr.append(df[labels[5]].reset_index(drop=True))
        cbfv_spo.append(df[labels[6]].reset_index(drop=True))
        cbfv_spp.append(df[labels[7]].reset_index(drop=True))
        cbfv_dn.append(df[labels[8]].reset_index(drop=True))
        cbfv_dpp.append(df[labels[9]].reset_index(drop=True))
    abp_spo = pd.concat(abp_spo, ignore_index=True)
    abp_spp = pd.concat(abp_spp, ignore_index=True)
    abp_dn = pd.concat(abp_dn, ignore_index=True)
    abp_dpp = pd.concat(abp_dpp, ignore_index=True)
    abp_rr = pd.concat(abp_rr, ignore_index=True)
    cbfv_spo = pd.concat(cbfv_spo, ignore_index=True)
    cbfv_spp = pd.concat(cbfv_spp, ignore_index=True)
    cbfv_dn = pd.concat(cbfv_dn, ignore_index=True)
    cbfv_dpp = pd.concat(cbfv_dpp, ignore_index=True)
    min_len_c1 = min(len(abp_spo), len(abp_spp), len(abp_rr), len(cbfv_spo), len(cbfv_spp))
    abp_spo = abp_spo[:min_len_c1]
    abp_spp = abp_spp[:min_len_c1]
    abp_rr = abp_rr[:min_len_c1]
    cbfv_spo = cbfv_spo[:min_len_c1]
    cbfv_spp = cbfv_spp[:min_len_c1]
    min_len_c2 = min(len(abp_dn), len(abp_dpp), len(cbfv_dn), len(cbfv_dpp))
    abp_dn = abp_dn[:min_len_c2]
    abp_dpp = abp_dpp[:min_len_c2]
    cbfv_dn = cbfv_dn[:min_len_c2]
    cbfv_dpp = cbfv_dpp[:min_len_c2]
    datetime_values_c1 = np.linspace(0, min_len_c1, min_len_c1)
    datetime_values_c2 = np.linspace(0, min_len_c2, min_len_c2)
    datetime_c1, col1_c1, col2_c1, col3_c1, col4_c1, col5_c1 = "DateTime", "ABP_SPO", "ABP_SPP", "ABP_RR", "CBFV_SPO", "CBFV_SPP"
    datetime_c2, col1_c2, col2_c2, col3_c2, col4_c2 = "DateTime", "ABP_DN", "ABP_DPP", "CBFV_DN", "CBFV_DPP"
    data_c1 = {
        datetime_c1: datetime_values_c1,
        col1_c1: abp_spo,
        col2_c1: abp_spp,
        col3_c1: abp_rr,
        col4_c1: cbfv_spo,
        col5_c1: cbfv_spp
    }
    data_c2 = {
        datetime_c2: datetime_values_c1,
        col1_c2: abp_dn,
        col2_c2: abp_dpp,
        col3_c2: cbfv_dn,
        col4_c2: cbfv_dpp
    }
    df_c1 = pd.DataFrame(data_c1)
    df_c2 = pd.DataFrame(data_c2)
    df_c1.to_csv(f"C:/Python/ZSSI/data2/dtw/{directory}_C1.csv", sep=';', index=False)
    df_c2.to_csv(f"C:/Python/ZSSI/data2/dtw/{directory}_C2.csv", sep=';', index=False)"""
