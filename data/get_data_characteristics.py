"""
@author: Radosław Pławecki
"""

import pandas as pd
import os


"""# MERGE DATA
# input_path = "C:/Python/ZSSI/data_v2/extracted"  # for signals
input_path = "C:/Python/ZSSI/data_v2/dtw/raw"  # for metrics
output_path = "C:/Python/ZSSI/data_v2/characteristics/metrics"

breaths = os.listdir(input_path)
for breath in breaths:
    # abp_per_breath, cbfv_per_breath = [], []
    (abp_spo_per_breath, cbfv_spo_per_breath, abp_spp_per_breath, cbfv_spp_per_breath,
     abp_rr_per_breath, cbfv_rr_per_breath) = [], [], [], [], [], []
    breath_path = os.path.join(input_path, breath)
    files = os.listdir(breath_path)
    for file in files:
        file_path = os.path.join(breath_path, file)
        data = pd.read_csv(file_path, delimiter=';')
        df = pd.DataFrame(data)
        # abp_per_breath.extend(data['ABP'].tolist())
        # cbfv_per_breath.extend(data['CBFV'].tolist())
        abp_spo_per_breath.extend(data['ABP_SPO'].tolist())
        abp_spp_per_breath.extend(data['ABP_SPP'].tolist())
        abp_rr_per_breath.extend(data['ABP_RR'].tolist())
        cbfv_spo_per_breath.extend(data['CBFV_SPO'].tolist())
        cbfv_spp_per_breath.extend(data['CBFV_SPP'].tolist())
        cbfv_rr_per_breath.extend(data['CBFV_RR'].tolist())
    new_data = {
        "ABP": abp_per_breath,
        "CBFV": cbfv_per_breath
    }
    new_data = {
        "ABP_SPO": abp_spo_per_breath,
        "ABP_SPP": abp_spp_per_breath,
        "ABP_RR": abp_rr_per_breath,
        "CBFV_SPO": cbfv_spo_per_breath,
        "CBFV_SPP": cbfv_spp_per_breath,
        "CBFV_RR": cbfv_rr_per_breath
    }
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(f"{output_path}/{breath}_all.csv", sep=';', index=True)"""


data = pd.read_csv("C:/Python/ZSSI/data_v2/ml-data/td-method/BAS/ABP_SPO-CBFV_SPO.csv.csv", delimiter=';')
df = pd.DataFrame(data)

median = df.median()

q1 = df.quantile(0.25)
q3 = df.quantile(0.75)

result = "$" + median.round(2).astype(str) + "~(" + q1.round(2).astype(str) + "\text{-}" + q3.round(2).astype(str) + ")$"

print(result)
