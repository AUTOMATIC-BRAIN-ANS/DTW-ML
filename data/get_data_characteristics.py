"""
@author: Radosław Pławecki
"""

import pandas as pd
import numpy as np
import os


# MERGE DATA
"""# input_path = "C:/Python/ZSSI/data/extracted"  # for signals
input_path = "C:/Python/ZSSI/data/dtw/raw"  # for metrics
output_path = "C:/Python/ZSSI/data/characteristics/metrics_for_anova"

allowed_ids = ["V1", "V2", "V4", "V5", "V6", "V7", "V10", "V11", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22",
               "V23", "V24", "V25", "V26", "V27", "V29", "V30", "V33", "V34", "V37", "V38", "V39", "V40", "V43", "V44",
               "V47", "V49", "V50", "V51", "V52", "V53", "V54", "V57", "V58", "V59", "V62", "V65", "V66", "V69", "V73",
               "V74", "V75", "V76", "V78", "V79", "V83", "V84", "V85", "V86", "V87", "V88", "V89", "V90"]

breaths = os.listdir(input_path)
for breath in breaths:
    abp_per_breath, cbfv_per_breath = [], []
    (abp_spo_per_breath, cbfv_spo_per_breath, abp_spp_per_breath, cbfv_spp_per_breath,
     abp_rr_per_breath, cbfv_rr_per_breath) = [], [], [], [], [], []
    breath_path = os.path.join(input_path, breath)
    files = os.listdir(breath_path)
    for file in files:
        file_id = file.split('_')[0].split('.')[0]
        if file_id not in allowed_ids:
            continue
        file_path = os.path.join(breath_path, file)
        data = pd.read_csv(file_path, delimiter=';')
        df = pd.DataFrame(data)
        abp_spo_median = df['ABP_SPO'].median()
        cbfv_spo_median = df['CBFV_SPO'].median()
        abp_spp_median = df['ABP_SPP'].median()
        cbfv_spp_median = df['CBFV_SPP'].median()
        abp_rr_median = df['ABP_RR'].median()
        cbfv_rr_median = df['CBFV_RR'].median()
        abp_spo_per_breath.append(abp_spo_median)
        cbfv_spo_per_breath.append(cbfv_spo_median)
        abp_spp_per_breath.append(abp_spp_median)
        cbfv_spp_per_breath.append(cbfv_spp_median)
        abp_rr_per_breath.append(abp_rr_median)
        cbfv_rr_per_breath.append(cbfv_rr_median)
        abp_per_breath.extend(data['ABP'].tolist())
        cbfv_per_breath.extend(data['CBFV'].tolist())
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

# MEDIANS DISPLAY
# data = pd.read_csv("C:/Python/ZSSI/data/characteristics/metrics/B15_all.csv", delimiter=';')
data = pd.read_csv("C:/Python/ZSSI/data/ml-data/d-method/BAS/ABP_RR-CBFV_RR.csv.csv", delimiter=';')
df = pd.DataFrame(data)

median = df.median()

q1 = df.quantile(0.25)
q3 = df.quantile(0.75)

result = "$" + median.round(3).astype(str) + "$$" + "(" + q1.round(3).astype(str) + "\text{-}" + q3.round(3).astype(str) + ")$"

print(result)

# GET DATA FOR ANOVA
"""b6 = pd.read_csv("C:/Python/ZSSI/data/characteristics/metrics_for_anova/B6_all.csv", delimiter=';')
b10 = pd.read_csv("C:/Python/ZSSI/data/characteristics/metrics_for_anova/B10_all.csv", delimiter=';')
b15 = pd.read_csv("C:/Python/ZSSI/data/characteristics/metrics_for_anova/B15_all.csv", delimiter=';')
bas = pd.read_csv("C:/Python/ZSSI/data/characteristics/metrics_for_anova/BAS_all.csv", delimiter=';')

b6 = b6[['ABP', 'CBFV']].rename(columns={'ABP': 'B6_ABP', 'CBFV': 'B6_CBFV'})
b10 = b10[['ABP', 'CBFV']].rename(columns={'ABP': 'B10_ABP', 'CBFV': 'B10_CBFV'})
b15 = b15[['ABP', 'CBFV']].rename(columns={'ABP': 'B15_ABP', 'CBFV': 'B15_CBFV'})
bas = bas[['ABP', 'CBFV']].rename(columns={'ABP': 'BAS_ABP', 'CBFV': 'BAS_CBFV'})

b6 = b6[['ABP_SPO', 'CBFV_SPO', 'ABP_SPP', 'CBFV_SPP', 'ABP_RR', 'CBFV_RR']].rename(
    columns={
        'ABP_SPO': 'B6_ABP_SPO',
        'CBFV_SPO': 'B6_CBFV_SPO',
        'ABP_SPP': 'B6_ABP_SPP',
        'CBFV_SPP': 'B6_CBFV_SPP',
        'ABP_RR': 'B6_ABP_RR',
        'CBFV_RR': 'B6_CBFV_RR'
    }
)

b10 = b10[['ABP_SPO', 'CBFV_SPO', 'ABP_SPP', 'CBFV_SPP', 'ABP_RR', 'CBFV_RR']].rename(
    columns={
        'ABP_SPO': 'B10_ABP_SPO',
        'CBFV_SPO': 'B10_CBFV_SPO',
        'ABP_SPP': 'B10_ABP_SPP',
        'CBFV_SPP': 'B10_CBFV_SPP',
        'ABP_RR': 'B10_ABP_RR',
        'CBFV_RR': 'B10_CBFV_RR'
    }
)

b15 = b15[['ABP_SPO', 'CBFV_SPO', 'ABP_SPP', 'CBFV_SPP', 'ABP_RR', 'CBFV_RR']].rename(
    columns={
        'ABP_SPO': 'B15_ABP_SPO',
        'CBFV_SPO': 'B15_CBFV_SPO',
        'ABP_SPP': 'B15_ABP_SPP',
        'CBFV_SPP': 'B15_CBFV_SPP',
        'ABP_RR': 'B15_ABP_RR',
        'CBFV_RR': 'B15_CBFV_RR'
    }
)

bas = bas[['ABP_SPO', 'CBFV_SPO', 'ABP_SPP', 'CBFV_SPP', 'ABP_RR', 'CBFV_RR']].rename(
    columns={
        'ABP_SPO': 'BAS_ABP_SPO',
        'CBFV_SPO': 'BAS_CBFV_SPO',
        'ABP_SPP': 'BAS_ABP_SPP',
        'CBFV_SPP': 'BAS_CBFV_SPP',
        'ABP_RR': 'BAS_ABP_RR',
        'CBFV_RR': 'BAS_CBFV_RR'
    }
)

merged = pd.concat([bas, b6, b10, b15], axis=1)
merged.to_csv("C:/Python/ZSSI/data/characteristics/metrics_for_anova.csv", sep=';')"""
