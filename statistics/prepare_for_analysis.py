"""
@author: Radosław Pławecki
"""

import pandas as pd
import os

# for every method, get each pair in a different file
"""breaths = "B6", "B10", "B15", "BAS"
base_dir = "C:/Python/ZSSI/data2/dtw/dtw"
methods = os.listdir(base_dir)
column = ""
i = 5
for method in methods:
    method_dir = os.path.join(base_dir, method)
    vals = []
    lengths = []
    for breath in breaths:
        path = os.path.join(method_dir, f"{breath}_C2_PP_DTW.csv")
        data = pd.read_csv(path, delimiter=';')
        df = pd.DataFrame(data)
        cols = df.columns
        vals.append(df[cols[i]])
        lengths.append(len(df[cols[i]]))
        column = cols[i]
    b6, b10, b15, bas = breaths[:4]
    min_len = min(lengths)
    print(min_len)
    data = {
        b6: vals[0][:min_len],
        b10: vals[1][:min_len],
        b15: vals[2][:min_len],
        bas: vals[3][:min_len]
    }
    df = pd.DataFrame(data)
    output_path = f"C:/Python/ZSSI/data2/statistics/{method}/{column}.csv"
    df.to_csv(output_path, sep=';', index=False)"""

# all-in-one
breaths = "B6", "B10", "B15", "BAS"
base_dir = "C:/Python/ZSSI/data/dtw/dtw"
methods = os.listdir(base_dir)
b6_c1, b10_c1, b15_c1, bas_c1 = pd.Series(), pd.Series(), pd.Series(), pd.Series()
b6_c2, b10_c2, b15_c2, bas_c2 = pd.Series(), pd.Series(), pd.Series(), pd.Series()
for method in methods:
    method_dir = os.path.join(base_dir, method)
    files = os.listdir(method_dir)
    for file in files:
        path = os.path.join(method_dir, file)
        data = pd.read_csv(path, delimiter=';')
        df = pd.DataFrame(data)
        if "B6_C1" in file:
            b6_c1 = pd.Series(pd.concat([df[c] for c in df.columns if c != "Window"], ignore_index=True))
        if "B6_C2" in file:
            b6_c2 = pd.Series(pd.concat([df[c] for c in df.columns if c != "Window"], ignore_index=True))
        if "B15_C1" in file:
            b15_c1 = pd.Series(pd.concat([df[c] for c in df.columns if c != "Window"], ignore_index=True))
        if "B15_C2" in file:
            b15_c2 = pd.Series(pd.concat([df[c] for c in df.columns if c != "Window"], ignore_index=True))
        if "B10_C1" in file:
            b10_c1 = pd.Series(pd.concat([df[c] for c in df.columns if c != "Window"], ignore_index=True))
        if "B10_C2" in file:
            b10_c2 = pd.Series(pd.concat([df[c] for c in df.columns if c != "Window"], ignore_index=True))
        if "BAS_C1" in file:
            bas_c1 = pd.Series(pd.concat([df[c] for c in df.columns if c != "Window"], ignore_index=True))
        if "BAS_C2" in file:
            bas_c2 = pd.Series(pd.concat([df[c] for c in df.columns if c != "Window"], ignore_index=True))
    print(method)
    b6 = pd.concat([b6_c1, b6_c2], ignore_index=True)
    b15 = pd.concat([b15_c1, b15_c2], ignore_index=True)
    b10 = pd.concat([b10_c1, b10_c2], ignore_index=True)
    bas = pd.concat([bas_c1, bas_c2], ignore_index=True)
    sizes = [b6.size, b15.size, b10.size, bas.size]
    min_size = min(sizes)
    b6_l, b10_l, b15_l, bas_l = breaths[:4]
    data = {
        b6_l: b6[:min_size],
        b10_l: b10[:min_size],
        b15_l: b15[:min_size],
        bas_l: bas[:min_size]
    }
    df = pd.DataFrame(data)
    output_path = f"C:/Python/ZSSI/data/statistics/all-in-one/{method}.csv"
    df.to_csv(output_path, sep=';', index=False)
