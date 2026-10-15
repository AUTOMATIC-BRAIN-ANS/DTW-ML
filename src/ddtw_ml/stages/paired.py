"""
@author: Radoslaw Plawecki
Stage `paired` (beats -> paired): pairs the ABP and CBFV beats by their number (M-01) in one file, cut to the shorter
signal: {condition}/{file}_M.csv. Subjects with at most max_beats_rejected (settings.toml) paired beats are copied to
the spoiled data.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import Stage


class PairBeatMetrics(Stage):
    PARAMS = {"max_beats_rejected": int}

    def run(self) -> None:
        ctx = self.ctx
        data_path = ctx.input()
        for directory in os.listdir(data_path):
            directory_path = data_path / directory
            output_path = ctx.output(directory)
            for breath in os.listdir(directory_path):
                breath_path = directory_path / breath
                df_abp = pd.read_csv(breath_path / "ABP.csv", delimiter=';')
                df_cbfv = pd.read_csv(breath_path / "CBFV.csv", delimiter=';')
                min_len = min(len(df_abp), len(df_cbfv))
                df_abp = df_abp.drop(columns=["DateTime"])
                df_cbfv = df_cbfv.drop(columns=["DateTime"])
                df_abp = df_abp.iloc[:min_len].reset_index(drop=True)
                df_cbfv = df_cbfv.iloc[:min_len].reset_index(drop=True)
                datetime_col = pd.Series(range(1, min_len + 1), name='DateTime')
                merged_df = pd.concat([datetime_col, df_abp, df_cbfv], axis=1)
                if len(merged_df) > ctx.params["max_beats_rejected"]:
                    merged_df.to_csv(output_path / f"{breath}_M.csv", index=False, sep=';')
                    ctx.processed(f"{directory}/{breath} ({len(merged_df)} beats)")
                else:
                    ctx.reject(breath_path, directory, f"Too few rows after merging: {len(merged_df)}")
