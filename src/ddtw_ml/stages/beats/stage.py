"""
@author: Radoslaw Plawecki
Stage `beats` (filtered -> beats): detects the heart beats of ABP and CBFV and computes the beat-to-beat metrics
(SPO, SPP, DN, DPP, RR): {condition}/{file}/{ABP,CBFV}.csv.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import Stage
from ddtw_ml.stages.beats.metrics import FromSignal


class ComputeBeatMetrics(Stage):
    def run(self) -> None:
        ctx = self.ctx
        input_path = ctx.input()
        for directory in os.listdir(input_path):
            directory_path = input_path / directory
            for file in os.listdir(directory_path):
                file_path = directory_path / file
                data = pd.read_csv(file_path, delimiter=';')
                df = pd.DataFrame(data)
                s = ['DateTime', 'ABP', 'CBFV']
                datetime, abp, cbfv = df[s[0]], df[s[1]], df[s[2]]
                from_signal_abp = FromSignal(datetime, signal=abp, signal_type="abp")
                from_signal_cbfv = FromSignal(datetime, signal=cbfv, signal_type="cbfv")
                out_dir = ctx.output(directory, file_path.stem)
                pd.DataFrame(from_signal_abp.get_metrics()).to_csv(out_dir / "ABP.csv", sep=';', index=False)
                pd.DataFrame(from_signal_cbfv.get_metrics()).to_csv(out_dir / "CBFV.csv", sep=';', index=False)
                ctx.processed(f"{directory}/{file}")
