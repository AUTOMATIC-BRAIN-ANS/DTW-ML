"""
@author: Radosław Plawecki
Stage `costs` (normalised -> costs): DDTW alignment costs in sliding windows (window_size and step in settings.toml)
for the ABP-CBFV pairs of the metrics of [analysis] pairs ("ABP_X-CBFV_Y": columns ABP_X and CBFV_Y), for every method
of [analysis] methods: {method}/{condition}/{file}_DTW.csv [Window, one column per pair].
Files that fail are copied to the spoiled data.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import Stage
from ddtw_ml.stages.costs.dtw import DTW


class ComputeDtwCosts(Stage):
    PARAMS = {"window_size": int, "step": int}

    def run(self) -> None:
        ctx = self.ctx
        data_path = ctx.input()
        pairs = ctx.analysis["pairs"]
        for method in ctx.analysis["methods"]:
            for breath in os.listdir(data_path):
                out_dir = ctx.output(method, breath)
                for file in os.listdir(data_path / breath):
                    try:
                        df = pd.read_csv(data_path / breath / file, delimiter=';')
                        costs: dict[str, list] = {}
                        for pair in pairs:
                            abp, cbfv = pair.split("-")
                            dtw = DTW(df[abp], df[cbfv], var="DDTW")
                            cost, windows, _ = dtw.sliding_window_dtw(window_size=ctx.params["window_size"],
                                                                      step=ctx.params["step"], method=method)
                            if not costs:
                                costs["Window"] = windows
                            costs[pair] = cost
                        file_no_extension = os.path.splitext(file)[0]
                        pd.DataFrame(costs).to_csv(out_dir / f"{file_no_extension}_DTW.csv", sep=';', index=False)
                    except Exception as e:
                        ctx.reject(data_path / breath / file, breath, f"{method}: {type(e).__name__}: {e}")
                        continue
                    ctx.processed(f"{method}/{breath}/{file}")
