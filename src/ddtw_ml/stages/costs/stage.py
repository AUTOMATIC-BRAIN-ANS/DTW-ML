"""
@author: Radosław Plawecki
Stage `costs` (normalised -> costs): DDTW alignment costs in sliding windows (window_size and step in settings.toml)
for the 9 ABP-CBFV pairs of the metrics, for every method of [analysis] methods: {method}/{condition}/{file}_DTW.csv.
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
        for method in ctx.analysis["methods"]:
            for breath in os.listdir(data_path):
                out_dir = ctx.output(method, breath)
                for file in os.listdir(data_path / breath):
                    try:
                        data = pd.read_csv(data_path / breath / file, delimiter=';')
                        df = pd.DataFrame(data)
                        (abp_spo, abp_spp, abp_rr,
                         cbfv_spo, cbfv_spp, cbfv_rr) = (df["ABP_SPO"], df["ABP_SPP"], df["ABP_RR"],
                                                         df["CBFV_SPO"], df["CBFV_SPP"], df["CBFV_RR"])

                        pairs = [(abp_spo, cbfv_spo), (abp_spp, cbfv_spp), (abp_spo, cbfv_spp), (abp_spp, cbfv_spo),
                                 (abp_rr, cbfv_spo), (abp_rr, cbfv_spp), (abp_spo, cbfv_rr), (abp_spp, cbfv_rr),
                                 (abp_rr, cbfv_rr)]

                        dtw_list = [DTW(x, y, var="DDTW") for x, y in pairs]
                        vals = []
                        windows = None
                        for i, dtw in enumerate(dtw_list):
                            cost, wds, trs = dtw.sliding_window_dtw(window_size=ctx.params["window_size"],
                                                                step=ctx.params["step"], method=method)
                            vals.append(cost)
                            if i == 0:
                                windows = wds
                        (col1, col2, col3, col4, col5,
                         col6, col7, col8, col9, col10) = ("Window", "ABP_SPO-CBFV_SPO", "ABP_SPP-CBFV_SPP",
                                                           "ABP_SPO-CBFV_SPP", "ABP_SPP-CBFV_SPO", "ABP_RR-CBFV_SPO",
                                                           "ABP_RR-CBFV_SPP", "ABP_SPO-CBFV_RR", "ABP_SPP-CBFV_RR",
                                                           "ABP_RR-CBFV_RR")
                        costs = {
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
                        df = pd.DataFrame(costs)
                        df.to_csv(out_dir / f"{file_no_extension}_DTW.csv", sep=';', index=False)
                    except Exception as e:
                        ctx.reject(data_path / breath / file, breath, f"{method}: {type(e).__name__}: {e}")
                        continue
                    ctx.processed(f"{method}/{breath}/{file}")
