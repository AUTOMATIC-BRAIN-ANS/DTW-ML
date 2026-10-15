"""
@author: Radoslaw Plawecki
Stage `normalised` (paired -> normalised): removes the outliers of the SPO, SPP and RR metrics, interpolates them and
normalises them with the Generalized Logistic function: {condition}/{file}_PP.csv. Files that fail are copied to the
spoiled data.
"""

import os

import pandas as pd

from ddtw_ml.pipeline import Stage
from ddtw_ml.signal.cleaning import PreprocessingUtils as prepUtils
from ddtw_ml.validation import check_path


class PreprocessMetrics:
    def __init__(self, filepath, label=None):
        """
        Method to initialize params of a class.
        :param filepath: path of a CSV file with the paired ABP and CBFV metrics.
        :param label: column of the metric to preprocess.
        """
        check_path(filepath)
        self.filepath = filepath
        data = pd.read_csv(filepath, delimiter=';')
        self.df = df = pd.DataFrame(data)
        if label is not None and label not in df.columns:
            raise ValueError(f"{label} isn't a label! Allowed values are: {list(df.columns)}.")
        else:
            self.label = label

    def get_metric(self):
        return self.df[self.label]

    def get_metric_outliers_removed(self):
        return prepUtils.iqr_outlier_removal(self.df[self.label], threshold=1.5)

    def get_metric_interpolated(self):
        return prepUtils.interpolate_data(self.get_metric_outliers_removed(), method='linear')

    def get_metric_normalized(self):
        return pd.Series(prepUtils.normalize(self.get_metric_interpolated(), method='generalized-logistic'))

    def get_preprocessed_metrics(self):
        """
        Method to get the preprocessed metrics (outliers removed, interpolated, Generalized Logistic normalisation).
        :return: DateTime and the preprocessed SPO, SPP and RR of ABP and CBFV.
        """
        df = self.df
        datetime, abp_spo, abp_spp, abp_rr, cbfv_spo, cbfv_spp, cbfv_rr = ('DateTime', 'ABP_SPO', 'ABP_SPP', 'ABP_RR', 'CBFV_SPO', 'CBFV_SPP', 'CBFV_RR')
        data = {
            datetime: df['DateTime'],
            abp_spo: PreprocessMetrics(self.filepath, label='ABP_SPO').get_metric_normalized(),
            abp_spp: PreprocessMetrics(self.filepath, label='ABP_SPP').get_metric_normalized(),
            abp_rr: PreprocessMetrics(self.filepath, label='ABP_RR').get_metric_normalized(),
            cbfv_spo: PreprocessMetrics(self.filepath, label='CBFV_SPO').get_metric_normalized(),
            cbfv_spp: PreprocessMetrics(self.filepath, label='CBFV_SPP').get_metric_normalized(),
            cbfv_rr: PreprocessMetrics(self.filepath, label='CBFV_RR').get_metric_normalized()
            # abp_dn: PreprocessMetrics(self.filepath, label='ABP_DN').get_metric_normalized(),
            # abp_dpp: PreprocessMetrics(self.filepath, label='ABP_DPP').get_metric_normalized(),
            # cbfv_dn: PreprocessMetrics(self.filepath, label='CBFV_DN').get_metric_normalized(),
            # cbfv_dpp: PreprocessMetrics(self.filepath, label='CBFV_DPP').get_metric_normalized()
        }
        return pd.DataFrame(data)


class NormaliseBeatMetrics(Stage):
    def run(self) -> None:
        ctx = self.ctx
        base_dir = ctx.input()
        for directory in os.listdir(base_dir):
            dir_path = base_dir / directory
            out_dir = ctx.output(directory)
            for file in os.listdir(dir_path):
                file_path = dir_path / file
                try:
                    df = PreprocessMetrics(str(file_path)).get_preprocessed_metrics()
                except Exception as e:
                    ctx.reject(file_path, directory, f"{type(e).__name__}: {e}")
                    continue
                df.to_csv(out_dir / f"{file_path.stem}_PP.csv", sep=';', index=False)
                ctx.processed(f"{directory}/{file}")
