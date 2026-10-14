"""
@author: Radosław Pławecki
"""

import pandas as pd

from ddtw_ml.core.signal.cleaning import PreprocessingUtils as prepUtils
from ddtw_ml.utils.validation import check_path


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
