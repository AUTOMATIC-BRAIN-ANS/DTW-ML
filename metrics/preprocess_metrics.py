"""
@author: Radosław Pławecki
"""

from project.utils.preprocessing_utils import PreprocessingUtils as prepUtils
from project.common import check_path
import pandas as pd


class PreprocessMetrics:
    def __init__(self, directory, filename, label=None):
        self.directory = directory
        filepath = f"C:/Python/ZSSI/data2/dtw/raw/{directory}/{filename}.csv"
        check_path(filepath)
        self.filename = filename
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

    def export_preprocessed_metrics(self):
        """
        Method to export preprocessed data.
        :return: None.
        """
        print(f"File: {self.filename} being processed...")
        df = self.df
        datetime, abp_spo, abp_spp, abp_rr, cbfv_spo, cbfv_spp = ('DateTime', 'ABP_SPO', 'ABP_SPP', 'ABP_RR', 'CBFV_SPO', 'CBFV_SPP')
        data = {
            datetime: df['DateTime'],
            abp_spo: PreprocessMetrics(directory=self.directory, filename=self.filename, label='ABP_SPO').get_metric_normalized(),
            abp_spp: PreprocessMetrics(directory=self.directory, filename=self.filename, label='ABP_SPP').get_metric_normalized(),
            abp_rr: PreprocessMetrics(directory=self.directory, filename=self.filename, label='ABP_RR').get_metric_normalized(),
            cbfv_spo: PreprocessMetrics(directory=self.directory, filename=self.filename, label='CBFV_SPO').get_metric_normalized(),
            cbfv_spp: PreprocessMetrics(directory=self.directory, filename=self.filename, label='CBFV_SPP').get_metric_normalized(),
            # abp_dn: PreprocessMetrics(directory=self.directory, filename=self.filename, label='ABP_DN').get_metric_normalized(),
            # abp_dpp: PreprocessMetrics(directory=self.directory, filename=self.filename, label='ABP_DPP').get_metric_normalized(),
            # cbfv_dn: PreprocessMetrics(directory=self.directory, filename=self.filename, label='CBFV_DN').get_metric_normalized(),
            # cbfv_dpp: PreprocessMetrics(directory=self.directory, filename=self.filename, label='CBFV_DPP').get_metric_normalized()
        }
        df = pd.DataFrame(data)
        df.to_csv(f"C:/Python/ZSSI/data2/dtw/preprocessed/{self.directory}/{self.filename}_PP.csv", sep=';', index=False)
        print("Data was exported!")
