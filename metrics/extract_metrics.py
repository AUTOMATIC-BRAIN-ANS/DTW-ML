"""
@author: Radoslaw Plawecki
"""

from project.common import use_latex
from detect_troughs import detect_peaks_troughs_optimized
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class FromSignal:
    def __init__(self, datetime, signal, signal_type):
        """
        Method to initialize params of the class.
        :param datetime: time of a simple.
        :param signal: cerebral blood flow velocity.
        :param signal_type: type of signal.
        """
        if signal_type.lower() in ('cbfv', 'abp'):
            self.signal_type = signal_type
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}!")
        self.datetime = np.array(datetime)
        self.signal = np.array(signal)
        self.trh, self.wd = self.get_troughs(), self.get_windows()

    def get_troughs(self):
        """
        Method to get locations of troughs from a signal.
        :return: locations of troughs.
        """
        return detect_peaks_troughs_optimized(self.signal)

    def get_windows(self):
        """
        Method to get windows based on found troughs.
        :return: windows based on found troughs.
        """
        trh_locs = self.get_troughs()
        return trh_locs[:]

    def find_spp(self):
        """
        Method to find systolic phase peaks (SPP) and their datetime.
        :return: spp_locs -- locations of systolic phase peaks.
        :return: values of systolic phase peaks.
        """
        wd = self.wd
        spp_locs, spp = [], []
        for i in range(len(wd) - 1):
            start, end = wd[i], wd[i + 1]
            wd_cbfv = self.signal[start:end]
            spp.append(np.max(wd_cbfv))
            spp_locs.append(np.argmax(wd_cbfv) + start)
        return spp_locs, self.signal[spp_locs]

    def find_dn(self):
        """
        Method to find dicrotic notches (DN) and their datetime.
        :return: dn_locs -- locations of dicrotic notches.
        :return: dn -- values of dicrotic notches.
        """
        wd = self.wd
        dn_locs, dn = [], []
        for i in range(len(wd) - 1):
            start, end = wd[i], wd[i + 1]
            wd_cbfv = self.signal[start:end]
            flip_cbfv = max(wd_cbfv) - wd_cbfv
            roi = flip_cbfv[int(0.25 * len(flip_cbfv)):int(0.5 * len(flip_cbfv))]
            pks, props = find_peaks(roi, distance=20, prominence=2)
            pks = pks + int(0.25 * len(flip_cbfv))
            if len(pks) > 0:
                first_peak = pks[0]
                dn_locs.append(first_peak + start)
                dn.append(wd_cbfv[first_peak])
            else:
                dn_locs.append(np.nan)
                dn.append(np.nan)
        return dn_locs, dn

    def find_dpp(self):
        """
        Method to find diastolic phase peaks (DPP) and their datetime.
        :return: dpp_locs -- locations of diastolic phase peaks.
        :return: dpp -- values of diastolic phase peaks.
        """
        wd = self.wd
        dn_locs, _ = self.find_dn()
        dpp_locs, dpp = [], []
        for i in range(len(wd) - 1):
            if not np.isnan(dn_locs[i]):
                start, end = dn_locs[i], wd[i + 1]
                wd_cbfv = self.signal[start:end]
                dpp.append(np.max(wd_cbfv))
                dpp_locs.append(np.argmax(wd_cbfv) + start)
            else:
                dpp_locs.append(np.nan)
                dpp.append(np.nan)
        return dpp_locs, dpp

    def find_spo(self):
        """
        Method to find systolic phase onsets (SPO) and their datetime.
        :return: spo_locs -- locations of systolic phase onsets.
        :return: spo -- values of systolic phase onsets.
        """
        wd = self.wd
        spp_locs, _ = self.find_spp()
        spo_locs, spo = [], []
        for i in range(len(wd) - 1):
            start, end = wd[i] - 10, spp_locs[i]
            wd_cbfv = self.signal[start:end]
            if wd_cbfv.size > 0:
                spo.append(np.min(wd_cbfv))
                spo_locs.append(np.argmin(wd_cbfv) + start)
            else:
                spo.append(np.nan)
        return spo_locs, spo

    def find_rr(self):
        """
        Method to find RR interval for the ABP signal.
        :return: rr -- RR interval.
        """
        spp_locs, _ = self.find_spp()
        rr = []
        for i in range(len(spp_locs) - 1):
            rr.append(spp_locs[i + 1] - spp_locs[i])
        return rr

    def plot_window(self, wd_number=None, file=None):
        """
        Method to plot a specific window from a signal with marked metrics.
        """
        output_path = os.path.join("C:/Python/ZSSI/plots/metrics", f"{file}.pdf")
        signal_type = self.signal_type

        use_latex()
        wd = self.wd
        if wd_number > len(wd) - 2:
            raise ValueError(f"The signal contains {len(wd) - 2} windows!")

        i = wd_number
        start, end = wd[i] - 3, wd[i + 1] + 103  # + 103 if abp
        wd_datetime, wd_signal = self.datetime[start:end], self.signal[start:end]

        sft = 0.7

        plt.figure(figsize=(10, 6))
        plt.grid(True, zorder=0)
        plt.plot(wd_datetime, wd_signal, label=signal_type.upper(), linewidth=2, zorder=3)
        spp_locs, spp = self.find_spp()
        dn_locs, dn = self.find_dn()
        dpp_locs, dpp = self.find_dpp()
        spo_locs, spo = self.find_spo()
        plt.scatter(spp_locs[i], spp[i] + sft, label="SPP", color="green", marker="v", s=80, zorder=2)
        plt.scatter(dn_locs[i], dn[i] + sft, label="DN", color="red", marker="v", s=80, zorder=2)
        plt.scatter(dpp_locs[i], dpp[i] + sft, label="DPP", color="orange", marker="v", s=80, zorder=2)
        plt.scatter(spo_locs[i], spo[i] + sft, label="SPO", color="purple", marker="v", s=80, zorder=2)
        plt.scatter(spo_locs[i + 1], spo[i + 1] + sft, label="DPE", color="brown", marker="v", s=80, zorder=2)
        if signal_type.lower() == "cbfv":
            plt.vlines(x=spp_locs[i], ymin=48.5, ymax=spp[i], color="darkolivegreen", linestyle="--")
            plt.vlines(x=spp_locs[i + 1], ymin=48.5, ymax=spp[i + 1], color="darkolivegreen", linestyle="--")
            plt.scatter(spp_locs[i] + 2.5, 50, color="darkolivegreen", marker="<", s=80, zorder=2)
            plt.scatter(spp_locs[i + 1] - 2.5, 50, color="darkolivegreen", marker=">", s=80, zorder=2)
            plt.scatter(spp_locs[i + 1], spp[i + 1] + sft, color="green", marker="v", s=80, zorder=2)
            plt.scatter(dn_locs[i + 1], dn[i + 1] + sft, color="red", marker="v", s=80, zorder=2)
            plt.scatter(dpp_locs[i + 1], dpp[i + 1] + sft, color="orange", marker="v", s=80, zorder=2)
            plt.plot([spp_locs[i] - 2.5, spp_locs[i + 1] + 2.5], [50, 50], color="darkolivegreen", linewidth=1.5,
                     label="Odstęp RR", zorder=2)
            plt.ylabel("CBFV [$\mathrm{cm \cdot s^{-1}}$]", fontsize=12)
        else:
            plt.vlines(x=spp_locs[i], ymin=98.5, ymax=spp[i], color="darkolivegreen", linestyle="--")
            plt.vlines(x=spp_locs[i + 1], ymin=98.5, ymax=spp[i + 1], color="darkolivegreen", linestyle="--")
            plt.scatter(spp_locs[i] + 2.5, 100, color="darkolivegreen", marker="<", s=80, zorder=2)
            plt.scatter(spp_locs[i + 1] - 2.5, 100, color="darkolivegreen", marker=">", s=80, zorder=2)
            plt.scatter(spp_locs[i + 1], spp[i + 1] + sft, color="green", marker="v", s=80, zorder=2)
            plt.scatter(dn_locs[i + 1], dn[i + 1] + sft, color="red", marker="v", s=80, zorder=2)
            plt.scatter(dpp_locs[i + 1], dpp[i + 1] + sft, color="orange", marker="v", s=80, zorder=2)
            plt.plot([spp_locs[i]-2.5, spp_locs[i + 1]+2.5], [100, 100], color="darkolivegreen", linewidth=1.5,
                     label="Odstęp RR", zorder=2)
            plt.ylabel("ABP [mmHg]", fontsize=12)
        plt.title(f"Metryki {signal_type.upper()}", fontsize=16)
        plt.xlabel("Czas [s]", fontsize=12)
        plt.xlim(left=wd_datetime[0], right=wd_datetime[-1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, format="pdf")
        plt.show()
        plt.close()

    def get_metrics(self):
        """
        Method to export metrics from a signal.
        :return: data -- metrics to save in a CSV file.
        """
        _, spo = self.find_spo()
        _, spp = self.find_spp()
        _, dn = self.find_dn()
        _, dpp = self.find_dpp()
        rr = self.find_rr()
        if self.signal_type.lower() == "abp":
            datetime, col1, col2, col3, col4, col5 = "DateTime", "ABP_SPO", "ABP_SPP", "ABP_DN", "ABP_DPP", "ABP_RR"
        else:
            datetime, col1, col2, col3, col4, col5 = "DateTime", "CBFV_SPO", "CBFV_SPP", "CBFV_DN", "CBFV_DPP", "CBFV_RR"
        min_len = min(len(spo), len(spp), len(dn), len(dpp), len(rr))
        spo = spo[:min_len]
        spp = spp[:min_len]
        dn = dn[:min_len]
        dpp = dpp[:min_len]
        rr = rr[:min_len]
        datetime_values = np.linspace(0, min_len, min_len)
        data = {
            datetime: datetime_values,
            col1: spo,
            col2: spp,
            col3: dn,
            col4: dpp,
            col5: rr
        }
        return data

    def export_to_csv(self, filename, directory):
        """
        Method to export metrics to a CSV file.
        """
        output_dir = os.path.join("C:/Python/ZSSI/data/metrics", directory)
        os.makedirs(output_dir, exist_ok=True)
        data = self.get_metrics()
        df = pd.DataFrame(data)
        output_path = f"C:/Python/ZSSI/data/metrics/{directory}"
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(f"{output_path}/{filename}", sep=';', index=False)
        print("Data was exported!")
