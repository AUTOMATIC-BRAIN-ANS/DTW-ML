"""
@author: Radoslaw Plawecki
"""

from extract_cbfv import detect_peaks_troughs
from project.common import use_latex
from scipy.signal import find_peaks
import numpy as np
import os
import matplotlib.pyplot as plt


class FromCBFV:
    def __init__(self, datetime, cbfv):
        """
        Method to initialize params of the class.
        :param datetime: time of a simple.
        :param cbfv: cerebral blood flow velocity.
        """
        self.datetime = np.array(datetime)
        self.cbfv = np.array(cbfv)
        self.pks, self.trh, self.wd = self.get_peaks(), self.get_troughs(), self.get_windows()

    def detect_peaks_troughs(self):
        """
        Method to detect peaks (pks) and troughs (trh) from a signal.
        :return: pks -- peaks from a signal.
        :return: trh -- troughs from a signal.
        """
        pks, trh = detect_peaks_troughs(self.cbfv)
        pks, trh = self.convert_array(pks), self.convert_array(trh)
        return pks, trh

    def get_peaks(self):
        """
        Method to get locations of peaks from a signal.
        :return: locations of peaks.
        """
        return self.detect_peaks_troughs()[0]

    def get_troughs(self):
        """
        Method to get locations of troughs from a signal.
        :return: locations of troughs.
        """
        return self.detect_peaks_troughs()[1]

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
        spp_locs = self.pks
        return spp_locs, self.cbfv[spp_locs]

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
            wd_cbfv = self.cbfv[start:end]
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
                wd_cbfv = self.cbfv[start:end]
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
            wd_cbfv = self.cbfv[start:end]
            spo.append(np.min(wd_cbfv))
            spo_locs.append(np.argmin(wd_cbfv) + start)
        return spo_locs, spo

    def plot_window(self, wd_number=None, directory=None, file=None):
        output_dir = os.path.join("plots/cbfv_metrics", directory)
        os.makedirs(output_dir, exist_ok=True)  # <-- this is the key
        output_path = os.path.join(output_dir, f"{file}.pdf")

        use_latex()
        wd = self.wd
        if wd_number > len(wd) - 2:
            raise ValueError(f"The signal contains {len(wd) - 2} windows!")

        i = wd_number
        start, end = wd[i] - 3, wd[i + 1] + 3
        wd_datetime, wd_cbfv = self.datetime[start:end], self.cbfv[start:end]

        spp_locs, spp = self.find_spp()
        dn_locs, dn = self.find_dn()
        dpp_locs, dpp = self.find_dpp()
        spo_locs, spo = self.find_spo()

        plt.figure(figsize=(10, 6))
        plt.grid(True, zorder=0)
        plt.plot(wd_datetime, wd_cbfv, label="CBFV", linewidth=2, zorder=3)
        plt.scatter(spp_locs[i], spp[i] + 1, label="SPP", color="green", marker="v", s=80, zorder=2)
        plt.scatter(dn_locs[i], dn[i] + 1, label="DN", color="red", marker="v", s=80, zorder=2)
        plt.scatter(dpp_locs[i], dpp[i] + 1, label="DPP", color="orange", marker="v", s=80, zorder=2)
        plt.scatter(spo_locs[i], spo[i] + 1, label="SPO", color="purple", marker="v", s=80, zorder=2)
        plt.scatter(spo_locs[i + 1], spo[i + 1] + 1, label="DPE", color="brown", marker="v", s=80, zorder=2)
        plt.title(f"Metrics of CBFV (window = {i})", fontsize=16)
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel("CBFV [$\mathrm{cm \cdot s^{-1}}$]", fontsize=12)
        plt.xlim(left=wd_datetime[0], right=wd_datetime[-1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, format="pdf")
        plt.show()

    def plot_signal(self):
        spp_locs, spp = self.find_spp()
        plt.plot(self.datetime, self.cbfv)
        plt.scatter(spp_locs, spp)
        plt.show()

    @staticmethod
    def convert_array(array):
        array = np.array(array)
        return [element[0] for element in array]
