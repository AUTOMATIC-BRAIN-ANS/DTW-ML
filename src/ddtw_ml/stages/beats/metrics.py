"""
@author: Radoslaw Plawecki
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from ddtw_ml.plotting import use_latex
from ddtw_ml.stages.beats.troughs import detect_peaks_troughs_optimized

# colours of the plots of the metrics: signal, SPO (violet), SPP and RR interval (green), grid and axes
INK = "#2b2b2a"
SPO_COLOUR = "#6250d6"
SPP_COLOUR = "#008300"
RR_COLOUR = "#008300"
GRID = "#e1e0d9"
AXIS = "#898781"
UNITS = {"abp": "ABP [mmHg]", "cbfv": r"CBFV [$\mathrm{cm \cdot s^{-1}}$]"}


def style_axes(ax):
    """
    Function to give the axes of a plot of the metrics a light style: horizontal hairline grid, no top and right spines.
    :param ax: axes.
    """
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelcolor=INK, labelsize=9, width=0.8)


def plot_cycle_metrics(abp, cbfv, abp_window, cbfv_window, output_path=None):
    """
    Function to plot a cardiac cycle of ABP and of CBFV with marked metrics (SPO, SPP, RR interval) on one figure;
    the time of each panel starts at the SPO of its cycle.
    :param abp: FromSignal of the ABP signal.
    :param cbfv: FromSignal of the CBFV signal of the same recording.
    :param abp_window: number of the ABP window (cycle).
    :param cbfv_window: number of the CBFV window (cycle).
    :param output_path: path of a file to save the plot to, format from its extension, e.g. PDF (None: shown instead).
    """
    use_latex()
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.6), sharex=True)
    _, (abp_left, abp_right), _ = abp.draw_window(axes[0], abp_window)
    _, (cbfv_left, cbfv_right), _ = cbfv.draw_window(axes[1], cbfv_window)
    axes[1].set_xlim(min(abp_left, cbfv_left), max(abp_right, cbfv_right))
    for ax, (letter, key) in zip(axes, (("a", "abp"), ("b", "cbfv"))):
        ax.set_ylabel(UNITS[key], fontsize=10)
        ax.set_title(rf"\textbf{{({letter})}} {key.upper()}", loc="left", fontsize=10, color=INK)
    axes[1].set_xlabel("Time [ms]", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close(fig)


class FromSignal:
    def __init__(self, datetime, signal, signal_type):
        """
        Method to initialize params of the class.
        :param datetime: time of each sample [ms].
        :param signal: cerebral blood flow velocity.
        :param signal_type: type of signal.
        """
        if signal_type.lower() in ('cbfv', 'abp'):
            self.signal_type = signal_type
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}!")
        self.datetime = np.array(datetime)
        self.signal = np.array(signal)
        # the windows are the troughs; get_windows() would detect them a second time (P-01)
        self.trh = self.get_troughs()
        self.wd = self.trh.copy()

    def to_time(self, locs):
        """
        Method to get the time of the samples at the given positions.
        :param locs: positions of samples (NaN: no sample).
        :return: time of the samples [ms] (NaN where there is no sample).
        """
        locs = np.asarray(locs, dtype=float)
        return np.array([np.nan if np.isnan(loc) else self.datetime[int(loc)] for loc in locs])

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
            start, end = max(wd[i] - 10, 0), spp_locs[i]
            wd_cbfv = self.signal[start:end]
            if wd_cbfv.size > 0:
                spo.append(np.min(wd_cbfv))
                spo_locs.append(np.argmin(wd_cbfv) + start)
            else:
                spo.append(np.nan)
                spo_locs.append(np.nan)
        return spo_locs, spo

    def find_rr(self):
        """
        Method to find RR interval for the ABP signal.
        :return: rr -- RR interval [ms] (time between consecutive systolic phase peaks).
        """
        spp_locs, _ = self.find_spp()
        rr = []
        for i in range(len(spp_locs) - 1):
            rr.append(self.datetime[spp_locs[i + 1]] - self.datetime[spp_locs[i]])
        return rr

    def draw_window(self, ax, wd_number, t0=None):
        """
        Method to draw a cardiac cycle with marked metrics: the cycle from its SPO to the SPO of the next cycle and the
        rise to the next SPP, with both SPO, both SPP and the RR interval between the SPP.
        :param ax: axes to draw on.
        :param wd_number: number of the window (cycle).
        :param t0: time [ms] shown as 0 on the x axis (None: the SPO of the cycle).
        :return: time [ms] of the SPO of the cycle, the shown time range [ms] relative to t0 and the RR interval [ms].
        """
        wd = self.wd
        if wd_number is None or not 0 <= wd_number <= len(wd) - 3:
            raise ValueError(f"The signal contains {len(wd) - 2} windows!")
        i = wd_number
        spp_locs, spp = self.find_spp()
        spo_locs, spo = self.find_spo()
        spp_t, spo_t = self.to_time(spp_locs), self.to_time(spo_locs)
        t0 = spo_t[i] if t0 is None else t0
        # from 80 ms before the SPO of the cycle to 120 ms after the SPP of the next cycle
        start = max(int(np.searchsorted(self.datetime, spo_t[i] - 80)), 0)
        end = int(np.searchsorted(self.datetime, spp_t[i + 1] + 120))
        t, s = self.datetime[start:end] - t0, self.signal[start:end]
        ax.plot(t, s, color=INK, linewidth=1.6, solid_capstyle="round", zorder=3)

        y_range = s.max() - s.min()
        gap = 0.05 * y_range
        marker = dict(s=56, edgecolors="white", linewidths=1.0, zorder=4, clip_on=False)
        ax.scatter(spo_t[i:i + 2] - t0, np.array(spo[i:i + 2]) - gap, marker="^", color=SPO_COLOUR, label="SPO",
                   **marker)
        ax.scatter(spp_t[i:i + 2] - t0, spp[i:i + 2] + gap, marker="v", color=SPP_COLOUR, label="SPP", **marker)

        # RR interval: an arrow above both SPP with dashed guides down to the markers
        rr = spp_t[i + 1] - spp_t[i]
        y_arrow = max(spp[i], spp[i + 1]) + 3.2 * gap
        for x, y in ((spp_t[i], spp[i]), (spp_t[i + 1], spp[i + 1])):
            ax.plot([x - t0, x - t0], [y + 1.8 * gap, y_arrow + 0.6 * gap], color=RR_COLOUR, linewidth=0.8,
                    linestyle=(0, (3, 2)), zorder=2)
        ax.annotate("", xy=(spp_t[i + 1] - t0, y_arrow), xytext=(spp_t[i] - t0, y_arrow),
                    arrowprops=dict(arrowstyle="<|-|>", color=RR_COLOUR, linewidth=1.2, mutation_scale=9,
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.plot([], [], color=RR_COLOUR, linewidth=1.2, label="R-R interval")
        ax.text((spp_t[i] + spp_t[i + 1]) / 2 - t0, y_arrow + 0.6 * gap, f"R-R = {rr:.0f} ms", ha="center",
                va="bottom", color=INK, fontsize=9)

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(s.min() - 3 * gap, y_arrow + 3 * gap)
        style_axes(ax)
        return spo_t[i], (t[0], t[-1]), rr

    def plot_window(self, wd_number=None, output_path=None):
        """
        Method to plot a specific window from a signal with marked metrics.
        :param wd_number: number of the window.
        :param output_path: path of a PDF file to save the plot to (None: not saved).
        """
        use_latex()
        fig, ax = plt.subplots(figsize=(6.3, 3.2))
        self.draw_window(ax, wd_number)
        ax.set_ylabel(UNITS[self.signal_type.lower()])
        ax.set_xlabel("Time [ms]")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, format="pdf")
        plt.show()
        plt.close(fig)

    def plot_pulsation(self, wd_number=None, output_path=None):
        """
        Method to plot a specific pulsation from a signal.
        :param wd_number: number of the window.
        :param output_path: path of an SVG file to save the plot to (None: not saved).
        """
        signal_type = self.signal_type
        use_latex()
        wd = self.wd
        if wd_number > len(wd) - 2:
            raise ValueError(f"The signal contains {len(wd) - 2} windows!")
        i = wd_number
        start, end = wd[i], wd[i + 1]
        wd_datetime, wd_signal = self.datetime[start:end], self.signal[start:end]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_box_aspect(1)
        ax.plot(wd_datetime, wd_signal, label=f"_{signal_type.upper()}", linewidth=16, zorder=3, color='black')
        # margin of 20 samples on both sides
        margin = 20 * (self.datetime[1] - self.datetime[0])
        plt.xlim(left=wd_datetime[0] - margin, right=wd_datetime[-1] + margin)
        plt.ylim(bottom=min(wd_signal) - 5, top=max(wd_signal) + 5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#191970")
            spine.set_linewidth(16)
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, format="svg", bbox_inches="tight")
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
