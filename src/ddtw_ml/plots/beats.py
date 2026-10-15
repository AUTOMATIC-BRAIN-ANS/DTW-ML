"""
@author: Radosław Pławecki
Plot `beats` (paired, normalised -> plots): beat-to-beat metrics of ABP and CBFV before normalisation (SPO, SPP, DN,
DPP, RR) and after it (SPO, SPP, RR), one figure per file and signal:
{paired,normalised}/{abp,cbfv}/{condition}/{file}.pdf.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from ddtw_ml.pipeline import Stage
from ddtw_ml.plotting import use_latex

UNITS = {"ABP": "mmHg", "CBFV": r"$\mathrm{cm \cdot s^{-1}}$"}
# (key of the input, titles of the metrics plotted in the upper rows, height ratios of the rows)
VARIANTS = {
    "paired": ({"SPO": "Systolic phase onset", "SPP": "Systolic phase peak", "DN": "Dicrotic notch",
                "DPP": "Diastolic phase peak"}, [1, 1, 0.7]),
    "normalised": ({"SPO": "Systolic phase onset", "SPP": "Systolic phase peak"}, [1, 0.7]),
}
COLOURS = {"SPO": "purple", "SPP": "green", "DN": "red", "DPP": "orange"}


def plot_metrics(df: pd.DataFrame, m: str, titles: dict[str, str], height_ratios: list[float]) -> plt.Figure:
    """
    Function to plot the metrics of one signal: two per row, the RR interval in a wide bottom row.
    :param df: metrics of one file.
    :param m: signal, "ABP" or "CBFV".
    :param titles: titles of the metrics plotted in the upper rows.
    :param height_ratios: height ratios of the rows.
    :return: figure.
    """
    datetime = df["DateTime"]
    rows = len(height_ratios)
    fig = plt.figure(figsize=(10, 8 if rows == 3 else 6))
    gs = fig.add_gridspec(rows, 2, height_ratios=height_ratios)
    axes = []
    for k, (metric, title) in enumerate(titles.items()):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.plot(datetime, df[f"{m}_{metric}"], color=COLOURS[metric])
        ax.set_title(title)
        ax.set_ylabel(f"Value [{UNITS[m]}]")
        axes.append(ax)
    ax_rr = fig.add_subplot(gs[rows - 1, :])
    ax_rr.plot(datetime, df[f"{m}_RR"], color="darkolivegreen")
    ax_rr.set_title("RR interval")
    ax_rr.set_ylabel("Value [s]")
    for ax in axes + [ax_rr]:
        ax.set_xlim(left=datetime[0], right=len(datetime))
        ax.set_xlabel("Window [-]")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth=0.7)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
    plt.suptitle(f"{m} metrics")
    plt.tight_layout()
    return fig


class PlotBeatMetrics(Stage):
    def run(self) -> None:
        ctx = self.ctx
        use_latex()
        for key, (titles, height_ratios) in VARIANTS.items():
            data_path = ctx.input(key)
            for directory in os.listdir(data_path):
                for file in os.listdir(data_path / directory):
                    df = pd.read_csv(data_path / directory / file, delimiter=';')
                    for m in ("ABP", "CBFV"):
                        fig = plot_metrics(df, m, titles, height_ratios)
                        fig.savefig(ctx.output(key, m.lower(), directory) / f"{os.path.splitext(file)[0]}.pdf",
                                    format="pdf")
                        plt.close(fig)
                    ctx.processed(f"{key}/{directory}/{file}")
