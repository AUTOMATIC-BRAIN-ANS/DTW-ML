"""
@author: Radosław Pławecki
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from project.common import use_latex

use_latex()

# BEFORE PREPROCESSING
"""data_path = "C:/Python/ZSSI/data2/dtw/raw"
directories = os.listdir(data_path)
for directory in directories:
    directory_path = os.path.join(data_path, directory)
    files = os.listdir(directory_path)
    for file in files:
        file_path = os.path.join(directory_path, file)
        data = pd.read_csv(file_path, delimiter=';')
        df = pd.DataFrame(data)
        m = "ABP"
        datetime, spo, spp, dn, dpp, rr = (df["DateTime"], df[f"{m}_SPO"], df[f"{m}_SPP"],
                                           df[f"{m}_DN"], df[f"{m}_DPP"], df[f"{m}_RR"])

        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])

        # top 4 plots (2x2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # bottom wide plot
        ax5 = fig.add_subplot(gs[2, :])

        # plot data
        ax1.plot(datetime, spo, color="purple")
        ax1.set_title("Początek fazy skurczowej")

        ax2.plot(datetime, spp, color="green")
        ax2.set_title("Szczyt fazy skurczowej")

        ax3.plot(datetime, dn, color="red")
        ax3.set_title("Wcięcie dykrotyczne")

        ax4.plot(datetime, dpp, color="orange")
        ax4.set_title("Szczyt fazy rozkurczowej")

        ax5.plot(datetime, rr, color="darkolivegreen")
        ax5.set_title("Odstęp RR")

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_xlim(left=datetime[0], right=len(datetime))
            ax.set_xlabel("Numer okna sygnału [-]")
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', linewidth=0.7)
            ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
            if ax != ax5:
                ax.set_ylabel("Wartość [mmHg]")
            else:
                ax.set_ylabel("Wartość [s]")

        plt.suptitle(f"Metryki {m}")
        plt.tight_layout()
        plt.grid()

        output_path = f"C:/Python/ZSSI/plots/metrics/before_prep/{m.lower()}/{directory}"
        os.makedirs(output_path, exist_ok=True)
        file = os.path.splitext(file)[0]
        plt.savefig(f"{output_path}/{file}.pdf", format="pdf")

        # plt.show()
        plt.close()"""

# AFTER PREPROCESSING
data_path = "C:/Python/ZSSI/data2/dtw/preprocessed"
directories = os.listdir(data_path)
for directory in directories:
    directory_path = os.path.join(data_path, directory)
    files = os.listdir(directory_path)
    for file in files:
        file_path = os.path.join(directory_path, file)
        data = pd.read_csv(file_path, delimiter=';')
        df = pd.DataFrame(data)
        m = "CBFV"
        datetime, spo, spp, rr = (df["DateTime"], df[f"{m}_SPO"], df[f"{m}_SPP"], df[f"{m}_RR"])

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.7])

        # top 4 plots (2x2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # bottom wide plot
        ax5 = fig.add_subplot(gs[1, :])

        # plot data
        ax1.plot(datetime, spo, color="purple")
        ax1.set_title("Początek fazy skurczowej")

        ax2.plot(datetime, spp, color="green")
        ax2.set_title("Szczyt fazy skurczowej")

        ax5.plot(datetime, rr, color="darkolivegreen")
        ax5.set_title("Odstęp RR")

        for ax in [ax1, ax2, ax5]:
            ax.set_xlim(left=datetime[0], right=len(datetime))
            ax.set_xlabel("Numer okna sygnału [-]")
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', linewidth=0.7)
            ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
            if ax != ax5:
                ax.set_ylabel("Wartość [mmHg]")
            else:
                ax.set_ylabel("Wartość [s]")

        plt.suptitle(f"Metryki {m}")
        plt.tight_layout()
        plt.grid()

        output_path = f"C:/Python/ZSSI/plots/metrics/after_prep/{m.lower()}/{directory}"
        os.makedirs(output_path, exist_ok=True)
        file = os.path.splitext(file)[0]
        plt.savefig(f"{output_path}/{file}.pdf", format="pdf")

        plt.show()
        plt.close()
