"""
Figure of the results of one classifier with one validation scheme (figure.pdf): confusion matrix, one-vs-rest ROC
curves and permutation feature importance, under a short title. Text is rendered with LaTeX (use_latex).
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.patches import Patch
from matplotlib.text import Text
from sklearn.metrics import roc_auc_score, roc_curve

from project.classification.config import TARGET
from project.classification.modeling.cross_validation import Results
from project.classification.scoring.classification_metrics import get_rates
from project.common import use_latex


def tex(text: str) -> str:
    """
    Escape the LaTeX special characters of a plain-text label (e.g. "_" in feature names) when the text is rendered
    with LaTeX (text.usetex); mathtext shows plain text as is.
    :param text: label.
    :return: label safe to render.
    """
    if not plt.rcParams["text.usetex"]:
        return text
    for char in "&%$#_{}":
        text = text.replace(char, "\\" + char)
    return text


class Visualizer:
    FPR_GRID = np.linspace(0, 1, 101)
    TITLE_SIZE = 20
    # gap (pt) between a plot and its title, the subplot titles and the figure title, the figure title and the edge;
    # also the margin at the other edges of the figure
    TITLE_GAP = 10
    # gap (pt) between neighbouring plots, measured between their labels (and the colorbar)
    PLOT_GAP = 30
    # labels of the features of feature_extraction/build_feature_matrix.py (other names are shown as they are)
    FEATURE_LABELS = {
        "mean": "Mean",
        "median": "Median",
        "p10": "10th percentile",
        "p25": "25th percentile",
        "p75": "75th percentile",
        "p90": "90th percentile",
        "std": "Standard deviation",
        "iqr": "Interquartile range",
        "coef_of_variation": "Coefficient of variation",
        "skewness": "Skewness",
        "kurtosis": "Excess kurtosis",
        "trend": "Trend (slope)",
        "mean_abs_diff": "Mean absolute successive difference",
        "autocorr_1": "Lag-1 autocorrelation",
    }

    @staticmethod
    def set_font_sizes() -> None:
        """
        Method to set consistent font sizes of the figure (call inside an rc_context).
        """
        plt.rc("font", size=11)
        plt.rc("axes", titlesize=14, labelsize=12)
        plt.rc("xtick", labelsize=11)
        plt.rc("ytick", labelsize=11)
        plt.rc("legend", fontsize=10)

    @staticmethod
    def get_signal_label(name: str) -> str:
        """
        Method to get the label of a signal metric with the metric as a subscript, e.g. "ABP_SPO" -> ABP with the
        subscript SPO.
        :param name: signal and metric joined by "_" (e.g. ABP_SPO).
        :return: label.
        """
        signal, _, metric = name.partition("_")
        return f"{tex(signal)}$_{{\\mathrm{{{tex(metric)}}}}}$" if metric else tex(signal)

    @staticmethod
    def get_title(info: dict[str, Any]) -> str:
        """
        Method to get the short title of the figure, e.g.
        "Classification performance: ABP_SPO × CBFV_SPO | td-method | LogReg | LOSO".
        :param info: description of the classifier (method, metric, model, validation).
        :return: title.
        """
        pair = r" $\times$ ".join(Visualizer.get_signal_label(part) for part in str(info["metric"]).split("-"))
        return (f"Classification performance: {pair} | {tex(info['method'])} | {tex(info['model'])} | "
                f"{tex(info['validation'])}")

    @staticmethod
    def space_titles(fig: Figure, layout: ConstrainedLayoutEngine, axes: Any, suptitle: Text, max_iterations: int = 10,
                     tolerance: float = 0.1) -> None:
        """
        Method to space the titles evenly: the gaps plot - subplot title, subplot title - figure title and figure
        title - top edge of the figure are all TITLE_GAP. Constrained layout cannot set these gaps and text extents are
        known only after drawing, so the pads of the subplot titles are corrected after one layout pass and the top of
        the layout area is found by the secant method (the square axes do not follow it one to one).
        :param fig: figure.
        :param layout: layout engine of the figure.
        :param axes: axes with titles.
        :param suptitle: figure title, left out of the layout and aligned to its top (va="top").
        :param max_iterations: maximum number of layout passes of the secant method.
        :param tolerance: accepted error of the gap between the subplot titles and the figure title (pt).
        """
        to_points = 72 / fig.dpi
        height = fig.bbox.height * to_points
        gap = Visualizer.TITLE_GAP
        suptitle.set_y(1 - gap / height)
        fig.draw_without_rendering()
        for ax in axes:
            # the pad is measured from the baseline; the bounding box also holds the descenders
            pad = plt.rcParams["axes.titlepad"] + gap - (ax.title.get_window_extent().y0 - ax.bbox.y1) * to_points
            ax.set_title(ax.get_title(), pad=pad)

        def get_error(top: float) -> float:
            layout.set(rect=(0, 0, 1, top))
            fig.draw_without_rendering()
            subtitles_top = max(ax.title.get_window_extent().y1 for ax in axes)
            return (suptitle.get_window_extent().y0 - subtitles_top) * to_points - gap

        previous_top, previous_error = 1.0, get_error(1.0)
        top = previous_top + previous_error / height
        for _ in range(max_iterations):
            error = get_error(top)
            if abs(error) < tolerance or error == previous_error:
                break
            previous_top, previous_error, top = top, error, top - error * (top - previous_top) / (error - previous_error)
        # freeze the measured layout: another pass of constrained layout moves the axes a little
        fig.set_layout_engine("none")

    @staticmethod
    def fit_figure(fig: Figure, axes: Any, suptitle: Text) -> None:
        """
        Method to fit the figure to the plots after space_titles: the plots in a row with PLOT_GAP between their
        labels, a margin of TITLE_GAP at the left, right and bottom edges (as at the top); the size of the plots and
        the vertical gaps of the titles stay as they are.
        :param fig: figure with a frozen layout (no layout engine).
        :param axes: axes in the row, from left to right.
        :param suptitle: figure title (va="top").
        """
        width, height = fig.get_size_inches()
        margin, plot_gap = Visualizer.TITLE_GAP / 72, Visualizer.PLOT_GAP / 72
        fig.draw_without_rendering()
        # in inches: plotting areas and their extents with labels, titles, colorbar and legend
        boxes = [ax.bbox.transformed(fig.dpi_scale_trans.inverted()) for ax in axes]
        extents = [ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted()) for ax in axes]
        shift = min(extent.y0 for extent in extents) - margin
        new_height = height - shift
        left = margin
        positions = []
        for box, extent in zip(boxes, extents, strict=True):
            x0 = left + box.x0 - extent.x0
            positions.append((x0, box.y0 - shift, box.width, box.height))
            left = x0 + box.width + extent.x1 - box.x1 + plot_gap
        new_width = max(left - plot_gap + margin, suptitle.get_window_extent().width / fig.dpi + 2 * margin)
        fig.set_size_inches(new_width, new_height)
        for ax, (x0, y0, w, h) in zip(axes, positions, strict=True):
            ax.set_position((x0 / new_width, y0 / new_height, w / new_width, h / new_height))
        # the top of the figure moved away from the plots by the cropped bottom: the figure title keeps its gap
        suptitle.set_y(1 - margin / new_height)

    @staticmethod
    def get_roc_curves(predictions: pd.DataFrame, classes: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
        """
        Method to get the one-vs-rest ROC curve of each class, averaged over runs on a common false positive rate grid.
        :param predictions: out-of-fold predictions (columns run, condition, proba_<class>).
        :param classes: class labels.
        :return: dictionary: class -> (mean TPR, std of TPR over runs, mean AUC over runs).
        """
        curves = {}
        for c in classes:
            tprs, aucs = [], []
            for _, run_df in predictions.groupby("run"):
                y_true, score = (run_df[TARGET] == c).to_numpy(), run_df[f"proba_{c}"].to_numpy()
                if y_true.all() or not y_true.any():
                    continue
                fpr, tpr, _ = roc_curve(y_true, score)
                tpr_grid = np.interp(Visualizer.FPR_GRID, fpr, tpr)
                tpr_grid[0] = 0.0
                tprs.append(tpr_grid)
                aucs.append(roc_auc_score(y_true, score))
            if tprs:
                curves[c] = (np.mean(tprs, axis=0), np.std(tprs, axis=0), float(np.mean(aucs)))
        return curves

    @staticmethod
    def plot_confusion(ax: Any, confusion: pd.DataFrame) -> None:
        """
        Method to plot the confusion matrix: colour = rate normalised by rows, text = rate and count.
        :param ax: axes.
        :param confusion: counts; index = true, columns = predicted.
        """
        rates = get_rates(confusion).to_numpy()
        image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        # colorbar as an inset: exactly as tall as the matrix
        ax.figure.colorbar(image, cax=ax.inset_axes((1.04, 0, 0.05, 1)), label="Row-normalised rate")
        for i in range(rates.shape[0]):
            for j in range(rates.shape[1]):
                color = "white" if rates[i, j] > 0.5 else "black"
                ax.text(j, i, f"{rates[i, j]:.2f}\n({confusion.iat[i, j]})", ha="center", va="center", color=color)
        ax.set_xticks(range(len(confusion.columns)), [tex(str(c)) for c in confusion.columns])
        ax.set_yticks(range(len(confusion.index)), [tex(str(c)) for c in confusion.index])
        ax.tick_params(length=0)
        ax.set_xlabel("Predicted condition")
        ax.set_ylabel("True condition")
        ax.set_title("(a) Confusion matrix")

    @staticmethod
    def get_class_colors(classes: list[str]) -> dict[str, str]:
        """
        Method to get the colour of each class, shared by the ROC curves and the feature importance.
        :param classes: class labels.
        :return: dictionary: class -> colour of the default colour cycle (C0, C1...).
        """
        return {c: f"C{i}" for i, c in enumerate(classes)}

    @staticmethod
    def plot_roc(ax: Any, result: Results) -> None:
        """
        Method to plot the one-vs-rest ROC curves of each class (mean ± std over runs) and their macro average.
        :param ax: axes.
        :param result: results.
        """
        classes = list(result.confusion.index)
        colors = Visualizer.get_class_colors(classes)
        curves = Visualizer.get_roc_curves(result.predictions, classes)
        for c, (tpr, tpr_std, auc) in curves.items():
            ax.plot(Visualizer.FPR_GRID, tpr, color=colors[c], label=f"{tex(c)} (AUC = {auc:.3f})")
            if tpr_std.any():
                ax.fill_between(Visualizer.FPR_GRID, np.clip(tpr - tpr_std, 0, 1), np.clip(tpr + tpr_std, 0, 1),
                                color=colors[c], alpha=0.15)
        if curves:
            macro = np.mean([tpr for tpr, _, _ in curves.values()], axis=0)
            macro_auc, lower, upper = result.summary.loc["ROC-AUC", ["mean", "ci_lower", "ci_upper"]].to_numpy(
                dtype=float)
            ax.plot(Visualizer.FPR_GRID, macro, color="black", linestyle="--", linewidth=2,
                    label=f"Macro (AUC = {macro_auc:.3f} [{lower:.3f}, {upper:.3f}])")
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.3)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("(b) ROC curves (one-vs-rest)")
        ax.legend(loc="lower right", framealpha=0.9)

    @staticmethod
    def plot_importance(ax: Any, importance: pd.DataFrame, classes: list[str]) -> None:
        """
        Method to plot the permutation importance of the features (mean over folds), most important on top; the colour
        of a bar is the class in which the mean of the feature is the highest.
        :param ax: axes.
        :param importance: features ranked by permutation importance, with the mean of each feature per class
        (columns mean_<class>, see CrossValidation.get_importance).
        :param classes: class labels; colours as in the ROC curves.
        """
        colors = Visualizer.get_class_colors(classes)
        means = importance[[f"mean_{c.lower()}" for c in classes]].set_axis(classes, axis=1)
        # a feature missing from every class (all NaN) has no highest class: grey
        highest = means.fillna(-np.inf).idxmax(axis=1).where(means.notna().any(axis=1))
        positions = np.arange(len(importance))[::-1]
        ax.barh(positions, importance["importance"], color=[colors.get(str(c), "tab:gray") for c in highest], alpha=0.85)
        ax.set_yticks(positions, [Visualizer.FEATURE_LABELS.get(f, tex(str(f))) for f in importance["feature"]])
        ax.set_ylim(-0.6, len(importance) - 0.4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("Decrease of balanced accuracy")
        ax.set_title("(c) Permutation feature importance")
        # every class, so that the colours read the same in every figure; lower right: the least important features
        # (shortest bars) are at the bottom
        handles = [Patch(color=colors[c], alpha=0.85, label=tex(c)) for c in classes]
        ax.legend(handles=handles, title="Highest mean", loc="lower right", framealpha=0.9)

    @staticmethod
    def save(result: Results, file_path: str, info: dict[str, Any]) -> None:
        """
        Method to save a figure with the confusion matrix, the ROC curves and the feature importance to a PDF.
        :param result: results.
        :param file_path: path to the PDF.
        :param info: description of the classifier (see get_title).
        """
        # rc_context: use_latex changes the global rcParams, restored after saving
        with plt.rc_context():
            use_latex()
            Visualizer.set_font_sizes()
            # Figure instead of pyplot: no GUI backend and no figures left open in a loop over classifiers
            layout = ConstrainedLayoutEngine()
            # the height sets the size of the plots; wide enough for the height to limit it (fit_figure then sets the
            # width and crops the bottom)
            fig = Figure(figsize=(23, 6.6), layout=layout)
            axes = fig.subplots(1, 3)
            Visualizer.plot_confusion(axes[0], result.confusion)
            Visualizer.plot_roc(axes[1], result)
            Visualizer.plot_importance(axes[2], result.importance, list(result.confusion.index))
            # square plotting areas of equal size, aligned in one row; anchored to the top, so that they follow the top
            # of the layout area in space_titles also when the width limits their size
            for ax in axes:
                ax.set_box_aspect(1)
                ax.set_anchor("N")
            suptitle = fig.suptitle(Visualizer.get_title(info), fontsize=Visualizer.TITLE_SIZE, va="top",
                                    in_layout=False)
            Visualizer.space_titles(fig, layout, axes, suptitle)
            Visualizer.fit_figure(fig, axes, suptitle)
            fig.savefig(file_path, format="pdf")
