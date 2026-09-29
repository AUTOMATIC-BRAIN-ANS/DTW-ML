"""
Unit tests for classification/reporting/visualizer.py (figures rendered without LaTeX).
"""

import pandas as pd
import pytest
from matplotlib.colors import same_color
from matplotlib.figure import Figure

from project.classification.config import CLASSES
from project.classification.modeling.cross_validation import CrossValidation
from project.classification.reporting.visualizer import Visualizer

pytestmark = pytest.mark.usefixtures("no_latex")


def test_title(info):
    assert (Visualizer.get_title(info)
            == r"Classification performance: ABP$_{\mathrm{RR}}$ $\times$ CBFV$_{\mathrm{RR}}$ | d-method | LogReg | LOSO")


def test_roc_curves(loso_result):
    curves = Visualizer.get_roc_curves(loso_result.predictions, CLASSES)
    assert list(curves) == CLASSES
    tpr, tpr_std, auc = curves["BAS"]
    assert tpr[0] == 0.0 and tpr[-1] == 1.0 and 0.5 < auc <= 1.0
    assert not tpr_std.any()  # one run


@pytest.mark.parametrize("validation", ["LOSO", "RepeatedSGKF"])
def test_save_pdf(data, info, tmp_path, validation):
    cv = CrossValidation(data, n_splits=3, n_repeats=2, permutation_repeats=2, n_bootstrap=10, n_jobs=1)
    result = cv.evaluate("LogReg", validation)
    path = tmp_path / "figure.pdf"
    Visualizer.save(result, str(path), {**info, "validation": validation})
    assert path.read_bytes().startswith(b"%PDF")


def test_importance_colour_is_class_with_highest_mean():
    importance = pd.DataFrame({"feature": ["a", "b", "c"], "importance": [0.3, 0.1, -0.05],
                               "mean_b6": [1.0, 5.0, float("nan")], "mean_b10": [2.0, 0.0, float("nan")],
                               "mean_b15": [0.0, 0.0, float("nan")], "mean_bas": [3.0, 1.0, float("nan")]})
    ax = Figure().subplots()
    Visualizer.plot_importance(ax, importance, CLASSES)
    # bars from the top: a -> BAS (C0), b -> B6 (C1), c: no means -> grey; legend: every class
    for bar, color in zip(ax.patches, ["C0", "C1", "tab:gray"], strict=True):
        assert same_color(bar.get_facecolor()[:3], color)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == CLASSES


def test_importance_feature_labels():
    importance = pd.DataFrame({"feature": ["autocorr_1", "p10", "other_name"], "importance": [0.3, 0.2, 0.1],
                               "mean_b6": 0.0, "mean_b10": 0.0, "mean_b15": 0.0, "mean_bas": 1.0})
    ax = Figure().subplots()
    Visualizer.plot_importance(ax, importance, CLASSES)
    # most important on top; names without a label as they are
    labels = [text.get_text() for text in ax.get_yticklabels()]
    assert labels == ["Lag-1 autocorrelation", "10th percentile", "other_name"]
