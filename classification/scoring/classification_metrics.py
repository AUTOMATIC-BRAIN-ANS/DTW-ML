"""
Classification metrics (one-vs-rest, macro-averaged over classes) and their summary over cross-validation runs.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from project.classification.config import TARGET


class ClassificationMetrics:
    @staticmethod
    def predict(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        """
        Method to predict labels as a 1-D array (CatBoost returns a column vector).
        :param estimator: trained estimator.
        :param X: features.
        :return: predicted labels.
        """
        return np.ravel(estimator.predict(X))

    @staticmethod
    def balanced_accuracy_scorer(estimator: BaseEstimator, X: pd.DataFrame, y: np.ndarray) -> float:
        """
        Method to score an estimator with balanced accuracy (scorer for permutation importance).
        :param estimator: trained estimator.
        :param X: features.
        :param y: true labels.
        :return: balanced accuracy.
        """
        return float(balanced_accuracy_score(y, ClassificationMetrics.predict(estimator, X)))

    @staticmethod
    def per_class_rates(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, np.ndarray]:
        """
        Method to compute one-vs-rest sensitivity, specificity and negative predictive value of each class
        (0 when the denominator is 0).
        :param y_true: true labels.
        :param y_pred: predicted labels.
        :param classes: class labels.
        :return: dictionary: rate -> values in the order of classes.
        """
        counts = confusion_matrix(y_true, y_pred, labels=classes)
        tp = np.diag(counts)
        fp = counts.sum(axis=0) - tp
        fn = counts.sum(axis=1) - tp
        tn = counts.sum() - tp - fp - fn

        def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.divide(a, b, out=np.zeros(len(a)), where=b > 0)

        return {"sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp), "npv": ratio(tn, tn + fn)}

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict[str, float]:
        """
        Method to compute classification metrics. Class-wise metrics are one-vs-rest and macro-averaged over classes;
        GM is the geometric mean of the sensitivities of the classes (sqrt(Sens * Spec) for two classes). Sens is the
        macro recall, so it equals Rec.
        :param y_true: true labels.
        :param y_pred: predicted labels.
        :param proba: predicted probabilities, columns in the order of classes.
        :param classes: class labels.
        :return: dictionary: metric (METRIC_NAMES) -> value.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(classes) == 2:
                roc_auc = roc_auc_score(y_true == classes[1], proba[:, 1])
            else:
                # roc_auc_score needs sorted labels, with the probability columns in the same order
                order = np.argsort(classes)
                roc_auc = roc_auc_score(y_true, proba[:, order], multi_class="ovr", average="macro",
                                        labels=np.array(classes)[order])
            pr_auc = np.mean([average_precision_score(y_true == c, proba[:, i]) if (y_true == c).any() else np.nan
                              for i, c in enumerate(classes)])
            rates = ClassificationMetrics.per_class_rates(y_true, y_pred, classes)
            return {
                "ROC-AUC": roc_auc,
                "BACC": balanced_accuracy_score(y_true, y_pred),
                "F1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
                "Prec": precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
                "Rec": recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
                "PR-AUC": pr_auc,
                "MCC": matthews_corrcoef(y_true, y_pred),
                "GM": float(np.prod(rates["sensitivity"]) ** (1 / len(classes))),
                "Spec": rates["specificity"].mean(),
                "Sens": rates["sensitivity"].mean(),
                "NPV": rates["npv"].mean(),
            }

    @staticmethod
    def compute_predictions(predictions: pd.DataFrame, classes: list[str]) -> dict[str, float]:
        """
        Method to compute the metrics of out-of-fold predictions (columns condition, predicted, proba_<class>).
        :param predictions: predictions of one run.
        :param classes: class labels.
        :return: dictionary: metric -> value.
        """
        proba = predictions[[f"proba_{c}" for c in classes]].to_numpy()
        return ClassificationMetrics.compute(predictions[TARGET].to_numpy(), predictions["predicted"].to_numpy(),
                                             proba, classes)


def summarise(metrics: pd.DataFrame, ci: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Summarise the metrics over runs: mean and, with more than one run, standard deviation over runs (variability
    between the splits; not defined for LOSO, which has one run), with the bootstrap standard errors and confidence
    intervals if given.
    :param metrics: metrics of each run.
    :param ci: bootstrap results (columns se, ci_lower and ci_upper, one row per metric).
    :return: one row per metric, columns mean, std (more than one run), and se, ci_lower, ci_upper (if ci is given).
    """
    values = metrics.drop(columns="run")
    summary = values.agg(["mean", "std"]).T if len(values) > 1 else values.agg(["mean"]).T
    return summary if ci is None else summary.join(ci)


def get_rates(confusion: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a confusion matrix by rows (recall of each class on the diagonal).
    :param confusion: counts; index = true, columns = predicted.
    :return: rates; rows of classes without samples are NaN.
    """
    return confusion.div(confusion.sum(axis=1).replace(0, np.nan), axis=0)
