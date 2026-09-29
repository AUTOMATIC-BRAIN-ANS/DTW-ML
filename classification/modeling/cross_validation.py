"""
Cross-validation of one model with one validation scheme: out-of-fold predictions, metrics of each run with
bootstrap confidence intervals, confusion matrix and permutation importance.
"""

import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from project.classification.config import (
    CI_LEVEL,
    ID,
    LOGGER_NAME,
    N_BOOTSTRAP,
    N_JOBS,
    N_REPEATS,
    N_SPLITS,
    PERMUTATION_REPEATS,
    RANDOM_STATE,
    TARGET,
    VALIDATIONS,
)
from project.classification.feature_extraction.build_feature_matrix import Identifiers
from project.classification.modeling.data import FeatureMatrixData
from project.classification.modeling.models import ModelFactory
from project.classification.scoring.bootstrap import Bootstrap
from project.classification.scoring.classification_metrics import ClassificationMetrics, summarise

logger = logging.getLogger(LOGGER_NAME)


@dataclass
class Results:
    """
    Results of one model with one validation scheme.
    """
    predictions: pd.DataFrame  # out-of-fold predictions of every run
    metrics: pd.DataFrame  # one row per run
    summary: pd.DataFrame  # one row per metric: mean, std over runs (more than one run), bootstrap se, ci_lower, ci_upper
    confusion: pd.DataFrame  # counts summed over runs; index = true, columns = predicted
    importance: pd.DataFrame  # permutation importance averaged over folds and mean of each feature per class

    @property
    def n_runs(self) -> int:
        return len(self.metrics)

    @property
    def n_folds(self) -> int:
        """
        Number of folds of one run.
        """
        return int(self.predictions.loc[self.predictions["run"] == 0, "fold"].nunique())


class CrossValidation:
    def __init__(self, data: FeatureMatrixData, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
                 permutation_repeats: int = PERMUTATION_REPEATS, n_bootstrap: int = N_BOOTSTRAP,
                 ci_level: float = CI_LEVEL, random_state: int = RANDOM_STATE, n_jobs: int = N_JOBS) -> None:
        """
        Constructor of the CrossValidation class.
        :param data: feature matrix.
        :param n_splits: number of folds of the repeated stratified group k-fold.
        :param n_repeats: number of repeats of the stratified group k-fold.
        :param permutation_repeats: number of shuffles of each feature in permutation importance.
        :param n_bootstrap: number of bootstrap resamples of the confidence intervals.
        :param ci_level: confidence level.
        :param random_state: seed.
        :param n_jobs: number of parallel jobs (-1 = all cores).
        """
        self.data = data
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.permutation_repeats = permutation_repeats
        self.bootstrap = Bootstrap(n_bootstrap, ci_level, random_state, n_jobs)
        self.random_state = random_state
        self.n_jobs = n_jobs

    def get_runs(self, validation: str) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        """
        Method to get the train/test splits. Each run predicts every row exactly once: LOSO is one run with one
        subject per fold; RepeatedSGKF is n_repeats runs of stratified k-fold, each shuffled differently.
        :param validation: validation scheme (one of VALIDATIONS).
        :return: list of runs, each a list of (train indices, test indices).
        :raise ValueError: if the validation scheme is unknown.
        """
        X, y, groups = self.data.get_features(), self.data.get_target(), self.data.get_groups()
        if validation == "LOSO":
            return [list(LeaveOneGroupOut().split(X, y, groups))]
        if validation == "RepeatedSGKF":
            return [
                list(StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.random_state + r)
                     .split(X, y, groups))
                for r in range(self.n_repeats)
            ]
        raise ValueError(f"Unknown validation: {validation}! Available: {', '.join(VALIDATIONS)}")

    def _run_fold(self, estimator: BaseEstimator, train: np.ndarray,
                  test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method to train a model on one fold and evaluate it on the held-out rows.
        :param estimator: untrained estimator.
        :param train: indices of the training rows.
        :param test: indices of the test rows.
        :return: predicted labels, probabilities (columns in the order of classes) and permutation importance of
        each feature (mean decrease of balanced accuracy on the test rows).
        """
        X, y = self.data.get_features(), self.data.get_target()
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.iloc[train], y[train])
            y_pred = ClassificationMetrics.predict(model, X.iloc[test])
            # reorder the probability columns from model.classes_ (alphabetical) to the order of classes
            model_proba = model.predict_proba(X.iloc[test])
            model_classes = list(model.classes_)
            proba = np.zeros((len(test), len(self.data.classes)))
            for i, c in enumerate(self.data.classes):
                if c in model_classes:
                    proba[:, i] = model_proba[:, model_classes.index(c)]
            importance = permutation_importance(
                model, X.iloc[test], y[test], scoring=ClassificationMetrics.balanced_accuracy_scorer,
                n_repeats=self.permutation_repeats, random_state=self.random_state,
            ).importances_mean
        return y_pred, proba, importance

    def _fold_predictions(self, run: int, fold: int, test: np.ndarray, y_pred: np.ndarray,
                          proba: np.ndarray) -> pd.DataFrame:
        """
        Method to collect the out-of-fold predictions of one fold.
        :param run: run number.
        :param fold: fold number.
        :param test: indices of the test rows.
        :param y_pred: predicted labels.
        :param proba: predicted probabilities, columns in the order of classes.
        :return: predictions (columns run, fold, id, condition, predicted, proba_<class>).
        """
        y, groups = self.data.get_target(), self.data.get_groups()
        predictions = pd.DataFrame({"run": run, "fold": fold, ID: groups[test], TARGET: y[test],
                                    "predicted": y_pred})
        predictions[[f"proba_{c}" for c in self.data.classes]] = proba
        return predictions

    def evaluate(self, model_name: str, validation: str) -> Results:
        """
        Method to cross-validate one model: out-of-fold predictions, metrics of each run (pooled predictions of the
        run) with bootstrap confidence intervals, confusion matrix summed over runs and permutation importance
        averaged over folds. The progress bar is shown on the console only; the log records each finished run.
        :param model_name: model name (one of MODELS).
        :param validation: validation scheme (one of VALIDATIONS).
        :return: results.
        """
        classes = self.data.classes
        estimator = ModelFactory.get(model_name, self.random_state)
        runs = self.get_runs(validation)
        jobs = [(r, f, train, test) for r, run in enumerate(runs) for f, (train, test) in enumerate(run)]
        test_sizes = [len(test) for _, _, _, test in jobs]
        logger.info(f"Cross-validation started: {len(runs)} run(s) × {len(runs[0])} folds = {len(jobs)} fits | "
                    f"test rows per fold: {min(test_sizes)}-{max(test_sizes)}")
        start = time.perf_counter()
        predictions, metrics, importances = [], [], []
        run_predictions: list[pd.DataFrame] = []
        with logging_redirect_tqdm(loggers=[logger]):
            outputs = Parallel(n_jobs=self.n_jobs, return_as="generator")(
                delayed(self._run_fold)(estimator, train, test) for _, _, train, test in jobs)
            for (r, f, _, test), (y_pred, proba, importance) in tqdm(
                    zip(jobs, outputs), total=len(jobs), desc=f"{validation} | {model_name}", unit="fold"):
                run_predictions.append(self._fold_predictions(r, f, test, y_pred, proba))
                importances.append(importance)
                if f == len(runs[r]) - 1:
                    # the run is complete: metrics on its pooled out-of-fold predictions
                    run_df = pd.concat(run_predictions, ignore_index=True)
                    row = ClassificationMetrics.compute_predictions(run_df, classes)
                    metrics.append({"run": r, **row})
                    predictions.append(run_df)
                    run_predictions = []
                    logger.info(f"Run {r + 1}/{len(runs)} finished ({time.perf_counter() - start:.1f} s): "
                                f"BACC {row['BACC']:.3f}, F1 {row['F1']:.3f}, MCC {row['MCC']:.3f}, "
                                f"ROC-AUC {row['ROC-AUC']:.3f}")
        logger.info(f"Cross-validation finished in {time.perf_counter() - start:.1f} s")
        predictions_df = pd.concat(predictions, ignore_index=True)
        metrics_df = pd.DataFrame(metrics)
        logger.info(f"Bootstrap started: {self.bootstrap.n_bootstrap} resamples of subjects, "
                    f"{self.bootstrap.ci_level:.0%} percentile intervals")
        start = time.perf_counter()
        ci = self.bootstrap.confidence_intervals(predictions_df, classes)
        logger.info(f"Bootstrap finished in {time.perf_counter() - start:.1f} s")
        counts = confusion_matrix(predictions_df[TARGET], predictions_df["predicted"], labels=classes)
        confusion = pd.DataFrame(counts, index=pd.Index(classes, name="true"),
                                 columns=pd.Index(classes, name="predicted"))
        return Results(predictions_df, metrics_df, summarise(metrics_df, ci), confusion,
                       self.get_importance(np.array(importances)))

    def get_importance(self, importances: np.ndarray) -> pd.DataFrame:
        """
        Method to get the permutation importance of each feature averaged over folds, with the mean of the feature in
        each class (whole feature matrix), classes in natural order: B6, B10, B15, BAS.
        :param importances: permutation importance, one row per fold, one column per feature.
        :return: columns feature, importance, mean_<class> (lower case), most important first.
        """
        importance = pd.DataFrame({"feature": self.data.features, "importance": importances.mean(axis=0)})
        means = self.data.df.groupby(TARGET)[self.data.features].mean()
        for c in sorted(self.data.classes, key=Identifiers.natural_key):
            importance[f"mean_{c.lower()}"] = means.loc[c].to_numpy()
        return importance.sort_values("importance", ascending=False, ignore_index=True)
