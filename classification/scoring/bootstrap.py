"""
Bootstrap confidence intervals of the classification metrics, resampling subjects (clusters of rows) with replacement.
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from project.classification.config import CI_LEVEL, ID, METRIC_NAMES, N_BOOTSTRAP, N_JOBS, RANDOM_STATE
from project.classification.scoring.classification_metrics import ClassificationMetrics


class Bootstrap:
    def __init__(self, n_bootstrap: int = N_BOOTSTRAP, ci_level: float = CI_LEVEL,
                 random_state: int = RANDOM_STATE, n_jobs: int = N_JOBS) -> None:
        """
        Constructor of the Bootstrap class: percentile confidence intervals of the metrics, resampling subjects with
        replacement (all rows of a drawn subject are kept together, as they are not independent). With several
        cross-validation runs, the metrics of each resample are averaged over runs, like the point estimate.
        :param n_bootstrap: number of resamples.
        :param ci_level: confidence level (e.g. 0.95).
        :param random_state: seed.
        :param n_jobs: number of parallel jobs (-1 = all cores).
        """
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.random_state = random_state
        self.n_jobs = n_jobs

    @staticmethod
    def _score_resamples(runs: list[pd.DataFrame], rows: list[dict[str, np.ndarray]], subjects: np.ndarray,
                         draws: np.ndarray, classes: list[str]) -> np.ndarray:
        """
        Method to compute the metrics of a chunk of resamples.
        :param runs: out-of-fold predictions of each run.
        :param rows: for each run, subject -> positions of its rows.
        :param subjects: subject identifiers.
        :param draws: indices of the drawn subjects, one resample per row.
        :param classes: class labels.
        :return: metrics averaged over runs, one row per resample, columns in the order of METRIC_NAMES.
        """
        values = np.full((len(draws), len(METRIC_NAMES)), np.nan)
        for b, draw in enumerate(draws):
            per_run = []
            for run_df, run_rows in zip(runs, rows):
                index = np.concatenate([run_rows[s] for s in subjects[draw] if s in run_rows])
                try:
                    metrics = ClassificationMetrics.compute_predictions(run_df.iloc[index], classes)
                except ValueError:
                    # e.g. a class missing from the resample
                    continue
                per_run.append([metrics[name] for name in METRIC_NAMES])
            if per_run:
                values[b] = np.nanmean(per_run, axis=0)
        return values

    def confidence_intervals(self, predictions: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
        """
        Method to get the bootstrap standard errors and confidence intervals of the metrics.
        :param predictions: out-of-fold predictions of every run (columns run, id, condition, predicted, proba_<class>).
        :param classes: class labels.
        :return: one row per metric, columns se (std of the bootstrap distribution), ci_lower and ci_upper.
        """
        runs = [run_df.reset_index(drop=True) for _, run_df in predictions.groupby("run")]
        rows = [{str(s): np.asarray(index) for s, index in run_df.groupby(ID).indices.items()} for run_df in runs]
        subjects = np.array(sorted(predictions[ID].astype(str).unique()))
        rng = np.random.default_rng(self.random_state)
        draws = rng.integers(0, len(subjects), size=(self.n_bootstrap, len(subjects)))
        chunks = np.array_split(draws, min(self.n_bootstrap, 64))
        values = np.vstack(Parallel(n_jobs=self.n_jobs)(
            delayed(self._score_resamples)(runs, rows, subjects, chunk, classes) for chunk in chunks))
        alpha = (1 - self.ci_level) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # all-NaN metric
            se = np.nanstd(values, axis=0, ddof=1)
            lower = np.nanpercentile(values, 100 * alpha, axis=0)
            upper = np.nanpercentile(values, 100 * (1 - alpha), axis=0)
        return pd.DataFrame({"se": se, "ci_lower": lower, "ci_upper": upper}, index=METRIC_NAMES)
