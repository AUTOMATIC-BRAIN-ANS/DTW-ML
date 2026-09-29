"""
Reporting of the classification: the log of a classifier (loaded data, feature ranking, confusion matrix, metrics)
and the result files metrics.json, confusion_matrix.json and feature_importance.csv.
"""

import json
import logging
import os
from typing import Any

import numpy as np

from project.classification.config import CONFUSION_FILE, IMPORTANCE_FILE, LOGGER_NAME, METRICS_FILE
from project.classification.feature_extraction.build_feature_matrix import SEPARATOR
from project.classification.modeling.cross_validation import Results
from project.classification.modeling.data import FeatureMatrixData
from project.classification.scoring.classification_metrics import get_rates

logger = logging.getLogger(LOGGER_NAME)


def to_json_number(value: float) -> float | None:
    """
    Convert a number to a JSON-compatible value (NaN and infinity are not valid JSON).
    :param value: number.
    :return: float, or None if the value is not finite.
    """
    return float(value) if np.isfinite(value) else None


class Reporter:
    FLOAT_FORMAT = "{:.4f}".format

    @staticmethod
    def log_table(title: str, table: str) -> None:
        """
        Method to log a table as one record: the timestamp only on the line of the title, the rows indented below it.
        :param title: title of the table.
        :param table: table as text (e.g. DataFrame.to_string()).
        """
        rows = "\n".join(f"    {line}" for line in table.splitlines())
        logger.info(f"{title}:\n{rows}")

    @staticmethod
    def log_data(data: FeatureMatrixData, input_path: str) -> None:
        """
        Method to log what was loaded: dimensions, class sizes, features and missing values.
        :param data: feature matrix.
        :param input_path: path to the feature matrix.
        """
        X, y = data.get_features(), data.get_target()
        described = data.describe()
        logger.info(f"Loaded: {input_path}")
        logger.info(f"X: {X.shape[0]} rows × {X.shape[1]} features | y: {len(y)} labels | "
                    f"subjects: {described['n_subjects']} | chance level: {described['chance_level']:.3f}")
        counts = {c: int((y == c).sum()) for c in data.classes}
        logger.info("Rows per class: " + ", ".join(f"{c} {n}" for c, n in counts.items()))
        logger.info(f"Features: {', '.join(data.features)}")
        missing = X.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            logger.info("Missing values: none")
        else:
            logger.info("Missing values: " + ", ".join(f"{feature} {n}" for feature, n in missing.items()))

    @staticmethod
    def log_results(result: Results, ci_level: float) -> None:
        """
        Method to log the feature ranking, the confusion matrix and the metrics of one model with one validation
        scheme.
        :param result: results.
        :param ci_level: confidence level of the bootstrap intervals.
        """
        Reporter.log_table("Feature ranking (permutation importance: mean decrease of balanced accuracy over folds; "
                           "mean of each feature per class)",
                           result.importance.to_string(index=False, float_format=Reporter.FLOAT_FORMAT))
        Reporter.log_table(f"Confusion matrix (rows = true, columns = predicted, counts summed over "
                           f"{result.n_runs} run(s))", result.confusion.to_string())
        Reporter.log_table("Confusion matrix (rates, normalised by rows)",
                           get_rates(result.confusion).to_string(float_format=Reporter.FLOAT_FORMAT))
        if result.n_runs > 1:
            Reporter.log_table("Metrics of each run",
                               result.metrics.to_string(index=False, float_format=Reporter.FLOAT_FORMAT))
        over_runs = f"mean and std over {result.n_runs} runs" if result.n_runs > 1 else "1 run: no std over runs"
        Reporter.log_table(f"Metrics ({over_runs}; bootstrap SE and {ci_level:.0%} CI)",
                           result.summary.to_string(float_format=Reporter.FLOAT_FORMAT))

    @staticmethod
    def save(result: Results, directory: str, info: dict[str, Any]) -> None:
        """
        Method to save the results of one model with one validation scheme: metrics.json (summary with confidence
        intervals and each run), confusion_matrix.json (counts summed over runs and row-normalised rates) and
        feature_importance.csv.
        :param result: results.
        :param directory: output directory (created if needed).
        :param info: description of the classifier, the data and the bootstrap (method, metric, model, validation,
        dimensions...).
        """
        os.makedirs(directory, exist_ok=True)
        metrics = {
            **info,
            "n_runs": result.n_runs,
            "n_folds": result.n_folds,
            "summary": {name: {stat: to_json_number(value) for stat, value in row.items()}
                        for name, row in result.summary.iterrows()},
            "runs": [{name: (int(value) if name == "run" else to_json_number(value)) for name, value in row.items()}
                     for row in result.metrics.to_dict("records")],
        }
        confusion = {
            **{key: value for key, value in info.items() if key != "classes"},
            "labels": list(result.confusion.index),
            "rows": "true",
            "columns": "predicted",
            "n_runs": result.n_runs,
            "counts": result.confusion.to_numpy().astype(int).tolist(),
            "rates": [[to_json_number(v) for v in row] for row in get_rates(result.confusion).to_numpy()],
        }
        for file_name, content in ((METRICS_FILE, metrics), (CONFUSION_FILE, confusion)):
            with open(os.path.join(directory, file_name), "w", encoding="utf-8") as file:
                json.dump(content, file, indent=2, ensure_ascii=False)
        result.importance.to_csv(os.path.join(directory, IMPORTANCE_FILE), sep=SEPARATOR, decimal=".", index=False)
