"""
Classify the breathing condition (BAS, B6, B10, B15) from the feature matrices (output of
feature_extraction/build_feature_matrix.py).

Models: LogReg, RandomForest, CatBoost, SVM-RBF, HistGradBoost. Validation: leave-one-subject-out (generalisation to
a new subject) and repeated stratified 5-fold with subjects as groups. Each method, metric and model gets its own
directory, {output_dir}/{method}/{metric}/{model}/, with a log of the classifier ({model}.log) and one subdirectory
per validation scheme holding metrics.json, confusion_matrix.json, feature_importance.csv and figure.pdf.
Metrics: ROC-AUC, BACC, F1, Prec, Rec, PR-AUC, MCC, GM, Spec, Sens, NPV (one-vs-rest, macro-averaged over classes),
with bootstrap confidence intervals (subjects resampled).

Uses: config.py (settings), modeling (data, models, cross-validation), scoring (metrics, bootstrap confidence
intervals), reporting (log, JSON/CSV, figure). The metrics of all classifiers are collected by
summary/collect_metrics.py.
"""

import argparse
import logging
import os
import time

import pandas as pd

from project.classification.config import (
    CI_LEVEL,
    CLASSES,
    FIGURE_FILE,
    INPUT_DIR,
    LOG_FILE,
    LOGGER_NAME,
    METHODS,
    MODELS,
    N_BOOTSTRAP,
    N_JOBS,
    N_REPEATS,
    N_SPLITS,
    OUTPUT_DIR,
    PERMUTATION_REPEATS,
    RANDOM_STATE,
    VALIDATIONS,
)
from project.classification.feature_extraction.build_feature_matrix import CONDITIONS, METRICS
from project.classification.modeling.cross_validation import CrossValidation
from project.classification.modeling.data import FeatureMatrixData
from project.classification.modeling.models import ModelFactory
from project.classification.reporting.reporter import Reporter
from project.classification.reporting.visualizer import Visualizer
from project.utils.logger import LoggerUtils

# log of one classifier (reconfigured for each method, metric and model)
logger = logging.getLogger(LOGGER_NAME)
# log of the whole run (one line per classifier and validation)
run_logger = logging.getLogger(f"{LOGGER_NAME}.run")


def run_classifier(method: str, metric: str, model_name: str, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """
    Cross-validate one model on the feature matrix of one method and metric, with every validation scheme; results
    and the log of the classifier go to {output_dir}/{method}/{metric}/{model}/.
    :param method: DTW cost method (e.g. d-method).
    :param metric: metric pair (feature matrix name).
    :param model_name: model name (one of MODELS).
    :param args: command-line arguments.
    :return: dictionary: validation -> metric summary (mean, std, SE, CI), for the schemes that succeeded.
    """
    model_dir = os.path.normpath(os.path.join(args.output_dir, method, metric, model_name))
    os.makedirs(model_dir, exist_ok=True)
    counter = LoggerUtils.setup(logger, os.path.join(model_dir, f"{model_name}.log"))
    start = time.perf_counter()
    logger.info(f"=== Method: {method} | metric: {metric} | model: {model_name} ===")
    input_path = os.path.normpath(os.path.join(args.input_dir, method, f"{metric}.csv"))
    logger.info(f"Loading the feature matrix: {input_path}")
    try:
        data = FeatureMatrixData(input_path, args.classes)
    except Exception as e:
        logger.error(f"✖ Loading {input_path} failed: {e}")
        return {}
    Reporter.log_data(data, input_path)
    # the repr of a pipeline spans several lines: one line, so that every line of the log has a timestamp
    logger.info(f"Model: {' '.join(str(ModelFactory.get(model_name, args.random_state)).split())}")
    cv = CrossValidation(data, N_SPLITS, args.n_repeats, args.permutation_repeats, args.n_bootstrap, args.ci_level,
                         args.random_state, args.n_jobs)
    summaries = {}
    for validation in args.validations:
        logger.info(f"=== Validation: {validation} ===")
        try:
            result = cv.evaluate(model_name, validation)
            Reporter.log_results(result, args.ci_level)
            info = {"method": method, "metric": metric, "model": model_name, "validation": validation,
                    **data.describe(), "ci_level": args.ci_level, "n_bootstrap": args.n_bootstrap,
                    "ci_method": "percentile bootstrap, subjects resampled with replacement"}
            validation_dir = os.path.join(model_dir, validation)
            Reporter.save(result, validation_dir, info)
            logger.info(f"Saved metrics, confusion matrix and feature importance to: {validation_dir}")
            figure_start = time.perf_counter()
            Visualizer.save(result, os.path.join(validation_dir, FIGURE_FILE), info)
            logger.info(f"Saved the figure ({time.perf_counter() - figure_start:.1f} s): {FIGURE_FILE}")
            summaries[validation] = result.summary
            logger.info(f"✔ {validation} completed")
        except Exception as e:
            logger.error(f"✖ {validation}: {e}")
    logger.info(f"=== Validations completed: {len(summaries)}/{len(args.validations)} in "
                f"{time.perf_counter() - start:.1f} s | warnings: {counter.warnings} | errors: {counter.errors} ===")
    return summaries


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments (defaults come from config.py).
    :param argv: arguments (None = sys.argv).
    :return: parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Classify the breathing condition from the feature matrices.")
    parser.add_argument("--methods", nargs="+", default=METHODS, help="DTW cost methods")
    parser.add_argument("--metrics", nargs="+", default=METRICS, choices=METRICS, help="feature matrices")
    parser.add_argument("--input-dir", default=INPUT_DIR,
                        help=f"feature matrices, {{input-dir}}/{{method}}/{{metric}}.csv (default: {INPUT_DIR})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="models to evaluate")
    parser.add_argument("--validations", nargs="+", default=VALIDATIONS, choices=VALIDATIONS,
                        help="validation schemes")
    parser.add_argument("--classes", nargs="+", default=CLASSES, choices=CONDITIONS, help="conditions to classify")
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS, help="repeats of stratified group k-fold")
    parser.add_argument("--permutation-repeats", type=int, default=PERMUTATION_REPEATS,
                        help="shuffles per feature in permutation importance")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="bootstrap resamples of the CIs")
    parser.add_argument("--ci-level", type=float, default=CI_LEVEL, help="confidence level of the CIs")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="seed")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="parallel jobs (-1 = all cores)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Cross-validate every chosen model on every chosen method and metric and save the results.
    :param argv: command-line arguments (None = sys.argv).
    """
    args = parse_arguments(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    counter = LoggerUtils.setup(run_logger, os.path.join(args.output_dir, LOG_FILE))
    run_logger.info(f"=== Methods: {', '.join(args.methods)} | metrics: {', '.join(args.metrics)} | "
                    f"models: {', '.join(args.models)} | validations: {', '.join(args.validations)} ===")
    n_completed = 0
    n_classifiers = len(args.methods) * len(args.metrics) * len(args.models)
    for method in args.methods:
        for metric in args.metrics:
            for model_name in args.models:
                name = f"{method} | {metric} | {model_name}"
                run_logger.info(f"Started: {name}")
                summaries = run_classifier(method, metric, model_name, args)
                for validation, summary in summaries.items():
                    scores = ", ".join(f"{m} {summary.loc[m, 'mean']:.3f} [{summary.loc[m, 'ci_lower']:.3f}, "
                                       f"{summary.loc[m, 'ci_upper']:.3f}]" for m in ["BACC", "F1", "MCC", "ROC-AUC"])
                    run_logger.info(f"✔ {name} | {validation}: {scores}")
                failed = [v for v in args.validations if v not in summaries]
                if failed:
                    log_path = os.path.normpath(
                        os.path.join(args.output_dir, method, metric, model_name, f"{model_name}.log"))
                    run_logger.error(f"✖ {name}: {', '.join(failed)} failed - see {log_path}")
                else:
                    n_completed += 1
    run_logger.info(f"=== Saved to: {args.output_dir} | classifiers completed: {n_completed}/{n_classifiers} | "
                    f"warnings: {counter.warnings} | errors: {counter.errors} ===")


if __name__ == "__main__":
    main()
