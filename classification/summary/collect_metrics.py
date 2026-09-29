"""
Collect the metrics of the classifiers (metrics.json, output of classify.py) into one table per model.

Input: {input_dir}/{method}/{metric}/{model}/{validation}/metrics.json. Output: {output_dir}/{model}.csv with the
full names of the metrics in the first column and one column per configuration, "{method} | {metric} | {validation}":
methods, then metric pairs, then validation schemes, each in the order of the configuration section. Values:
"mean ± SE [CI lower–CI upper]", e.g. "0.594 ± 0.024 [0.548–0.637]" (mean over runs for RepeatedSGKF, bootstrap SE and
confidence interval); a missing or unreadable metrics.json gives empty cells and a warning.
"""

import argparse
import json
import logging
import os

import pandas as pd

from project.classification.config import (
    METRIC_LABELS,
    METRIC_NAMES,
    METRICS_FILE,
    MODELS,
    VALIDATIONS,
)
from project.classification.config import OUTPUT_DIR as RESULTS_DIR
from project.classification.feature_extraction.build_feature_matrix import SEPARATOR
from project.utils.logger import LoggerUtils

# ============================== CONFIGURATION ==============================
INPUT_DIR = RESULTS_DIR
OUTPUT_DIR = "C:/Python/ZSSI/data/classification/summary"
# order of the columns: distance method first, SPO first, LOSO first
METHODS = ["d-method", "td-method"]
METRICS = ["ABP_SPO-CBFV_SPO", "ABP_SPP-CBFV_SPP", "ABP_RR-CBFV_RR"]
DECIMALS = 3
# UTF-8 with BOM: "±" and "–" shown correctly in Excel
ENCODING = "utf-8-sig"
LOG_FILE = "summary.log"
# ===========================================================================

METRIC_COLUMN = "metric"

logger = logging.getLogger("summary")


class MetricsCollector:
    def __init__(self, input_dir: str, methods: list[str] = METHODS, metrics: list[str] = METRICS,
                 validations: list[str] = VALIDATIONS) -> None:
        """
        Constructor of the MetricsCollector class.
        :param input_dir: results of classify.py, {input_dir}/{method}/{metric}/{model}/{validation}/metrics.json.
        :param methods: DTW cost methods, in the order of the columns.
        :param metrics: metric pairs, in the order of the columns.
        :param validations: validation schemes, in the order of the columns.
        """
        self.input_dir = input_dir
        self.configurations = [(method, metric, validation) for method in methods for metric in metrics
                               for validation in validations]

    @staticmethod
    def get_column(method: str, metric: str, validation: str) -> str:
        """
        Method to get the name of the column of one configuration.
        :param method: DTW cost method.
        :param metric: metric pair.
        :param validation: validation scheme.
        :return: e.g. "d-method | ABP_SPO-CBFV_SPO | LOSO".
        """
        return f"{method} | {metric} | {validation}"

    @staticmethod
    def format_value(stats: dict[str, float | None]) -> str | None:
        """
        Method to format one metric as "mean ± SE [CI lower–CI upper]"; missing parts (JSON null) are left out.
        :param stats: statistics of the metric from metrics.json (mean, se, ci_lower, ci_upper).
        :return: e.g. "0.594 ± 0.024 [0.548–0.637]"; None if there is no mean.
        """
        mean, se, lower, upper = (stats.get(key) for key in ("mean", "se", "ci_lower", "ci_upper"))
        if mean is None:
            return None
        text = f"{mean:.{DECIMALS}f}"
        if se is not None:
            text += f" ± {se:.{DECIMALS}f}"
        if lower is not None and upper is not None:
            text += f" [{lower:.{DECIMALS}f}–{upper:.{DECIMALS}f}]"
        return text

    def read_values(self, file_path: str) -> dict[str, str | None] | None:
        """
        Method to read and format each metric from metrics.json.
        :param file_path: path to metrics.json.
        :return: dictionary: metric (short name) -> formatted value (None if missing); None if the file is missing or
        unreadable.
        """
        if not os.path.isfile(file_path):
            logger.warning(f"Missing file: {file_path}")
            return None
        try:
            with open(file_path, encoding="utf-8") as file:
                summary = json.load(file)["summary"]
        except (OSError, ValueError, KeyError) as e:
            logger.error(f"✖ {file_path}: {e}")
            return None
        missing = [name for name in METRIC_NAMES if name not in summary]
        if missing:
            logger.warning(f"{file_path}: no {', '.join(missing)}")
        return {name: self.format_value(summary.get(name, {})) for name in METRIC_NAMES}

    def get_table(self, model: str) -> pd.DataFrame:
        """
        Method to get the table of one model: full names of the metrics in the first column, one column per
        configuration.
        :param model: model name.
        :return: table; empty cells for the configurations without results.
        """
        table = pd.DataFrame({METRIC_COLUMN: [METRIC_LABELS[name] for name in METRIC_NAMES]})
        for method, metric, validation in self.configurations:
            values = self.read_values(os.path.join(self.input_dir, method, metric, model, validation, METRICS_FILE))
            table[self.get_column(method, metric, validation)] = (
                [values[name] for name in METRIC_NAMES] if values else None)
        return table


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments (defaults come from the configuration section and config.py).
    :param argv: arguments (None = sys.argv).
    :return: parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Collect the metrics of the classifiers into one table per model.")
    parser.add_argument("--input-dir", default=INPUT_DIR, help=f"results of classify.py (default: {INPUT_DIR})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--methods", nargs="+", default=METHODS, help="DTW cost methods (order of the columns)")
    parser.add_argument("--metrics", nargs="+", default=METRICS, help="metric pairs (order of the columns)")
    parser.add_argument("--validations", nargs="+", default=VALIDATIONS,
                        help="validation schemes (order of the columns)")
    parser.add_argument("--models", nargs="+", default=MODELS, help="models (one table each)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Collect the metrics of every model and save one table per model.
    :param argv: command-line arguments (None = sys.argv).
    """
    args = parse_arguments(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    counter = LoggerUtils.setup(logger, os.path.join(args.output_dir, LOG_FILE))
    logger.info(f"=== Input: {args.input_dir} ===")
    collector = MetricsCollector(args.input_dir, args.methods, args.metrics, args.validations)
    for model in args.models:
        table = collector.get_table(model)
        n_found = int(table.drop(columns=METRIC_COLUMN).notna().any().sum())
        output_path = os.path.join(args.output_dir, f"{model}.csv")
        table.to_csv(output_path, sep=SEPARATOR, index=False, encoding=ENCODING)
        logger.info(f"✔ {model}: {n_found}/{len(collector.configurations)} configurations | saved: {output_path}")
    logger.info(f"=== Saved to: {args.output_dir} | warnings: {counter.warnings} | errors: {counter.errors} ===")


if __name__ == "__main__":
    main()
