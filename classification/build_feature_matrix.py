"""
Build feature matrices (one per metric pair) from sliding-window DTW costs for statistical analysis and machine learning.

Input: {input_dir}/{condition}/{metric}.csv (output of other/reorganise.py). Output: {output_dir}/{metric}.csv,
one row per subject and condition; the n_obs column is diagnostic and should be dropped before training a model.
"""

import argparse
import logging
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, linregress, skew

from project.utils.logger import LoggerUtils

# ============================== CONFIGURATION ==============================
METHOD = "d-method"
INPUT_DIR = f"C:/Python/ZSSI/data/dtw/reorganised/{METHOD}"
OUTPUT_DIR = f"C:/Python/ZSSI/data/classification/{METHOD}"
CONDITIONS = ["BAS", "B6", "B10", "B15"]
METRICS = ["ABP_RR-CBFV_RR", "ABP_SPO-CBFV_SPO", "ABP_SPP-CBFV_SPP"]
SEPARATOR = ";"
ENCODING = "utf-8-sig"
LOG_FILE = "feature_matrix.log"
# ===========================================================================

# minimum number of observations needed to compute each feature (the order defines the column order)
MIN_OBSERVATIONS = {
    "mean": 1,
    "median": 1,
    "p10": 1,
    "p25": 1,
    "p75": 1,
    "p90": 1,
    "std": 1,
    "iqr": 1,
    "coef_of_variation": 1,  # std / mean
    "skewness": 3,
    "kurtosis": 4,
    "trend": 2,
    "mean_abs_diff": 2,
    "autocorr_1": 3,
}
FEATURES = list(MIN_OBSERVATIONS)
# diagnostic column (recording length) - exclude before training a model
N_OBS = "n_obs"
ZERO_MEAN_TOLERANCE = 1e-12

# (condition, metric) -> {subject -> costs}
Costs = dict[tuple[str, str], dict[str, np.ndarray]]

logger = logging.getLogger("feature_matrix")


class Identifiers:
    @staticmethod
    def subject(header: str) -> str:
        """
        Method to get a subject identifier from a column header.
        :param header: column header, e.g. "V10".
        :return: subject identifier ("" if the header is empty).
        """
        return header.strip()

    @staticmethod
    def condition(directory: str) -> str:
        """
        Method to get a breathing condition from a directory path.
        :param directory: path to a condition directory, e.g. ".../reorganised/td-method/B6".
        :return: breathing condition, e.g. "B6".
        """
        return os.path.basename(os.path.normpath(directory))

    @staticmethod
    def natural_key(text: str) -> list[int | str]:
        """
        Method to get a key for natural sorting, so that V2 comes before V10.
        :param text: text to sort.
        :return: list of alternating text and integer parts.
        """
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


class CostReader:
    def __init__(self, input_dir: str, conditions: list[str] = CONDITIONS, metrics: list[str] = METRICS,
                 sep: str = SEPARATOR) -> None:
        """
        Constructor of the CostReader class.
        :param input_dir: directory with one folder per breathing condition.
        :param conditions: breathing conditions to read.
        :param metrics: metric pairs (file names without the extension).
        :param sep: column separator.
        """
        self.input_dir = input_dir
        self.conditions = conditions
        self.metrics = metrics
        self.sep = sep
        self.__costs: Costs | None = None

    @staticmethod
    def parse_number(cell: str) -> float | None:
        """
        Method to parse a number written with either a dot or a comma as the decimal separator.
        :param cell: cell content (already stripped).
        :return: finite float, or None if the cell is not a finite number.
        """
        try:
            value = float(cell.replace(",", "."))
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def __parse_column(cells: list[str], context: str) -> np.ndarray:
        """
        Method to convert one subject column to numbers: drop trailing padding silently, drop (and log) empty or
        non-numeric cells in the middle of the column.
        :param cells: raw cell contents under the header.
        :param context: description used in log messages (file and subject).
        :return: array of costs in the original order.
        """
        stripped = [cell.strip() for cell in cells]
        last = max((i for i, cell in enumerate(stripped) if cell), default=-1)
        values = []
        # data row i is line i + 2 of the file (line 1 is the header)
        for i, cell in enumerate(stripped[: last + 1]):
            if not cell:
                logger.warning(f"{context}: empty cell in the middle of the column (line {i + 2}) - removed")
                continue
            value = CostReader.parse_number(cell)
            if value is None:
                logger.warning(f"{context}: non-numeric value '{cell}' (line {i + 2}) - removed")
                continue
            values.append(value)
        return np.array(values, dtype=float)

    def __read_file(self, file_path: str) -> dict[str, np.ndarray]:
        """
        Method to read a file with DTW costs (one column per subject).
        :param file_path: path to the CSV file.
        :return: dictionary: subject -> costs.
        :raise ValueError: if the file is empty, its header has no subject identifiers or a subject appears in it
        twice.
        """
        try:
            df = pd.read_csv(
                file_path, sep=self.sep, header=None, dtype=str, keep_default_na=False, skip_blank_lines=False,
                encoding=ENCODING,
            )
        except pd.errors.EmptyDataError:
            raise ValueError("empty file")
        df = df.map(lambda cell: cell if isinstance(cell, str) else "")
        subjects = [Identifiers.subject(header) for header in df.iloc[0]]
        if not any(subjects):
            raise ValueError("no subject identifiers in the header")
        # each subject has exactly one column; a duplicate means a broken file
        duplicates = sorted({p for p in subjects if p and subjects.count(p) > 1}, key=Identifiers.natural_key)
        if duplicates:
            raise ValueError(f"Duplicated subjects in the header: {', '.join(duplicates)}!")
        data = {}
        for col, subject in enumerate(subjects):
            if not subject:
                logger.warning(f"{file_path}: column {col + 1} has an empty header - skipped")
                continue
            data[subject] = self.__parse_column(df.iloc[1:, col].tolist(), f"{file_path} [{subject}]")
        return data

    def __load(self) -> Costs:
        """
        Method to load all cost files; a missing or broken folder/file is logged and skipped.
        :return: dictionary: (condition, metric) -> {subject -> costs}.
        """
        costs = {}
        for folder in self.conditions:
            directory = os.path.join(self.input_dir, folder)
            if not os.path.isdir(directory):
                logger.error(f"Missing folder: {directory}")
                continue
            condition = Identifiers.condition(directory)
            for metric in self.metrics:
                file_path = os.path.join(directory, f"{metric}.csv")
                if not os.path.isfile(file_path):
                    logger.error(f"Missing file: {file_path}")
                    continue
                try:
                    costs[(condition, metric)] = self.__read_file(file_path)
                    logger.info(f"✔ {file_path}: {len(costs[(condition, metric)])} subjects")
                except Exception as e:
                    logger.error(f"✖ {file_path}: {e}")
        return costs

    def get_costs(self) -> Costs:
        """
        Method to get the costs of all files that were read (loaded once, on the first call).
        :return: dictionary: (condition, metric) -> {subject -> costs}.
        """
        if self.__costs is None:
            self.__costs = self.__load()
        return self.__costs

    def get_subjects(self) -> set[str]:
        """
        Method to get all subjects found in the files that were read.
        :return: set of subject identifiers.
        """
        return {subject for data in self.get_costs().values() for subject in data}


class FeatureMatrix:
    def __init__(self, costs: Costs, conditions: list[str] = CONDITIONS) -> None:
        """
        Constructor of the FeatureMatrix class.
        :param costs: dictionary: (condition, metric) -> {subject -> costs}.
        :param conditions: breathing conditions, in the output order.
        """
        self.costs = costs
        self.conditions = conditions

    @staticmethod
    def compute_features(x: np.ndarray) -> tuple[dict[str, float], list[str]]:
        """
        Method to compute descriptive features of one cost series (one subject, one metric, one condition).
        :param x: costs in the window order, without padding.
        :return: features (NaN where there are too few observations) and names of the features that were not
        computed.
        """
        n = len(x)
        features = dict.fromkeys(FEATURES, math.nan)
        skipped = [name for name, minimum in MIN_OBSERVATIONS.items() if n < minimum]
        # constant series trigger "precision loss" warnings in scipy; the results are handled explicitly below
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            if n >= 1:
                p10, p25, p75, p90 = np.percentile(x, [10, 25, 75, 90])
                mean = float(np.mean(x))
                std = float(np.std(x, ddof=0))
                features.update(
                    mean=mean,
                    median=float(np.median(x)),
                    p10=float(p10),
                    p25=float(p25),
                    p75=float(p75),
                    p90=float(p90),
                    std=std,
                    iqr=float(p75 - p25),
                    coef_of_variation=std / mean if abs(mean) >= ZERO_MEAN_TOLERANCE else math.nan,
                )
            if n >= 2:
                features["trend"] = float(linregress(np.arange(n), x).slope)
                features["mean_abs_diff"] = float(np.mean(np.abs(np.diff(x))))
            if n >= 3:
                features["skewness"] = float(skew(x, bias=True))
                previous, following = x[:-1], x[1:]
                if np.ptp(previous) > 0 and np.ptp(following) > 0:
                    features["autocorr_1"] = float(np.corrcoef(previous, following)[0, 1])
            if n >= 4:
                features["kurtosis"] = float(kurtosis(x, fisher=True, bias=True))
        return features, skipped

    def log_incomplete_subjects(self) -> None:
        """
        Method to log subjects missing from some of the files that were read.
        """
        subjects = {subject for data in self.costs.values() for subject in data}
        for subject in sorted(subjects, key=Identifiers.natural_key):
            missing = [f"{condition}/{metric}" for (condition, metric), data in self.costs.items()
                       if subject not in data]
            if missing:
                logger.warning(f"Subject {subject} is missing from: {', '.join(missing)}")

    def get_matrix(self, metric: str) -> pd.DataFrame:
        """
        Method to get the feature matrix of one metric: one row per subject and condition, the features and n_obs.
        A subject gets a row in a condition if it is present in any metric file of that condition, so the matrices
        of all metrics have the same rows; features are NaN if the subject is missing from this metric.
        :param metric: metric pair, e.g. "ABP_RR-CBFV_RR".
        :return: feature matrix sorted by condition and subject (natural order).
        """
        rows = []
        for condition in self.conditions:
            subjects = {subject for (c, _), data in self.costs.items() if c == condition for subject in data}
            for subject in sorted(subjects, key=Identifiers.natural_key):
                row: dict[str, object] = {"id": subject, "condition": condition}
                x = self.costs.get((condition, metric), {}).get(subject)
                if x is not None:
                    features, skipped = self.compute_features(x)
                    if skipped:
                        logger.warning(f"{condition}/{metric} [{subject}]: only {len(x)} observations - "
                                       f"NaN for: {', '.join(skipped)}")
                    row.update(features)
                    row[N_OBS] = len(x)
                rows.append(row)
        matrix = pd.DataFrame(rows, columns=["id", "condition", *FEATURES, N_OBS])
        matrix[FEATURES] = matrix[FEATURES].astype(float)
        matrix[N_OBS] = matrix[N_OBS].astype("Int64")
        return matrix


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments (defaults come from the configuration section).
    :param argv: arguments (None = sys.argv).
    :return: parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build feature matrices from sliding-window DTW costs.")
    parser.add_argument("--input-dir", default=INPUT_DIR, help=f"folders with conditions (default: {INPUT_DIR})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"output directory (default: {OUTPUT_DIR})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Load the costs, build and save one feature matrix per metric and print a summary.
    :param argv: command-line arguments (None = sys.argv).
    """
    args = parse_arguments(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    counter = LoggerUtils.setup(logger, os.path.join(args.output_dir, LOG_FILE))
    logger.info(f"=== Input: {args.input_dir} ===")
    reader = CostReader(args.input_dir)
    costs = reader.get_costs()
    feature_matrix = FeatureMatrix(costs)
    feature_matrix.log_incomplete_subjects()
    n_rows = 0
    for metric in METRICS:
        matrix = feature_matrix.get_matrix(metric)
        output_path = os.path.join(args.output_dir, f"{metric}.csv")
        matrix.to_csv(output_path, sep=SEPARATOR, decimal=".", index=False)
        n_rows = len(matrix)
        logger.info(f"=== Saved: {output_path} ===")
    logger.info(f"Files read: {len(costs)}/{len(CONDITIONS) * len(METRICS)} | "
                f"subjects: {len(reader.get_subjects())} | rows per matrix: {n_rows} | warnings: {counter.warnings} | errors: {counter.errors}")


if __name__ == "__main__":
    main()
