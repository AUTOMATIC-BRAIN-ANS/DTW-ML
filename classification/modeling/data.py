"""
Feature matrix of one DTW cost method and metric pair prepared for classification: features X, breathing
condition y and subjects as groups.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from project.classification.config import CLASSES, ID, LOGGER_NAME, TARGET
from project.classification.feature_extraction.build_feature_matrix import N_OBS, SEPARATOR

logger = logging.getLogger(LOGGER_NAME)


class FeatureMatrixData:
    def __init__(self, file_path: str, classes: list[str] = CLASSES) -> None:
        """
        Constructor of the FeatureMatrixData class: load a feature matrix and keep the rows of the chosen classes.
        Rows without any feature (subject missing from the metric file) are dropped; n_obs is not used as a feature.
        :param file_path: path to the feature matrix (CSV).
        :param classes: breathing conditions to classify.
        :raise ValueError: if fewer than two classes are left.
        """
        df = pd.read_csv(file_path, sep=SEPARATOR, dtype={ID: str, TARGET: str})
        df = df[df[TARGET].isin(classes)]
        self.features = [col for col in df.columns if col not in (ID, TARGET, N_OBS)]
        empty = df[self.features].isna().all(axis=1)
        for _, row in df[empty].iterrows():
            logger.warning(f"{row[ID]} / {row[TARGET]}: no features - row dropped")
        self.df = df[~empty].reset_index(drop=True)
        self.classes = [c for c in classes if c in set(self.df[TARGET])]
        if len(self.classes) < 2:
            raise ValueError(f"At least two classes are needed, found: {self.classes}!")

    def get_features(self) -> pd.DataFrame:
        """
        Method to get the feature matrix X.
        :return: features, one row per subject and condition.
        """
        return self.df[self.features]

    def get_target(self) -> np.ndarray:
        """
        Method to get the target y.
        :return: breathing conditions.
        """
        return self.df[TARGET].to_numpy()

    def get_groups(self) -> np.ndarray:
        """
        Method to get the groups (subjects), so that one subject is never in both the training and the test set.
        :return: subject identifiers.
        """
        return self.df[ID].to_numpy()

    def describe(self) -> dict[str, Any]:
        """
        Method to describe the loaded data (dimensions, classes, subjects).
        :return: dictionary: property -> value.
        """
        return {
            "n_rows": len(self.df),
            "n_subjects": int(self.df[ID].nunique()),
            "n_features": len(self.features),
            "classes": self.classes,
            "chance_level": 1 / len(self.classes),
        }
