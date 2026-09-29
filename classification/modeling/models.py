"""
Models classifying the breathing condition (LogReg, RandomForest, CatBoost, SVM-RBF, HistGradBoost).
"""

from collections.abc import Callable

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from project.classification.config import RANDOM_STATE


class ModelFactory:
    @staticmethod
    def get(name: str, random_state: int = RANDOM_STATE) -> BaseEstimator:
        """
        Method to get an untrained model. LogReg and SVM-RBF need imputation and scaling (fitted on the training folds
        only, inside the pipeline); the tree ensembles handle NaN natively.
        :param name: model name (one of MODELS).
        :param random_state: seed.
        :return: estimator.
        :raise ValueError: if the model is unknown.
        """
        models: dict[str, Callable[[], BaseEstimator]] = {
            "LogReg": lambda: make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000)),
            "RandomForest": lambda: RandomForestClassifier(n_estimators=500, random_state=random_state),
            "CatBoost": lambda: CatBoostClassifier(
                iterations=500, random_seed=random_state, verbose=0, thread_count=1, allow_writing_files=False),
            # Platt scaling for predict_proba (replaces the deprecated SVC(probability=True))
            "SVM-RBF": lambda: make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                CalibratedClassifierCV(SVC(kernel="rbf", random_state=random_state), ensemble=False)),
            "HistGradBoost": lambda: HistGradientBoostingClassifier(random_state=random_state),
        }
        if name not in models:
            raise ValueError(f"Unknown model: {name}! Available: {', '.join(models)}")
        return models[name]()
