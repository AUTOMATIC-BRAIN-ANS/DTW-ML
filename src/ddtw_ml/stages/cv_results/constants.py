"""
Constants of the classification of the breathing condition: the available models and validation schemes, the defaults
of the cross-validation (the values used by the pipeline are in settings.toml, [stages.cv_results]) and the names of the
output files, shared by classify.py, the modeling, scoring and reporting packages and the stage `scores`.
"""

from ddtw_ml.stages.features import CONDITIONS

# ============================== CONFIGURATION ==============================
METHODS = ["td-method", "d-method"]
CLASSES = CONDITIONS
MODELS = ["LogReg", "RandomForest", "CatBoost", "SVM-RBF", "HistGradBoost"]
VALIDATIONS = ["LOSO", "RepeatedSGKF"]
N_SPLITS = 5
N_REPEATS = 10
PERMUTATION_REPEATS = 10
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
RANDOM_STATE = 42
N_JOBS = -1
LOG_FILE = "classification.log"
# ===========================================================================

ID = "id"
TARGET = "condition"
METRICS_FILE = "metrics.json"
CONFUSION_FILE = "confusion_matrix.json"
IMPORTANCE_FILE = "feature_importance.csv"
FIGURE_FILE = "figure.pdf"
METRIC_NAMES = ["ROC-AUC", "BACC", "F1", "Prec", "Rec", "PR-AUC", "MCC", "GM", "Spec", "Sens", "NPV"]
# full names of the metrics (tables of the results), in the order of METRIC_NAMES
METRIC_LABELS = {
    "ROC-AUC": "ROC-AUC",
    "BACC": "Balanced accuracy",
    "F1": "F1-score",
    "Prec": "Precision",
    "Rec": "Recall",
    "PR-AUC": "PR-AUC",
    "MCC": "Matthews correlation coefficient",
    "GM": "Geometric mean",
    "Spec": "Specificity",
    "Sens": "Sensitivity",
    "NPV": "Negative predictive value",
}
# name of the logger of one classifier (shared by the modules, reconfigured for each method, metric and model); it
# writes to {model}.log in the results, not to pipeline.log
LOGGER_NAME = "stage.cv_results.classifier"
