"""
Configuration of the classification of the breathing condition: paths, models, validation schemes, bootstrap and
names of the output files, shared by classify.py and the modeling, scoring, reporting and summary packages.
"""

from project.classification.feature_extraction.build_feature_matrix import CONDITIONS

# ============================== CONFIGURATION ==============================
# feature matrices: {INPUT_DIR}/{method}/{metric}.csv
INPUT_DIR = "C:/Python/ZSSI/data/classification/features"
# results: {OUTPUT_DIR}/{method}/{metric}/{model}/{validation}/
OUTPUT_DIR = "C:/Python/ZSSI/data/classification/results"
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
# name of the logger of one classifier (shared by the modules, reconfigured for each method, metric and model)
LOGGER_NAME = "classification"
