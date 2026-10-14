"""
Stage `cv_results` (features -> cv_results): cross-validation of every model (core/ml/settings.py) on the feature
matrix of every method and metric pair: {method}/{pair}/{model}/{validation}/ (see core/ml/classify.py).
"""

from ddtw_ml.core.ml import classify
from ddtw_ml.pipeline import StageContext


def run(ctx: StageContext) -> None:
    classify.main(["--methods", *ctx.params["methods"], "--metrics", *ctx.params["pairs"],
                   "--input-dir", str(ctx.input()), "--output-dir", str(ctx.output())])
    ctx.processed("all classifiers (see classification.log)")
