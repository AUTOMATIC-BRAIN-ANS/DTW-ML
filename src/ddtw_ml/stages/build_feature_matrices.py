"""
Stage `features` (pairs -> features): one feature matrix per method and metric pair, one row per subject and condition,
built from the sliding-window DTW costs: {method}/{pair}.csv (see core/ml/features.py).
"""

from ddtw_ml.core.ml import features
from ddtw_ml.pipeline import StageContext


def run(ctx: StageContext) -> None:
    for method in ctx.params["methods"]:
        features.main(["--method", method, "--metrics", *ctx.params["pairs"],
                       "--input-dir", str(ctx.input() / method), "--output-dir", str(ctx.output(method))])
        ctx.processed(method)
