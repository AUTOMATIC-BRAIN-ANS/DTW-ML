"""
Stage `scores` (cv_results -> scores): metrics of every configuration collected into one table per model:
{model}.csv (see core/ml/summary.py).
"""

from ddtw_ml.core.ml import summary
from ddtw_ml.pipeline import StageContext


def run(ctx: StageContext) -> None:
    summary.main(["--metrics", *ctx.params["pairs"], "--input-dir", str(ctx.input()), "--output-dir", str(ctx.output())])
    ctx.processed("all models (see summary.log)")
