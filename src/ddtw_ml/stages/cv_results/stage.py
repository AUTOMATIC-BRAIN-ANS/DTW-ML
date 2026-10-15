"""
Stage `cv_results` (features -> cv_results): cross-validation of every model of [stages.cv_results] models
(settings.toml) on the feature matrix of every method and metric pair: {method}/{pair}/{model}/{validation}/
(see classify.py).
"""

from typing import Any

from ddtw_ml.config import ConfigError
from ddtw_ml.pipeline import Stage
from ddtw_ml.stages.cv_results.classify import Options, classify
from ddtw_ml.stages.cv_results.constants import MODELS, VALIDATIONS
from ddtw_ml.stages.features import CONDITIONS


class CrossValidateClassifiers(Stage):
    PARAMS = {"models": list, "validations": list, "classes": list, "n_repeats": int, "permutation_repeats": int,
              "n_bootstrap": int, "ci_level": float, "random_state": int, "n_jobs": int}

    @classmethod
    def check(cls, params: dict[str, Any], where: str) -> None:
        for key, allowed in (("models", MODELS), ("validations", VALIDATIONS), ("classes", CONDITIONS)):
            unknown = [v for v in params[key] if v not in allowed]
            if unknown:
                raise ConfigError(f"{where}: unknown {key} {unknown}; allowed: {allowed}.")

    def run(self) -> None:
        ctx = self.ctx
        p = ctx.params
        classify(Options(input_dir=str(ctx.input()), output_dir=str(ctx.output()), methods=ctx.analysis["methods"],
                         metrics=ctx.analysis["pairs"], models=p["models"], validations=p["validations"],
                         classes=p["classes"], n_repeats=p["n_repeats"], permutation_repeats=p["permutation_repeats"],
                         n_bootstrap=p["n_bootstrap"], ci_level=float(p["ci_level"]), random_state=p["random_state"],
                         n_jobs=p["n_jobs"]))
        ctx.processed("all classifiers (see classification.log)")
