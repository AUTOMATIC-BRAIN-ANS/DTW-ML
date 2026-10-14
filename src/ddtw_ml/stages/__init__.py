"""
Stages of the pipeline: one module per stage of config.toml, each with run(ctx: StageContext) -> None.
Run them with python -m ddtw_ml run; the order follows from their inputs.
"""
