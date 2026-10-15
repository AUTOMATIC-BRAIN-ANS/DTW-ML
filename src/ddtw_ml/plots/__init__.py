"""
Plots of the pipeline: one module per plot of config.toml ([plots.<key>]), defining a subclass of
ddtw_ml.pipeline.Stage with run(), writing to {plots root}/{version}/. Enable them in settings.toml or run them with
python main.py run --plots.
"""
