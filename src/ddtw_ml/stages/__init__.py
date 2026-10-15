"""
Stages of the pipeline, one per stage of config.toml ([stages.<key>]): a module <key>.py, or a package <key>/ when
the implementation is split into several files (stage.py with the class, exported by __init__.py). Each defines a
subclass of ddtw_ml.pipeline.Stage with run(). Run them with python main.py; the order follows from their inputs (config.toml),
their parameters are in settings.toml.
"""
