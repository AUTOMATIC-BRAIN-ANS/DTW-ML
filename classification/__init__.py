"""
Classification of the breathing condition (BAS, B6, B10, B15) from sliding-window DTW costs of ABP-CBFV metric pairs.

Run in order (python -m <module>, --help for the options):
1. project.classification.feature_extraction.build_feature_matrix - DTW costs -> one feature matrix per method and
   metric pair,
2. project.classification.classify - cross-validation of every model -> metrics.json, confusion matrix, feature
   importance, figure,
3. project.classification.summary.collect_metrics - metrics.json of every configuration -> one table per model.

Packages: modeling (feature matrix data, models, cross-validation), scoring (classification metrics, bootstrap
confidence intervals), reporting (log, result files, figure). Settings shared by the steps: config.py.
"""
