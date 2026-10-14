"""
Tests of the pipeline scripts and stages: importing them has no side effects, every stage has run(ctx), and the legacy
scripts that overwrite the raw data refuse to run.
"""

import importlib
import os
import pkgutil
import runpy
import sys
import types

import pandas as pd
import pytest

SCRIPTS = [
    "ddtw_ml.data.get_data_characteristics",
    "ddtw_ml.execution.metrics_extraction",
    "ddtw_ml.execution.perform_dtw",
    "ddtw_ml.execution.preprocessing",
    "ddtw_ml.other.comma_to_dot",
    "ddtw_ml.other.get_data_for_ml",
    "ddtw_ml.other.get_summary",
    "ddtw_ml.other.merging",
    "ddtw_ml.other.parse_dot_separated_data",
    "ddtw_ml.other.reorganise",
    "ddtw_ml.statistics.prepare_for_analysis",
    "ddtw_ml.statistics.rm-anova",
    "ddtw_ml.statistics.statistical_analysis",
    "ddtw_ml.visualisation.alignment_cost",
    "ddtw_ml.visualisation.plot_metrics",
]
STAGES = [f"ddtw_ml.stages.{m.name}" for m in pkgutil.iter_modules(importlib.import_module("ddtw_ml.stages").__path__)]
LEGACY = ["ddtw_ml.other.comma_to_dot", "ddtw_ml.other.parse_dot_separated_data"]


def fail(*args, **kwargs):
    raise AssertionError("the data must not be touched")


@pytest.fixture
def no_data_access(monkeypatch):
    """
    Any access to the data fails the test; detect_troughs (outside the repository) is replaced with a stub.
    """
    # pingouin reads its own dataset list with pandas on import
    importlib.import_module("pingouin")
    monkeypatch.setattr(os, "listdir", fail)
    monkeypatch.setattr(pd, "read_csv", fail)
    stub = types.ModuleType("detect_troughs")
    stub.detect_peaks_troughs_optimized = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "detect_troughs", stub)


@pytest.mark.parametrize("name", SCRIPTS)
def test_import_has_no_side_effects(name, no_data_access, monkeypatch):
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    assert callable(module.main)


@pytest.mark.parametrize("name", STAGES)
def test_stage_import_has_no_side_effects(name, no_data_access, monkeypatch):
    monkeypatch.delitem(sys.modules, name, raising=False)
    assert callable(importlib.import_module(name).run)


@pytest.mark.parametrize("name", LEGACY)
def test_legacy_script_refuses_to_run(name, no_data_access, monkeypatch):
    monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(SystemExit, match="Legacy script disabled"):
        runpy.run_module(name, run_name="__main__")
