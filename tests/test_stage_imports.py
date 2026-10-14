"""
Every stage and plot of config.toml exists, has run(ctx) and can be imported without touching the data.
"""

import importlib
import os
import sys
import types

import pandas as pd
import pytest

from ddtw_ml.config import REPO_ROOT, load_config


def fail(*args, **kwargs):
    raise AssertionError("the data must not be touched")


def modules() -> list[str]:
    """
    Modules of the stages and plots of the config.toml of the repository (with stub roots).
    """
    from ddtw_ml.config import merge, parse_nodes, read_config
    raw = merge(read_config(REPO_ROOT / "config.toml"), {"data": {"root": "."}, "plots": {"root": "."}})
    # stages and plots may share keys (e.g. beats), so their modules are collected separately
    nodes = [*parse_nodes(raw["stages"], "stage", "stages").values(), *parse_nodes(raw["plots"], "plot", "plots").values()]
    return sorted(node.module for node in nodes)


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


@pytest.mark.parametrize("name", modules())
def test_stage_import_has_no_side_effects(name, no_data_access, monkeypatch):
    monkeypatch.delitem(sys.modules, name, raising=False)
    assert callable(importlib.import_module(name).run)


def test_every_stage_module_is_in_the_config(tmp_path):
    configured = set(modules())
    for package in ("ddtw_ml.stages", "ddtw_ml.plots"):
        directory = os.path.dirname(importlib.import_module(package).__file__ or "")
        for file in os.listdir(directory):
            if file.endswith(".py") and file != "__init__.py":
                assert f"{package}.{file[:-3]}" in configured, f"{package}.{file[:-3]} is not used in config.toml"


def test_load_config_needs_the_local_roots(tmp_path):
    # the repository config alone has no machine-specific roots
    (tmp_path / "config.toml").write_text((REPO_ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        load_config(tmp_path / "config.toml")
