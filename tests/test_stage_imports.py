"""
Every stage and plot of config.toml is a module or package with a Stage class that can be imported without touching the
data, every package of stages/ and plots/ is in config.toml, and the settings.toml of the repository matches the
classes.
"""

import importlib
import os
import shutil
import sys
import types

import pandas as pd
import pytest

from ddtw_ml.config import REPO_ROOT, load_config, load_settings
from ddtw_ml.pipeline import Stage
from ddtw_ml.pipeline.registry import create_stage


def fail(*args, **kwargs):
    raise AssertionError("the data must not be touched")


def nodes() -> list:
    """
    Stages and plots of the config.toml of the repository (with stub roots).
    """
    from ddtw_ml.config import merge, parse_nodes, read_config
    raw = merge(read_config(REPO_ROOT / "config.toml"), {"data": {"root": "."}, "plots": {"root": "."}})
    # stages and plots may share keys (e.g. beats), so their modules are collected separately
    return [*parse_nodes(raw["stages"], "stage", "stages").values(), *parse_nodes(raw["plots"], "plot", "plots").values()]


def modules() -> list[str]:
    return sorted(node.module for node in nodes())


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


@pytest.mark.parametrize("node", nodes(), ids=lambda node: f"{node.kind}:{node.key}")
def test_stage_import_has_no_side_effects(node, no_data_access, monkeypatch):
    monkeypatch.delitem(sys.modules, node.module, raising=False)
    cls = getattr(importlib.import_module(node.module), node.class_name)
    assert issubclass(cls, Stage) and cls.run is not Stage.run


def test_repository_settings_match_the_stages(tmp_path):
    shutil.copy(REPO_ROOT / "config.toml", tmp_path / "config.toml")
    shutil.copy(REPO_ROOT / "settings.toml", tmp_path / "settings.toml")
    (tmp_path / "config.local.toml").write_text('[data]\nroot = "."\n[plots]\nroot = "."\n', encoding="utf-8")
    config = load_config(tmp_path / "config.toml")
    settings = load_settings(config, tmp_path / "config.toml")
    for kind, table in (("stage", config.stages), ("plot", config.plots)):
        for key in table:
            assert isinstance(create_stage(config, settings, key, kind), Stage)


def test_every_stage_module_is_in_the_config(tmp_path):
    configured = set(modules())
    for package in ("ddtw_ml.stages", "ddtw_ml.plots"):
        directory = os.path.dirname(importlib.import_module(package).__file__ or "")
        for name in os.listdir(directory):
            if name.endswith(".py") and name != "__init__.py":
                name = name[:-3]
            elif not os.path.isfile(os.path.join(directory, name, "__init__.py")):
                continue
            assert f"{package}.{name}" in configured, f"{package}.{name} is not used in config.toml"


def test_load_config_needs_the_local_roots(tmp_path):
    # the repository config alone has no machine-specific roots
    (tmp_path / "config.toml").write_text((REPO_ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        load_config(tmp_path / "config.toml")
