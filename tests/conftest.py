"""
Shared fixtures of the stage and plot tests: the config.toml and settings.toml of the repository with temporary data
and plot roots, and the context of a stage in version 2.0 (its logger: stage.<key> or plot.<key>).
"""

import shutil
import sys
import types

import pytest

from ddtw_ml.config import REPO_ROOT, Config, Settings, load_config, load_settings
from ddtw_ml.pipeline import StageContext

# detect_troughs lives outside the repository (not available in CI); the tests that need troughs patch the function
try:
    import detect_troughs  # noqa: F401
except ImportError:
    stub = types.ModuleType("detect_troughs")
    stub.detect_peaks_troughs_optimized = None  # type: ignore[attr-defined]
    sys.modules["detect_troughs"] = stub


@pytest.fixture
def config(tmp_path) -> Config:
    shutil.copy(REPO_ROOT / "config.toml", tmp_path / "config.toml")
    shutil.copy(REPO_ROOT / "settings.toml", tmp_path / "settings.toml")
    (tmp_path / "config.local.toml").write_text(
        f'[data]\nroot = "{(tmp_path / "data").as_posix()}"\n[plots]\nroot = "{(tmp_path / "plots").as_posix()}"\n',
        encoding="utf-8")
    return load_config(tmp_path / "config.toml")


@pytest.fixture
def settings(config, tmp_path) -> Settings:
    return load_settings(config, tmp_path / "config.toml")


@pytest.fixture
def make_ctx(config, settings):
    """
    Factory of the context of a stage (or plot) of the repository configuration, in version 2.0; run a stage with
    e.g. ComputeDtwCosts(make_ctx("costs")).run().
    """

    def make(key: str, kind: str = "stage") -> StageContext:
        node = config.plots[key] if kind == "plot" else config.stages[key]
        return StageContext(config, node, "2.0", settings)

    return make


@pytest.fixture
def no_latex(monkeypatch):
    """
    Render test figures without LaTeX (not installed in CI).
    """
    import matplotlib.pyplot as plt
    monkeypatch.setattr("ddtw_ml.plotting.plt.rc", lambda *args, **kwargs: None)
    monkeypatch.setitem(plt.rcParams, "text.usetex", False)
