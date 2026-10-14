"""
Shared fixtures of the stage tests: the config.toml of the repository with temporary data and plot roots, and the
context of a stage in version 2.0.
"""

import logging
import shutil

import pytest

from ddtw_ml.config import REPO_ROOT, Config, load_config
from ddtw_ml.pipeline import StageContext


@pytest.fixture
def config(tmp_path) -> Config:
    shutil.copy(REPO_ROOT / "config.toml", tmp_path / "config.toml")
    (tmp_path / "config.local.toml").write_text(
        f'[data]\nroot = "{(tmp_path / "data").as_posix()}"\n[plots]\nroot = "{(tmp_path / "plots").as_posix()}"\n',
        encoding="utf-8")
    return load_config(tmp_path / "config.toml")


@pytest.fixture
def make_ctx(config):
    """
    Factory of the context of a stage (or plot) of the repository configuration, in version 2.0.
    """
    log = logging.getLogger("test_stages")
    log.handlers.clear()

    def make(key: str, kind: str = "stage") -> StageContext:
        node = config.plots[key] if kind == "plot" else config.stages[key]
        return StageContext(config, node, "2.0", log)

    return make


@pytest.fixture
def no_latex(monkeypatch):
    """
    Render test figures without LaTeX (not installed in CI).
    """
    import matplotlib.pyplot as plt
    monkeypatch.setattr("ddtw_ml.utils.plotting.plt.rc", lambda *args, **kwargs: None)
    monkeypatch.setitem(plt.rcParams, "text.usetex", False)
