"""
Unit tests for config.py, on temporary configuration files and on the config.toml of the repository.
"""

import shutil
from pathlib import Path

import pytest

from ddtw_ml.config import REPO_ROOT, ConfigError, load_config, merge

BASE = """
[data]
version = "2.0"
[data.raw]
path = "raw"

[versions."1.0"]
frozen = true
[versions."1.0".paths]
first = "old_first"
[versions."2.0"]

[run]
conditions = ["B6", "BAS"]

[stages.second]
input = ["first"]
module = "m.second"
[stages.first]
input = ["raw"]
module = "m.first"
[stages.third]
input = ["first", "second"]
module = "m.third"
path = "nested/third"

[plots.overview]
input = ["third"]
module = "m.overview"
"""


def write_config(tmp_path: Path, text: str = BASE, local: str | None = None) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    if local is None:
        local = f'[data]\nroot = "{(tmp_path / "data").as_posix()}"\n[plots]\nroot = "{(tmp_path / "plots").as_posix()}"\n'
    (tmp_path / "config.local.toml").write_text(local, encoding="utf-8")
    return path


def test_paths_of_stages_versions_and_plots(tmp_path):
    config = load_config(write_config(tmp_path))
    data = tmp_path / "data"
    assert config.raw_dir() == data / "raw"
    assert config.stage_dir("raw") == data / "raw"
    assert config.stage_dir("first") == data / "2.0" / "first"
    assert config.stage_dir("third") == data / "2.0" / "nested" / "third"
    # the paths of a version override the default ones
    assert config.stage_dir("first", "1.0") == data / "1.0" / "old_first"
    assert config.stage_dir("second", "1.0") == data / "1.0" / "second"
    assert config.spoiled_dir() == data / "2.0" / "spoiled"
    assert config.plot_dir("overview", "1.0") == tmp_path / "plots" / "1.0" / "overview"
    assert config.get_version("1.0").frozen and not config.get_version().frozen
    assert config.run["conditions"] == ["B6", "BAS"]


def test_order_follows_inputs_then_file_order(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.order() == ["first", "second", "third"]
    assert config.order("plot") == ["overview"]


def test_local_config_is_required_for_the_roots(tmp_path):
    with pytest.raises(ConfigError, match="root"):
        load_config(write_config(tmp_path, local=""))


@pytest.mark.parametrize("change, message", [
    (('version = "2.0"', 'version = "3.0"'), "not described"),
    (('[versions."2.0"]', '[versions."2"]'), "MAJOR.MINOR"),
    (('first = "old_first"', 'fourth = "old"'), "unknown stages"),
    (('input = ["first", "second"]', 'input = ["missing"]'), "unknown inputs"),
    (('input = ["raw"]', 'input = ["third"]'), "cycle"),
    (('module = "m.second"', ''), "Missing 'module'"),
])
def test_invalid_configurations(tmp_path, change, message):
    old, new = change
    assert old in BASE
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, BASE.replace(old, new, 1)))


def test_unknown_stage(tmp_path):
    config = load_config(write_config(tmp_path))
    with pytest.raises(ConfigError, match="Unknown stage"):
        config.stage_dir("fourth")


def test_merge_is_recursive():
    assert merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}, "c": 4}) == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}


def test_repository_config_is_valid(tmp_path):
    # the config.toml of the repository with temporary roots instead of the machine-specific ones
    shutil.copy(REPO_ROOT / "config.toml", tmp_path / "config.toml")
    write_config(tmp_path, (tmp_path / "config.toml").read_text(encoding="utf-8"))
    config = load_config(tmp_path / "config.toml")
    assert config.order()[0] == "recordings"
    assert set(config.order()) == set(config.stages)
    # version 1.0 keeps the directories from before the refactoring
    assert config.stage_dir("costs", "1.0") == tmp_path / "data" / "1.0" / "dtw" / "dtw"
    assert config.stage_dir("costs", "2.0") == tmp_path / "data" / "2.0" / "costs"
