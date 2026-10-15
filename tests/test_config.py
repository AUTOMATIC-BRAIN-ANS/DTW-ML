"""
Unit tests for config.py, on temporary configuration files and on the config.toml of the repository.
"""

import shutil
from pathlib import Path

import pytest

from ddtw_ml.config import REPO_ROOT, ConfigError, load_config, load_settings, merge

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

[stages.second]
input = ["first"]
class = "m.second.Second"
[stages.first]
input = ["raw"]
class = "m.first.First"
[stages.third]
input = ["first", "second"]
class = "m.third.Third"
path = "nested/third"

[plots.overview]
input = ["third"]
class = "m.overview.Overview"
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
    (('class = "m.second.Second"', ''), "Missing 'class'"),
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


def test_missing_local_config_says_how_to_create_it(tmp_path):
    path = write_config(tmp_path)
    (tmp_path / "config.local.toml").unlink()
    with pytest.raises(ConfigError, match="copy config.local.example.toml to config.local.toml"):
        load_config(path)


SETTINGS = '''
[pipeline]
force = false
dry_run = true
[pipeline.logging]
level = "WARNING"

[analysis]
conditions = ["B6", "BAS"]
methods = ["d-method"]
pairs = ["ABP_RR-CBFV_RR"]
metrics = ["RR"]

[stages.first]
enabled = true
window = 10
[stages.second]
enabled = false
[stages.third]
enabled = true

[plots.overview]
enabled = false
'''


def write_settings(tmp_path: Path, text: str = SETTINGS) -> Path:
    path = write_config(tmp_path)
    (tmp_path / "settings.toml").write_text(text, encoding="utf-8")
    return path


def test_settings(tmp_path):
    path = write_settings(tmp_path)
    config = load_config(path)
    settings = load_settings(config, path)
    assert settings.dry_run and not settings.force and settings.log_level == "WARNING"
    assert settings.analysis["conditions"] == ["B6", "BAS"]
    assert settings.enabled(config.stages["first"]) and not settings.enabled(config.stages["second"])
    assert not settings.enabled(config.plots["overview"])
    # the parameters of a stage: its section without "enabled"
    assert settings.params(config.stages["first"]) == {"window": 10}
    assert settings.params(config.stages["third"]) == {}


def test_settings_are_required(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ConfigError, match="settings.toml not found"):
        load_settings(load_config(path), path)


@pytest.mark.parametrize("change, message", [
    (("force = false", "force = 1"), "'force' must be true or false"),
    (("force = false", ""), r"missing setting\(s\) \['force'\]"),
    (("dry_run = true", "dryrun = true"), r"missing setting\(s\) \['dry_run'\]; unknown setting\(s\) \['dryrun'\]"),
    (('level = "WARNING"', 'level = "LOUD"'), "'level' must be one of"),
    (('methods = ["d-method"]', 'methods = []'), "'methods' must be a non-empty list"),
    (('metrics = ["RR"]', ''), r"missing setting\(s\) \['metrics'\]"),
    (("[stages.third]\nenabled = true", ""), r"missing section\(s\) \['stages.third'\]"),
    (("[stages.third]", "[stages.fourth]"),
     r"missing section\(s\) \['stages.third'\]; unknown section\(s\) \['stages.fourth'\]"),
    (("[stages.second]\nenabled = false", '[stages.second]\nenabled = "no"'), "'enabled' must be true or false"),
    (("[plots.overview]", "[plots.summary]"),
     r"missing section\(s\) \['plots.overview'\]; unknown section\(s\) \['plots.summary'\]"),
    (("force = false", "force = "), "not valid TOML"),
])
def test_invalid_settings(tmp_path, change, message):
    old, new = change
    assert old in SETTINGS
    path = write_settings(tmp_path, SETTINGS.replace(old, new, 1))
    with pytest.raises(ConfigError, match=message):
        load_settings(load_config(path), path)
