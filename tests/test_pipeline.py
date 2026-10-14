"""
Unit tests for pipeline.py and the command line (__main__.py), with fake stage modules and a temporary configuration.
"""

import importlib.machinery
import logging
import sys
import types

import pytest

from ddtw_ml.__main__ import main
from ddtw_ml.config import CONFIG_ENV, load_config
from ddtw_ml.pipeline import PipelineError, StageContext, describe, run_pipeline, select

CONFIG = """
[data]
version = "2.0"
[data.raw]
path = "raw"
[versions."1.0"]
frozen = true
[versions."2.0"]
[run]
conditions = ["B6"]

[stages.first]
input = ["raw"]
module = "fake_stages.first"
[stages.second]
input = ["first"]
module = "fake_stages.second"
[stages.third]
input = ["second"]
module = "fake_stages.not_implemented"

[plots.overview]
input = ["second"]
module = "fake_stages.overview"
"""


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    (tmp_path / "config.local.toml").write_text(
        f'[data]\nroot = "{(tmp_path / "data").as_posix()}"\n[plots]\nroot = "{(tmp_path / "plots").as_posix()}"\n',
        encoding="utf-8")
    (tmp_path / "data" / "raw" / "B6").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "B6" / "V1.csv").write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV, str(path))
    return load_config(path)


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    """
    Fake stage modules recording the order in which they run.
    """
    calls: list[str] = []

    def first(ctx: StageContext) -> None:
        calls.append("first")
        for file in (ctx.input() / "B6").iterdir():
            (ctx.output("B6") / file.name).write_text(file.read_text())
            ctx.processed(file.name)

    def second(ctx: StageContext) -> None:
        calls.append("second")
        ctx.output("B6")
        ctx.reject(ctx.input() / "B6" / "V1.csv", "B6", "too short")

    def overview(ctx: StageContext) -> None:
        calls.append("overview")
        ctx.output()

    for name, run in (("first", first), ("second", second), ("overview", overview)):
        module = types.ModuleType(f"fake_stages.{name}")
        module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)
        module.run = run  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.fixture
def log():
    logger = logging.getLogger("test_pipeline")
    logger.handlers.clear()
    return logger


def test_select(config):
    assert select(config) == ["first", "second", "third"]
    assert select(config, start="second") == ["second", "third"]
    assert select(config, end="second") == ["first", "second"]
    assert select(config, keys=["third", "first"]) == ["first", "third"]
    assert select(config, "plot") == ["overview"]
    with pytest.raises(PipelineError, match="Unknown stage"):
        select(config, start="fourth")


def test_run_in_order_with_outputs_and_rejections(config, calls, log, tmp_path):
    reports = run_pipeline(config, ["first", "second"], log=log)
    assert calls == ["first", "second"]
    assert (tmp_path / "data" / "2.0" / "first" / "B6" / "V1.csv").read_text() == "x\n1\n"
    assert reports[0].processed == 1 and reports[1].rejected == [str(tmp_path / "data" / "2.0" / "first" / "B6" / "V1.csv")]
    # the rejected input is copied, not moved
    assert (tmp_path / "data" / "2.0" / "spoiled" / "B6" / "V1.csv").exists()
    assert (tmp_path / "data" / "2.0" / "first" / "B6" / "V1.csv").exists()


def test_plots_are_written_to_the_plots_root(config, calls, log, tmp_path):
    run_pipeline(config, ["first", "second"], log=log)
    run_pipeline(config, ["overview"], kind="plot", log=log)
    assert (tmp_path / "plots" / "2.0" / "overview").is_dir()


def test_missing_input_stops_the_run(config, calls, log):
    with pytest.raises(PipelineError, match="Missing input of 'second'"):
        run_pipeline(config, ["second"], log=log)
    assert calls == []


def test_frozen_version_is_read_only(config, calls, log):
    with pytest.raises(PipelineError, match="frozen"):
        run_pipeline(config, ["first"], version="1.0", log=log)
    ctx = StageContext(config, config.stages["first"], "1.0", log)
    with pytest.raises(PipelineError, match="frozen"):
        ctx.output()


def test_dry_run_runs_and_creates_nothing(config, calls, log, tmp_path):
    assert run_pipeline(config, ["first", "second", "third"], dry_run=True, log=log) == []
    assert calls == []
    assert not (tmp_path / "data" / "2.0").exists()


def test_context_input(config, log):
    ctx = StageContext(config, config.stages["second"], "2.0", log)
    assert ctx.input() == ctx.input("first") == config.stage_dir("first")
    with pytest.raises(PipelineError, match="not an input"):
        ctx.input("third")


def test_describe_marks_missing_modules(config, calls):
    lines = describe(config)
    assert lines[0].startswith("Version 2.0")
    assert "first" in lines[1] and "not implemented" not in lines[1]
    assert "third" in lines[3] and "[module not implemented]" in lines[3]


def test_command_line(config, calls, capsys, tmp_path):
    assert main(["list"]) == 0
    assert "1. first" in capsys.readouterr().out
    assert main(["run", "--to", "first"]) == 0
    assert calls == ["first"]
    assert (tmp_path / "data" / "2.0" / "pipeline.log").is_file()
    assert main(["run", "--version", "1.0"]) == 1
    assert "frozen" in capsys.readouterr().err
    logging.getLogger("pipeline").handlers.clear()
