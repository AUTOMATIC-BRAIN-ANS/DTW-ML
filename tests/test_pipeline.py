"""
Unit tests for the pipeline package (Stage, StageContext, registry, state and orchestrator) and the entry point
(main.py), with fake stage classes and a temporary configuration.
"""

import importlib.machinery
import logging
import shutil
import sys
import types

import pytest

from ddtw_ml.config import CONFIG_ENV, ConfigError, load_config, load_settings
from ddtw_ml.log import PIPELINE_LOGGERS
from ddtw_ml.pipeline import PipelineError, Stage, StageContext, StageReport
from ddtw_ml.pipeline.orchestrator import Orchestrator
from ddtw_ml.pipeline.registry import create_stage, select
from ddtw_ml.pipeline.state import describe, read_state
from main import main

CONFIG = """
[data]
version = "2.0"
[data.raw]
path = "raw"
[versions."1.0"]
frozen = true
[versions."2.0"]

[stages.first]
input = ["raw"]
class = "fake_stages.first.First"
[stages.second]
input = ["first"]
class = "fake_stages.second.Second"
[stages.third]
input = ["second"]
class = "fake_stages.not_implemented.Third"

[plots.overview]
input = ["second"]
class = "fake_stages.overview.Overview"
"""

SETTINGS = """
[pipeline]
force = false
dry_run = false
[pipeline.logging]
level = "INFO"

[analysis]
conditions = ["B6"]
methods = ["d-method"]
pairs = ["ABP_RR-CBFV_RR"]
metrics = ["RR"]

[stages.first]
enabled = true
size = 1
[stages.second]
enabled = true
[stages.third]
enabled = false

[plots.overview]
enabled = false
"""


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG, encoding="utf-8")
    (tmp_path / "config.local.toml").write_text(
        f'[data]\nroot = "{(tmp_path / "data").as_posix()}"\n[plots]\nroot = "{(tmp_path / "plots").as_posix()}"\n',
        encoding="utf-8")
    (tmp_path / "settings.toml").write_text(SETTINGS, encoding="utf-8")
    (tmp_path / "data" / "raw" / "B6").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "B6" / "V1.csv").write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV, str(path))
    return load_config(path)


def write_settings(tmp_path, *changes: tuple[str, str]) -> None:
    text = SETTINGS
    for old, new in changes:
        assert old in text
        text = text.replace(old, new, 1)
    (tmp_path / "settings.toml").write_text(text, encoding="utf-8")


@pytest.fixture
def settings(config, tmp_path):
    return load_settings(config, tmp_path / "config.toml")


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    """
    Fake stage classes recording the order in which they run.
    """
    calls: list[str] = []

    class First(Stage):
        PARAMS = {"size": int}

        def run(self) -> None:
            calls.append("first")
            ctx = self.ctx
            self.log.info(f"size {ctx.params['size']}, conditions {ctx.analysis['conditions']}")
            for file in (ctx.input() / "B6").iterdir():
                (ctx.output("B6") / file.name).write_text(file.read_text())
                ctx.processed(file.name)

    class Second(Stage):
        def run(self) -> None:
            calls.append("second")
            self.ctx.output("B6")
            self.ctx.reject(self.ctx.input() / "B6" / "V1.csv", "B6", "too short")

    class Overview(Stage):
        def run(self) -> None:
            calls.append("overview")
            self.ctx.output()

    for name, cls in (("first", First), ("second", Second), ("overview", Overview)):
        module = types.ModuleType(f"fake_stages.{name}")
        module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)
        setattr(module, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def run(config, settings, keys: list[str], kind: str = "stage", version: str | None = None, dry_run: bool = False,
        force: bool = False, log: logging.Logger | None = None) -> list[StageReport]:
    """
    Create the stages with the registry and let the orchestrator run them, as main.py does.
    """
    stages = [create_stage(config, settings, key, kind, version) for key in keys]
    return Orchestrator(config, version, log).run(stages, dry_run, force)


@pytest.fixture
def log():
    logger = logging.getLogger("test_pipeline")
    logger.handlers.clear()
    return logger


@pytest.fixture(autouse=True)
def reset_loggers():
    """
    The command line sets up the loggers of the pipeline (handlers, no propagation); restore them after a test.
    """
    yield
    for name in PIPELINE_LOGGERS:
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_select(config):
    assert select(config) == ["first", "second", "third"]
    assert select(config, start="second") == ["second", "third"]
    assert select(config, end="second") == ["first", "second"]
    assert select(config, keys=["third", "first"]) == ["first", "third"]
    assert select(config, "plot") == ["overview"]
    with pytest.raises(PipelineError, match="Unknown stage"):
        select(config, start="fourth")


def test_run_in_order_with_outputs_and_rejections(config, settings, calls, log, tmp_path):
    reports = run(config, settings, ["first", "second"], log=log)
    assert calls == ["first", "second"]
    assert (tmp_path / "data" / "2.0" / "first" / "B6" / "V1.csv").read_text() == "x\n1\n"
    assert reports[0].processed == 1 and reports[1].rejected == [str(tmp_path / "data" / "2.0" / "first" / "B6" / "V1.csv")]
    # the rejected input is copied, not moved
    assert (tmp_path / "data" / "2.0" / "spoiled" / "B6" / "V1.csv").exists()
    assert (tmp_path / "data" / "2.0" / "first" / "B6" / "V1.csv").exists()


def test_a_stage_gets_its_parameters_and_logs_under_its_name(config, settings, calls, log, caplog):
    with caplog.at_level(logging.INFO):
        run(config, settings, ["first", "second"], log=log)
    first = [r.getMessage() for r in caplog.records if r.name == "stage.first"]
    assert first == ["size 1, conditions ['B6']", "✔ V1.csv"]
    assert [r.levelname for r in caplog.records if r.name == "stage.second"] == ["WARNING"]
    # the orchestrator logs the transitions under its own name
    assert any("[1/2] first" in r.getMessage() for r in caplog.records if r.name == "test_pipeline")


def test_a_stage_runs_on_its_own(config, settings, calls, tmp_path):
    stage = create_stage(config, settings, "first")
    assert isinstance(stage, Stage) and stage.ctx.params == {"size": 1} and stage.log.name == "stage.first"
    stage.run()
    assert calls == ["first"] and stage.ctx.report.processed == 1
    assert (tmp_path / "data" / "2.0" / "first" / "B6" / "V1.csv").is_file()


@pytest.mark.parametrize("change, message", [
    (("size = 1", 'size = "1"'), "'size' must be an integer"),
    (("size = 1", ""), r"missing parameter\(s\) \['size'\]"),
    (("size = 1", "size = 1\nsize2 = 2"), r"unknown parameter\(s\) \['size2'\]"),
    (("[stages.second]\nenabled = true", "[stages.second]\nenabled = true\nsize = 1"),
     r"\[stages.second\]: unknown parameter\(s\) \['size'\]; parameters of Second: none"),
])
def test_invalid_parameters(config, calls, tmp_path, capsys, change, message):
    write_settings(tmp_path, change)
    settings = load_settings(config, tmp_path / "config.toml")
    with pytest.raises(ConfigError, match=message):
        for key in ("first", "second"):
            create_stage(config, settings, key)
    # the command line stops before the first stage
    assert main([]) == 1 and calls == []


def test_a_missing_stage_class_stops_the_run_before_it_starts(config, calls, capsys, tmp_path):
    write_settings(tmp_path, ("[stages.third]\nenabled = false", "[stages.third]\nenabled = true"))
    assert main([]) == 1
    assert "Module 'fake_stages.not_implemented' of stage 'third' not found" in capsys.readouterr().err
    assert calls == []


def test_plots_are_written_to_the_plots_root(config, settings, calls, log, tmp_path):
    run(config, settings, ["first", "second"], log=log)
    run(config, settings, ["overview"], kind="plot", log=log)
    assert (tmp_path / "plots" / "2.0" / "overview").is_dir()


def test_missing_input_stops_the_run(config, settings, calls, log):
    with pytest.raises(PipelineError, match="Missing input of 'second'"):
        run(config, settings, ["second"], log=log)
    assert calls == []


def test_frozen_version_is_read_only(config, settings, calls, log):
    with pytest.raises(PipelineError, match="frozen"):
        run(config, settings, ["first"], version="1.0", log=log)
    ctx = StageContext(config, config.stages["first"], "1.0", settings, log)
    with pytest.raises(PipelineError, match="frozen"):
        ctx.output()


def test_dry_run_runs_and_creates_nothing(config, settings, calls, log, tmp_path):
    assert run(config, settings, ["first", "second"], dry_run=True, log=log) == []
    assert calls == []
    assert not (tmp_path / "data" / "2.0").exists()


def test_context_input(config, settings, log):
    ctx = StageContext(config, config.stages["second"], "2.0", settings, log)
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
    log = (tmp_path / "data" / "2.0" / "pipeline.log").read_text(encoding="utf-8")
    # one log for the orchestrator and the stages; every line names its logger
    assert "[INFO] [pipeline] === [1/1] first" in log and "[INFO] [stage.first] size 1" in log
    assert main(["run", "--version", "1.0"]) == 1
    assert "frozen" in capsys.readouterr().err


def test_done_stages_are_skipped_unless_forced(config, settings, calls, log):
    run(config, settings, ["first", "second"], log=log)
    assert set(read_state(config, "2.0")) == {"stage:first", "stage:second"}
    assert run(config, settings, ["first", "second"], log=log) == []
    assert calls == ["first", "second"]
    reports = run(config, settings, ["second"], force=True, log=log)
    assert [r.key for r in reports] == ["second"] and calls == ["first", "second", "second"]


def test_a_stage_runs_again_after_its_input(config, settings, calls, log, tmp_path):
    run(config, settings, ["first", "second"], log=log)
    # an input run again in a later run
    run(config, settings, ["first"], force=True, log=log)
    run(config, settings, ["first", "second"], log=log)
    assert calls == ["first", "second", "first", "second"]
    # an input run again in the same run (its output was removed)
    shutil.rmtree(tmp_path / "data" / "2.0" / "first")
    run(config, settings, ["first", "second"], log=log)
    assert calls[4:] == ["first", "second"]


def test_a_failed_stage_is_not_done(config, settings, calls, log, monkeypatch):
    run(config, settings, ["first", "second"], log=log)

    def fail(self) -> None:
        raise ValueError("broken")

    monkeypatch.setattr(sys.modules["fake_stages.second"].Second, "run", fail)
    with pytest.raises(ValueError, match="broken"):
        run(config, settings, ["second"], force=True, log=log)
    assert "stage:second" not in read_state(config, "2.0")
    assert "stage:first" in read_state(config, "2.0")


def test_parameters_are_recorded_and_a_change_is_warned(config, settings, calls, log, tmp_path, caplog):
    run(config, settings, ["first"], log=log)
    assert read_state(config, "2.0")["stage:first"]["params"] == {"size": 1}
    write_settings(tmp_path, ("size = 1", "size = 2"))
    changed = load_settings(config, tmp_path / "config.toml")
    with caplog.at_level(logging.INFO, logger="test_pipeline"):
        run(config, changed, ["first"], log=log)
    # the skipping rule does not change: the stage is skipped, with a warning
    assert calls == ["first"]
    assert any(r.levelname == "WARNING" and "other parameters {'size': 1}" in r.getMessage() for r in caplog.records)


def test_plots_are_skipped_until_their_input_runs_again(config, settings, calls, log):
    run(config, settings, ["first", "second"], log=log)
    run(config, settings, ["overview"], kind="plot", log=log)
    run(config, settings, ["overview"], kind="plot", log=log)
    assert calls == ["first", "second", "overview"]
    run(config, settings, ["second"], force=True, log=log)
    run(config, settings, ["overview"], kind="plot", log=log)
    assert calls[3:] == ["second", "overview"]


def test_dry_run_shows_the_skipped_stages(config, settings, calls, log, caplog):
    run(config, settings, ["first", "second"], log=log)
    state = read_state(config, "2.0")
    with caplog.at_level(logging.INFO, logger="test_pipeline"):
        assert run(config, settings, ["first", "second"], dry_run=True, log=log) == []
    assert caplog.text.count(": skipped, done") == 2
    assert read_state(config, "2.0") == state and calls == ["first", "second"]


def test_describe_marks_done_stages(config, settings, calls, log):
    run(config, settings, ["first"], log=log)
    lines = describe(config)
    assert "(done " in lines[1] and "(missing)" in lines[2]


def test_command_line_force(config, calls):
    assert main(["run", "--to", "first"]) == 0
    assert main(["run", "--to", "first"]) == 0
    assert main(["run", "--to", "first", "--force"]) == 0
    assert calls == ["first", "first"]


def test_one_command_runs_the_pipeline_as_set_in_the_settings(config, calls, tmp_path):
    # the enabled stages (third is disabled), no plots
    assert main([]) == 0
    assert calls == ["first", "second"]
    # the stages that are done are skipped; an enabled plot is made after the stages
    write_settings(tmp_path, ("[plots.overview]\nenabled = false", "[plots.overview]\nenabled = true"))
    assert main([]) == 0
    assert calls == ["first", "second", "overview"]
    # a disabled stage does not run; force in the settings runs the others again
    write_settings(tmp_path, ("[stages.second]\nenabled = true", "[stages.second]\nenabled = false"),
                   ("force = false", "force = true"))
    assert main([]) == 0
    assert calls[3:] == ["first"]
    # the options of run override the settings for one run: --stages runs a stage even if it is disabled
    assert main(["run", "--stages", "second"]) == 0
    assert calls[4:] == ["second"]


def test_invalid_settings_stop_the_run(config, calls, capsys, tmp_path):
    write_settings(tmp_path, ("force = false", "forced = false"))
    assert main([]) == 1
    assert "unknown setting(s) ['forced']" in capsys.readouterr().err and calls == []


def test_run_needs_no_input(tmp_path):
    # a run reads nothing from the standard input (closed here): no questions, no prompts
    import os
    import subprocess

    from ddtw_ml.config import REPO_ROOT
    shutil.copy(REPO_ROOT / "config.toml", tmp_path / "config.toml")
    (tmp_path / "config.local.toml").write_text(
        f'[data]\nroot = "{(tmp_path / "data").as_posix()}"\n[plots]\nroot = "{(tmp_path / "plots").as_posix()}"\n',
        encoding="utf-8")
    text = (REPO_ROOT / "settings.toml").read_text(encoding="utf-8")
    assert "dry_run = false" in text
    text = text.replace("dry_run = false", "dry_run = true").replace("enabled = false", "enabled = true")
    (tmp_path / "settings.toml").write_text(text, encoding="utf-8")
    # detect_troughs lives outside the repository (imported by the stage beats)
    (tmp_path / "stub").mkdir()
    (tmp_path / "stub" / "detect_troughs.py").write_text("detect_peaks_troughs_optimized = None\n", encoding="utf-8")
    env = {**os.environ, CONFIG_ENV: str(tmp_path / "config.toml"), "PYTHONPATH": str(tmp_path / "stub")}
    result = subprocess.run([sys.executable, str(REPO_ROOT / "main.py")], stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stderr
    # the 15 stages and the 3 plots (all enabled here)
    assert "18 would run, 0 skipped" in result.stderr
    assert not (tmp_path / "data").exists()


def test_an_import_error_of_a_stage_is_reported(config, calls, capsys, monkeypatch):
    def broken(name, *args, **kwargs):
        raise ModuleNotFoundError("No module named 'detect_troughs'")

    monkeypatch.setattr("ddtw_ml.pipeline.registry.importlib.import_module", broken)
    assert main([]) == 1
    assert ("Module 'fake_stages.first' of stage 'first' cannot be imported: No module named 'detect_troughs' (install"
            in capsys.readouterr().err)
    assert calls == []
