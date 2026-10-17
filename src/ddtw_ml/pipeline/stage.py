"""
Interface of a stage (or plot): a subclass of Stage named in config.toml ([stages.<key>] class), with run(). The stage
reads its inputs and writes its output only through its context (StageContext), which also gives it its parameters
(its section of settings.toml), the parameters of the analysis, its logger (stage.<key> or plot.<key>) and records the
processed and rejected inputs.
"""

import logging
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ddtw_ml.config import Config, ConfigError, Node, Settings
from ddtw_ml.pipeline.spoiled import copy_to_spoiled


class PipelineError(RuntimeError):
    """
    A stage cannot be run: unknown, missing input or output in a frozen version.
    """


class UsedAnalysis(Mapping[str, list[str]]):
    """
    Read-only view of the parameters of the analysis that records every parameter read (key -> value).
    """

    def __init__(self, analysis: Mapping[str, list[str]], used: dict[str, list[str]]) -> None:
        self._analysis = analysis
        self._used = used

    def __getitem__(self, key: str) -> list[str]:
        value = self._analysis[key]
        self._used[key] = list(value)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._analysis)

    def __len__(self) -> int:
        return len(self._analysis)


@dataclass
class StageReport:
    """
    Summary of one run of a stage.
    """

    key: str
    processed: int = 0
    rejected: list[str] = field(default_factory=list)


class StageContext:
    """
    Everything a stage may use: paths of its inputs and output in the selected version, its parameters and the
    parameters of the analysis, rejection of inputs and its logger.
    """

    def __init__(self, config: Config, node: Node, version: str, settings: Settings,
                 log: logging.Logger | None = None) -> None:
        self.config = config
        self.node = node
        self.version = version
        self.settings = settings
        self.log = log or logging.getLogger(f"{node.kind}.{node.key}")
        self.report = StageReport(node.key)
        # the parameters of the analysis the stage read, recorded in the state of the run
        self.analysis_used: dict[str, list[str]] = {}

    @property
    def params(self) -> dict[str, Any]:
        """
        Parameters of the stage ([stages.<key>] or [plots.<key>] in settings.toml, without "enabled").
        """
        return self.settings.params(self.node)

    @property
    def analysis(self) -> Mapping[str, list[str]]:
        """
        Parameters of the analysis shared by the stages ([analysis] in settings.toml); the ones read are recorded in
        analysis_used.
        """
        return UsedAnalysis(self.settings.analysis, self.analysis_used)

    def input(self, key: str | None = None) -> Path:
        """
        Method to get the directory of an input of the stage.
        :param key: key of the input (may be omitted when the stage has one input).
        :return: path of the input in the version of the run.
        :raise PipelineError: if the key is not an input of the stage.
        """
        if key is None:
            if len(self.node.inputs) != 1:
                raise PipelineError(f"Stage '{self.node.key}' has several inputs {self.node.inputs}; name one.")
            key = self.node.inputs[0]
        if key not in self.node.inputs:
            raise PipelineError(f"'{key}' is not an input of '{self.node.key}' (inputs: {self.node.inputs}).")
        return self.config.stage_dir(key, self.version)

    def output(self, *parts: str) -> Path:
        """
        Method to get (and create) the output directory of the stage, or a subdirectory of it.
        :param parts: names of the subdirectories, e.g. (method, condition).
        :return: path of the output directory.
        :raise PipelineError: if the version is frozen.
        """
        if self.config.get_version(self.version).frozen:
            raise PipelineError(f"Version {self.version} is frozen; '{self.node.key}' cannot write to it.")
        out = output_dir(self.config, self.node, self.version)
        if is_file_output(out):
            raise PipelineError(f"The output of '{self.node.key}' is a file ({out.name}); use output_file().")
        path = out.joinpath(*parts)
        os.makedirs(path, exist_ok=True)
        return path

    def output_file(self) -> Path:
        """
        Method to get the output file of a stage whose output is one file (its path has an extension, e.g.
        recordings_metadata.csv); its directory is created.
        :return: path of the output file.
        :raise PipelineError: if the version is frozen or the output of the stage is a directory.
        """
        if self.config.get_version(self.version).frozen:
            raise PipelineError(f"Version {self.version} is frozen; '{self.node.key}' cannot write to it.")
        out = output_dir(self.config, self.node, self.version)
        if not is_file_output(out):
            raise PipelineError(f"The output of '{self.node.key}' is a directory ({out}); use output().")
        os.makedirs(out.parent, exist_ok=True)
        return out

    def processed(self, name: str) -> None:
        """
        Method to record a successfully processed input.
        :param name: name of the input (file or subject).
        """
        self.report.processed += 1
        self.log.info(f"✔ {name}")

    def reject(self, src: str | Path, condition: str, reason: str) -> None:
        """
        Method to record a rejected input: it is copied to the spoiled data of the version, with the reason.
        :param src: path of the rejected file or directory (left in place).
        :param condition: condition of the input (subdirectory of the spoiled data).
        :param reason: reason of the rejection.
        """
        if self.config.get_version(self.version).frozen:
            raise PipelineError(f"Version {self.version} is frozen; '{self.node.key}' cannot write to it.")
        dst = copy_to_spoiled(str(src), str(self.config.spoiled_dir(self.version) / condition), reason)
        self.report.rejected.append(str(src))
        self.log.warning(f"✖ {Path(src).name}: {reason} (copied to {dst})")


class Stage:
    """
    Base of the stages and plots. A subclass implements run() and declares its parameters in PARAMS (name -> type);
    their values are in its section of settings.toml, checked before the pipeline runs (check_params()).
    """

    # parameters of the stage in settings.toml: name -> type (bool, int, float, str, list of str, dict of numbers)
    PARAMS: ClassVar[dict[str, type]] = {}

    def __init__(self, ctx: StageContext) -> None:
        self.ctx = ctx

    @property
    def name(self) -> str:
        """
        Name of the stage in the log of the run: its key ("plot <key>" for a plot).
        """
        node = self.ctx.node
        return node.key if node.kind == "stage" else f"plot {node.key}"

    @property
    def log(self) -> logging.Logger:
        return self.ctx.log

    def run(self) -> None:
        raise NotImplementedError

    @classmethod
    def check_params(cls, params: dict[str, Any], where: str) -> None:
        """
        Method to check the parameters of the stage: the names and types of PARAMS, then check() of the subclass.
        :param params: parameters (section of settings.toml without "enabled").
        :param where: name of the section, for the error messages.
        :raise ConfigError: if a parameter is missing, unknown or invalid.
        """
        missing = [k for k in cls.PARAMS if k not in params]
        unknown = sorted(set(params) - set(cls.PARAMS))
        problems = ([f"missing parameter(s) {missing}"] if missing else []) + (
            [f"unknown parameter(s) {unknown}"] if unknown else [])
        if problems:
            raise ConfigError(f"{where}: {'; '.join(problems)}; parameters of {cls.__name__}: "
                              f"{list(cls.PARAMS) or 'none'}.")
        for key, expected in cls.PARAMS.items():
            if not has_type(params[key], expected):
                raise ConfigError(f"{where}: '{key}' must be {TYPE_NAMES[expected]}, got {params[key]!r}.")
        cls.check(params, where)

    @classmethod
    def check(cls, params: dict[str, Any], where: str) -> None:
        """
        Method of a subclass to check the values of its parameters (e.g. allowed names).
        :raise ConfigError: if a value is not allowed.
        """


TYPE_NAMES = {bool: "true or false", int: "an integer", float: "a number", str: "a string",
              list: "a list of names", dict: "a table of numbers"}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def has_type(value: Any, expected: type) -> bool:
    """
    Function to check the type of a parameter of settings.toml.
    :param value: value read from TOML.
    :param expected: bool, int, float (an integer is accepted), str, list (of strings) or dict (of numbers).
    :return: True if the value has the type.
    """
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return is_number(value)
    if expected is str:
        return isinstance(value, str)
    if expected is list:
        return isinstance(value, list) and len(value) > 0 and all(isinstance(v, str) for v in value)
    if expected is dict:
        return isinstance(value, dict) and all(is_number(v) for v in value.values())
    raise TypeError(f"Unsupported type of a parameter: {expected}")


def is_file_output(path: Path) -> bool:
    """
    Function to check whether the output of a stage is one file: its path has an extension (e.g. .csv).
    :param path: output path of the stage (see output_dir()).
    :return: True for a file, False for a directory.
    """
    return path.suffix != ""


def output_dir(config: Config, node: Node, version: str) -> Path:
    """
    Function to get the output directory of a stage or plot (or its output file, see is_file_output()).
    :param config: configuration.
    :param node: stage or plot.
    :param version: data version.
    :return: path of the output directory (or file).
    """
    if node.kind == "plot":
        return config.plot_dir(node.key, version)
    return config.stage_dir(node.key, version)
