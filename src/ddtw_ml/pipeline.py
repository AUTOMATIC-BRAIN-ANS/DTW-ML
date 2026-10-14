"""
Runner of the pipeline: runs the stages (or plots) of config.toml in the order of their inputs, in one data version.
Every stage is a module with run(ctx: StageContext) -> None; it reads its inputs and writes its output only through
the context, which also records the processed and rejected inputs.
"""

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from ddtw_ml.config import Config, Node
from ddtw_ml.utils.logger import LoggerUtils
from ddtw_ml.utils.spoiled import copy_to_spoiled

LOGGER_NAME = "pipeline"
LOG_FILE = "pipeline.log"


class PipelineError(RuntimeError):
    """
    A stage cannot be run: unknown, missing input or output in a frozen version.
    """


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
    Everything a stage may use: paths of its inputs and output in the selected version, run parameters, rejection of
    inputs and the logger.
    """

    def __init__(self, config: Config, node: Node, version: str, log: logging.Logger) -> None:
        self.config = config
        self.node = node
        self.version = version
        self.log = log
        self.report = StageReport(node.key)

    @property
    def params(self) -> dict:
        """
        Run parameters ([run] in config.toml).
        """
        return self.config.run

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
        path = output_dir(self.config, self.node, self.version).joinpath(*parts)
        os.makedirs(path, exist_ok=True)
        return path

    def processed(self, name: str) -> None:
        """
        Method to record a successfully processed input.
        :param name: name of the input (file or subject).
        """
        self.report.processed += 1
        self.log.info(f"[{self.node.key}] ✔ {name}")

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
        self.log.warning(f"[{self.node.key}] ✖ {Path(src).name}: {reason} (copied to {dst})")


def output_dir(config: Config, node: Node, version: str) -> Path:
    """
    Function to get the output directory of a stage or plot.
    :param config: configuration.
    :param node: stage or plot.
    :param version: data version.
    :return: path of the output directory.
    """
    if node.kind == "plot":
        return config.plot_dir(node.key, version)
    return config.stage_dir(node.key, version)


def nodes_of(config: Config, kind: str) -> dict[str, Node]:
    return config.plots if kind == "plot" else config.stages


def select(config: Config, kind: str = "stage", keys: list[str] | None = None, start: str | None = None,
           end: str | None = None) -> list[str]:
    """
    Function to select the stages (or plots) to run, in the order of execution.
    :param config: configuration.
    :param kind: "stage" or "plot".
    :param keys: stages to run (default: all).
    :param start: first stage to run (with everything after it).
    :param end: last stage to run (with everything before it).
    :return: ordered keys.
    :raise PipelineError: if a key is unknown.
    """
    order = config.order(kind)
    for key in (keys or []) + [k for k in (start, end) if k is not None]:
        if key not in order:
            raise PipelineError(f"Unknown {kind} '{key}'. Known: {order}.")
    first = order.index(start) if start else 0
    last = order.index(end) if end else len(order) - 1
    selected = order[first:last + 1]
    return [k for k in selected if keys is None or k in keys]


def module_available(node: Node) -> bool:
    """
    Function to check whether the module of a stage exists (without importing it).
    :param node: stage or plot.
    :return: True if the module can be imported.
    """
    try:
        return importlib.util.find_spec(node.module) is not None
    except ModuleNotFoundError:
        return False


def describe(config: Config, kind: str = "stage", version: str | None = None) -> list[str]:
    """
    Function to describe the stages (or plots) in the order of execution: inputs, output directory, whether the
    output exists and whether the module of the stage exists.
    :param config: configuration.
    :param kind: "stage" or "plot".
    :param version: data version (default: the active one).
    :return: lines of the description.
    """
    v = config.get_version(version)
    nodes = nodes_of(config, kind)
    lines = [f"Version {v.name}{' (frozen)' if v.frozen else ''}: {v.description}"]
    for i, key in enumerate(config.order(kind), start=1):
        node = nodes[key]
        out = output_dir(config, node, v.name)
        status = "exists" if out.is_dir() else "missing"
        module = "" if module_available(node) else "  [module not implemented]"
        lines.append(f"{i:2}. {key:<12} <- {', '.join(node.inputs):<22} {out} ({status}){module}")
    return lines


def run_pipeline(config: Config, keys: list[str], kind: str = "stage", version: str | None = None,
                 dry_run: bool = False, log: logging.Logger | None = None) -> list[StageReport]:
    """
    Function to run stages (or plots) in the given order.
    :param config: configuration.
    :param keys: ordered keys of the stages to run (see select()).
    :param kind: "stage" or "plot".
    :param version: data version (default: the active one).
    :param dry_run: only log the inputs and outputs, run nothing.
    :param log: logger (default: console + {data root}/{version}/pipeline.log, appended).
    :return: reports of the stages that ran.
    :raise PipelineError: if the version is frozen or an input of a stage is missing.
    """
    v = config.get_version(version)
    if v.frozen and not dry_run:
        raise PipelineError(f"Version {v.name} is frozen; set [data] version to another one.")
    if log is None:
        log = logging.getLogger(LOGGER_NAME)
        if dry_run:
            log.handlers.clear()
            log.addHandler(logging.StreamHandler())
            log.setLevel(logging.INFO)
            log.propagate = False
        else:
            os.makedirs(config.data_root / v.name, exist_ok=True)
            LoggerUtils.setup(log, str(config.data_root / v.name / LOG_FILE), mode="a")
    nodes = nodes_of(config, kind)
    reports = []
    for key in keys:
        node = nodes[key]
        log.info(f"=== {key} ({node.module}) ===")
        for i in node.inputs:
            path = config.stage_dir(i, v.name)
            log.info(f"  input  {i}: {path}{'' if path.is_dir() else ' (missing)'}")
        log.info(f"  output {key}: {output_dir(config, node, v.name)}")
        if dry_run:
            continue
        missing = [i for i in node.inputs if not config.stage_dir(i, v.name).is_dir()]
        if missing:
            raise PipelineError(f"Missing input of '{key}': {missing}. Run the earlier stages first.")
        ctx = StageContext(config, node, v.name, log)
        importlib.import_module(node.module).run(ctx)
        log.info(f"=== {key}: {ctx.report.processed} processed, {len(ctx.report.rejected)} rejected ===")
        reports.append(ctx.report)
    return reports
