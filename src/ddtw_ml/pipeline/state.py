"""
State of the stages and plots of a data version: the finished runs, recorded in {data root}/{version}/pipeline_state.json
next to pipeline.log (outside the stage directories, which the next stages list as their inputs).

A stage is done when its last run finished, its output directory exists, none of its inputs was run after it and the
parameters of the analysis it read ([analysis] in settings.toml) did not change. The state also records the parameters
of the last run of every stage.
"""

import json
import os
from collections.abc import Collection, Mapping
from pathlib import Path

from ddtw_ml.config import Config, Node
from ddtw_ml.pipeline.registry import module_available, nodes_of
from ddtw_ml.pipeline.stage import output_dir

STATE_FILE = "pipeline_state.json"


def state_path(config: Config, version: str) -> Path:
    return config.data_root / version / STATE_FILE


def state_key(node: Node) -> str:
    # stages and plots may share keys (e.g. beats)
    return f"{node.kind}:{node.key}"


def read_state(config: Config, version: str) -> dict[str, dict]:
    """
    Function to read the finished runs of the stages and plots of a version.
    :param config: configuration.
    :param version: data version.
    :return: {"{kind}:{key}": {"finished": ISO time, "run": number of the run in the version, "processed": n,
        "rejected": n, "params": {...},
        "analysis": {parameter of the analysis read: value}}}; empty if nothing ran yet.
    """
    path = state_path(config, version)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(config: Config, version: str, state: dict[str, dict]) -> None:
    path = state_path(config, version)
    os.makedirs(path.parent, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def next_run(state: dict[str, dict]) -> int:
    """
    Function to get the number of the next finished run of a version: one more than the last one.
    :param state: finished runs (see read_state()).
    :return: number of the run.
    """
    return max((entry.get("run", 0) for entry in state.values()), default=0) + 1


def ran_after(source: dict, entry: dict) -> bool:
    """
    Function to check whether a stage finished after another one, by the numbers of their runs (the clock may give the
    same time to runs that follow each other quickly); runs recorded without a number are compared by their time.
    :param source: state of the first stage.
    :param entry: state of the second stage.
    :return: True if the first stage finished after the second one.
    """
    if "run" in source and "run" in entry:
        return source["run"] > entry["run"]
    return source["finished"] > entry["finished"]


def analysis_changes(entry: dict, analysis: Mapping[str, list[str]]) -> list[str]:
    """
    Function to find the parameters of the analysis that changed since the last run of a stage.
    :param entry: state of the stage (see read_state()).
    :param analysis: current parameters of the analysis ([analysis] in settings.toml).
    :return: names of the changed parameters; ["unknown"] if the run did not record them.
    """
    if "analysis" not in entry:
        return ["unknown"]
    return [key for key, value in entry["analysis"].items() if list(analysis.get(key, [])) != value]


def is_done(config: Config, node: Node, version: str, state: dict[str, dict], ran: Collection[str] = (),
            analysis: Mapping[str, list[str]] | None = None) -> bool:
    """
    Function to check whether a stage (or plot) is done: its last run finished, its output exists, none of its
    inputs ran after it and the parameters of the analysis it read did not change.
    :param config: configuration.
    :param node: stage or plot.
    :param version: data version.
    :param state: finished runs (see read_state()).
    :param ran: stages run (or to be run) earlier in the current run.
    :param analysis: current parameters of the analysis (None: not checked).
    :return: True if the stage can be skipped.
    """
    entry = state.get(state_key(node))
    if entry is None or not output_dir(config, node, version).is_dir():
        return False
    if analysis is not None and analysis_changes(entry, analysis):
        return False
    for key in node.inputs:
        if key not in config.stages:  # raw data
            continue
        if key in ran:
            return False
        source = state.get(state_key(config.stages[key]))
        if source is None or ran_after(source, entry):
            return False
    return True


def describe(config: Config, kind: str = "stage", version: str | None = None,
             analysis: Mapping[str, list[str]] | None = None) -> list[str]:
    """
    Function to describe the stages (or plots) in the order of execution: inputs, output directory, whether the
    stage is done or its output exists and whether the module of the stage exists.
    :param config: configuration.
    :param kind: "stage" or "plot".
    :param version: data version (default: the active one).
    :param analysis: current parameters of the analysis (None: their changes are not checked).
    :return: lines of the description.
    """
    v = config.get_version(version)
    nodes = nodes_of(config, kind)
    state = read_state(config, v.name)
    lines = [f"Version {v.name}{' (frozen)' if v.frozen else ''}: {v.description}"]
    for i, key in enumerate(config.order(kind), start=1):
        node = nodes[key]
        out = output_dir(config, node, v.name)
        if is_done(config, node, v.name, state, analysis=analysis):
            status = f"done {state[state_key(node)]['finished']}"
        else:
            status = "exists" if out.is_dir() else "missing"
        module = "" if module_available(node) else "  [module not implemented]"
        lines.append(f"{i:2}. {key:<12} <- {', '.join(node.inputs):<22} {out} ({status}){module}")
    return lines
