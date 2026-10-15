"""
State of the stages and plots of a data version: the finished runs, recorded in {data root}/{version}/pipeline_state.json
next to pipeline.log (outside the stage directories, which the next stages list as their inputs).

A stage is done when its last run finished, its output directory exists and none of its inputs was run after it.
The state also records the parameters of the last run of every stage.
"""

import json
import os
from collections.abc import Collection
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
    :return: {"{kind}:{key}": {"finished": ISO time, "processed": n, "rejected": n, "params": {...}}}; empty if
        nothing ran yet.
    """
    path = state_path(config, version)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(config: Config, version: str, state: dict[str, dict]) -> None:
    path = state_path(config, version)
    os.makedirs(path.parent, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def is_done(config: Config, node: Node, version: str, state: dict[str, dict], ran: Collection[str] = ()) -> bool:
    """
    Function to check whether a stage (or plot) is done: its last run finished, its output exists and none of its
    inputs ran after it.
    :param config: configuration.
    :param node: stage or plot.
    :param version: data version.
    :param state: finished runs (see read_state()).
    :param ran: stages run (or to be run) earlier in the current run.
    :return: True if the stage can be skipped.
    """
    entry = state.get(state_key(node))
    if entry is None or not output_dir(config, node, version).is_dir():
        return False
    for key in node.inputs:
        if key not in config.stages:  # raw data
            continue
        if key in ran:
            return False
        source = state.get(state_key(config.stages[key]))
        if source is None or source["finished"] > entry["finished"]:
            return False
    return True


def describe(config: Config, kind: str = "stage", version: str | None = None) -> list[str]:
    """
    Function to describe the stages (or plots) in the order of execution: inputs, output directory, whether the
    stage is done or its output exists and whether the module of the stage exists.
    :param config: configuration.
    :param kind: "stage" or "plot".
    :param version: data version (default: the active one).
    :return: lines of the description.
    """
    v = config.get_version(version)
    nodes = nodes_of(config, kind)
    state = read_state(config, v.name)
    lines = [f"Version {v.name}{' (frozen)' if v.frozen else ''}: {v.description}"]
    for i, key in enumerate(config.order(kind), start=1):
        node = nodes[key]
        out = output_dir(config, node, v.name)
        if is_done(config, node, v.name, state):
            status = f"done {state[state_key(node)]['finished']}"
        else:
            status = "exists" if out.is_dir() else "missing"
        module = "" if module_available(node) else "  [module not implemented]"
        lines.append(f"{i:2}. {key:<12} <- {', '.join(node.inputs):<22} {out} ({status}){module}")
    return lines
