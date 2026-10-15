"""
Registry of the stages and plots: their selection in the order of config.toml and their creation from the classes
named there ([stages.<key>] class), with their parameters from settings.toml.
"""

import importlib
import importlib.util

from ddtw_ml.config import Config, Node, Settings
from ddtw_ml.pipeline.stage import PipelineError, Stage, StageContext


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


def stage_class(node: Node) -> type[Stage]:
    """
    Function to import the class of a stage or plot.
    :param node: stage or plot.
    :return: subclass of Stage.
    :raise PipelineError: if the module or class does not exist or the class is not a Stage.
    """
    where = f"[{node.kind}s.{node.key}] class = \"{node.target}\" (config.toml)"
    if not module_available(node):
        raise PipelineError(f"Module '{node.module}' of {node.kind} '{node.key}' not found: {where}.")
    try:
        module = importlib.import_module(node.module)
    except ImportError as e:
        raise PipelineError(f"Module '{node.module}' of {node.kind} '{node.key}' cannot be imported: {e} (install the "
                            f"missing package or put its directory on PYTHONPATH).") from e
    cls = getattr(module, node.class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, Stage)):
        raise PipelineError(f"{where} is not a subclass of ddtw_ml.pipeline.Stage.")
    return cls


def create_stage(config: Config, settings: Settings, key: str, kind: str = "stage",
                 version: str | None = None) -> Stage:
    """
    Function to create one stage (or plot) with its parameters (checked against its class), e.g. to run it on its own:
        create_stage(config, settings, "costs").run()
    Its inputs must exist (outputs of the earlier stages); run on its own, it does not record its state, so the
    orchestrator does not know that it ran.
    :param config: configuration.
    :param settings: settings.
    :param key: key of the stage.
    :param kind: "stage" or "plot".
    :param version: data version (default: the active one).
    :return: the stage.
    :raise ConfigError: if its parameters are invalid.
    :raise PipelineError: if its class cannot be imported.
    """
    node = nodes_of(config, kind)[key]
    cls = stage_class(node)
    cls.check_params(settings.params(node), f"settings.toml [{kind}s.{key}]")
    return cls(StageContext(config, node, config.get_version(version).name, settings))
