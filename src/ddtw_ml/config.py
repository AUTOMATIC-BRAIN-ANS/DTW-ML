"""
Configuration of the pipeline, in three files next to each other (repository root):
- config.toml -- data versions, stages, their inputs and order and the class of every stage (technical),
- config.local.toml -- paths of the data and plots on this machine, merged over config.toml (not versioned),
- settings.toml -- how the pipeline runs, the parameters of the analysis and of every stage and plot (versioned).
"""

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# repository root of the editable install: src/ddtw_ml/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = "config.toml"
LOCAL_CONFIG_FILE = "config.local.toml"
LOCAL_CONFIG_EXAMPLE = "config.local.example.toml"
SETTINGS_FILE = "settings.toml"
# parameters of the analysis shared by several stages ([analysis] in settings.toml), each a list of names
ANALYSIS_KEYS = ("conditions", "methods", "pairs", "metrics")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
# environment variable with the path of an alternative config.toml (its config.local.toml is looked up next to it)
CONFIG_ENV = "DDTW_ML_CONFIG"
RAW = "raw"
VERSION_PATTERN = re.compile(r"^\d+\.\d+$")


class ConfigError(ValueError):
    """
    Invalid or incomplete configuration.
    """


@dataclass(frozen=True)
class Node:
    """
    Stage of the pipeline or plot: what it is computed from and which class computes it.
    """

    key: str
    kind: str  # "stage" or "plot"
    description: str
    inputs: tuple[str, ...]
    target: str  # class of the stage, "package.module.Class"
    path: str

    @property
    def module(self) -> str:
        return self.target.rpartition(".")[0]

    @property
    def class_name(self) -> str:
        return self.target.rpartition(".")[2]


@dataclass(frozen=True)
class Version:
    """
    Data version: a complete run of the pipeline from raw.
    """

    name: str
    description: str
    frozen: bool
    paths: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    data_root: Path
    plots_root: Path
    version: str
    raw_path: str
    spoiled_path: str
    versions: dict[str, Version]
    stages: dict[str, Node]
    plots: dict[str, Node]

    def get_version(self, version: str | None = None) -> Version:
        """
        Method to get a data version.
        :param version: name of the version (default: the active one).
        :return: the version.
        :raise ConfigError: if the version is not described in [versions].
        """
        name = self.version if version is None else version
        if name not in self.versions:
            raise ConfigError(f"Version '{name}' is not described in [versions]. Known: {sorted(self.versions)}.")
        return self.versions[name]

    def raw_dir(self) -> Path:
        """
        Method to get the directory of the raw data (outside the versions).
        :return: path of the raw data.
        """
        return self.data_root / self.raw_path

    def stage_dir(self, key: str, version: str | None = None) -> Path:
        """
        Method to get the directory of a stage in a version.
        :param key: key of the stage ("raw" for the raw data).
        :param version: name of the version (default: the active one).
        :return: path of the stage.
        :raise ConfigError: if the stage is not defined.
        """
        if key == RAW:
            return self.raw_dir()
        if key not in self.stages:
            raise ConfigError(f"Unknown stage '{key}'. Known: {list(self.stages)}.")
        v = self.get_version(version)
        return self.data_root / v.name / v.paths.get(key, self.stages[key].path)

    def spoiled_dir(self, version: str | None = None) -> Path:
        """
        Method to get the directory of the rejected inputs in a version.
        :param version: name of the version (default: the active one).
        :return: path of the rejected inputs.
        """
        v = self.get_version(version)
        return self.data_root / v.name / v.paths.get("spoiled", self.spoiled_path)

    def plot_dir(self, key: str, version: str | None = None) -> Path:
        """
        Method to get the directory of a plot in a version.
        :param key: key of the plot.
        :param version: name of the version (default: the active one).
        :return: path of the plot.
        :raise ConfigError: if the plot is not defined.
        """
        if key not in self.plots:
            raise ConfigError(f"Unknown plot '{key}'. Known: {list(self.plots)}.")
        return self.plots_root / self.get_version(version).name / self.plots[key].path

    def order(self, kind: str = "stage") -> list[str]:
        """
        Method to get the keys of the stages (or plots) in the order of execution: every node after its inputs, ties
        kept in the order of config.toml.
        :param kind: "stage" or "plot".
        :return: ordered keys.
        """
        nodes = self.stages if kind == "stage" else self.plots
        done = set(self.stages) if kind == "plot" else set()
        done.add(RAW)
        ordered: list[str] = []
        pending = list(nodes)
        while pending:
            ready = next(k for k in pending if all(i in done for i in nodes[k].inputs))
            ordered.append(ready)
            done.add(ready)
            pending.remove(ready)
        return ordered


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Function to merge two TOML tables recursively; values of the override win.
    :param base: base table.
    :param override: table merged over the base.
    :return: merged table (new object).
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_file(path: str | Path | None = None) -> Path:
    """
    Function to get the path of config.toml; config.local.toml and settings.toml are looked up next to it.
    :param path: path of config.toml (default: $DDTW_ML_CONFIG, then the repository root).
    :return: path of config.toml.
    """
    return Path(path or os.environ.get(CONFIG_ENV) or REPO_ROOT / CONFIG_FILE)


def read_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Function to read config.toml and merge config.local.toml (if present) over it.
    :param path: path of config.toml (default: $DDTW_ML_CONFIG, then the repository root).
    :return: merged raw configuration.
    :raise ConfigError: if config.toml does not exist.
    """
    config_path = config_file(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    local_path = config_path.with_name(LOCAL_CONFIG_FILE)
    if local_path.is_file():
        with open(local_path, "rb") as f:
            raw = merge(raw, tomllib.load(f))
    return raw


def require(table: dict[str, Any], key: str, where: str) -> Any:
    """
    Function to get a required value of a TOML table.
    :param table: TOML table.
    :param key: key of the value.
    :param where: name of the table, for the error message.
    :return: the value.
    :raise ConfigError: if the value is missing.
    """
    if key not in table:
        raise ConfigError(f"Missing '{key}' in [{where}].")
    return table[key]


def parse_nodes(table: dict[str, Any], kind: str, section: str) -> dict[str, Node]:
    """
    Function to parse the stages or plots of the configuration.
    :param table: [stages] or [plots] table (without plain values such as plots.root).
    :param kind: "stage" or "plot".
    :param section: name of the section, for the error messages.
    :return: nodes by key.
    """
    nodes = {}
    for key, value in table.items():
        if not isinstance(value, dict):
            continue
        where = f"{section}.{key}"
        nodes[key] = Node(key=key, kind=kind, description=value.get("description", ""),
                          inputs=tuple(require(value, "input", where)), target=require(value, "class", where),
                          path=value.get("path", key))
    return nodes


def validate(config: Config) -> None:
    """
    Function to check the consistency of the configuration.
    :param config: parsed configuration.
    :raise ConfigError: on the first inconsistency found.
    """
    for name in config.versions:
        if not VERSION_PATTERN.match(name):
            raise ConfigError(f"Version '{name}' must have the form MAJOR.MINOR, e.g. \"2.0\" (a quoted string).")
        unknown = set(config.versions[name].paths) - set(config.stages) - {"spoiled"}
        if unknown:
            raise ConfigError(f"[versions.\"{name}\".paths] names unknown stages: {sorted(unknown)}.")
    config.get_version()
    for kind, nodes, known in (("stages", config.stages, set(config.stages) | {RAW}),
                               ("plots", config.plots, set(config.stages))):
        for node in nodes.values():
            missing = [i for i in node.inputs if i not in known]
            if missing:
                raise ConfigError(f"[{kind}.{node.key}] has unknown inputs: {missing}.")
    for kind in ("stage", "plot"):
        try:
            config.order(kind)
        except StopIteration:
            raise ConfigError(f"The inputs of the {kind}s form a cycle.") from None


def load_config(path: str | Path | None = None) -> Config:
    """
    Function to load and validate the configuration of the pipeline.
    :param path: path of config.toml (default: $DDTW_ML_CONFIG, then the repository root).
    :return: configuration.
    :raise ConfigError: if the configuration is missing, incomplete or inconsistent.
    """
    raw = read_config(path)
    data = require(raw, "data", "root of config.toml")
    local_path = config_file(path).with_name(LOCAL_CONFIG_FILE)
    if "root" not in data and not local_path.is_file():
        raise ConfigError(f"{local_path} not found: copy {LOCAL_CONFIG_EXAMPLE} to {LOCAL_CONFIG_FILE} and set the "
                          f"paths of the data ([data] root) and plots ([plots] root) on this machine.")
    plots = raw.get("plots", {})
    versions = {name: Version(name=name, description=v.get("description", ""), frozen=bool(v.get("frozen", False)),
                              paths=dict(v.get("paths", {})))
                for name, v in raw.get("versions", {}).items()}
    config = Config(
        data_root=Path(require(data, "root", "data (set it in config.local.toml)")),
        plots_root=Path(require(plots, "root", "plots (set it in config.local.toml)")),
        version=str(require(data, "version", "data")),
        raw_path=require(require(data, "raw", "data"), "path", "data.raw"),
        spoiled_path=data.get("spoiled", {}).get("path", "spoiled"),
        versions=versions,
        stages=parse_nodes(raw.get("stages", {}), "stage", "stages"),
        plots=parse_nodes(plots, "plot", "plots"),
    )
    validate(config)
    return config


@dataclass(frozen=True)
class Settings:
    """
    Settings of the pipeline (settings.toml): how it runs, the parameters of the analysis and the section of every
    stage and plot ({"enabled": bool, parameters of the stage}).
    """

    force: bool
    dry_run: bool
    log_level: str
    analysis: dict[str, list[str]]
    sections: dict[str, dict[str, dict[str, Any]]]  # kind ("stage", "plot") -> key -> section

    def enabled(self, node: Node) -> bool:
        return self.sections[node.kind][node.key]["enabled"]

    def params(self, node: Node) -> dict[str, Any]:
        """
        Method to get the parameters of a stage or plot: its section without "enabled".
        """
        return {k: v for k, v in self.sections[node.kind][node.key].items() if k != "enabled"}


def table(raw: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: missing table [{key}].")
    return value


def check_keys(value: dict[str, Any], allowed: tuple[str, ...] | list[str], where: str) -> None:
    """
    Function to check that a table has exactly the allowed keys.
    :raise ConfigError: if a key is missing or unknown.
    """
    missing = [k for k in allowed if k not in value]
    unknown = sorted(set(value) - set(allowed))
    problems = ([f"missing setting(s) {missing}"] if missing else []) + (
        [f"unknown setting(s) {unknown}"] if unknown else [])
    if problems:
        raise ConfigError(f"{where}: {'; '.join(problems)}; expected: {list(allowed)}.")


def load_settings(config: Config, path: str | Path | None = None) -> Settings:
    """
    Function to load settings.toml and check its structure: [pipeline], [analysis] and one section with "enabled"
    for every stage and plot of config.toml. The parameters of a stage are checked against its class by the runner
    (pipeline.runner.check_settings()).
    :param config: configuration (for the stages and plots).
    :param path: path of config.toml (settings.toml is next to it; default: see config_file()).
    :return: settings.
    :raise ConfigError: if settings.toml is missing, is not valid TOML or has missing, unknown or invalid settings.
    """
    settings_path = config_file(path).with_name(SETTINGS_FILE)
    if not settings_path.is_file():
        raise ConfigError(f"{settings_path} not found; it is part of the repository (git checkout {SETTINGS_FILE}).")
    try:
        with open(settings_path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{settings_path} is not valid TOML: {e}") from None
    where = str(settings_path)
    check_keys(raw, ("pipeline", "analysis", "stages", "plots"), where)

    pipeline = table(raw, "pipeline", where)
    check_keys(pipeline, ("force", "dry_run", "logging"), f"{where} [pipeline]")
    for key in ("force", "dry_run"):
        if not isinstance(pipeline[key], bool):
            raise ConfigError(f"{where} [pipeline]: '{key}' must be true or false, got {pipeline[key]!r}.")
    logging_table = table(pipeline, "logging", f"{where} [pipeline]")
    check_keys(logging_table, ("level",), f"{where} [pipeline.logging]")
    if logging_table["level"] not in LOG_LEVELS:
        raise ConfigError(f"{where} [pipeline.logging]: 'level' must be one of {list(LOG_LEVELS)}, "
                          f"got {logging_table['level']!r}.")

    analysis = table(raw, "analysis", where)
    check_keys(analysis, ANALYSIS_KEYS, f"{where} [analysis]")
    for key in ANALYSIS_KEYS:
        value = analysis[key]
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{where} [analysis]: '{key}' must be a non-empty list of names, got {value!r}.")

    sections: dict[str, dict[str, dict[str, Any]]] = {}
    for kind, name, nodes in (("stage", "stages", config.stages), ("plot", "plots", config.plots)):
        tables = table(raw, name, where)
        missing = [f"{name}.{k}" for k in nodes if k not in tables]
        unknown = [f"{name}.{k}" for k in sorted(set(tables) - set(nodes))]
        problems = ([f"missing section(s) {missing}"] if missing else []) + (
            [f"unknown section(s) {unknown}"] if unknown else [])
        if problems:
            raise ConfigError(f"{where}: {'; '.join(problems)}; the {name} are those of config.toml: {list(nodes)}.")
        for key, section in tables.items():
            if not isinstance(section, dict) or not isinstance(section.get("enabled"), bool):
                raise ConfigError(f"{where} [{name}.{key}]: 'enabled' must be true or false.")
        sections[kind] = tables
    return Settings(force=pipeline["force"], dry_run=pipeline["dry_run"],
                    log_level=logging_table["level"],
                    analysis={k: list(analysis[k]) for k in ANALYSIS_KEYS}, sections=sections)
